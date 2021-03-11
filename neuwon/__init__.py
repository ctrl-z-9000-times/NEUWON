"""
NEUWON is a simulation framework for neuroscience and artificial intelligence
specializing in conductance based models. This software is a modern remake of
the NEURON simulator. It is accurate, efficient, and easy to use.

All units are prefix-less:
* Meters, Grams, Seconds
* Volts, Siemens, Farads
* Liters, Moles, Molar
"""
# Public API Entry Points:
__all__ = """Segment Mechanism Reaction Species Model Geometry Neighbor""".split()
# Numeric/Scientific Library Imports.
import numpy as np
import cupy as cp
import numba.cuda
# Standard Library Imports.
from collections.abc import Callable, Iterable, Mapping
from collections import namedtuple

F = 96485.3321233100184 # Faraday's constant, Coulumbs per Mole of electrons
R = 8.31446261815324 # Universal gas constant
celsius = 37 # Human body temperature
T = celsius + 273.15 # Human body temperature in Kelvins

Real = np.dtype('f8')
epsilon = np.finfo(Real).eps
Location = np.dtype('u4')

def docstring_wrapper(property_name, docstring):
        def get_prop(self):
            return self.__dict__[property_name]
        def set_prop(self, value):
            self.__dict__[property_name] = value
        return property(get_prop, set_prop, None, docstring)

from neuwon.geometry import Neighbor, Geometry
from neuwon.species import Species, Diffusion, Electrics, _init_species
from neuwon.mechanisms import Mechanism, _init_mechansisms
from neuwon.reactions import Reaction
from neuwon.segments import Segment, _serialize_segments

class Model:
    def __init__(self, time_step,
            neurons,
            reactions=(),
            mechanisms=(),
            species=(),
            stagger=True):
        self.time_step = float(time_step)
        self.stagger = bool(stagger)
        coordinates, parents, diameters, insertions = _serialize_segments(self, neurons)
        assert(len(coordinates) > 0)
        self.geometry = Geometry(coordinates, parents, diameters)
        self.reactions = tuple(reactions)
        assert(all(issubclass(r, Reaction) for r in self.reactions))
        self.mechanisms = _init_mechansisms(mechanisms, insertions, self.time_step, self.geometry)
        self.species = _init_species(species, self.time_step, self.geometry, self.reactions, self.mechanisms)
        self.electrics = Electrics(self.time_step, self.geometry)
        self._injected_currents = Model._InjectedCurrents()
        numba.cuda.synchronize()

        # Setup the reaction input & output data structures.
        self.ReactionInputs = namedtuple("ReactionInputs", "v intra extra")
        self.ReactionOutputs = namedtuple("ReactionOutputs", "conductances intra extra")
        self.IntraSpecies = namedtuple("IntraSpecies",
                [n for n, s in self.species.items() if s.intra_diffusivity is not None])
        self.ExtraSpecies = namedtuple("ExtraSpecies",
                [n for n, s in self.species.items() if s.extra_diffusivity is not None])
        self.Conductances = namedtuple("Conductances",
                [n for n, s in self.species.items() if s.transmembrane])

    def _setup_reaction_io(self):
        r_in = self.ReactionInputs(
            v = self.electrics.previous_voltages,
            intra = self.IntraSpecies(**{
                n: s.intra.previous_concentrations
                    for n, s in self.species.items() if s.intra is not None}),
            extra = self.ExtraSpecies(**{
                n: s.extra.previous_concentrations
                    for n, s in self.species.items() if s.extra is not None}))
        r_out = self.ReactionOutputs(
            conductances = self.Conductances(**{
                n: s.conductances
                    for n, s in self.species.items() if s.transmembrane}),
            intra = self.IntraSpecies(**{
                n: s.intra.release_rates
                    for n, s in self.species.items() if s.intra is not None}),
            extra = self.ExtraSpecies(**{
                n: s.extra.release_rates
                    for n, s in self.species.items() if s.extra is not None}))
        for outter in r_out:
            for inner in outter:
                inner.fill(0)
        numba.cuda.synchronize()
        return r_in, r_out

    def __len__(self):
        return len(self.geometry)

    def advance(self):
        # Calculate the externally applied currents.
        self._injected_currents.advance(self.time_step, self.electrics)
        if self.stagger:
            """
            All systems (reactions & mechanisms, diffusions & electrics) are
            integrated using input values from halfway through their time step.
            Tracing through the exact sequence of operations is difficult because
            both systems see the other system as staggered halfway through their
            time step.

            For more information see: The NEURON Book, 2003.
            Chapter 4, Section: Efficient handling of nonlinearity.
            """
            self._diffusions_advance()
            self._reactions_advance()
            self._diffusions_advance()
        else:
            """
            Naive integration strategy, for reference only.
            """
            # Update diffusions & electrics for the whole time step using the
            # state of the reactions at the start of the time step.
            self._diffusions_advance()
            self._diffusions_advance()
            # Update the reactions for the whole time step using the
            # concentrations & voltages from halfway through the time step.
            self._reactions_advance()
        # self._check_data()

    def _check_data(self):
        for mech_type, (locations, instances) in self.mechanisms.items():
            if isinstance(instances, Mapping):
                for key, array in instances.items():
                    assert cp.all(cp.isfinite(array)), (mech_type, key)
            elif instances.dtype.kind in "fc":
                assert cp.all(cp.isfinite(instances)), mech_type
            elif instances.dtype.fields is not None:
                instances = instances.copy_to_host()
                for name in instances.dtype.fields:
                    assert np.all(np.isfinite(instances[name])), (mech_type, name)
        for s in self.species.values():
            if s.transmembrane:
                assert cp.all(cp.isfinite(s.conductances)), s.name
            if s.intra is not None:
                assert cp.all(cp.isfinite(s.intra.concentrations)), s.name
                assert cp.all(cp.isfinite(s.intra.previous_concentrations)), s.name
                assert cp.all(cp.isfinite(s.intra.release_rates)), s.name
            if s.extra is not None:
                assert cp.all(cp.isfinite(s.extra.concentrations)), s.name
                assert cp.all(cp.isfinite(s.extra.previous_concentrations)), s.name
                assert cp.all(cp.isfinite(s.extra.release_rates)), s.name
        assert(cp.all(cp.isfinite(self.electrics.voltages)))
        assert(cp.all(cp.isfinite(self.electrics.previous_voltages)))
        assert(cp.all(cp.isfinite(self.electrics.driving_voltages)))
        assert(cp.all(cp.isfinite(self.electrics.conductances)))

    def _reactions_advance(self):
        dt = self.time_step
        reaction_inputs, reaction_outputs = self._setup_reaction_io()
        for reaction in self.reactions:
            f = reaction.advance_reaction
            for location in range(len(self)):
                f(dt, location, reaction_inputs, reaction_outputs)
            numba.cuda.synchronize()
        for container in self.mechanisms.values():
            container.mechanism.advance(
                    container.locations, container.instances,
                    dt, reaction_inputs, reaction_outputs)
            numba.cuda.synchronize()

    def _diffusions_advance(self):
        """ Note: Each call to this method integrates over half a time step. """
        dt = self.electrics.time_step
        # Save prior state.
        self.electrics.previous_voltages = cp.array(self.electrics.voltages, copy=True)
        for s in self.species.values():
            for x in (s.intra, s.extra):
                if x is not None:
                    x.previous_concentrations = cp.array(x.concentrations, copy=True)
        numba.cuda.synchronize()
        # Accumulate the net conductances and driving voltages from the chemical data.
        self.electrics.conductances.fill(0)     # Zero accumulator.
        self.electrics.driving_voltages.fill(0) # Zero accumulator.
        for s in self.species.values():
            if not s.transmembrane: continue
            s.reversal_potential = s._reversal_potential_method(
                s.intra_concentration if s.intra is None else s.intra.concentrations,
                s.extra_concentration if s.extra is None else s.extra.concentrations,
                self.electrics.voltages)
            self.electrics.conductances += s.conductances
            self.electrics.driving_voltages += s.conductances * s.reversal_potential
            numba.cuda.synchronize()
        self.electrics.driving_voltages /= self.electrics.conductances
        numba.cuda.synchronize()
        self.electrics.driving_voltages = cp.nan_to_num(self.electrics.driving_voltages)
        numba.cuda.synchronize()
        # Calculate the transmembrane currents.
        diff_v = self.electrics.driving_voltages - self.electrics.voltages
        recip_rc = self.electrics.conductances / self.electrics.capacitances
        alpha = cp.exp(-dt * recip_rc)
        self.electrics.voltages += diff_v * (1.0 - alpha)
        numba.cuda.synchronize()
        # Calculate the lateral currents throughout the neurons.
        self.electrics.voltages = self.electrics.irm.dot(self.electrics.voltages)
        # Calculate the transmembrane ion flows.
        numba.cuda.synchronize()
        for s in self.species.values():
            if not s.transmembrane: continue
            if s.intra is None and s.extra is None: continue
            integral_v = dt * (s.reversal_potential - self.electrics.driving_voltages)
            integral_v += rc * diff_v * alpha
            moles = s.conductances * integral_v / (s.charge * F)
            if s.intra is not None:
                s.intra.concentrations += moles / self.geometry.intra_volumes
            if s.extra is not None:
                s.extra.concentrations -= moles / self.geometry.extra_volumes
        # Calculate the local release / removal of chemicals.
        for s in self.species.values():
            for x in (s.intra, s.extra):
                if x is None: continue
                x.concentrations = cp.maximum(0, x.concentrations + x.release_rates * dt)
                # Calculate the lateral diffusion throughout the space.
                x.concentrations = x.irm.dot(x.concentrations)
                numba.cuda.synchronize()

    class _InjectedCurrents:
        def __init__(self):
            self.currents = []
            self.locations = []
            self.remaining = []

        def advance(self, time_step, electrics):
            for idx, (amps, location, t) in enumerate(
                    zip(self.currents, self.locations, self.remaining)):
                dv = amps * min(time_step, t) / electrics.capacitances[location]
                electrics.voltages[location] += dv
                self.remaining[idx] -= time_step
            keep = [t > 0 for t in self.remaining]
            self.currents  = [x for k, x in zip(keep, self.currents) if k]
            self.locations = [x for k, x in zip(keep, self.locations) if k]
            self.remaining = [x for k, x in zip(keep, self.remaining) if k]

    def inject_current(self, location, current = None, duration = 1.4e-3):
        location = int(location)
        assert(location < len(self))
        duration = float(duration)
        assert(duration >= 0)
        if current is None:
            target_voltage = 200e-3
            current = target_voltage * self.electrics.capacitances[location] / duration
        else:
            current = float(current)
        self._injected_currents.currents.append(current)
        self._injected_currents.locations.append(location)
        self._injected_currents.remaining.append(duration)

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
__all__ = """Segment Reaction Pointer Species Model Geometry Neighbor""".split()
# Numeric/Scientific Library Imports.
import numpy as np
import cupy as cp
import numba.cuda
# Standard Library Imports.
from collections.abc import Callable, Iterable, Mapping

from neuwon.common import *
from neuwon.geometry import Neighbor, Geometry
from neuwon.species import Species, Diffusion, Electrics, _init_species
from neuwon.reactions import Reaction, Pointer, _init_reactions
from neuwon.segments import Segment, _serialize_segments

class Model:
    def __init__(self, time_step,
            neurons,
            reactions=(),
            species=(),
            stagger=True):
        self.time_step = float(time_step)
        self.stagger = bool(stagger)
        coordinates, parents, diameters, insertions = _serialize_segments(self, neurons)
        assert(len(coordinates) > 0)
        self.geometry = Geometry(coordinates, parents, diameters)
        self.reactions = _init_reactions(reactions, insertions, self.time_step, self.geometry)
        self.species = _init_species(species, self.time_step, self.geometry, self.reactions)
        self.electrics = Electrics(self.time_step, self.geometry)
        self._injected_currents = Model._InjectedCurrents()

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
        self._check_data()

    def _check_data(self):
        for r in self.reactions.values():
            for ptr_name, array in r.state.items():
                if array.dtype.kind in "fc":
                    assert cp.all(cp.isfinite(array)), (r.name(), ptr_name)
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
        for x in self.species.values():
            if x.transmembrane: x.conductances.fill(0)
            if x.extra: x.extra.release_rates.fill(0)
            if x.intra: x.intra.release_rates.fill(0)
        for container in self.reactions.values():
            args = {}
            for name, ptr in container.pointers.items():
                if ptr.voltage:
                    args[name] = self.electrics.previous_voltages
                    continue
                elif ptr.dtype:
                    args[name] = container.state[name]
                    continue
                species = self.species[ptr.species]
                if ptr.conductance: args[name] = species.conductances
                elif ptr.intra_concentration: args[name] = species.intra.previous_concentrations
                elif ptr.extra_concentration: args[name] = species.extra.previous_concentrations
                elif ptr.intra_release_rate: args[name] = species.intra.previous_release_rates
                elif ptr.extra_release_rate: args[name] = species.extra.previous_release_rates
                else: raise NotImplementedError
            container.reaction.advance(dt, container.locations, **args)

    def _diffusions_advance(self):
        """ Note: Each call to this method integrates over half a time step. """
        dt = self.electrics.time_step
        # Save prior state.
        self.electrics.previous_voltages = cp.array(self.electrics.voltages, copy=True)
        for s in self.species.values():
            for x in (s.intra, s.extra):
                if x is not None:
                    x.previous_concentrations = cp.array(x.concentrations, copy=True)
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
        self.electrics.driving_voltages /= self.electrics.conductances
        self.electrics.driving_voltages = cp.nan_to_num(self.electrics.driving_voltages)
        # Calculate the transmembrane currents.
        diff_v = self.electrics.driving_voltages - self.electrics.voltages
        recip_rc = self.electrics.conductances / self.electrics.capacitances
        alpha = cp.exp(-dt * recip_rc)
        self.electrics.voltages += diff_v * (1.0 - alpha)
        # Calculate the lateral currents throughout the neurons.
        self.electrics.voltages = self.electrics.irm.dot(self.electrics.voltages)
        # Calculate the transmembrane ion flows.
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

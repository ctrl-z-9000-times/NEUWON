import numpy as np
import cupy as cp
from collections.abc import Callable, Iterable, Mapping

from neuwon.common import *
from neuwon.segments import _serialize_segments
from neuwon.geometry import Geometry
from neuwon.species import _AllSpecies, _Electrics
from neuwon.reactions import _AllReactions

# TODO: Consider switching to use NEURON's units? It makes my code a bit more
# complicated, but it should make the users code simpler and more intuitive.

class Model:
    def __init__(self, time_step,
            neurons,
            reactions=(),
            species=(),
            stagger=True,
            intracellular_resistance = 1,
            membrane_capacitance = 1e-2,
            initial_voltage = -70e-3,):
        self.time_step = float(time_step)
        self.stagger = bool(stagger)
        coordinates, parents, diameters, insertions = _serialize_segments(self, neurons)
        assert(len(coordinates) > 0)
        self.geometry = Geometry(coordinates, parents, diameters)
        self._reactions = _AllReactions(reactions, insertions)
        all_pointers = self._reactions.pointers()
        self._species = _AllSpecies(species, self.time_step, self.geometry, all_pointers)
        self._electrics = _Electrics(self.time_step, self.geometry,
            intracellular_resistance, membrane_capacitance, initial_voltage)
        initial_values = {}
        for ptr in all_pointers:
            if ptr.read:
                if ptr.reaction_instance:
                    initial_values[ptr] = 0
                else:
                    initial_values[ptr] = self.read_pointer(ptr, 0)
        self._reactions.bake(self.time_step, self.geometry, initial_values)
        self._injected_currents = Model._InjectedCurrents()

    def __len__(self):
        return len(self.geometry)

    def read_pointer(self, pointer, location=None):
        """ Returns the current value of a pointer.

        If location is not given, then this returns an array containing all
        values in the system, indexed by location. """
        assert(isinstance(pointer, Pointer) and pointer.read)
        if pointer.intra_concentration:
            data = self._species[pointer.species].intra.concentrations
        elif pointer.extra_concentration:
            data = self._species[pointer.species].extra.concentrations
        elif pointer.voltage:
            data = self._electrics.voltages
        elif pointer.reaction_reference:
            reaction_name, pointer_name = pointer.reaction_reference
            data = self._reactions[reaction_name].state[pointer_name]
        elif pointer.reaction_instance: raise ValueError(pointer)
        else: raise NotImplementedError(pointer)
        if location is None: return np.array(data, copy=True)
        else: return data[location]

    def write_pointer(self, pointer, location, value):
        """ Write a new value to a pointer at the given location in the system. """
        assert(isinstance(pointer, Pointer) and pointer.write)
        if pointer.species: species = self._species[pointer.species]
        if pointer.reaction_reference:
            reaction_name, pointer_name = pointer.reaction_reference
            array = self._reactions[reaction_name].state[pointer_name]
            array[location] = value
        elif pointer.conductance:
            # TODO: These updates need to be defered until after the reaction IO
            # has zeroed the writable data buffers.
            1/0
            # species.conductances[location] += value
        elif pointer.intra_release_rate:
            1/0
        elif pointer.extra_release_rate:
            1/0
        else: raise NotImplementedError(pointer)

    def advance(self):
        # Calculate the externally applied currents.
        self._injected_currents.advance(self.time_step, self._electrics)
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
            self._species_advance()
            self._reactions.advance(self)
            self._species_advance()
        else:
            """
            Naive integration strategy, for reference only.
            """
            # Update diffusions & electrics for the whole time step using the
            # state of the reactions at the start of the time step.
            self._species_advance()
            self._species_advance()
            # Update the reactions for the whole time step using the
            # concentrations & voltages from halfway through the time step.
            self._reactions.advance(self)
        self._check_data()

    def _check_data(self):
        self._reactions.check_data()
        self._species.check_data()
        self._electrics.check_data()

    def _species_advance(self):
        """ Note: Each call to this method integrates over half a time step. """
        dt = self._electrics.time_step
        # Save prior state.
        self._electrics.previous_voltages = cp.array(self._electrics.voltages, copy=True)
        for s in self._species.values():
            for x in (s.intra, s.extra):
                if x is not None:
                    x.previous_concentrations = cp.array(x.concentrations, copy=True)
        # Accumulate the net conductances and driving voltages from the chemical data.
        self._electrics.conductances.fill(0)     # Zero accumulator.
        self._electrics.driving_voltages.fill(0) # Zero accumulator.
        for s in self._species.values():
            if not s.transmembrane: continue
            s.reversal_potential = s._reversal_potential_method(
                s.intra_concentration if s.intra is None else s.intra.concentrations,
                s.extra_concentration if s.extra is None else s.extra.concentrations,
                self._electrics.voltages)
            self._electrics.conductances += s.conductances
            self._electrics.driving_voltages += s.conductances * s.reversal_potential
        self._electrics.driving_voltages /= self._electrics.conductances
        self._electrics.driving_voltages = cp.nan_to_num(self._electrics.driving_voltages)
        # Calculate the transmembrane currents.
        diff_v = self._electrics.driving_voltages - self._electrics.voltages
        recip_rc = self._electrics.conductances / self._electrics.capacitances
        alpha = cp.exp(-dt * recip_rc)
        self._electrics.voltages += diff_v * (1.0 - alpha)
        # Calculate the lateral currents throughout the neurons.
        self._electrics.voltages = self._electrics.irm.dot(self._electrics.voltages)
        # Calculate the transmembrane ion flows.
        for s in self._species.values():
            if not s.transmembrane: continue
            if s.intra is None and s.extra is None: continue
            integral_v = dt * (s.reversal_potential - self._electrics.driving_voltages)
            integral_v += rc * diff_v * alpha
            moles = s.conductances * integral_v / (s.charge * F)
            if s.intra is not None:
                s.intra.concentrations += moles / self.geometry.intra_volumes
            if s.extra is not None:
                s.extra.concentrations -= moles / self.geometry.extra_volumes
        # Calculate the local release / removal of chemicals.
        for s in self._species.values():
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
            current = target_voltage * self._electrics.capacitances[location] / duration
        else:
            current = float(current)
        self._injected_currents.currents.append(current)
        self._injected_currents.locations.append(location)
        self._injected_currents.remaining.append(duration)

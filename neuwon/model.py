import numpy as np
import cupy as cp
from collections.abc import Callable, Iterable, Mapping

from neuwon.common import *
from neuwon.segments import _serialize_segments
from neuwon.geometry import _Geometry
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
            celsius = 37,
            intracellular_resistance = 1,
            membrane_capacitance = 1e-2,
            initial_voltage = -70e-3,):
        self.time_step = float(time_step)
        self.stagger = bool(stagger)
        self.celsius = float(celsius)
        coordinates, parents, diameters, insertions = _serialize_segments(self, neurons)
        assert(len(coordinates) > 0)
        self.geometry = _Geometry(coordinates, parents, diameters)
        self._reactions = _AllReactions(reactions, insertions)
        all_pointers = self._reactions.pointers()
        self._species = _AllSpecies(species, self.time_step / 2, self.geometry, all_pointers)
        self._electrics = _Electrics(self.time_step / 2, self.geometry,
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

    def is_root(self, location):
        return self.geometry.is_root(location)

    def nearest_neighbors(self, coordinates, k, maximum_distance=np.inf):
        return self.geometry.nearest_neighbors(coordinates, k, maximum_distance)

    # TODO: rename this to just "read"
    def read_pointer(self, handle, location=None):
        """ Returns the current value of a pointer.

        If location is not given, then this returns an array containing all
        values in the system, indexed by location. """
        assert(isinstance(handle, AccessHandle) and handle.read)
        if handle.reaction_reference:
            reaction_name, handle_name = handle.reaction_reference
            data = self._reactions[reaction_name].state[handle_name]
        elif handle.reaction_instance: raise ValueError("Use reaction_reference in this context!")
        elif handle.intra_concentration:   data = self._species[handle.species].intra.concentrations
        elif handle.extra_concentration:   data = self._species[handle.species].extra.concentrations
        elif handle.voltage:               data = self._electrics.voltages
        elif handle.coordinates:           data = self.geometry.coordinates
        elif handle.diameters:             data = self.geometry.diameters
        elif handle.parents:               data = self.geometry.parents
        elif handle.children:              data = self.geometry.children
        elif handle.surface_areas:         data = self.geometry.surface_areas
        elif handle.cross_sectional_areas: data = self.geometry.cross_sectional_areas
        elif handle.intra_volumes:         data = self.geometry.intra_volumes
        elif handle.extra_volumes:         data = self.geometry.extra_volumes
        elif handle.neighbors:             data = self.geometry.neighbors
        else: raise NotImplementedError(handle)
        if location is None: return np.array(data, copy=True)
        else: return data[location]

    # TODO: rename this to just "write"
    def write_pointer(self, handle, location, value):
        """ Write a new value to a pointer at the given location in the system. """
        assert(isinstance(handle, AccessHandle) and handle.write)
        if handle.species: species = self._species[handle.species]
        if handle.reaction_reference:
            reaction_name, pointer_name = handle.reaction_reference
            array = self._reactions[reaction_name].state[pointer_name]
            array[location] = value
        elif handle.reaction_instance:
            raise ValueError("Use reaction_reference in this context!")
        elif handle.conductance:
            # TODO: These updates need to be defered until after the reaction IO
            # has zeroed the writable data buffers.
            1/0
            # species.conductances[location] += value
        elif handle.intra_release_rate:
            1/0
        elif handle.extra_release_rate:
            1/0
        else: raise NotImplementedError(handle)

    def advance(self):
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
            self._injected_currents.advance(self.time_step, self._electrics)
            self._species.advance(self)
            self._reactions.advance(self)
            self._species.advance(self)
        else:
            """ Naive integration strategy, for reference only. """
            self._injected_currents.advance(self.time_step / 2, self._electrics)
            self._species.advance(self)
            self._injected_currents.advance(self.time_step / 2, self._electrics)
            self._species.advance(self)
            self._reactions.advance(self)

    def check_data(self):
        self._reactions.check_data()
        self._species.check_data()
        self._electrics.check_data()

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

import numpy as np
import cupy as cp
from collections.abc import Callable, Iterable, Mapping

from neuwon.database import *
from neuwon.segments import _serialize_segments
from neuwon.geometry import _Geometry
from neuwon.species import _AllSpecies, _Electrics
from neuwon.reactions import _AllReactions, Reaction

# TODO: Consider switching to use NEURON's units? It makes my code a bit more
# complicated, but it should make the users code simpler and more intuitive.

class Model:
    def __init__(self, time_step, species, reactions,
            celsius = 37,
            intracellular_resistance = 1,
            membrane_capacitance = 1e-2,
            initial_voltage = -70e-3,):
        """
        Argument species is a list of:
          * An instance of the Species class,
          * A dictionary of arguments for initializing a new instance of the Species class,
          * The species name, to be filled in from a standard library.

        Argument reactions is a list of either:
          * An instance or subclass of the Reaction class, or
          * The name of a reaction from the standard library.
        """
        self.db = Database()
        self.db.add_global_constant("time_step", float(time_step))
        self.db.add_global_constant("celsius", float(celsius))

        self.db.add_entity_type("Segment")
        self.db.add_component("Segment/intra", reference="Intracellular")
        self.db.add_component("Segment/extra", reference="Extracellular")
        self.db.add_component("Segment/coordinates", shape=(3,))
        self.db.add_component("Segment/parents", reference="Segment", check=False)
        # self.db.add_component("Location", "children") # TODO: How to deal with sparse matrixes?
        self.db.add_component("Segment/diameters")
        self.db.add_component("Segment/lengths", check=False)
        self.db.add_component("Segment/surface_areas")
        self.db.add_component("Segment/cross_sectional_areas")

        self.db.add_entity_type("Intracellular")
        self.db.add_component("Intracellular/volumes")
        self.db.add_component("Intracellular/segment", reference="Segment")

        self.db.add_entity_type("Extracellular")
        self.db.add_component("Extracellular/volumes")
        # self.db.add_component("Extracellular/neighbors")
        # self.db.add_component("Extracellular/neighbor_distances")
        # self.db.add_component("Extracellular/border_surface_areas")

        self.db.add_global_constant("maximum_extracellular_radius", float(maximum_extracellular_radius))
        self.db.add_global_constant("extracellular_volume_fraction", float(extracellular_volume_fraction))
        self.db.add_global_constant("extracellular_tortuosity", float(extracellular_tortuosity))

        self._reactions = _AllReactions(reactions, self.db)
        self._species = _AllSpecies(species, self.db)

        self.db.add_component("Location", "voltages", initial_value=float(initial_voltage))
        self.db.add_component("Location", "axial_resistances", check=False)
        self.db.add_component("Location", "capacitances")
        self.db.add_global_constant("intracellular_resistance", float(intracellular_resistance))
        self.db.add_global_constant("membrane_capacitance", float(membrane_capacitance))

        self._injected_currents = Model._InjectedCurrents()

    def __len__(self):
        return len(self.geometry)

    def add_segment(self, coordinates, diameter, shells=0):
        1/0

    def insert_reaction(self):
        1/0

    def is_root(self, location):
        return self.geometry.is_root(location)

    def nearest_neighbors(self, coordinates, k, maximum_distance=np.inf):
        return self.geometry.nearest_neighbors(coordinates, k, maximum_distance)

    def read(self, component_name, location=None):
        """

        If argument location is not given then this returns an array containing
        all values in the system. """
        data = self.db.access(component_name)
        if location is None:    return data.get()
        else:                   return data[location]

    def write(self, component_name, location, value):
        """ Write a new value to a pointer at the given location in the system. """
        data = self.db.access(component_name)
        data[location] = value

    def advance(self):
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

    def _advance_lockstep(self):
        """ Naive integration strategy, for reference only. """
        self._injected_currents.advance(self.time_step / 2, self._electrics)
        self._species.advance(self)
        self._injected_currents.advance(self.time_step / 2, self._electrics)
        self._species.advance(self)
        self._reactions.advance(self)

    def check_data(self):
        self.db.check()

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

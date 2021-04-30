import numpy as np
import cupy as cp
from collections.abc import Callable, Iterable, Mapping

from neuwon.database import *
from neuwon.segments import _serialize_segments
from neuwon.geometry import _Geometry
from neuwon.species import _Electrics, species_library
from neuwon.reactions import Reaction, reactions_library

# TODO: Consider switching to use NEURON's units? It makes my code a bit more
# complicated, but it should make the users code simpler and more intuitive.

class Model:
    def __init__(self, time_step,
            celsius = 37,
            intracellular_resistance = 1,
            membrane_capacitance = 1e-2,
            initial_voltage = -70e-3,):
        self.db = Database()
        self.db.add_global_constant("time_step", float(time_step),
            doc="Units: Seconds")
        self.db.add_global_constant("celsius", float(celsius))

        self.db.add_entity_type("membrane", doc="")
        self.db.add_component("membrane/parents", reference="membrane", check=False,
            doc="Cell membranes are connected in a tree.")
        self.db.add_component("membrane/coordinates", shape=(3,),
            doc="Units: ")
        self.db.add_component("membrane/diameters",
            doc="Units: ")
        self.db.add_component("membrane/children", dtype="membrane", sparse=True,
            doc="")
        self.db.add_component("membrane/inside", dtype="inside")
        self.db.add_component("membrane/outside", dtype="outside")
        self.db.add_component("membrane/lengths", check=False,
            doc="Units: ")
        self.db.add_component("membrane/surface_areas",
            doc="Units: ")
        self.db.add_component("membrane/cross_sectional_areas",
            doc="Units: ")
        self.db.add_component("membrane/voltages", initial_value=float(initial_voltage),
            doc="Units: ")
        self.db.add_component("membrane/conductances")
        self.db.add_component("membrane/driving_voltages")
        self.db.add_component("membrane/axial_resistances", check=False,
            doc="Units: ")
        self.db.add_component("membrane/capacitances",
            doc="Units: Farads")
        self.db.add_global_constant("membrane/capacitance", float(membrane_capacitance),
            doc="Units: Farads / Meter^2")
        self.db.add_component("membrane/diffusion", shape="sparse")

        self.db.add_entity_type("inside")
        self.db.add_component("inside/volumes")
        self.db.add_component("inside/membrane", dtype="membrane")
        self.db.add_global_constant("inside/resistance", float(intracellular_resistance),
            doc="Units: ")

        self.db.add_entity_type("outside",
            doc="Extracellular space.")
        self.db.add_global_constant("outside/volume_fraction", float(extracellular_volume_fraction),
            doc="")
        self.db.add_global_constant("outside/tortuosity", float(extracellular_tortuosity),
            doc="")
        self.db.add_global_constant("outside/maximum_radius", float(maximum_extracellular_radius))
        self.db.add_component("outside/volumes")
        self.db.add_component("outside/neighbors", dtype="outside", shape="sparse")
        self.db.add_component("outside/neighbor_distances", shape="sparse")
        self.db.add_component("outside/border_surface_areas", shape="sparse")

        self.reactions = {}
        self.species = {}

        self._injected_currents = Model._InjectedCurrents()

    def add_species(self, species):
        """
        Argument species is a list of:
          * An instance of the Species class,
          * A dictionary of arguments for initializing a new instance of the Species class,
          * The species name, to be filled in from a standard library.
        """
        if isinstance(species, Mapping):
            species = Species(**species)
        elif isinstance(species, str):
            if species in species_library: species = Species(species, **species_library[species])
            else: raise ValueError("Unresolved species: %s."%species)
        else:
            assert(isinstance(species, Species))
        assert(species.name not in self.species)
        self.species[species.name] = species
        if species.intra_diffusivity is not None:
            self.db.add_component("inside/%s/concentrations"%species.name)
            self.db.add_component("inside/%s/release_rates"%species.name, initial_value=0)
            self.db.add_component("inside/%s/diffusion"%species.name, shape="sparse")
        if species.extra_diffusivity is not None:
            self.db.add_component("outside/%s/concentrations"%species.name)
            self.db.add_component("outside/%s/release_rates"%species.name, initial_value=0)
            self.db.add_component("outside/%s/diffusion"%species.name, shape="sparse")
        if species.transmembrane:
            self.db.add_component("membrane/%s/conductances"%species.name, initial_value=0)

    def add_reaction(self, reaction):
        """
        Argument reactions is a list of either:
          * An instance or subclass of the Reaction class, or
          * The name of a reaction from the standard library.
        """
        if not isinstance(r, Reaction) and not (isinstance(r, type) and issubclass(r, Reaction)):
            try: nmodl_file_path, kw_args = reactions_library[str(r)]
            except IndexError: raise ValueError("Unrecognized Reaction: %s."%str(r))
            r = NmodlMechanism(nmodl_file_path, **kw_args)
        if hasattr(r, "initialize"):
            r = copy.deepcopy(r)
            retval = r.initialize(self.db)
            if retval is not None: r = retval
        name = str(r.name())
        assert(name not in self.reactions)
        self.reactions[name] = r

    def __len__(self):
        return len(self.geometry)

    def create_segment(self, parents, coordinates, diameters, shells=0, maximum_segment_length=np.inf):
        1/0
        # TODO: Create "touched" lists for both create & destroy which allow
        # things to be recomputed as needed. Also, how can the user touch
        # segments? For example after changing the diameter?

    def destroy_segment(self, segments):
        1/0

    def insert_reaction(self, reaction, *args, **kwargs):
        1/0

    def remove_reaction(self, reaction, segments):
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

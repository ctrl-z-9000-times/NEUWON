import numpy as np
import cupy as cp
from collections.abc import Callable, Iterable, Mapping

from neuwon.database import *
from neuwon.segments import _serialize_segments
from neuwon.geometry import _Geometry
from neuwon.species import _Electrics, species_library

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
        self._dirty = False

    def __len__(self):
        return len(self.geometry)

    def check_data(self):
        self.db.check()

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
            self.db.add_component("inside/%s/concentrations"%species.name, initial_value=species.intra_concentration)
            self.db.add_component("inside/%s/release_rates"%species.name, initial_value=0)
            self.db.add_component("inside/%s/diffusion"%species.name, shape="sparse")
        if species.extra_diffusivity is not None:
            self.db.add_component("outside/%s/concentrations"%species.name, initial_value=species.extra_concentration)
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

    def create_segment(self, parents, coordinates, diameters, shells=0, maximum_segment_length=np.inf):
        self.dirty = True

        # STEPS:
        #   1) Immediately allocate space for the entities in the database.

        #   2) whats wrong with just updating everything? if some areas get
        #   rebuilt a few extra times durring the init sequence, so what? As
        #   long as it does not repeatedly do the cubic matrix rebuild it should
        #   be fine. Expecially since durring *normal* operation, there will be
        #   a relatively few & batched calls to this.

        #   So before returning, full compute: Geometry (in & out), passive
        #   electric properties

        1/0

    def destroy_segment(self, segments):
        self.dirty = True

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
        if self.dirty: 1/0 # TODO: Recompute all diffusion matrixes.
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

    def _advance_species(model):
        1/0 # TODO: Update this to use the database.
        """ Note: Each call to this method integrates over half a time step. """
        dt = model._electrics.time_step
        # Accumulate the net conductances and driving voltages from the chemical data.
        # model._electrics.conductances     = cp.zeros(len(geometry), dtype=Real)
        # model._electrics.driving_voltages = cp.zeros(len(geometry), dtype=Real)
        model._electrics.conductances.fill(0)     # Zero accumulator.
        model._electrics.driving_voltages.fill(0) # Zero accumulator.
        T = model.celsius + 273.15
        for s in model._species.values():
            if not s.transmembrane: continue
            s.reversal_potential = s._reversal_potential_method(
                T,
                s.intra_concentration if s.intra is None else s.intra.concentrations,
                s.extra_concentration if s.extra is None else s.extra.concentrations,
                model._electrics.voltages)
            model._electrics.conductances += s.conductances
            model._electrics.driving_voltages += s.conductances * s.reversal_potential
        model._electrics.driving_voltages /= model._electrics.conductances
        model._electrics.driving_voltages = cp.nan_to_num(model._electrics.driving_voltages)
        # Calculate the transmembrane currents.
        diff_v = model._electrics.driving_voltages - model._electrics.voltages
        recip_rc = model._electrics.conductances / model._electrics.capacitances
        alpha = cp.exp(-dt * recip_rc)
        model._electrics.voltages += diff_v * (1.0 - alpha)
        # Calculate the lateral currents throughout the neurons.
        model._electrics.voltages = model._electrics.irm.dot(model._electrics.voltages)
        # Calculate the transmembrane ion flows.
        for s in model._species.values():
            if not s.transmembrane: continue
            if s.intra is None and s.extra is None: continue
            integral_v = dt * (s.reversal_potential - model._electrics.driving_voltages)
            integral_v += rc * diff_v * alpha
            moles = s.conductances * integral_v / (s.charge * F)
            if s.intra is not None:
                s.intra.concentrations += moles / model.geometry.intra_volumes
            if s.extra is not None:
                s.extra.concentrations -= moles / model.geometry.extra_volumes
        # Calculate the local release / removal of chemicals.
        for s in model._species.values():
            for x in (s.intra, s.extra):
                if x is None: continue
                x.concentrations = cp.maximum(0, x.concentrations + x.release_rates * dt)
                # Calculate the lateral diffusion throughout the space.
                x.concentrations = x.irm.dot(x.concentrations)

    def _advance_reactions(self):
        1/0 # TODO: Update this to use the database.
        for s in self.species.values():
            if s.transmembrane: s.conductances.fill(0)
            if s.extra: s.extra.release_rates.fill(0)
            if s.intra: s.intra.release_rates.fill(0)
        for r in self.reactions.values():
            r.advance(self.db.access)

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

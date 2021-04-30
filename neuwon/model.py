from collections.abc import Callable, Iterable, Mapping
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
import copy
import cupy as cp
import cupyx.scipy.sparse
import math
import numpy as np
from neuwon import *
from neuwon.database import *
import neuwon.voronoi
Neighbor = neuwon.voronoi.Neighbor


# TODO: Consider switching to use NEURON's units? It makes my code a bit more
# complicated, but it should make the users code simpler and more intuitive.


# TODO: Consider having an intracellular neighbor & border_surface_area?
#       This would replace children?

# TODO: Split Neighbor into three separate properties. Then remove my custom Neighbor dtype.




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
        return self.db.num_entity("membrane")

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
            self.db.add_component("inside/%s/concentrations"%species.name, initial_value=species.intra_concentration,
                doc="Units: Molar")
            self.db.add_component("inside/%s/release_rates"%species.name, initial_value=0,
                doc="Units: Molar / Second")
            self.db.add_component("inside/%s/diffusion"%species.name, shape="sparse")
        if species.extra_diffusivity is not None:
            self.db.add_component("outside/%s/concentrations"%species.name, initial_value=species.extra_concentration,
                doc="Units: Molar")
            self.db.add_component("outside/%s/release_rates"%species.name, initial_value=0,
                doc="Units: Molar / Second")
            self.db.add_component("outside/%s/diffusion"%species.name, shape="sparse")
        if species.transmembrane:
            self.db.add_component("membrane/%s/conductances"%species.name, initial_value=0,
                doc="Units: Siemens")

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
        """ Returns a list of Segments. """
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

        # Note: do NOT return raw/unstable DB Indexes, make a Segment class for
        # each handle to that the user has a *nice* thing to hold onto. I can
        # put lots of convenience methods on the Segment handle...

        1/0

        # TODO: Rework this code!
        # coordinates = tuple(float(x) for x in coordinates)
        # diameter = float(diameter)
        # maximum_segment_length = float(maximum_segment_length)
        # assert(maximum_segment_length > 0)
        # parent = self
        # parent_diameter = self.diameter
        # parent_coordinates = self.coordinates
        # length = np.linalg.norm(np.subtract(parent_coordinates, coordinates))
        # divisions = max(1, math.ceil(length / maximum_segment_length))
        # segments = []
        # for i in range(divisions):
        #     x = (i + 1) / divisions
        #     _x = 1 - x
        #     coords = (  coordinates[0] * x + parent_coordinates[0] * _x,
        #                 coordinates[1] * x + parent_coordinates[1] * _x,
        #                 coordinates[2] * x + parent_coordinates[2] * _x)
        #     diam = diameter * x + parent_diameter * _x
        #     child = Segment(coords, diam, parent)
        #     segments.append(child)
        #     parent = child
        # return segments


    def destroy_segment(self, segments):
        self.dirty = True

        1/0

    def insert_reaction(self, reaction, *args, **kwargs):
        1/0

    def remove_reaction(self, reaction, segments):
        1/0

    def is_root(self, location):
        return self.parents[location] == ROOT

    def nearest_neighbors(self, coordinates, k, maximum_distance=np.inf):
        coordinates = np.array(coordinates, dtype=Real)
        assert(coordinates.shape == (3,))
        assert(all(np.isfinite(x) for x in coordinates))
        k = int(k)
        assert(k >= 1)
        d, i = self._tree.query(coordinates, k, distance_upper_bound=maximum_distance)
        return i


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


class Segment:
    """ This class is returned by model.create_segment() """
    def __init__(self):
        1/0
        # self.parent = parent
        # assert(isinstance(self.parent, Segment) or self.parent is None)
        # self.children = []
        # self.coordinates = tuple(float(x) for x in coordinates)
        # assert(len(self.coordinates) == 3)
        # self.diameter = float(diameter)
        # assert(diameter >= 0)
        # self.insertions = []
        # if self.parent is None:
        #     self.path_length = 0
        # else:
        #     parent.children.append(self)
        #     segment_length = np.linalg.norm(np.subtract(parent.coordinates, self.coordinates))
        #     self.path_length = parent.path_length + segment_length

    def read(self, component):
        1/0

    def get_voltage(self):
        1/0
        assert(self.model is not None)
        return self.model.read_pointer(_v, self.location)

    def inject_current(self, current=None, duration=1e-3):
        1/0
        assert(self.model is not None)
        self.model.inject_current(self.location, current, duration)



# TODO: Merge the Geometry class into the main model.
class _Geometry:
    """ Physical shapes & structures of neurons """
    def __init__(self, coordinates, parents, diameters,
            maximum_extracellular_radius=3e-6,
            extracellular_volume_fraction=.20,
            extracellular_tortuosity=1.55,):
        # Save the arguments.
        self.coordinates = np.array(coordinates, dtype=Real)
        self.parents = np.array([ROOT if p is None else p for p in parents], dtype=Location)
        self.diameters = np.array(diameters, dtype=Real)
        self.maximum_extracellular_radius = float(maximum_extracellular_radius)
        self.extracellular_volume_fraction = float(extracellular_volume_fraction)
        self.extracellular_tortuosity = float(extracellular_tortuosity)
        # Check the arguments.
        assert(len(self.coordinates) == len(self))
        assert(len(self.parents)     == len(self))
        assert(len(self.diameters)   == len(self))
        assert(all(all(np.isfinite(c)) for c in self.coordinates))
        assert(all(p < len(self) or p == ROOT for p in self.parents))
        assert(all(d >= 0 for d in self.diameters))
        assert(self.maximum_extracellular_radius > epsilon * 1e-6)
        assert(1 >= self.extracellular_volume_fraction >= 0)
        assert(self.extracellular_tortuosity >= 1)
        # Initialize the geometric properties.
        self._init_tree_properties()
        self._init_cellular_properties()
        self._init_extracellular_properties()

    def _init_tree_properties(self):
        # Compute the children lists.
        self.children = np.empty(len(self), dtype=object)
        for location in range(len(self)):
            self.children[location] = []
        for location, parent in enumerate(self.parents):
            if not self.is_root(location):
                self.children[parent].append(location)
        # Root must have at least one child, because cylinder is defined as between two points.
        assert(all(len(self.children[x]) >= 1 for x in range(len(self)) if self.is_root(x)))
        # The child with the largest diameter is special and is always kept at
        # the start of the children list.
        for siblings in self.children:
            siblings.sort(reverse=True, key=lambda x: self.diameters[x])
        # Compute lengths, which are the distances between each node and its
        # parent node. All root node lengths are NAN.
        self.lengths = np.empty(len(self), dtype=Real)
        for location in range(len(self)):
            if self.is_root(location):
                self.lengths[location] = np.nan
            else:
                self.lengths[location] = np.linalg.norm(
                    self.coordinates[location] - self.coordinates[self.parents[location]])
        assert(all(l >= epsilon * (1e-6)**1 or self.is_root(idx) for idx, l in enumerate(self.lengths)))

    def _init_cellular_properties(self):
        self.cross_sectional_areas = np.array([np.pi * (d / 2) ** 2 for d in self.diameters], dtype=Real)
        self.surface_areas = np.empty(len(self), dtype=Real)
        self.intra_volumes = np.empty(len(self), dtype=Real)
        for location, parent in enumerate(self.parents):
            radius = self.diameters[location] / 2
            if self.is_root(location):
                # Root of new tree. The body of this segment is half of the
                # cylinder spanning between this node and its first child.
                eldest = self.children[location][0]
                length = self.diameters[eldest] / 2
            elif self.is_root(parent) and self.children[parent][0] == location:
                length = self.lengths[location] / 2
            else:
                length = self.lengths[location]
            # Primary segments are straightforward extensions of the parent
            # branch. Non-primary segments are lateral branchs off to the side
            # of the parent branch. Subtract the parent's radius from the
            # secondary nodes length, to avoid excessive overlap between
            # segments.
            if self.is_root(location):
                primary = True
            else:
                siblings = self.children[parent]
                if siblings[0] == location or (self.is_root(parent) and siblings[1] == location):
                    primary = True
                else:
                    primary = False
            if not primary:
                parent_radius = self.diameters[parent] / 2
                if length > parent_radius + epsilon * 1e-6:
                    length -= parent_radius
                else:
                    # This segment is entirely enveloped within its parent. In
                    # this corner case allow the segment to protrude directly
                    # from the center of the parent instead of the surface.
                    pass
            self.surface_areas[location] = 2 * np.pi * radius * length
            self.intra_volumes[location] = np.pi * radius ** 2 * length * 1e3
            # Account for the surface area on the tips of terminal/leaf segments.
            num_children = len(self.children[location])
            if num_children == 0 or (self.is_root(location) and num_children == 1):
                self.surface_areas[location] += np.pi * radius ** 2
        assert(all(x  >= epsilon * (1e-6)**2 for x in self.cross_sectional_areas))
        assert(all(sa >= epsilon * (1e-6)**2 for sa in self.surface_areas))
        assert(all(v  >= epsilon * (1e-6)**3 for v in self.intra_volumes))

    def _init_extracellular_properties(self):
        # TODO: Consider https://en.wikipedia.org/wiki/Power_diagram
        self._tree = scipy.spatial.cKDTree(self.coordinates)
        self.extra_volumes = np.empty(len(self), dtype=Real)
        self.neighbors = np.zeros(len(self), dtype=object)
        for location in range(len(self)):
            coords = self.coordinates[location]
            max_dist = self.maximum_extracellular_radius + self.diameters[location] / 2
            neighbors = self._tree.query_ball_point(coords, 2 * max_dist)
            neighbors.remove(location)
            neighbors = np.array(neighbors, dtype=Location)
            v, n = neuwon.voronoi.voronoi_cell(location, max_dist,
                    neighbors, self.coordinates)
            self.extra_volumes[location] = v * self.extracellular_volume_fraction * 1e3
            self.neighbors[location] = n
            for n in self.neighbors[location]:
                n["distance"] = np.linalg.norm(coords - self.coordinates[n["location"]])
        # TODO: Cast neighbors from list of lists to a sparse array.



F = 96485.3321233100184 # Faraday's constant, Coulombs per Mole of electrons
R = 8.31446261815324 # Universal gas constant

class _Diffusion:
    def __init__(self, time_step, geometry, species, where):
        self.time_step = time_step
        # Compute the coefficients of the derivative function:
        # dX/dt = C * X, where C is Coefficients matrix and X is state vector.
        cols = [] # Source
        rows = [] # Destintation
        data = [] # Weight
        # derivative(Destintation) += Source * Weight
        if where == "intracellular":
            for location in range(len(geometry)):
                if geometry.is_root(location):
                    continue
                parent = geometry.parents[location]
                l = geometry.lengths[location]
                flux = species.intra_diffusivity * geometry.cross_sectional_areas[location] / l
                cols.append(location)
                rows.append(parent)
                data.append(+1 * flux / geometry.intra_volumes[parent])
                cols.append(location)
                rows.append(location)
                data.append(-1 * flux / geometry.intra_volumes[location])
                cols.append(parent)
                rows.append(location)
                data.append(+1 * flux / geometry.intra_volumes[location])
                cols.append(parent)
                rows.append(parent)
                data.append(-1 * flux / geometry.intra_volumes[parent])
            for location in range(len(geometry)):
                cols.append(location)
                rows.append(location)
                data.append(-1 / species.intra_decay_period)
        elif where == "extracellular":
            D = species.extra_diffusivity / geometry.extracellular_tortuosity ** 2
            for location in range(len(geometry)):
                for neighbor in geometry.neighbors[location]:
                    flux = D * neighbor["border_surface_area"] / neighbor["distance"]
                    cols.append(location)
                    rows.append(neighbor["location"])
                    data.append(+1 * flux / geometry.extra_volumes[neighbor["location"]])
                    cols.append(location)
                    rows.append(location)
                    data.append(-1 * flux / geometry.extra_volumes[location])
            for location in range(len(geometry)):
                cols.append(location)
                rows.append(location)
                data.append(-1 / species.extra_decay_period)
        # Note: always use double precision floating point for building the impulse response matrix.
        coefficients = csc_matrix((data, (rows, cols)), shape=(len(geometry), len(geometry)), dtype=float)
        coefficients.data *= self.time_step
        self.irm = expm(coefficients)
        # Prune the impulse response matrix at epsilon nanomolar (mol/L).
        self.irm.data[np.abs(self.irm.data) < epsilon * 1e-6] = 0
        self.irm.eliminate_zeros()
        if True: print(where, species.name, "IRM NNZ per Location", self.irm.nnz / len(geometry))
        self.irm = cupyx.scipy.sparse.csr_matrix(self.irm, dtype=Real)

def nerst_potential(charge, T, intra_concentration, extra_concentration):
    """ Returns the reversal voltage for an ionic species. """
    xp = cp.get_array_module(intra_concentration)
    if charge == 0: return xp.full_like(intra_concentration, xp.nan)
    ratio = xp.divide(extra_concentration, intra_concentration)
    return xp.nan_to_num(R * T / F / charge * np.log(ratio))

@cp.fuse()
def _efun(z):
    if abs(z) < 1e-4:
        return 1 - z / 2
    else:
        return z / (math.exp(z) - 1)

def goldman_hodgkin_katz(charge, T, intra_concentration, extra_concentration, voltages):
    """ Returns the reversal voltage for an ionic species. """
    xp = cp.get_array_module(intra_concentration)
    if charge == 0: return xp.full_like(intra_concentration, np.nan)
    z = (charge * F / (R * T)) * voltages
    return (charge * F) * (intra_concentration * _efun(-z) - extra_concentration * _efun(z))

class _Electrics:
    def __init__(self, time_step, geometry,
            intracellular_resistance = 1,
            membrane_capacitance = 1e-2,
            initial_voltage = -70e-3):
        # Save and check the arguments.
        self.time_step                  = time_step
        self.intracellular_resistance   = float(intracellular_resistance)
        self.membrane_capacitance       = float(membrane_capacitance)
        assert(self.intracellular_resistance > 0)
        assert(self.membrane_capacitance > 0)
        # Initialize data buffers.
        self.voltages           = cp.full(len(geometry), initial_voltage, dtype=Real)
        # Compute passive properties.
        self.axial_resistances  = np.empty(len(geometry), dtype=Real)
        self.capacitances       = np.empty(len(geometry), dtype=Real)
        for location in range(len(geometry)):
            l = geometry.lengths[location]
            sa = geometry.surface_areas[location]
            xa = geometry.cross_sectional_areas[location]
            self.axial_resistances[location] = self.intracellular_resistance * l / xa
            self.capacitances[location] = self.membrane_capacitance * sa
        # Compute the coefficients of the derivative function:
        # dX/dt = C * X, where C is Coefficients matrix and X is state vector.
        cols = [] # Source
        rows = [] # Destintation
        data = [] # Weight
        for location in range(len(geometry)):
            if geometry.is_root(location):
                continue
            parent = geometry.parents[location]
            r = self.axial_resistances[location]
            cols.append(location)
            rows.append(parent)
            data.append(+1 / r / self.capacitances[parent])
            cols.append(location)
            rows.append(location)
            data.append(-1 / r / self.capacitances[location])
            cols.append(parent)
            rows.append(location)
            data.append(+1 / r / self.capacitances[location])
            cols.append(parent)
            rows.append(parent)
            data.append(-1 / r / self.capacitances[parent])
        # Note: always use double precision floating point for building the impulse response matrix.
        coefficients = csc_matrix((data, (rows, cols)), shape=(len(geometry), len(geometry)), dtype=np.float64)
        coefficients.data *= self.time_step
        self.irm = expm(coefficients)
        # Prune the impulse response matrix at epsilon millivolts.
        self.irm.data[np.abs(self.irm.data) < epsilon * 1e-3] = 0
        self.irm.eliminate_zeros()
        if True: print("Electrics IRM NNZ per Location", self.irm.nnz / len(geometry))
        # Move this data to the GPU now that the CPU is done with it.
        self.irm = cupyx.scipy.sparse.csr_matrix(self.irm, dtype=Real)

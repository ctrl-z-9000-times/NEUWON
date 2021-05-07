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

F = 96485.3321233100184 # Faraday's constant, Coulombs per Mole of electrons
R = 8.31446261815324 # Universal gas constant

# TODO: Consider switching to use NEURON's units? It makes my code a bit more
# complicated, but it should make the users code simpler and more intuitive.

# TODO: Rename intra & extra into inside & outside

Neighbor = np.dtype([
    ("distance", Real),
    ("border_surface_area", Real),
])

class Model:
    def __init__(self, time_step,
            celsius = 37,
            maximum_extracellular_radius=3e-6,
            extracellular_volume_fraction=.20,
            extracellular_tortuosity=1.55,
            intracellular_resistance = 1,
            membrane_capacitance = 1e-2,
            initial_voltage = -70e-3,):
        assert(maximum_extracellular_radius > epsilon * 1e-6)
        assert(1 >= extracellular_volume_fraction >= 0)
        assert(extracellular_tortuosity >= 1)
        assert(intracellular_resistance > 0)
        assert(membrane_capacitance > 0)
        self.species = {}
        self.reactions = {}
        self._injected_currents = Model._InjectedCurrents()
        self.db = db = Database()
        db.add_global_constant("time_step", float(time_step), doc="Units: Seconds")
        db.add_global_constant("celsius", float(celsius))
        db.add_function("create_segment", self.create_segment)
        db.add_function("destroy_segment", self.destroy_segment)
        db.add_function("insert_reaction", self.insert_reaction)
        db.add_function("remove_reaction", self.remove_reaction)
        # Basic entities and their relations.
        db.add_entity_type("membrane", doc="")
        db.add_attribute("membrane/parents", dtype="membrane", check=False,
            doc="Cell membranes are connected in a tree.")
        db.add_attribute("membrane/children", dtype="membrane", shape="sparse", doc="")
        db.add_entity_type("inside", doc="Intracellular space.")
        db.add_attribute("membrane/inside", dtype="inside",
                doc="""A reference to the outermost shell.
                The shells and the innermost core are allocated in a contiguous block
                with this referencing the start of range of length "membrane/shells" + 1.
                """)
        db.add_attribute("membrane/shells", dtype=np.uint8)
        db.add_attribute("inside/membrane", dtype="membrane")
        db.add_attribute("inside/shell_radius")
        # Geometric properties.
        db.add_attribute("membrane/coordinates", shape=(3,), doc="Units: ")
        db.add_attribute("membrane/diameters", doc="Units: ")
        db.add_attribute("membrane/shape", dtype=np.bool, doc="True for Frustum, False for Cylinder.")
        db.add_attribute("membrane/primary", dtype=np.bool)
        db.add_attribute("membrane/lengths", check=False, doc="Units: Meters")
        db.add_attribute("membrane/surface_areas", doc="Units: Meters ^ 2")
        db.add_attribute("membrane/cross_sectional_areas", doc="Units: Meters ^ 2")
        db.add_attribute("inside/volumes", doc="Units: Litres")
        db.add_attribute("inside/neighbors", dtype=Neighbor, shape="sparse")
        # Extracellular space properties.
        db.add_entity_type("outside", doc="Extracellular space using a voronoi diagram.")
        db.add_attribute("membrane/outside", dtype="outside", doc="")
        db.add_attribute("outside/coordinates", shape=(3,), doc="Units: ")
        db.add_kd_tree(  "outside/tree", "outside/coordinates")
        db.add_attribute("outside/volumes", doc="Units: Litres")
        db.add_attribute("outside/neighbors", dtype=Neighbor, shape="sparse")
        db.add_global_constant("outside/volume_fraction", float(extracellular_volume_fraction))
        db.add_global_constant("outside/tortuosity", float(extracellular_tortuosity))
        db.add_global_constant("outside/maximum_radius", float(maximum_extracellular_radius))
        # Electric properties.
        db.add_global_constant("inside/resistance", float(intracellular_resistance), doc="Units: ")
        db.add_global_constant("membrane/capacitance", float(membrane_capacitance), doc="Units: Farads / Meter^2")
        db.add_attribute("membrane/voltages", initial_value=float(initial_voltage))
        db.add_attribute("membrane/axial_resistances", check=False, doc="Units: ")
        db.add_attribute("membrane/capacitances", doc="Units: Farads")
        db.add_attribute("membrane/conductances")
        db.add_attribute("membrane/driving_voltages")
        db.add_linear_system("membrane/diffusion", function=_electric_coefficients,
            epsilon=epsilon * 1e-3,) # Epsilon millivolts.

        # TODO: Merge these checks into the database.
        assert(cp.all(access("membrane/diameters")[membrane] >= 0.0))
        assert(all(l  >= epsilon * (1e-6)**1 or self.is_root(idx) for idx, l in enumerate(self.lengths)))
        assert(all(x  >= epsilon * (1e-6)**2 for x in self.cross_sectional_areas))
        assert(all(sa >= epsilon * (1e-6)**2 for sa in self.surface_areas))
        assert(all(v  >= epsilon * (1e-6)**3 for v in self.intra_volumes))

    def __len__(self):
        return self.db.num_entity("membrane")

    def check_data(self):
        self.db.check()

    def add_species(self, species):
        """
        Argument species is one of:
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
        add_attribute = self.db.add_attribute
        if species.intra_diffusivity is not None:
            add_attribute("inside/%s/concentrations"%species.name,
                    initial_value=species.intra_concentration,
                    doc="Units: Molar")
            add_attribute("inside/%s/release_rates"%species.name,
                    initial_value=0,
                    doc="Units: Molar / Second")
            add_attribute("inside/%s/diffusion"%species.name, shape="sparse")
        if species.extra_diffusivity is not None:
            add_attribute("outside/%s/concentrations"%species.name,
                    initial_value=species.extra_concentration,
                    doc="Units: Molar")
            add_attribute("outside/%s/release_rates"%species.name,
                    initial_value=0,
                    doc="Units: Molar / Second")
            add_attribute("outside/%s/diffusion"%species.name, shape="sparse")
        if species.transmembrane:
            add_attribute("membrane/%s/conductances"%species.name,
                    initial_value=0,
                    doc="Units: Siemens")

    def add_reaction(self, reaction):
        """
        Argument reactions is one of:
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

    # TODO: Rework this code!
    # Special case to break long segments into multiple shorter ones.
    # 
    # maximum_segment_length = float(maximum_segment_length)
    # assert(maximum_segment_length > 0)
    # if maximum_segment_length != np.inf: ???
    # coordinates = tuple(float(x) for x in coordinates)
    # diameter = float(diameter)
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

    def create_segment(self, parents, coordinates, diameters, shape="cylinder", shells=0):
        """
        Argument parents:
        Argument coordinates:
        Argument diameters:
        Argument shape: either "cylinder" or "frustum".
        Argument shells: unimplemented.

        Returns a list of Segments.
        """
        if not isinstance(parents, Iterable):
            parents     = [parents]
            coordinates = [coordinates]
            diameters   = [diameters]
        num_new_segs = len(parents)
        parents_clean = np.empty(num_new_segs, dtype=Index)
        for idx, p in enumerate(parents):
            parents_clean[idx] = NULL if p is None else p
        assert(shape in ("cylinder", "frustum"))
        membrane = self.db.create_entity("membrane", num_new_segs)
        inside   = self.db.create_entity("inside", num_new_segs * (shells + 1))
        outside  = self.db.create_entity("outside", num_new_segs)
        access   = self.db.access
        access("membrane/parents")[membrane]     = parents_clean
        access("membrane/coordinates")[membrane] = coordinates
        access("membrane/diameters")[membrane]   = diameters
        access("membrane/shape")[membrane]       = shape == "frustum"
        access("membrane/shells")[membrane]      = shells
        access("membrane/inside")[membrane]      = inside[slice(None,None,shells + 1)]
        access("inside/membrane")[inside]        = cp.repeat(membrane, shells + 1)
        shell_radius = [1.0] # TODO
        access("inside/shell_radius")[inside]    = cp.tile(shell_radius, membrane)
        access("membrane/outside")[membrane]     = outside

        1/0 # Update children here!
        children = access("membrane/children")
        self.db.sparse_matrix_write_rows(
            rows = parents,
            cols = [],
            )

        self._initialize_membrane(membrane)
        self._initialize_membrane(parents)
        self._initialize_outside(outside)
        1/0 # TODO: Also re-initialize all of the neighbors of new outside points.

        # Note: do NOT return raw/unstable DB Indexes, make a Segment class for
        # each handle to that the user has a *nice* thing to hold onto. I can
        # put lots of convenience methods on the Segment handle...
        return membrane

    def destroy_segment(self, segments):
        """ """
        1/0

    def _initialize_membrane(self, locations):
        # Compute lengths, which are the distances between each node and its
        # parent node. All root node lengths are NAN.
        for location in membrane:
            if self.is_root(location):
                self.lengths[location] = np.nan
            else:
                self.lengths[location] = np.linalg.norm(
                    self.coordinates[location] - self.coordinates[self.parents[location]])

        xa = access("membrane/cross_sectional_areas")
        if shape:
            xa[membrane] = 1/0 # Frustum.
        else:
            xa[membrane] = np.array([np.pi * (d / 2) ** 2 for d in diameters], dtype=Real)


        access("membrane/primary")[membrane] = 1/0

        # The child with the largest diameter is special and is always kept at
        # the start of the children list.
        for siblings in self.children:
            siblings.sort(reverse=True, key=lambda x: self.diameters[x])

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

        # Passive electric properties.
        l  = self.db.access("membrane/lengths")
        sa = self.db.access("membrane/surface_areas")
        xa = self.db.access("membrane/cross_sectional_areas")
        R  = self.db.access("inside/resistance")
        rx = self.db.access("membrane/axial_resistances")
        C_ = self.db.access("membrane/capacitance")
        c  = self.db.access("membrane/capacitances")
        rx[membrane] = R * l[membrane] / xa[membrane]
        c[membrane] = C_ * sa[membrane]

    def _initialize_outside(self, locations):
        # TODO: Update this code to use the database. Also consider: how will it
        # get the coordinates? Does this use "membrane/coordinates"? if so then
        # it can't really have extra tracking points which are not associated
        # with the membrane... Or should the extracellular stuff have its own
        # system, complete with its own coordinates?
        1/0

        tree = self.db.access("outside/tree")
        # TODO: Consider https://en.wikipedia.org/wiki/Power_diagram
        for location in locations:
            coords = self.coordinates[location]
            max_dist = self.maximum_extracellular_radius + self.diameters[location] / 2
            neighbors = tree.query_ball_point(coords, 2 * max_dist)
            neighbors.remove(location)
            neighbors = np.array(neighbors, dtype=Index)
            v, n = neuwon.voronoi.voronoi_cell(location, max_dist,
                    neighbors, self.coordinates)
            self.extra_volumes[location] = v * self.extracellular_volume_fraction * 1e3
            self.neighbors[location] = n
            for n in self.neighbors[location]:
                n["distance"] = np.linalg.norm(coords - self.coordinates[n["location"]])

    def insert_reaction(self, reaction, *args, **kwargs):
        r = self.reactions[str(reaction)]
        r.initialize(segment, *args, **kw_args)

    def remove_reaction(self, reaction, segments):
        1/0

    def is_root(self, locations):
        return self.db.access("membrane/parents")[locations] == NULL

    def nearest_neighbors(self, coordinates, k, maximum_distance=np.inf):
        coordinates = np.array(coordinates, dtype=Real)
        assert(coordinates.shape == (3,))
        assert(all(np.isfinite(x) for x in coordinates))
        k = int(k)
        assert(k >= 1)
        d, i = self._tree.query(coordinates, k, distance_upper_bound=maximum_distance)
        return i

    def access(self, component_name):
        return self.db.access(component_name)

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

    def _advance_species(self):
        1/0 # TODO: Update this to use the database.
        """ Note: Each call to this method integrates over half a time step. """
        access = self.db.access
        dt = access("time_step") / 2
        # Accumulate the net conductances and driving voltages from the chemical data.
        access("membrane/conductances").fill(0.0)     # Zero accumulator.
        access("membrane/driving_voltages").fill(0.0) # Zero accumulator.
        T = access("celsius") + 273.15
        for s in self._species.values():
            if not s.transmembrane: continue
            # TODO: Rework this loop so that it does not need to save a private lambda method.
            #       Just put an if-switch just here, its not that expensive...
            s.reversal_potential = s._reversal_potential_method(
                T,
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

    def _advance_reactions(self):
        access = self.db.access
        for name, species in self.species.items():
            if species.transmembrane:
                access("membrane/%s/conductances"%species).fill(0.0)
            if species.intra_diffusivity is not None:
                access("inside/%s/release_rates"%species).fill(0.0)
            if species.extra_diffusivity is not None:
                access("outside/%s/release_rates"%species).fill(0.0)
        for r in self.reactions.values():
            r.advance(access)

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

    def inject_current(self, location, current, duration = 1.4e-3):
        location = int(location)
        assert(location < len(self))
        duration = float(duration)
        assert(duration >= 0)
        current = float(current)
        self._injected_currents.currents.append(current)
        self._injected_currents.locations.append(location)
        self._injected_currents.remaining.append(duration)

class Segment:
    """ This class is returned by model.create_segment() """
    def __init__(self, model, membrane_index):
        self.model = model
        self.entity = Entity(model.db, "membrane", membrane_index)

    @property
    def parent(self):
        parent = self.entity.read("membrane/parents")
        if parent != NULL:  return Segment(self.model, parent)
        else:               return None

    @property
    def children(self):
        children = self.entity.read("membrane/children")
        return [Segment(self.model, c) for c in children]

    @property
    def coordinates(self):
        return self.entity.read("membrane/coordinates")

    @property
    def diameter(self):
        return self.entity.read("membrane/diameters")

    def read(self, component):
        return self.entity.read(component)

    def write(self, component, value):
        return self.entity.write(component, value)

    @property
    def voltage(self):
        return self.entity.read("membrane/voltages")

    def inject_current(self, current, duration=1e-3):
        self.model.inject_current(self.entity.index, current, duration)

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

def _electric_coefficients(database_access):
    """
    Model the electric currents over the membrane surface.
    Compute the coefficients of the derivative function:
    dV/dt = C * V, where C is Coefficients matrix and V is voltage vector.
    """
    dt           = database_access("time_step")
    parents      = database_access("membrane/parents").get()
    resistances  = database_access("membrane/axial_resistances").get()
    capacitances = database_access("membrane/capacitances").get()
    src = []; dst = []; coef = []
    for child, parent in enumerate(parents):
        if parent == NULL: continue
        r        = resistances[child]
        c_parent = capacitances[parent]
        c_child  = capacitances[child]
        src.append(child)
        dst.append(parent)
        coef.append(+dt / (r * c_parent))
        src.append(child)
        dst.append(child)
        coef.append(-dt / (r * c_child))
        src.append(parent)
        dst.append(child)
        coef.append(+dt / (r * c_child))
        src.append(parent)
        dst.append(parent)
        coef.append(-dt / (r * c_parent))
    return (coef, (dst, src))

def _inside_diffusion_coefficients(database_access, species):
    src = []; dst = []; coef = []
    for location in range(len(geometry)):
        if geometry.is_root(location):
            continue
        parent = geometry.parents[location]
        l = geometry.lengths[location]
        flux = species.intra_diffusivity * geometry.cross_sectional_areas[location] / l
        src.append(location)
        dst.append(parent)
        coef.append(+dt * flux / geometry.intra_volumes[parent])
        src.append(location)
        dst.append(location)
        coef.append(-dt * flux / geometry.intra_volumes[location])
        src.append(parent)
        dst.append(location)
        coef.append(+dt * flux / geometry.intra_volumes[location])
        src.append(parent)
        dst.append(parent)
        coef.append(-dt * flux / geometry.intra_volumes[parent])
    for location in range(len(geometry)):
        src.append(location)
        dst.append(location)
        coef.append(-dt / species.intra_decay_period)
    return (coef, (dst, src))

def _outside_diffusion_coefficients(database_access, species):
    src = []; dst = []; coef = []
    D = species.extra_diffusivity / geometry.extracellular_tortuosity ** 2
    for location in range(len(geometry)):
        for neighbor in geometry.neighbors[location]:
            flux = D * neighbor["border_surface_area"] / neighbor["distance"]
            src.append(location)
            dst.append(neighbor["location"])
            coef.append(+dt * flux / geometry.extra_volumes[neighbor["location"]])
            src.append(location)
            dst.append(location)
            coef.append(-dt * flux / geometry.extra_volumes[location])
    for location in range(len(geometry)):
        src.append(location)
        dst.append(location)
        coef.append(-dt / species.extra_decay_period)
    return (coef, (dst, src))

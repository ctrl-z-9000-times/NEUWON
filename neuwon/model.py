from collections.abc import Callable, Iterable, Mapping
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
import copy
import numba.cuda
import cupy as cp
import math
import numpy as np
from neuwon.database import *
import neuwon.voronoi

# TODO: Consider switching to use NEURON's units? It makes my code a bit more
# complicated, but it should make the users code simpler and more intuitive.
# Also, if I do this then I can get rid of a lot of nmodl shims...

F = 96485.3321233100184 # Faraday's constant, Coulombs per Mole of electrons
R = 8.31446261815324 # Universal gas constant

Neighbor = np.dtype([
    ("distance", Real),
    ("border_surface_area", Real),
])

class Species:
    """ """

    # TODO: Consider getting rid of the standard library of species and mechanisms.
    # Instead provide it in code examples which the user can copy paste into their
    # code, or possible import directly from an "examples" sub-module (like with
    # htm.core: `import htm.examples`). The problem with this std-lib is that there
    # is no real consensus on what's standard? Species have a lot of arguments and
    # while there may be one scientifically correct value for each argument, the
    # user might want to omit options for run-speed. Mechanisms come in so many
    # different flavors too, with varying levels of bio-accuracy vs run-speed.
    _library = {
        "na": {
            "charge": 1,
            "transmembrane": True,
            "reversal_potential": "nerst",
            "inside_concentration":  15e-3,
            "outside_concentration": 145e-3,
        },
        "k": {
            "charge": 1,
            "transmembrane": True,
            "reversal_potential": "nerst",
            "inside_concentration": 150e-3,
            "outside_concentration":   4e-3,
        },
        "ca": {
            "charge": 2,
            "transmembrane": True,
            "reversal_potential": "goldman_hodgkin_katz",
            "inside_concentration": 70e-9,
            "outside_concentration": 2e-3,
        },
        "cl": {
            "charge": -1,
            "transmembrane": True,
            "reversal_potential": "nerst",
            "inside_concentration":  10e-3,
            "outside_concentration": 110e-3,
        },
        "glu": {
            # "outside_concentration": 1/0, # TODO!
            "outside_diffusivity": 1e-6, # TODO!
            # "outside_decay_period": 1/0, # TODO!
        },
    }

    def __init__(self, name,
            charge = 0,
            transmembrane = False,
            reversal_potential = "nerst",
            inside_concentration = 0.0,
            outside_concentration = 0.0,
            inside_diffusivity = None,
            outside_diffusivity = None,
            inside_decay_period = float("inf"),
            outside_decay_period = float("inf"),
            use_shells = False,
            use_grid = None,):
        """
        If diffusivity is not given, then the concentration is constant.
        Argument reversal_potential is one of: number, "nerst", "goldman_hodgkin_katz"
        """
        self.name = str(name)
        self.charge = int(charge)
        self.transmembrane = bool(transmembrane)
        try: self.reversal_potential = float(reversal_potential)
        except ValueError:
            self.reversal_potential = str(reversal_potential)
            assert(self.reversal_potential in ("nerst", "goldman_hodgkin_katz"))
        self.inside_concentration  = float(inside_concentration)
        self.outside_concentration = float(outside_concentration)
        self.inside_diffusivity  = float(inside_diffusivity)  if inside_diffusivity is not None else None
        self.outside_diffusivity = float(outside_diffusivity) if outside_diffusivity is not None else None
        self.inside_decay_period  = float(inside_decay_period)
        self.outside_decay_period = float(outside_decay_period)
        self.use_shells = bool(use_shells)
        self.use_grid = None if use_grid is None else tuple(float(x) for x in use_grid)
        assert(self.inside_concentration  >= 0.0)
        assert(self.outside_concentration >= 0.0)
        assert(self.inside_diffusivity  is None or self.inside_diffusivity >= 0)
        assert(self.outside_diffusivity is None or self.outside_diffusivity >= 0)
        assert(self.inside_decay_period  > 0.0)
        assert(self.outside_decay_period > 0.0)
        if self.use_shells: assert(self.inside_diffusivity is not None)
        if self.use_grid: assert(len(self.use_grid) == 3 and all(x > 0 for x in self.use_grid))

    def _initialize(self, database):
        db = database
        concentrations_doc = "Units: Molar"
        release_rates_doc = "Units: Molar / Second"
        reversal_potentials_doc = "Units:"
        conductances_doc = "Units: Siemens"
        if self.inside_diffusivity is None:
            db.add_global_constant("inside/concentrations/%s"%self.name,
                    self.inside_concentration, doc=concentrations_doc)
        else:
            if self.use_shells:
                db.add_attribute("inside/concentrations/%s"%self.name,
                        initial_value=self.inside_concentration, doc=concentrations_doc)
                db.add_attribute("inside/release_rates/%s"%self.name,
                        initial_value=0, doc=release_rates_doc)
                db.add_linear_system("inside/%s/diffusion"%self.name,
                        function=_inside_diffusion_coefficients, epsilon=epsilon * 1e-9,)
            else:
                db.add_attribute("membrane/inside/concentrations/%s"%self.name,
                        initial_value=self.inside_concentration, doc=concentrations_doc)
                db.add_attribute("membrane/inside/release_rates/%s"%self.name,
                        initial_value=0, doc=release_rates_doc)
                db.add_linear_system("membrane/inside/diffusion/%s"%self.name,
                        function=_inside_diffusion_coefficients, epsilon=epsilon * 1e-9,)
        if self.outside_diffusivity is None:
            db.add_global_constant("outside/concentrations/%s"%self.name,
                    self.outside_concentration, doc=concentrations_doc)
        else:
            db.add_attribute("outside/concentrations/%s"%self.name,
                    initial_value=self.outside_concentration, doc=concentrations_doc)
            db.add_attribute("outside/release_rates/%s"%self.name,
                    initial_value=0, doc=release_rates_doc)
            db.add_linear_system("outside/diffusion/%s"%self.name,
                    function=_outside_diffusion_coefficients, epsilon=epsilon * 1e-9,)
        if self.transmembrane:
            db.add_attribute("membrane/conductances/%s"%self.name,
                    initial_value=0, doc=conductances_doc)
            if isinstance(self.reversal_potential, float):
                db.add_global_constant("membrane/reversal_potentials/%s"%self.name,
                        self.reversal_potential, doc=reversal_potentials_doc)
            elif (self.inside_diffusivity is None and self.outside_diffusivity is None
                    and self.reversal_potential == "nerst"):
                x = self._nerst_potential(
                        db.access("T"), self.inside_concentration, self.outside_concentration)
                db.add_global_constant("membrane/reversal_potentials/%s"%self.name,
                        x, doc=reversal_potentials_doc)
            else:
                db.add_attribute("membrane/reversal_potentials/%s"%self.name,
                        doc=reversal_potentials_doc)

    def _reversal_potential(self, database_access):
        if isinstance(self.reversal_potential, float): return self.reversal_potential
        T = database_access("T")
        try:             inside = database_access("inside/concentrations/%s"%self.name)
        except KeyError: inside = database_access("membrane/inside/concentrations/%s"%self.name)
        else:
            if isinstance(inside, Iterable):
                index = database_access("membrane/inside")
                inside = inside[index]
        outside = database_access("outside/concentrations/%s"%self.name)
        if self.reversal_potential == "nerst":
            return self._nerst_potential(T, inside, outside)
        elif self.reversal_potential == "goldman_hodgkin_katz":
            voltages = database_access("membrane/voltages")
            return self._goldman_hodgkin_katz(T, inside, outside, voltages)

    def _nerst_potential(self, T, inside_concentration, outside_concentration):
        """ Returns the reversal voltage for an ionic species. """
        xp = cp.get_array_module(inside_concentration)
        if self.charge == 0: return xp.zeros_like(inside_concentration)
        ratio = xp.divide(outside_concentration, inside_concentration)
        return xp.nan_to_num(R * T / F / self.charge * xp.log(ratio))

    def _goldman_hodgkin_katz(self, T, inside_concentration, outside_concentration, voltages):
        """ Returns the reversal voltage for an ionic species. """
        xp = cp.get_array_module(inside_concentration)
        if self.charge == 0: return xp.full_like(inside_concentration, np.nan)
        z = (self.charge * F / (R * T)) * voltages
        return (self.charge * F) * (inside_concentration * self._efun(-z) - outside_concentration * self._efun(z))

    @cp.fuse()
    def _efun(z):
        if abs(z) < 1e-4:
            return 1 - z / 2
        else:
            return z / (math.exp(z) - 1)

class Reaction:
    """ Abstract class for specifying reactions and mechanisms. """
    @classmethod
    def name(self):
        """ A unique name for this reaction and all of its instances. """
        raise TypeError("Abstract method called by %s."%repr(self))

    @classmethod
    def initialize(self, database):
        """ (Optional) This method is called after the Model has been created.
        This method is called on a deep copy of each Reaction object.

        Argument database is a function(name) -> value

        (Optional) Returns a new Reaction object to use in place of this one. """
        pass

    @classmethod
    def new_instances(self, database, *args, **kwargs):
        """ """
        pass

    @classmethod
    def advance(self, database_access):
        """ Advance all instances of this reaction.

        Argument database_access is function: f(component_name) -> value
        """
        raise TypeError("Abstract method called by %s."%repr(self))

    _library = {
        "hh": ("nmodl_library/hh.mod",
            dict(pointers={"gl": ("membrane/conductances/L", 'a')},
                 parameter_overrides = {"celsius": 6.3})),

        # "na11a": ("neuwon/nmodl_library/Balbi2017/Nav11_a.mod", {}),

        # "Kv11_13States_temperature2": ("neuwon/nmodl_library/Kv-kinetic-models/hbp-00009_Kv1.1/hbp-00009_Kv1.1__13States_temperature2/hbp-00009_Kv1.1__13States_temperature2_Kv11.mod", {}),

        # "AMPA5": ("neuwon/nmodl_library/Destexhe1994/ampa5.mod",
        #     dict(pointers={"C": AccessHandle("Glu", outside_concentration=True)})),

        # "caL": ("neuwon/nmodl_library/Destexhe1994/caL3d.mod",
        #     dict(pointers={"g": AccessHandle("ca", conductance=True)})),
    }

class Model:
    def __init__(self, time_step,
            celsius = 37,
            maximum_extracellular_radius=3e-6,
            extracellular_volume_fraction=.20,
            extracellular_tortuosity=1.55,
            intracellular_resistance = 1,
            membrane_capacitance = 1e-2,
            initial_voltage = -70e-3,):
        self.species = {}
        self.reactions = {}
        self._injected_currents = _InjectedCurrents()
        # TODO: Either make the db private or rename it to the full name "database"
        self.db = db = Database()
        db.add_global_constant("time_step", float(time_step), doc="Units: Seconds")
        db.add_global_constant("celsius", float(celsius))
        db.add_global_constant("T", db.access("celsius") + 273.15, doc="Temperature in Kelvins.")
        db.add_function("create_segment", self.create_segment)
        db.add_function("destroy_segment", self.destroy_segment)
        db.add_function("insert_reaction", self.insert_reaction)
        db.add_function("remove_reaction", self.remove_reaction)
        # Basic entities and their relations.
        db.add_archetype("membrane", doc=""" """)
        db.add_archetype("inside", doc="Intracellular space.")
        db.add_archetype("outside", doc="Extracellular space using a voronoi diagram.")
        db.add_attribute("membrane/parents", dtype="membrane", check=False,
            doc="Cell membranes are connected in a tree.")
        db.add_sparse_matrix("membrane/children", "membrane", dtype=np.bool, doc="")
        db.add_attribute("membrane/inside", dtype="inside", doc="""
                A reference to the outermost shell.
                The shells and the innermost core are allocated in a contiguous block
                with this referencing the start of range of length "membrane/shells" + 1.
                """)
        db.add_attribute("membrane/shells", dtype=np.uint8)
        db.add_attribute("inside/membrane", dtype="membrane")
        db.add_attribute("inside/shell_radius")
        # Geometric properties.
        db.add_attribute("membrane/coordinates", shape=(3,), doc="Units: ")
        db.add_attribute("membrane/diameters", check=(">=", 0.0), doc="""
            Units: """)
        db.add_attribute("membrane/shapes", dtype=np.uint8, doc="""
            0 - Sphere
            1 - Cylinder
            2 - Frustum

            Note: all & only root segments are spheres. """)
        db.add_attribute("membrane/primary", dtype=np.bool, doc="""

            Primary segments are straightforward extensions of the parent
            branch. Non-primary segments are lateral branches off to the side of
            the parent branch.  """)

        db.add_attribute("membrane/lengths", doc="""
            The distance between each node and its parent node.
            Root node lengths are their radius.\n
            Units: Meters""", initial_value=np.nan)
        db.add_attribute("membrane/surface_areas", check=(">=", epsilon * (1e-6)**2), doc="""
            Units: Meters ^ 2""", initial_value=np.nan)
        db.add_attribute("membrane/cross_sectional_areas", doc="Units: Meters ^ 2",
            check = (">=", epsilon * (1e-6)**2))
        db.add_attribute("membrane/volumes", check=(">=", epsilon * (1e-6)**3), doc="""
            Units: Liters""")
        db.add_sparse_matrix("inside/neighbors", "inside", dtype=Neighbor)
        # Extracellular space properties.
        db.add_attribute("membrane/outside", dtype="outside", doc="")
        db.add_attribute("outside/coordinates", shape=(3,), doc="Units: ")
        db.add_kd_tree(  "outside/tree", "outside/coordinates")
        db.add_attribute("outside/volumes", doc="Units: Litres")
        db.add_sparse_matrix("outside/neighbors", "outside", dtype=Neighbor)
        db.add_global_constant("outside/volume_fraction", float(extracellular_volume_fraction),
            check=("in", (0.0, 1.0)), doc="")
        db.add_global_constant("outside/tortuosity", float(extracellular_tortuosity), check=(">=", 1.0))
        db.add_global_constant("outside/maximum_radius", float(maximum_extracellular_radius),
            check=(">=", epsilon * 1e-6))
        # Electric properties.
        db.add_global_constant("inside/resistance", float(intracellular_resistance), check=(">", 0.0), doc="""
            Cytoplasmic resistance,
            In NEURON this variable is named Ra?
            Units: """)
        db.add_global_constant("membrane/capacitance", float(membrane_capacitance), check=(">", 0.0), doc="""
            Units: Farads / Meter^2""")
        db.add_attribute("membrane/voltages", initial_value=float(initial_voltage))
        db.add_attribute("membrane/axial_resistances", check=False, doc="Units: ")
        db.add_attribute("membrane/capacitances", doc="Units: Farads")
        db.add_attribute("membrane/conductances")
        db.add_attribute("membrane/driving_voltages")
        db.add_linear_system("membrane/diffusion", function=_electric_coefficients,
            epsilon=epsilon * 1e-3,) # Epsilon millivolts.

    def __len__(self):
        return self.db.num_entity("membrane")

    def __str__(self):
        return str(self.db)

    def __repr__(self):
        return repr(self.db)

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
            if species in Species._library: species = Species(species, **Species._library[species])
            else: raise ValueError("Unrecognized species: %s."%species)
        else: assert(isinstance(species, Species))
        assert(species.name not in self.species)
        self.species[species.name] = species
        species._initialize(self.db)

    def add_reaction(self, reaction):
        """
        Argument reactions is one of:
          * An instance or subclass of the Reaction class, or
          * The name of a reaction from the standard library.
        """
        r = reaction
        if not isinstance(r, Reaction) and not (isinstance(r, type) and issubclass(r, Reaction)):
            try: nmodl_file_path, kwargs = Reaction._library[str(r)]
            except KeyError: raise ValueError("Unrecognized Reaction: %s."%str(r))
            from neuwon.nmodl import NmodlMechanism
            r = NmodlMechanism(nmodl_file_path, **kwargs)
        if hasattr(r, "initialize"):
            r = copy.deepcopy(r)
            retval = r.initialize(self.db)
            if retval is not None: r = retval
        name = str(r.name())
        assert(name not in self.reactions)
        self.reactions[name] = r

    def create_segment(self, parents, coordinates, diameters,
                shape="cylinder", shells=0, maximum_segment_length=np.inf):
        """
        Argument parents:
        Argument coordinates:
        Argument diameters:
        Argument shape: either "cylinder", "frustum".
        Argument shells: unimplemented.
        Argument maximum_segment_length

        Returns a list of Segments.
        """
        if not isinstance(parents, Iterable):
            parents     = [parents]
            coordinates = [coordinates]
        parents_clean = np.empty(len(parents), dtype=Index)
        for idx, p in enumerate(parents):
            if p is None:
                parents_clean[idx] = NULL
            elif isinstance(p, Segment):
                parents_clean[idx] = p.entity.index
            else:
                parents_clean[idx] = p
        parents = parents_clean
        if not isinstance(diameters, Iterable):
            diameters = np.full(len(parents), diameters, dtype=Real)
        if   shape == "cylinder":   shape = 1
        elif shape == "frustum":    shape = 2
        else: raise ValueError("Invalid argument 'shape'")
        # This method only deals with the "maximum_segment_length" argument and
        # delegates the remaining work to the method: "_create_segment_batch".
        maximum_segment_length = float(maximum_segment_length)
        assert(maximum_segment_length > 0)
        if maximum_segment_length == np.inf:
            tips = self._create_segment_batch(parents, coordinates, diameters,
                    shape=shape, shells=shells,)
            return [Segment(self, x) for x in tips]
        # Accept defeat... Batching operations is too complicated.
        tips = []
        old_coords = self.db.access("membrane/coordinates").get()
        old_diams = self.db.access("membrane/diameters").get()
        for p, c, d in zip(parents, coordinates, diameters):
            if p == NULL:
                tips.append(self._create_segment_batch([p], [c], [d], shape=shape, shells=shells,))
                continue
            length = np.linalg.norm(np.subtract(old_coords[p], c))
            divisions = np.maximum(1, int(np.ceil(length / maximum_segment_length)))
            x = np.linspace(0.0, 1.0, num=divisions + 1)[1:].reshape(-1, 1)
            _x = np.subtract(1.0, x)
            seg_coords = c * x + old_coords[p] * _x
            seg_diams  = d * x + old_diams[p] * _x
            cursor = p
            for i in range(divisions):
                cursor = self._create_segment_batch([cursor],
                    seg_coords[i], seg_diams[i], shape=shape, shells=shells,)[0]
                tips.append(cursor)
        return [Segment(self, x) for x in tips]

    def _create_segment_batch(self, parents, coordinates, diameters, shape, shells):
        access = self.db.access
        # Allocate memory.
        num_new_segs = len(parents)
        membrane_idx = self.db.create_entity("membrane", num_new_segs, return_entity=False)
        inside_idx   = self.db.create_entity("inside", num_new_segs * (shells + 1), return_entity=False)
        outside_idx  = self.db.create_entity("outside", num_new_segs, return_entity=False)
        membrane_idx = np.array(membrane_idx, dtype=int)
        inside_idx   = np.array(inside_idx, dtype=int)
        outside_idx  = np.array(outside_idx, dtype=int)
        # Save segment arguments.
        access("membrane/parents")[membrane_idx]     = parents
        access("membrane/coordinates")[membrane_idx] = coordinates
        access("membrane/diameters")[membrane_idx]   = diameters
        access("membrane/shells")[membrane_idx]      = shells
        #
        shapes = access("membrane/shapes")
        for p, m in zip(parents, membrane_idx):
            # Shape of root is always sphere.
            if p == NULL:   shapes[m] = 0
            else:           shapes[m] = shape
        # Cross-link the membrane parent to child to form a doubly linked tree.
        children = access("membrane/children")
        write_rows = []
        write_cols = []
        for p, m in zip(parents, membrane_idx):
            if p != NULL:
                siblings = list(children[p].indices)
                siblings.append(m)
                write_rows.append(p)
                write_cols.append(siblings)
        data = [np.ones(len(x), dtype=np.bool) for x in write_cols]
        access("membrane/children", sparse_matrix_write=(write_rows, write_cols, data))
        # Set some branches as primary.
        primary  = access("membrane/primary")
        parents  = access("membrane/parents")
        children = access("membrane/children")
        for m in membrane_idx:
            p = parents[m]
            # Shape of root is always sphere.
            if p == NULL: # Root.
                primary[m] = True # Value does not matter.
            elif parents[p] == NULL: # Parent is root.
                primary[m] = False # Spheres have no primary branches off of a them.
            else:
                # Set the first child added to a segment as the primary extension,
                # and all subsequent children as secondary branches.
                primary[m] = (children.getrow(p).getnnz() == 1)
        # 
        self._initialize_membrane(membrane_idx)
        self._initialize_membrane([p for p in parents[membrane_idx] if p != NULL])
        # 
        access("membrane/inside")[membrane_idx] = inside_idx[slice(None,None,shells + 1)]
        access("inside/membrane")[inside_idx]   = cp.repeat(membrane_idx, shells + 1)
        # shell_radius = [1.0] # TODO
        # access("inside/shell_radius")[inside_idx] = cp.tile(shell_radius, membrane_idx)
        # 
        access("membrane/outside")[membrane_idx] = outside_idx
        # self._initialize_outside(outside)
        # 1/0 # TODO: Also re-initialize all of the neighbors of new outside points.
        return membrane_idx

    def _initialize_membrane(self, membrane_idx):
        access   = self.db.access
        parents  = access("membrane/parents")
        children = access("membrane/children")
        coords   = access("membrane/coordinates")
        diams    = access("membrane/diameters")
        shapes   = access("membrane/shapes")
        primary  = access("membrane/primary")
        lengths  = access("membrane/lengths")
        s_areas  = access("membrane/surface_areas")
        x_areas  = access("membrane/cross_sectional_areas")
        volumes  = access("membrane/volumes")
        # Compute lengths.
        for idx in membrane_idx:
            p = parents[idx]
            if shapes[idx] == 0: # Root sphere.
                lengths[idx] = 0.5 * diams[idx] # Spheres have no defined length, so use the radius.
            else:
                distance = np.linalg.norm(coords[idx] - coords[p])
                # Subtract the parent's radius from the secondary nodes length,
                # to avoid excessive overlap between segments.
                if not primary[idx]:
                    parent_radius = diams[p] / 2.0
                    if distance < parent_radius + epsilon * 1e-6:
                        # This segment is entirely enveloped within its parent. In
                        # this corner case allow the segment to protrude directly
                        # from the center of the parent instead of the surface.
                        pass
                    else:
                        distance -= parent_radius
                lengths[idx] = distance
        # Compute surface areas.
        for idx in membrane_idx:
            p = parents[idx]
            d = diams[idx]
            shape = shapes[idx]
            if shape == 0: # Sphere.
                s_areas[idx] = np.pi * (d ** 2)
            else:
                l = lengths[idx]
                if shape == 1: # Cylinder.
                    s_areas[idx] = np.pi * d * l
                elif shape == 2: # Frustum.
                    s_areas[idx] = 1/0
                # Account for the surface area on the tips of terminal/leaf segments.
                if children.getrow(idx).getnnz() == 0:
                    s_areas[idx] += _area_circle(d)
        # Compute cross-sectional areas.
        for idx in membrane_idx:
            p = parents[idx]
            d = diams[idx]
            shape = shapes[idx]
            if shape == 0: # Sphere.
                x_areas[idx] = _area_circle(d)
            else:
                if shape == 1: # Cylinder.
                    x_areas[idx] = _area_circle(d)
                elif shape == 2: # Frustum.
                    x_areas[idx] = 1/0
        # Compute intracellular volumes.
        for idx in membrane_idx:
            p = parents[idx]
            shape = shapes[idx]
            if shape == 0: # Sphere.
                volumes[idx] = (4/3) * np.pi * (d/2) ** 3
            else:
                if shape == 1: # Cylinder.
                    volumes[idx] = np.pi * (d/2) ** 2 * lengths[idx]
                elif shape == 2: # Frustum.
                    volumes[idx] = 1/0
        # Compute passive electric properties
        Ra       = access("inside/resistance")
        r        = access("membrane/axial_resistances")
        Cm       = access("membrane/capacitance")
        c        = access("membrane/capacitances")
        # Compute axial membrane resistance.
        # TODO: This formula only works for cylinders.
        r[membrane_idx] = Ra * lengths[membrane_idx] / x_areas[membrane_idx]
        # Compute membrane capacitance.
        c[membrane_idx] = Cm * s_areas[membrane_idx]

        # TODO: Currently, diffusion from non-primary branches omits the section
        # of diffusive material between the parents surface and center.
        # Model the parents side of intracellular volume as a frustum to diffuse through.
        # Implementation:
        # -> Directly include this frustum into the resistance calculations.
        # -> For chemical diffusion: make a new attribute for the geometry terms
        #    in the diffusion equation, and include the new frustum in the new attribute.

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
            self.outside_volumes[location] = v * self.extracellular_volume_fraction * 1e3
            self.neighbors[location] = n
            for n in self.neighbors[location]:
                n["distance"] = np.linalg.norm(coords - self.coordinates[n["location"]])

    def destroy_segment(self, segments):
        """ """
        1/0

    def insert_reaction(self, reaction, *args, **kwargs):
        self.reactions[str(reaction)].new_instances(self.db, *args, **kwargs)

    def remove_reaction(self, reaction, segments):
        1/0

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
        self._injected_currents.advance(self.db)
        self._advance_species()
        self._advance_reactions()
        self._advance_species()

    def _advance_lockstep(self):
        """ Naive integration strategy, for reference only. """
        self._injected_currents.advance(self.db)
        self._advance_species()
        self._advance_species()
        self._advance_reactions()

    def _advance_species(self):
        """ Note: Each call to this method integrates over half a time step. """
        access = self.db.access
        dt     = access("time_step") / 2
        conductances        = access("membrane/conductances")
        driving_voltages    = access("membrane/driving_voltages")
        voltages            = access("membrane/voltages")
        capacitances        = access("membrane/capacitances")
        # Accumulate the net conductances and driving voltages from the chemical data.
        conductances.fill(0.0) # Zero accumulator.
        driving_voltages.fill(0.0) # Zero accumulator.
        for s in self.species.values():
            if not s.transmembrane: continue
            reversal_potential = s._reversal_potential(access)
            g = access("membrane/conductances/%s"%s.name)
            conductances += g
            driving_voltages += g * reversal_potential
        driving_voltages /= conductances
        driving_voltages[:] = cp.nan_to_num(driving_voltages)
        # Calculate the transmembrane currents.
        diff_v   = driving_voltages - voltages
        recip_rc = conductances / capacitances
        alpha    = cp.exp(-dt * recip_rc)
        voltages += diff_v * (1.0 - alpha)
        # Calculate the lateral currents throughout the neurons.
        voltages[:] = access("membrane/diffusion").dot(voltages)

        return

        # Calculate the transmembrane ion flows.
        for s in self.species.values():
            if not s.transmembrane: continue
            if s.intra is None and s.extra is None: continue
            integral_v = dt * (s.reversal_potential - driving_voltages)
            integral_v += rc * diff_v * alpha
            moles = s.conductances * integral_v / (s.charge * F)
            if s.intra is not None:
                s.intra.concentrations += moles / self.geometry.inside_volumes
            if s.extra is not None:
                s.extra.concentrations -= moles / self.geometry.outside_volumes
        # Calculate the local release / removal of chemicals.
        for s in self.species.values():
            for x in (s.intra, s.extra):
                if x is None: continue
                x.concentrations = cp.maximum(0, x.concentrations + x.release_rates * dt)
                # Calculate the lateral diffusion throughout the space.
                x.concentrations = x.irm.dot(x.concentrations)

    def _advance_reactions(self):
        access = self.db.access
        for name, species in self.species.items():
            if species.transmembrane:
                access("membrane/conductances/%s"%name).fill(0.0)
            if species.inside_diffusivity is not None:
                access("inside/release_rates/%s"%name).fill(0.0)
            if species.outside_diffusivity is not None:
                access("outside/release_rates/%s"%name).fill(0.0)
        for r in self.reactions.values():
            r.advance(access)

class _InjectedCurrents:
    def __init__(self):
        self.currents = []
        self.locations = []
        self.remaining = []

    def advance(self, database):
        time_step = database.access("time_step")
        capacitances = database.access("membrane/capacitances")
        voltages = database.access("membrane/voltages")
        for idx, (amps, location, t) in enumerate(
                zip(self.currents, self.locations, self.remaining)):
            dv = amps * min(time_step, t) / capacitances[location]
            voltages[location] += dv
            self.remaining[idx] -= time_step
        keep = [t > 0 for t in self.remaining]
        self.currents  = [x for k, x in zip(keep, self.currents) if k]
        self.locations = [x for k, x in zip(keep, self.locations) if k]
        self.remaining = [x for k, x in zip(keep, self.remaining) if k]

    def inject_current(self, location, current, duration = 1.4e-3):
        location = int(location)
        # assert(location < len(self))
        duration = float(duration)
        assert(duration >= 0)
        current = float(current)
        self.currents.append(current)
        self.locations.append(location)
        self.remaining.append(duration)

class Segment:
    """ This class is returned by model.create_segment() """
    def __init__(self, model, membrane_index):
        self.model = model
        self.entity = Entity(model.db, "membrane", membrane_index)

    def insert_reaction(self, reaction, *args, **kwargs):
        self.model.insert_reaction(reaction, [self.entity.index], *args, **kwargs)

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

    def voltage(self):
        return self.entity.read("membrane/voltages")

    def inject_current(self, current, duration=1e-3):
        self.model._injected_currents.inject_current(self.entity.index, current, duration)

@cp.fuse()
def _area_circle(diameter):
    return np.pi * (diameter / 2.0) ** 2

@cp.fuse()
def _volume_sphere(diameter):
    return 1/0

@cp.fuse()
def _surface_area_sphere(diameter):
    return 1/0

def surface_area_frustum(radius_1, radius_2, length):
    """ Lateral surface area, does not include the ends. """
    s = sqrt((radius_1 - radius_2) ** 2 + length ** 2)
    return np.pi * (radius_1 + radius_2) * s

def volume_of_frustum(radius_1, radius_2, length):
    return np.pi / 3.0 * length * (radius_1 * radius_1 + radius_1 * radius_2 + radius_2 * radius_2)

def _electric_coefficients(access):
    """
    Model the electric currents over the membrane surface.
    Compute the coefficients of the derivative function:
    dV/dt = C * V, where C is Coefficients matrix and V is voltage vector.
    """
    dt           = access("time_step")
    parents      = access("membrane/parents").get()
    resistances  = access("membrane/axial_resistances").get()
    capacitances = access("membrane/capacitances").get()
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
        flux = species.inside_diffusivity * geometry.cross_sectional_areas[location] / l
        src.append(location)
        dst.append(parent)
        coef.append(+dt * flux / geometry.inside_volumes[parent])
        src.append(location)
        dst.append(location)
        coef.append(-dt * flux / geometry.inside_volumes[location])
        src.append(parent)
        dst.append(location)
        coef.append(+dt * flux / geometry.inside_volumes[location])
        src.append(parent)
        dst.append(parent)
        coef.append(-dt * flux / geometry.inside_volumes[parent])
    for location in range(len(geometry)):
        src.append(location)
        dst.append(location)
        coef.append(-dt / species.inside_decay_period)
    return (coef, (dst, src))

def _outside_diffusion_coefficients(database_access, species):
    src = []; dst = []; coef = []
    D = species.outside_diffusivity / geometry.extracellular_tortuosity ** 2
    for location in range(len(geometry)):
        for neighbor in geometry.neighbors[location]:
            flux = D * neighbor["border_surface_area"] / neighbor["distance"]
            src.append(location)
            dst.append(neighbor["location"])
            coef.append(+dt * flux / geometry.outside_volumes[neighbor["location"]])
            src.append(location)
            dst.append(location)
            coef.append(-dt * flux / geometry.outside_volumes[location])
    for location in range(len(geometry)):
        src.append(location)
        dst.append(location)
        coef.append(-dt / species.outside_decay_period)
    return (coef, (dst, src))

if __name__ == "__main__":
    print("#"*30, "REPR")
    print(repr(Model(1e-4).db))
    print("#"*30, "STR")
    print(str(Model(1e-4).db))
    Model(1e-4).db.check()

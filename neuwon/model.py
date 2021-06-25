import copy
import cupy as cp
import math
import neuwon.voronoi
import numba.cuda
import numpy as np
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping
from neuwon.database import *
from PIL import Image, ImageFont, ImageDraw
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm

# TODO: Consider switching to use NEURON's units? It makes my code a bit more
# complicated, but it should make the users code simpler and more intuitive.
# Also, if I do this then I can get rid of a lot of nmodl shims...

# TODO: How to write species derivatives? as concentrations or absolute amounts?
#       Either way, my current code is broken BC of it.

F = 96485.3321233100184 # Faraday's constant, Coulombs per Mole of electrons
R = 8.31446261815324 # Universal gas constant

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
    # 
    # Also, modify add_reactions to accept a whole dictionary of reactions so
    # that this works: model.add_reactions(neuwon.examples.Hodgkin_Huxley.reactions)
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
            "reversal_potential": "nerst",
            # "reversal_potential": "goldman_hodgkin_katz", # TODO: Does not work...
            "inside_concentration": 70e-9,
            "outside_concentration": 2e-3,
            "inside_diffusivity": 1e-9,
        },
        "cl": {
            "charge": -1,
            "transmembrane": True,
            "reversal_potential": "nerst",
            "inside_concentration":   10e-3,
            "outside_concentration": 110e-3,
        },
        "glu": {
            "outside_diffusivity": 1e-9,
            "outside_decay_period": .5e-3,
        },
    }

    def __init__(self, name,
            charge = 0,
            transmembrane = False,
            reversal_potential = "nerst",
            inside_concentration  = 0.0,
            outside_concentration = 0.0,
            inside_diffusivity    = None,
            outside_diffusivity   = None,
            inside_decay_period   = float("inf"),
            outside_decay_period  = float("inf"),
            use_shells = False,
            outside_grid = None,):
        """
        Arguments
        * inside_concentration:  initial value, units millimolar.
        * outside_concentration: initial value, units millimolar.
        * reversal_potential: is one of: number, "nerst", "goldman_hodgkin_katz"

        If diffusivity is not given, then the concentration is a global constant.
        """
        self.name = str(name)
        self.charge = int(charge)
        self.transmembrane = bool(transmembrane)
        try: self.reversal_potential = float(reversal_potential)
        except ValueError:
            self.reversal_potential = str(reversal_potential)
            assert(self.reversal_potential in ("nerst", "goldman_hodgkin_katz"))
        self.inside_concentration   = float(inside_concentration)
        self.outside_concentration  = float(outside_concentration)
        self.inside_global_const    = inside_diffusivity is None
        self.outside_global_const   = outside_diffusivity is None
        self.inside_diffusivity     = float(inside_diffusivity) if not self.inside_global_const else 0.0
        self.outside_diffusivity    = float(outside_diffusivity) if not self.outside_global_const else 0.0
        self.inside_decay_period    = float(inside_decay_period)
        self.outside_decay_period   = float(outside_decay_period)
        self.use_shells             = bool(use_shells)
        self.inside_archetype       = "inside" if self.use_shells else "membrane/inside"
        self.outside_grid           = tuple(float(x) for x in outside_grid) if outside_grid else None
        assert(self.inside_concentration  >= 0.0)
        assert(self.outside_concentration >= 0.0)
        assert(self.inside_diffusivity    >= 0)
        assert(self.outside_diffusivity   >= 0)
        assert(self.inside_decay_period   > 0.0)
        assert(self.outside_decay_period  > 0.0)
        if self.use_shells: assert(self.inside_diffusivity > 0.0)
        if self.inside_global_const:  assert self.inside_decay_period == np.inf
        if self.inside_global_const:  assert not self.use_shells
        if self.outside_global_const: assert self.outside_decay_period == np.inf

    def __repr__(self):
        return "neuwon.Species(%s)"%self.name

    def _initialize(self, database):
        db = database
        concentrations_doc = "Units: Molar"
        release_rates_doc = "Units: Molar / Second"
        reversal_potentials_doc = "Units:"
        conductances_doc = "Units: Siemens"
        if self.inside_diffusivity == 0.0:
            db.add_global_constant(self.inside_archetype+"/concentrations/" + self.name,
                    self.inside_concentration, doc=concentrations_doc)
        else:
            db.add_attribute(self.inside_archetype+"/concentrations/" + self.name,
                    initial_value=self.inside_concentration, doc=concentrations_doc)
            db.add_attribute(self.inside_archetype+"/release_rates/" + self.name,
                    initial_value=0.0, doc=release_rates_doc)
            db.add_linear_system(self.inside_archetype+"/diffusions/" + self.name,
                    function=self._inside_diffusion_coefficients, epsilon=epsilon * 1e-9,)
        if self.outside_diffusivity == 0.0:
            db.add_global_constant("outside/concentrations/" + self.name,
                    self.outside_concentration, doc=concentrations_doc)
        else:
            db.add_attribute("outside/concentrations/" + self.name,
                    initial_value=self.outside_concentration, doc=concentrations_doc)
            db.add_attribute("outside/release_rates/" + self.name,
                    initial_value=0.0, doc=release_rates_doc)
            db.add_linear_system("outside/diffusions/" + self.name,
                    function=self._outside_diffusion_coefficients, epsilon=epsilon * 1e-9,)
        if self.transmembrane:
            db.add_attribute("membrane/conductances/" + self.name,
                    initial_value=0.0, bounds=(0, np.inf), doc=conductances_doc)
            if isinstance(self.reversal_potential, float):
                db.add_global_constant("membrane/reversal_potentials/" + self.name,
                        self.reversal_potential, doc=reversal_potentials_doc)
            elif (self.inside_global_const and self.outside_global_const
                    and self.reversal_potential == "nerst"):
                x = self._nerst_potential(
                        db.access("T"), self.inside_concentration, self.outside_concentration)
                db.add_global_constant("membrane/reversal_potentials/" + self.name,
                        x, doc=reversal_potentials_doc)
            else:
                db.add_attribute("membrane/reversal_potentials/" + self.name,
                        doc=reversal_potentials_doc)

    def _reversal_potential(self, access):
        x = access("membrane/reversal_potentials/" + self.name)
        if isinstance(x, float): return x
        inside  = access(self.inside_archetype+"/concentrations/"+self.name)
        outside = access("outside/concentrations/"+self.name)
        if not isinstance(inside, float) and self.use_shells:
            inside = inside[access("membrane/inside")]
        if not isinstance(outside, float):
            outside = outside[access("membrane/outside")]
        T = access("T")
        if self.reversal_potential == "nerst":
            x[:] = self._nerst_potential(T, inside, outside)
        elif self.reversal_potential == "goldman_hodgkin_katz":
            voltages = access("membrane/voltages")
            x[:] = self._goldman_hodgkin_katz(T, inside, outside, voltages)
        else: raise NotImplementedError(self.reversal_potential)
        return x

    def _nerst_potential(self, T, inside_concentration, outside_concentration):
        xp = cp.get_array_module(inside_concentration)
        ratio = xp.divide(outside_concentration, inside_concentration)
        return xp.nan_to_num(R * T / F / self.charge * xp.log(ratio))

    def _goldman_hodgkin_katz(self, T, inside_concentration, outside_concentration, voltages):
        xp = cp.get_array_module(inside_concentration)
        z = (self.charge * F / (R * T)) * voltages
        return (self.charge * F) * (inside_concentration * self._efun(-z) - outside_concentration * self._efun(z))

    @cp.fuse()
    def _efun(z):
        if abs(z) < 1e-4:
            return 1 - z / 2
        else:
            return z / (math.exp(z) - 1)

    def _inside_diffusion_coefficients(self, access):
        dt      = access("time_step") / 1000 / 2
        parents = access("membrane/parents").get()
        lengths = access("membrane/lengths").get()
        xareas  = access("membrane/cross_sectional_areas").get()
        volumes = access("membrane/inside/volumes").get()
        if self.use_shells: raise NotImplementedError
        src = []; dst = []; coef = []
        for location in range(len(parents)):
            parent = parents[location]
            if parent == NULL: continue
            flux = self.inside_diffusivity * xareas[location] / lengths[location]
            src.append(location)
            dst.append(parent)
            coef.append(+dt * flux / volumes[parent])
            src.append(location)
            dst.append(location)
            coef.append(-dt * flux / volumes[location])
            src.append(parent)
            dst.append(location)
            coef.append(+dt * flux / volumes[location])
            src.append(parent)
            dst.append(parent)
            coef.append(-dt * flux / volumes[parent])
        for location in range(len(parents)):
            src.append(location)
            dst.append(location)
            coef.append(-dt / self.inside_decay_period)
        return (coef, (dst, src))

    def _outside_diffusion_coefficients(self, access):
        extracellular_tortuosity = 1.4 # TODO: FIXME: put this one back in the db?
        D = self.outside_diffusivity / extracellular_tortuosity ** 2
        dt          = access("time_step") / 1000 / 2
        decay       = -dt / self.outside_decay_period
        recip_vol   = (1.0 / access("outside/volumes")).get()
        area        = access("outside/neighbor_border_areas")
        dist        = access("outside/neighbor_distances")
        flux_data   = D * area.data / dist.data
        src         = np.empty(2*len(flux_data))
        dst         = np.empty(2*len(flux_data))
        coef        = np.empty(2*len(flux_data))
        write_idx   = 0
        for location in range(len(recip_vol)):
            for ii in range(area.indptr[location], area.indptr[location+1]):
                neighbor = area.indices[ii]
                flux     = flux_data[ii]
                src[write_idx] = location
                dst[write_idx] = neighbor
                coef[write_idx] = +dt * flux * recip_vol[neighbor]
                write_idx += 1
                src[write_idx] = location
                dst[write_idx] = location
                coef[write_idx] = -dt * flux * recip_vol[location] + decay
                write_idx += 1
        return (coef, (dst, src))

class Reaction:
    """ Abstract class for specifying reactions and mechanisms. """
    @classmethod
    def name(self):
        """ A unique name for this reaction and all of its instances. """
        return type(self).__name__

    @classmethod
    def initialize(self, database):
        """ (Optional) This method is called after the Model has been created.
        This method is called on a deep copy of each Reaction object.

        Argument database is a function(name) -> value

        (Optional) Returns a new Reaction object to use in place of this one. """
        pass

    @classmethod
    def advance(self, database_access):
        """ Advance all instances of this reaction.

        Argument database_access is function: f(component_name) -> value
        """
        raise TypeError("Abstract method called by %s."%repr(self))

    _library = {
        "hh": ("nmodl_library/hh.mod",
            dict(parameter_overrides = {"celsius": 6.3})),

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
            fh_space = 300e-10, # Frankenhaeuser Hodgkin Space, in Angstroms
            max_outside_radius=20e-6,
            outside_volume_fraction=.20,
            outside_tortuosity=1.55,
            cytoplasmic_resistance = 1,
            membrane_capacitance = 1e-2,
            initial_voltage = -70e-3,):
        """
        Argument cytoplasmic_resistance
                Units:

        Argument outside_volume_fraction

        Argument outside_tortuosity

        Argument max_outside_radius

        Argument membrane_capacitance
                Units: Farads / Meter^2
        """
        self.species = {}
        self.reactions = {}
        self._injected_currents = _InjectedCurrents()
        self.animations = set()

        # TODO: Rename "db" to the full "database", add method model.get_database().
        self.db = db = Database()
        db.add_global_constant("time_step", float(time_step), doc="Units: Milliseconds")
        db.add_global_constant("celsius", float(celsius))
        db.add_global_constant("T", db.access("celsius") + 273.15, doc="Temperature in Kelvins.")
        db.add_function("create_segment", self.create_segment)
        db.add_function("destroy_segment", self.destroy_segment)
        self._initialize_database_membrane(db)
        self._initialize_database_inside(db)
        self._initialize_database_outside(db)
        self._initialize_database_electric(db, initial_voltage)

        self.fh_space = float(fh_space)
        self.cytoplasmic_resistance = float(cytoplasmic_resistance)
        self.max_outside_radius = float(max_outside_radius)
        self.membrane_capacitance = float(membrane_capacitance)
        self.outside_tortuosity = float(outside_tortuosity)
        self.outside_volume_fraction = float(outside_volume_fraction)
        assert(self.fh_space >= epsilon * 1e-10)
        assert(self.cytoplasmic_resistance >= epsilon)
        assert(self.max_outside_radius >= epsilon * 1e-6)
        assert(self.membrane_capacitance >= epsilon)
        assert(self.outside_tortuosity >= 1.0)
        assert(1.0 >= self.outside_volume_fraction >= 0.0)

    def _initialize_database_membrane(self, db):
        db.add_archetype("membrane", doc=""" """)
        db.add_attribute("membrane/parents", dtype="membrane", allow_invalid=True,
                doc="Cell membranes are connected in a tree.")
        db.add_sparse_matrix("membrane/children", "membrane", dtype=np.bool, doc="")
        db.add_attribute("membrane/coordinates", shape=(3,), doc="Units: ")
        db.add_attribute("membrane/diameters", bounds=(0.0, np.inf), doc="""
                Units: """)
        db.add_attribute("membrane/shapes", dtype=np.uint8, doc="""
                0 - Sphere
                1 - Cylinder
                2 - Frustum

                Note: only and all root segments are spheres. """)
        db.add_attribute("membrane/primary", dtype=np.bool, doc="""

                Primary segments are straightforward extensions of the parent
                branch. Non-primary segments are lateral branches off to the side of
                the parent branch.  """)

        db.add_attribute("membrane/lengths", doc="""
                The distance between each node and its parent node.
                Root node lengths are their radius.\n
                Units: Meters""", initial_value=np.nan)
        db.add_attribute("membrane/surface_areas", bounds=(epsilon * (1e-6)**2, np.inf), doc="""
                Units: Meters ^ 2""", initial_value=np.nan)
        db.add_attribute("membrane/cross_sectional_areas", doc="Units: Meters ^ 2",
                bounds = (epsilon * (1e-6)**2, np.inf))
        db.add_attribute("membrane/inside/volumes", bounds=(epsilon * (1e-6)**3, np.inf), doc="""
                Units: Liters""")

    def _initialize_database_inside(self, db):
        db.add_archetype("inside", doc="Intracellular space.")
        db.add_attribute("membrane/inside", dtype="inside", doc="""
                A reference to the outermost shell.
                The shells and the innermost core are allocated in a contiguous block
                with this referencing the start of range of length "membrane/shells" + 1.
                """)
        db.add_attribute("membrane/shells", dtype=np.uint8)
        db.add_attribute("inside/membrane", dtype="membrane")
        db.add_attribute("inside/shell_radius")
        db.add_attribute("inside/volumes",
                # bounds=(epsilon * (1e-6)**3, np.inf),
                allow_invalid=True,
                doc="""Units: Liters""")
        db.add_sparse_matrix("inside/neighbor_distances", "inside")
        db.add_sparse_matrix("inside/neighbor_border_areas", "inside")

    def _initialize_database_outside(self, db):
        db.add_archetype("outside", doc="Extracellular space using a voronoi diagram.")
        db.add_attribute("membrane/outside", dtype="outside", doc="")
        db.add_attribute("outside/coordinates", shape=(3,), doc="Units: ")
        db.add_kd_tree(  "outside/tree", "outside/coordinates")
        db.add_attribute("outside/volumes", doc="Units: Litres")
        db.add_sparse_matrix("outside/neighbor_distances", "outside")
        db.add_sparse_matrix("outside/neighbor_border_areas", "outside")

    def _initialize_database_electric(self, db, initial_voltage):
        db.add_attribute("membrane/voltages", initial_value=float(initial_voltage))
        db.add_attribute("membrane/axial_resistances", allow_invalid=True, doc="Units: ")
        db.add_attribute("membrane/capacitances", doc="Units: Farads", bounds=(0, np.inf))
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

    def check(self):
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

    def get_species(self, species_name):
        return self.species[str(species_name)]

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
        return r

    def get_reaction(self, reaction_name):
        return self.reactions[str(reaction_name)]

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
        parents_clean = np.empty(len(parents), dtype=Pointer)
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
        if   shape == "cylinder":   shape = np.full(len(parents), 1, dtype=np.uint8)
        elif shape == "frustum":    shape = np.full(len(parents), 2, dtype=np.uint8)
        else: raise ValueError("Invalid argument 'shape'")
        shape[parents == NULL] = 0 # Shape of root is always sphere.
        # This method only deals with the "maximum_segment_length" argument and
        # delegates the remaining work to the method: "_create_segment_batch".
        maximum_segment_length = float(maximum_segment_length)
        assert(maximum_segment_length > 0)
        if maximum_segment_length == np.inf:
            tips = self._create_segment_batch(parents, coordinates, diameters,
                    shapes=shape, shells=shells,)
            return [Segment(self, x) for x in tips]
        # Accept defeat... Batching operations is too complicated.
        tips = []
        old_coords = self.db.access("membrane/coordinates").get()
        old_diams = self.db.access("membrane/diameters").get()
        for p, c, d in zip(parents, coordinates, diameters):
            if p == NULL:
                tips.append(self._create_segment_batch([p], [c], [d], shapes=shape, shells=shells,))
                continue
            length = np.linalg.norm(np.subtract(old_coords[p], c))
            divisions = np.maximum(1, int(np.ceil(length / maximum_segment_length)))
            x = np.linspace(0.0, 1.0, num=divisions + 1)[1:].reshape(-1, 1)
            _x = np.subtract(1.0, x)
            seg_coords = c * x + old_coords[p] * _x
            if shape == 1:
                seg_diams = np.full(divisions, d)
            elif shape == 2:
                seg_diams = d * x + old_diams[p] * _x
            cursor = p
            for i in range(divisions):
                cursor = self._create_segment_batch([cursor],
                    seg_coords[i], seg_diams[i], shapes=shape, shells=shells,)[0]
                tips.append(cursor)
        return [Segment(self, x) for x in tips]

    def _create_segment_batch(self, parents, coordinates, diameters, shapes, shells):
        num_new_segs = len(parents)
        m_idx = self.db.create_entity("membrane", num_new_segs, return_entity=False)
        i_idx = self.db.create_entity("inside", num_new_segs * (shells + 1), return_entity=False)
        o_idx = self.db.create_entity("outside", num_new_segs, return_entity=False)
        access = self.db.access
        access("membrane/inside")[m_idx]      = i_idx[slice(None,None,shells + 1)]
        access("inside/membrane")[i_idx]      = cp.repeat(m_idx, shells + 1)
        access("membrane/outside")[m_idx]     = o_idx
        access("membrane/parents")[m_idx]     = parents
        access("membrane/coordinates")[m_idx] = coordinates
        access("membrane/diameters")[m_idx]   = diameters
        access("membrane/shapes")[m_idx]      = shapes
        access("membrane/shells")[m_idx]      = shells
        children = access("membrane/children")
        write_rows = []
        write_cols = []
        for p, c in zip(parents, m_idx):
            if p != NULL:
                siblings = list(children[p].indices)
                siblings.append(c)
                write_rows.append(p)
                write_cols.append(siblings)
        data = [np.full(len(x), True) for x in write_cols]
        access("membrane/children", sparse_matrix_write=(write_rows, write_cols, data))
        primary    = access("membrane/primary")
        all_parent = access("membrane/parents").get()
        children   = access("membrane/children")
        for p, m in zip(parents, m_idx):
            if p == NULL: # Root.
                primary[m] = False # Shape of root is always sphere, value does not matter.
            elif all_parent[p] == NULL: # Parent is root.
                primary[m] = False # Spheres have no primary branches off of a them.
            else:
                # Set the first child added to a segment as the primary extension,
                # and all subsequent children as secondary branches.
                primary[m] = (children[p].getnnz() == 1)
        self._initialize_membrane_geometry(m_idx)
        self._initialize_membrane_geometry([p for p in parents if p != NULL])

        # shell_radius = [1.0] # TODO
        # access("inside/shell_radius")[i_idx] = cp.tile(shell_radius, m_idx)
        # 

        # TODO: Consider moving the extracellular tracking point to over the
        # center of the cylinder, instead of the tip. Using the tip only really
        # makes sense for synapses. Also sphere are special case.
        access("outside/coordinates")[m_idx] = coordinates
        self._initialize_outside(o_idx)
        return m_idx

    def _initialize_membrane_geometry(self, membrane_idx):
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
        volumes  = access("membrane/inside/volumes")
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
            d = diams[idx]
            l = lengths[idx]
            shape = shapes[idx]
            if shape == 0: # Sphere.
                volumes[idx] = 1000 * (4/3) * np.pi * (d/2) ** 3
            else:
                if shape == 1: # Cylinder.
                    volumes[idx] = 1000 * np.pi * (d/2) ** 2 * l
                elif shape == 2: # Frustum.
                    volumes[idx] = 1/0
        # Compute passive electric properties
        Ra       = self.cytoplasmic_resistance
        r        = access("membrane/axial_resistances")
        Cm       = self.membrane_capacitance
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

        outside_volumes = access("outside/volumes")
        fh_space = self.fh_space * s_areas[membrane_idx] * 1000
        outside_volumes[access("membrane/outside")[membrane_idx]] = fh_space

    def _initialize_outside(self, locations):
        self._initialize_outside_inner(locations)
        touched = set()
        for neighbors in self.db.access("outside/neighbor_distances")[locations]:
            touched.update(neighbors.indices)
        touched.difference_update(set(locations))
        self._initialize_outside_inner(list(touched))

    def _initialize_outside_inner(self, locations):
        # TODO: Consider https://en.wikipedia.org/wiki/Power_diagram
        coordinates     = self.db.access("outside/coordinates").get()
        tree            = self.db.access("outside/tree")
        write_neighbor_cols = []
        write_neighbor_dist = []
        write_neighbor_area = []
        for location in locations:
            coords = coordinates[location]
            potential_neighbors = tree.query_ball_point(coords, 2 * self.max_outside_radius)
            potential_neighbors.remove(location)
            volume, neighbors = neuwon.voronoi.voronoi_cell(location,
                    self.max_outside_radius, np.array(potential_neighbors, dtype=Pointer), coordinates)
            write_neighbor_cols.append(list(neighbors['location']))
            write_neighbor_dist.append(list(neighbors['distance']))
            write_neighbor_area.append(list(neighbors['border_surface_area']))
        self.db.access("outside/neighbor_distances",
                sparse_matrix_write=(locations, write_neighbor_cols, write_neighbor_dist))
        self.db.access("outside/neighbor_border_areas",
                sparse_matrix_write=(locations, write_neighbor_cols, write_neighbor_area))

    def destroy_segment(self, segments):
        """ """
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
        for x in self.animations: x._advance()

    def _advance_lockstep(self):
        """ Naive integration strategy, for reference only. """
        self._injected_currents.advance(self.db)
        self._advance_species()
        self._advance_species()
        self._advance_reactions()

    def _advance_species(self):
        """ Note: Each call to this method integrates over half a time step. """
        access = self.db.access
        dt     = access("time_step") / 1000 / 2
        conductances        = access("membrane/conductances")
        driving_voltages    = access("membrane/driving_voltages")
        voltages            = access("membrane/voltages")
        capacitances        = access("membrane/capacitances")
        # Accumulate the net conductances and driving voltages from each ion species' data.
        conductances.fill(0.0) # Zero accumulator.
        driving_voltages.fill(0.0) # Zero accumulator.
        for s in self.species.values():
            if not s.transmembrane: continue
            reversal_potential = s._reversal_potential(access)
            g = access("membrane/conductances/"+s.name)
            conductances += g
            driving_voltages += g * reversal_potential
        driving_voltages /= conductances
        driving_voltages[:] = cp.nan_to_num(driving_voltages)
        # Update voltages.
        exponent    = -dt * conductances / capacitances
        alpha       = cp.exp(exponent)
        diff_v      = driving_voltages - voltages
        irm         = access("membrane/diffusion")
        voltages[:] = irm.dot(driving_voltages - diff_v * alpha)
        integral_v  = dt * driving_voltages - diff_v * exponent * alpha
        # Calculate the transmembrane ion flows.
        for s in self.species.values():
            if not s.transmembrane: continue
            if s.inside_diffusivity == 0 and s.outside_diffusivity == 0: continue
            assert s.charge != 0, s.name
            reversal_potential = access("membrane/reversal_potentials/"+s.name)
            g = access("membrane/conductances/"+s.name)
            moles = g * (dt * reversal_potential - integral_v) / (s.charge * F)
            if s.inside_diffusivity != 0:
                if s.use_shells:
                    1/0
                else:
                    volumes        = access("membrane/inside/volumes")
                    concentrations = access("membrane/inside/concentrations/"+s.name)
                    concentrations += moles / volumes
            if s.outside_diffusivity != 0:
                volumes = access("outside/volumes")
                s.outside.concentrations -= moles / self.geometry.outside_volumes
        # Update chemical concentrations: release rates and diffusion.
        for s in self.species.values():
            if s.inside_diffusivity != 0:
                x    = access("membrane/inside/concentrations/"+s.name)
                rr   = access("membrane/inside/release_rates/"+s.name)
                irm  = access("membrane/inside/diffusions/"+s.name)
                x[:] = irm.dot(cp.maximum(0, x + rr * dt))
            if s.outside_diffusivity != 0:
                x    = access("outside/concentrations/"+s.name)
                rr   = access("outside/release_rates/"+s.name)
                irm  = access("outside/diffusions/"+s.name)
                x[:] = irm.dot(cp.maximum(0, x + rr * dt))

    def _advance_reactions(self):
        access = self.db.access
        for name, species in self.species.items():
            if species.transmembrane:
                access("membrane/conductances/"+name).fill(0.0)
            if species.inside_diffusivity != 0.0:
                access(species.inside_archetype + "/release_rates/"+name).fill(0.0)
            if species.outside_diffusivity != 0.0:
                access("outside/release_rates/"+name).fill(0.0)
        for name, r in self.reactions.items():
            try:
                r.advance(access)
            except Exception:
                print("Error in reaction: " + name)
                raise

    def render_frame(self, membrane_colors,
            output_filename, resolution,
            camera_coordinates,
            camera_look_at=(0,0,0),
            fog_color=(1,1,1),
            fog_distance=np.inf,
            lights=((1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),)):
        """ Use POVRAY to render an image of the model. """
        qq = 1e6 # Convert to microns for POVRAY.
        pov = ""
        pov += '#include "shapes.inc"\n'
        pov += '#include "colors.inc"\n'
        pov += '#include "textures.inc"\n'
        pov += "camera { location <%s> look_at  <%s> }\n"%(
            ", ".join(str(x*qq) for x in camera_coordinates),
            ", ".join(str(x*qq) for x in camera_look_at))
        pov += "global_settings { ambient_light rgb<1, 1, 1> }\n"
        for coords in lights:
            pov += "light_source { <%s> color rgb<1, 1, 1>}\n"%", ".join(str(x*qq) for x in coords)
        if fog_distance == np.inf:
            pov += "background { color rgb<%s> }\n"%", ".join(str(x) for x in fog_color)
        else:
            pov += "fog { distance %s color rgb<%s>}\n"%(str(fog_distance*qq),
            ", ".join(str(x) for x in fog_color))
        all_parent = self.access("membrane/parents").get()
        all_coords = self.access("membrane/coordinates").get()
        all_diams  = self.access("membrane/diameters").get()
        for location, (parent, coords, diam) in enumerate(zip(all_parent, all_coords, all_diams)):
            # Special cases for root of tree, which is a sphere.
            if parent == NULL:
                pov += "sphere { <%s>, %s "%(
                    ", ".join(str(x*qq) for x in coords),
                    str(diam / 2 * qq))
            else:
                parent_coords = all_coords[parent]
                pov += "cylinder { <%s>, <%s>, %s "%(
                    ", ".join(str(x*qq) for x in coords),
                    ", ".join(str(x*qq) for x in parent_coords),
                    str(diam / 2 * qq))
            pov += "texture { pigment { color rgb <%s> } } }\n"%", ".join(str(x) for x in membrane_colors[location])
        pov_file = tempfile.NamedTemporaryFile(suffix=".pov", mode='w+t', delete=False)
        pov_file.write(pov); pov_file.flush(); os.fsync(pov_file.fileno())
        subprocess.run(["povray",
            "-D", # Disables immediate graphical output, save to file instead.
            "+O" + output_filename,
            "+W" + str(resolution[0]),
            "+H" + str(resolution[1]),
            pov_file.name,],
            stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL,
            check=True,)
        os.remove(pov_file.name)

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
            dv = amps * min(time_step, t)/1000 / capacitances[location]
            voltages[location] += dv
            self.remaining[idx] -= time_step
        keep = [t > 0 for t in self.remaining]
        self.currents  = [x for k, x in zip(keep, self.currents) if k]
        self.locations = [x for k, x in zip(keep, self.locations) if k]
        self.remaining = [x for k, x in zip(keep, self.remaining) if k]

    def inject_current(self, location, current, duration = 1.4):
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

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.model == other.model and
                self.index == other.index)

    def __hash__(self):
        # TODO: This is a terrible design, BC the entity index is not immutable.
        return self.index

    @property
    def index(self):
        return self.entity.index

    @property
    def parent(self):
        parent = self.entity.read("membrane/parents")
        if parent != NULL:  return Segment(self.model, parent)
        else:               return None

    @property
    def children(self):
        children = self.entity.read("membrane/children")
        return [Segment(self.model, c) for c in children.indices]

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

    # TODO: Consider renaming this to "get_voltage" so that its name contains a verb.
    def voltage(self):
        return self.entity.read("membrane/voltages")

    def inject_current(self, current, duration=1e-3):
        self.model._injected_currents.inject_current(self.entity.index, current, duration)

class Animation:
    def __init__(self, model, color_function, text_function = None,
            skip = 0,
            scale = None,
            **render_args):
        """
        Argument scale: multiplier on the image dimensions to reduce filesize.
        Argument skip: Don't render this many frames between every actual render.
        """
        self.model = model
        model.animations.add(self)
        self.color_function = color_function
        self.text_function = text_function
        self.render_args = render_args
        self.frames_dir = tempfile.TemporaryDirectory()
        self.frames = []
        self.skip = int(skip)
        self.ticks = 0
        self.scale = scale

    def _advance(self):
        """
        Argument text: is overlayed on the top right corner.
        """
        self.ticks += 1
        if self.ticks % (self.skip+1) != 0: return
        colors = self.color_function(self.model.db.access)
        if self.text_function:  text = self.text_function(self.model.db.access)
        else:                   text = None
        self.frames.append(os.path.join(self.frames_dir.name, str(len(self.frames))+".png"))
        self.model.render_frame(colors, self.frames[-1], **self.render_args)
        if text or self.scale:
            img = Image.open(self.frames[-1])
            if self.scale is not None:
                new_size = (int(round(img.size[0] * self.scale)), int(round(img.size[1] * self.scale)))
                img = img.resize(new_size, resample=Image.LANCZOS)
            if text:
                draw = ImageDraw.Draw(img)
                draw.text((5, 5), text, (0, 0, 0))
            img.save(self.frames[-1])

    def start(self):
        self.model.animations.add(self)

    def pause(self):
        if self in self.model.animations:
            self.model.animations.remove(self)

    def save(self, output_filename):
        """ Save into a GIF file that loops forever. """
        self.frames = [Image.open(i) for i in self.frames] # Load all of the frames.
        dt = (self.skip+1) * self.model.access("time_step")
        self.frames[0].save(output_filename, format='GIF',
                append_images=self.frames[1:], save_all=True,
                duration=int(round(dt * 1e3)), # Milliseconds per frame.
                optimize=True, quality=0,
                loop=0,) # Loop forever.

    def stop(self, output_filename):
        self.pause()
        self.save(output_filename)
        self.frames.clear()
        self.ticks = 0

@cp.fuse()
def _area_circle(diameter):
    return np.pi * (diameter / 2.0) ** 2

@cp.fuse()
def _volume_sphere(diameter):
    return 1/0

@cp.fuse()
def _surface_area_sphere(diameter):
    return 1/0

def _surface_area_frustum(radius_1, radius_2, length):
    """ Lateral surface area, does not include the ends. """
    s = sqrt((radius_1 - radius_2) ** 2 + length ** 2)
    return np.pi * (radius_1 + radius_2) * s

def _volume_of_frustum(radius_1, radius_2, length):
    return np.pi / 3.0 * length * (radius_1 * radius_1 + radius_1 * radius_2 + radius_2 * radius_2)

def _electric_coefficients(access):
    """
    Model the electric currents over the membrane surface.
    Compute the coefficients of the derivative function:
    dV/dt = C * V, where C is Coefficients matrix and V is voltage vector.
    """
    dt           = access("time_step") / 1000 / 2
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

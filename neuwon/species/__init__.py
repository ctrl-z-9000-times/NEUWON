from collections.abc import Callable, Iterable, Mapping
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
import cupy as cp
import math
import neuwon.species.voronoi
import numba.cuda
import numpy as np

F = 96485.3321233100184 # Faraday's constant, Coulombs per Mole of electrons
R = 8.31446261815324 # Universal gas constant
zero_c = 273.15 # Temperature, in Kelvins.

class Species:
    """ """
    def __init__(self, name,
            charge = 0,
            reversal_potential    = None,
            inside_concentration: "millimolar"  = None,
            outside_concentration: "millimolar" = None,
            inside_diffusivity    = None,
            outside_diffusivity   = None,
            inside_decay_period   = float("inf"),
            outside_decay_period  = float("inf"),
            use_shells = False,
            outside_grid = None,):
        """
        Arguments
        * inside_concentration:  initial value.
        * outside_concentration: initial value.
        * reversal_potential: is one of: number, "nerst", "goldman_hodgkin_katz"

        If diffusivity is not given, then the concentration is a global constant.
        """
        self.name = str(name)
        self.charge = int(charge)
        if reversal_potential is None:
            self.reversal_potential = None
        else:
            try:
                self.reversal_potential = float(reversal_potential)
            except ValueError:
                self.reversal_potential = str(reversal_potential)
                assert self.reversal_potential in ("nerst", "goldman_hodgkin_katz")
        self.electric = (self.charge != 0) or (self.reversal_potential is not None)
        if self.electric: assert self.reversal_potential is not None
        self.inside_global_const    = inside_diffusivity is None
        self.outside_global_const   = outside_diffusivity is None
        self.inside_diffusivity     = float(inside_diffusivity) if not self.inside_global_const else 0.0
        self.outside_diffusivity    = float(outside_diffusivity) if not self.outside_global_const else 0.0
        self.inside_decay_period    = float(inside_decay_period)
        self.outside_decay_period   = float(outside_decay_period)
        self.use_shells             = bool(use_shells)
        self.inside_archetype       = "Inside" if self.use_shells else "Segment"
        self.outside_grid           = tuple(float(x) for x in outside_grid) if outside_grid is not None else None
        if inside_concentration is not None:
            self.inside_concentration = float(inside_concentration)
            assert self.inside_concentration  >= 0.0
        else:
            self.inside_concentration = None
            assert self.inside_global_const
            assert self.inside_decay_period == float('inf')
        if outside_concentration is not None:
            self.outside_concentration = float(outside_concentration)
            assert self.outside_concentration >= 0.0
        else:
            self.outside_concentration = None
            assert self.outside_global_const
            assert self.outside_decay_period == float('inf')
        assert self.inside_diffusivity    >= 0
        assert self.outside_diffusivity   >= 0
        assert self.inside_decay_period   > 0.0
        assert self.outside_decay_period  > 0.0
        if self.inside_global_const:  assert self.inside_decay_period == np.inf
        if self.inside_global_const:  assert not self.use_shells
        if self.outside_global_const: assert self.outside_decay_period == np.inf

    def get_name(self) -> str:
        return self.name

    def __repr__(self):
        return "neuwon.species.Species(%s)"%self.name

    def _initialize(self, database, time_step, celsius, input_clock):
        inside_cls = database.get(self.inside_archetype)
        if self.inside_concentration is not None:
            if self.inside_global_const:
                inside_cls.add_class_attribute(f"{self.name}_concentration",
                        initial_value=self.inside_concentration,
                        units="millimolar")
            else:
                inside_cls.add_attribute(f"{self.name}_concentration",
                        initial_value=self.inside_concentration,
                        units="millimolar")
                inside_cls.add_attribute(f"{self.name}_delta_concentration",
                        initial_value=0.0,
                        units="millimolar / timestep")
        if self.outside_concentration is not None:
            outside_cls = database.get("Outside")
            if self.outside_global_const:
                outside_cls.add_class_attribute(f"{self.name}_concentration",
                        self.outside_concentration,
                        units="millimolar")
            else:
                outside_cls.add_attribute(f"{self.name}_concentration",
                        initial_value=self.outside_concentration,
                        units="millimolar")
                outside_cls.add_attribute(f"{self.name}_delta_concentration",
                        initial_value=0.0,
                        units="millimolar / timestep")
        if self.electric:
            segment_cls = database.get("Segment")
            segment_cls.add_attribute(f"{self.name}_conductance",
                    initial_value=0.0,
                    valid_range=(0, np.inf),
                    units="Siemens")
            if isinstance(self.reversal_potential, float):
                segment_cls.add_class_attribute(f"{self.name}_reversal_potential",
                        initial_value=self.reversal_potential,
                        units="mV")
            else:
                segment_cls.add_attribute(f"{self.name}_reversal_potential",
                        initial_value=np.nan,
                        units="mV")
            input_clock.register_callback(lambda: self._accumulate_conductance(database, celsius) or True)

    def _compute_reversal_potential(self, database, celsius):
        x = database.get_data(f"Segment.{self.name}_reversal_potential")
        if isinstance(x, float): return x
        1/0 # The following code needs to be rewritten for the new database & schema.
        inside  = access(self.inside_archetype+"/concentrations/"+self.name)
        outside = access("outside/concentrations/"+self.name)
        if not isinstance(inside, float) and self.use_shells:
            inside = inside[access("membrane/inside")]
        if not isinstance(outside, float):
            outside = outside[access("membrane/outside")]
        T = access("T")
        if self.reversal_potential == "nerst":
            x[:] = self._nerst_potential(self.charge, T, inside, outside)
        elif self.reversal_potential == "goldman_hodgkin_katz":
            voltages = access("membrane/voltages")
            x[:] = self._goldman_hodgkin_katz(self.charge, T, inside, outside, voltages)
        else: raise NotImplementedError(self.reversal_potential)
        return x

    def _zero_accumulators(self, database):
        def zero(component_name):
            database.get_data(component_name).fill(0.0)
        if self.electric:
            zero(f"Segment.{self.name}_conductance")
        if self.inside_diffusivity != 0.0:
            zero(self.inside_archetype + "/delta_concentrations/"+self.name)
        if self.outside_diffusivity != 0.0:
            zero("outside/delta_concentrations/"+self.name)

    def _accumulate_conductance(self, database, celsius):
        sum_conductance     = database.get_data("Segment.sum_conductance")
        driving_voltage     = database.get_data("Segment.driving_voltage")
        species_conductance = database.get_data(f"Segment.{self.name}_conductance")
        reversal_potential  = self._compute_reversal_potential(database, celsius)
        sum_conductance += species_conductance
        driving_voltage += species_conductance * reversal_potential

    def _advance(self, database):
        # Calculate the transmembrane ion flows.
        if self.inside_global_const and self.outside_global_const: return
        if not (self.electric and self.charge != 0): return
        reversal_potential = access("membrane/reversal_potentials/"+self.name)
        g = access("membrane/conductances/"+self.name)
        millimoles = g * (dt * reversal_potential - integral_v) / (self.charge * F)
        if self.inside_diffusivity != 0:
            if self.use_shells:
                1/0
            else:
                volumes        = access("membrane/inside/volumes")
                concentrations = access("membrane/inside/concentrations/"+self.name)
                concentrations += millimoles / volumes
        if self.outside_diffusivity != 0:
            volumes = access("outside/volumes")
            self.outside.concentrations -= millimoles / self.geometry.outside_volumes
        # Update chemical concentrations with local changes and diffusion.
        if not self.inside_global_const:
            x    = access("membrane/inside/concentrations/"+self.name)
            rr   = access("membrane/inside/delta_concentrations/"+self.name)
            irm  = access("membrane/inside/diffusions/"+self.name)
            x[:] = irm.dot(cp.maximum(0, x + rr * 0.5))
        if not self.outside_global_const:
            x    = access("outside/concentrations/"+self.name)
            rr   = access("outside/delta_concentrations/"+self.name)
            irm  = access("outside/diffusions/"+self.name)
            x[:] = irm.dot(cp.maximum(0, x + rr * 0.5))

    def _inside_diffusion_coefficients(self, access):
        dt      = access("time_step") / 1000 / _ITERATIONS_PER_TIMESTEP
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
        dt          = access("time_step") / 1000 / _ITERATIONS_PER_TIMESTEP
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

def _nerst_potential(charge, T, inside_concentration, outside_concentration):
    xp = cp.get_array_module(inside_concentration)
    ratio = xp.divide(outside_concentration, inside_concentration)
    return xp.nan_to_num(1e3 * R * T / F / charge * xp.log(ratio))

def _goldman_hodgkin_katz(charge, T, inside_concentration, outside_concentration, voltages):
    xp = cp.get_array_module(inside_concentration)
    inside_concentration  = inside_concentration * 1e-3  # Convert from millimolar to molar
    outside_concentration = outside_concentration * 1e-3 # Convert from millimolar to molar
    z = (charge * F / (R * T)) * voltages
    return ((1e3 * charge * F) *
            (inside_concentration * Species._efun(-z) - outside_concentration * Species._efun(z)))

@cp.fuse()
def _efun(z):
    if abs(z) < 1e-4:
        return 1 - z / 2
    else:
        return z / (math.exp(z) - 1)

class InsideMethods:
    @classmethod
    def _initialize(cls):
        db.add_archetype("inside", doc="Intracellular space.")
        db.add_attribute("membrane/inside", dtype="inside", doc="""
                A reference to the outermost shell.
                The shells and the innermost core are allocated in a contiguous block
                with this referencing the start of range of length "membrane/shells" + 1.
                """)
        db.add_attribute("membrane/shells", dtype=np.uint8)
        db.add_attribute("inside/membrane", dtype="membrane")
        db.add_attribute("inside/shell_radius", units="μm")
        db.add_attribute("inside/volumes",
                # bounds=(epsilon * (1e-6)**3, None),
                allow_invalid=True,
                units="Liters")
        db.add_sparse_matrix("inside/neighbor_distances", "inside")
        db.add_sparse_matrix("inside/neighbor_border_areas", "inside")

class OutsideMethods:
    @classmethod
    def _initialize(cls):
        db.add_archetype("outside", doc="Extracellular space using a voronoi diagram.")
        db.add_attribute("membrane/outside", dtype="outside", doc="")
        db.add_attribute("outside/coordinates", shape=(3,), units="μm")
        db.add_kd_tree(  "outside/tree", "outside/coordinates")
        db.add_attribute("outside/volumes", units="Liters")
        db.add_sparse_matrix("outside/neighbor_distances", "outside")
        db.add_sparse_matrix("outside/neighbor_border_areas", "outside")


    def _initialize_outside(self, locations):
        self._initialize_outside_inner(locations)
        touched = set()
        for neighbors in self.db.access("outside/neighbor_distances")[locations]:
            touched.update(neighbors.indices)
        touched.difference_update(set(locations))
        self._initialize_outside_inner(list(touched))

        outside_volumes = access("outside/volumes")
        fh_space = self.fh_space * s_areas[membrane_idx] * 1000
        outside_volumes[access("membrane/outside")[membrane_idx]] = fh_space

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
            volume, neighbors = neuwon.species.voronoi.voronoi_cell(location,
                    self.max_outside_radius, np.array(potential_neighbors, dtype=Pointer), coordinates)
            write_neighbor_cols.append(list(neighbors['location']))
            write_neighbor_dist.append(list(neighbors['distance']))
            write_neighbor_area.append(list(neighbors['border_surface_area']))
        self.db.access("outside/neighbor_distances",
                sparse_matrix_write=(locations, write_neighbor_cols, write_neighbor_dist))
        self.db.access("outside/neighbor_border_areas",
                sparse_matrix_write=(locations, write_neighbor_cols, write_neighbor_area))



class _Linear_System:
    def __init__(self, class_type, name, function, epsilon, doc="", allow_invalid=False,):
        """ Add a system of linear & time-invariant differential equations.

        Argument function(database_access) -> coefficients

        For equations of the form: dX/dt = C * X
        Where X is a component, of the same archetype as this linear system.
        Where C is a matrix of coefficients, returned by the argument "function".

        The database computes the propagator matrix but does not apply it.
        The matrix is updated after any of the entity are created or destroyed.
        """
        _Component.__init__(self, class_type, name, doc, allow_invalid=allow_invalid)
        self.function   = function
        self.epsilon    = float(epsilon)
        self.data       = None

        coef = self.function(self.cls.database)
        coef = scipy.sparse.csc_matrix(coef, shape=(self.archetype.size, self.archetype.size))
        # Note: always use double precision floating point for building the impulse response matrix.
        # TODO: Detect if the user returns f32 and auto-convert it to f64.
        matrix = scipy.sparse.linalg.expm(coef)
        # Prune the impulse response matrix.
        matrix.data[np.abs(matrix.data) < self.epsilon] = 0
        matrix.eliminate_zeros()
        self.data = cupyx.scipy.sparse.csr_matrix(matrix, dtype=Real)


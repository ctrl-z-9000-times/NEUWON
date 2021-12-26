from collections.abc import Callable, Iterable, Mapping
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse
import scipy.sparse.linalg
from neuwon.database import Clock
from neuwon.parameters import Parameters
import cupy as cp
import math
import numba.cuda
import numpy as np

F = 96485.3321233100184 # Faraday's constant, Coulombs per Mole of electrons
R = 8.31446261815324 # Universal gas constant
zero_c = 273.15 # Temperature, in Kelvins.

class SpeciesInstance:
    """ A species in a location. """
    def __init__(self, time_step, db_class, name, initial_concentration, *,
            diffusivity=None,
            geometry_component=None,
            decay_period=math.inf,
            constant=False):
        self.name = name
        self.initial_concentration = float(initial_concentration)
        assert self.initial_concentration >= 0.0
        if diffusivity is None:
            self.diffusivity = None 
        else:
            self.diffusivity = float(diffusivity)
            self.diffusion_matrix = db_class.add_sparse_matrix(
                    self.name + '_diffusion',
                    db_class,
                    doc='Propagator matrix.')
        if geometry_component is None:
            self.geometry = None
            assert self.diffusivity is None
        else:
            self.geometry = db_class.get(geometry_component)
            assert self.diffusivity >= 0
            self._matrix_valid = False
        self.decay_period = float(decay_period)
        assert self.decay_period > 0.0
        self.constant = bool(constant)
        if self.constant:
            assert self.decay_period == math.inf
            assert self.diffusivity is None

        add_attr = db_class.add_class_attribute if self.constant else db_class.add_attribute
        self.concentrations = add_attr(f"{self.name}",
                self.initial_concentration,
                units="millimolar")
        if not self.constant:
            self.deltas = db_class.add_attribute(f"{self.name}_delta",
                    initial_value=0.0,
                    units="millimolar / timestep")

    def _zero_accumulators(self):
        if not self.constant:
            self.deltas.get_data().fill(0.0)

    def _advance(self):
        """ Update the chemical concentrations with local changes and diffusion. """
        if self.constant:
            return
        c  = self.concentrations.get_data()
        rr = self.deltas.get_data()
        xp = self.concentrations.get_database().get_memory_space().get_array_module()
        xp.maximum(0, c + rr * 0.5, out=c)
        if self.diffusivity is None:
            return
        if not self._matrix_valid:
            self._compute_matrix()
        m = self.diffusion_matrix.get_data()
        c = m.dot(c)
        self.concentrations.set_data(c)

    # Who sets _matrix_valid=False?

    def _compute_matrix(self):
        1/0
        # Note: always use double precision floating point for building the impulse response matrix.
        coef = scipy.sparse.csc_matrix(coef, shape=(len(db_cls), len(db_cls)), dtype=np.float64)
        matrix = scipy.sparse.linalg.expm(coef)
        # Prune the impulse response matrix.
        matrix.data[np.abs(matrix.data) < epsilon] = 0.0
        matrix.eliminate_zeros()
        db_cls.get("electric_propagator_matrix").to_csr().set_data(matrix)
        self._matrix_valid = True

class SpeciesType:
    def __init__(self, name, factory,
                charge = 0,
                reversal_potential: '(None|float|"nerst"|"goldman_hodgkin_katz")' = None,
                initial_concentration = None,
                diffusivity = None,
                inside = None,
                outside = None,
                ):
        self.name       = str(name)
        self.factory    = factory
        self.charge     = int(charge)
        if reversal_potential is None:
            self.reversal_potential = None
        else:
            try:
                self.reversal_potential = float(reversal_potential)
            except ValueError:
                self.reversal_potential = str(reversal_potential)
                assert self.reversal_potential in ("nerst", "goldman_hodgkin_katz")
        self.electric = (self.charge != 0) or (self.reversal_potential is not None)
        if self.electric:
            assert self.reversal_potential is not None
            segment_data = factory.database.get("Segment")
            self.conductance = segment_data.add_attribute(f"{self.name}_conductance",
                    initial_value=0.0,
                    valid_range=(0, np.inf),
                    units="Siemens")
            if isinstance(self.reversal_potential, float):
                self.reversal_potential_data = segment_data.add_class_attribute(
                        f"{self.name}_reversal_potential",
                        initial_value=self.reversal_potential,
                        units="mV")
            else:
                self.reversal_potential_data = segment_data.add_attribute(
                        f"{self.name}_reversal_potential",
                        initial_value=0.0,
                        units="mV")
            factory.input_hook.register_callback(self._accumulate_conductance)

        self.inside = None
        self.outside = None
        if inside is not None:
            inside = Parameters(inside)
            db_class = self.factory.database.get_class("Segment")
            geometry_component = 1/0
            self.inside = SpeciesInstance(self.factory.time_step, db_class, self.name,
                    geometry_component=None,
                    **inside)

        if outside is not None:
            self.outside_grid       = tuple(float(x) for x in outside_grid) if outside_grid is not None else None
            self.outside = SpeciesInstance(**parameters['outside'])

    def get_name(self) -> str:
        return self.name

    def _zero_accumulators(self):
        if self.electric:
            self.conductance.free()
        for instance in (self.inside, self.outside):
            if instance is not None:
                instance._zero_accumulators()

    def _accumulate_conductance(self):
        database            = self.factory.database
        sum_conductance     = database.get_data("Segment.sum_conductance")
        driving_voltage     = database.get_data("Segment.driving_voltage")
        species_conductance = self.conductance.get_data()
        reversal_potential  = self._compute_reversal_potential()
        sum_conductance += species_conductance
        driving_voltage += species_conductance * reversal_potential
        return True

    def _compute_reversal_potential(self):
        x = self.reversal_potential_data.get_data()
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

    def _advance(self):
        return # This function is not needed yet.

        # Calculate the transmembrane ion flows.
        # if (self.electric and self.charge != 0):
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

        if self.inside: self.inside._advance()
        if self.outside: self.outside._advance()

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

class SpeciesFactory(dict):
    def __init__(self, parameters: dict, database, time_step, celsius):
        super().__init__()
        self.database   = database
        self.time_step  = time_step
        self.celsius    = celsius
        # TODO: Come up with a better name than "input_hook". Maybe "accumulate_conductance_hook"?
        self.input_hook = Clock(time_step)
        self.add_parameters(parameters)

    def add_parameters(self, parameters: dict):
        for name, species in Parameters(parameters).items():
            self.add_species(name, species)

    def add_species(self, name, species) -> SpeciesType:
        if name in self:
            return self[name]
        if not isinstance(species, SpeciesType):
            species = SpeciesType(name, species, self)
        self[species.name] = species
        return species

    def _zero_accumulators(self):
        """ Zero all data buffers which the mechanisms can write to. """
        for species in self.values():
            species._zero_accumulators()

    def _advance(self):
        """ """
        for species in self.values():
            species._advance()

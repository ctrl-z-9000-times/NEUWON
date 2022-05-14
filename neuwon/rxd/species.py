from collections.abc import Callable, Iterable, Mapping
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse
import scipy.sparse.linalg
from neuwon.database import Clock
import cupy as cp
import math
import numba.cuda
import numpy as np

F = 96485.3321233100184 # Faraday's constant, Coulombs per Mole of electrons
R = 8.31446261815324 # Universal gas constant
zero_c = 273.15 # Temperature, in Kelvins.

class SpeciesInstance:
    """ A species in a location. """
    def __init__(self, time_step, db_class, name, *, initial_concentration,
                 global_constant, decay_period, diffusivity, geometry_component):
        self.time_step          = time_step
        self.db_class           = db_class
        self.name               = name
        self.global_constant    = bool(global_constant)
        self.decay_period       = float(decay_period)
        self.diffusivity        = float(diffusivity)
        assert self.decay_period > 0.0
        assert self.diffusivity >= 0.0
        if self.global_constant:
            assert self.decay_period == math.inf
            assert self.diffusivity == 0.0

        if self.diffusivity != 0.0:
            self.geometry = db_class.get(geometry_component)
            self.diffusion_matrix = db_class.add_sparse_matrix(
                    self.name + '_diffusion',
                    db_class,
                    doc='Propagator matrix.')
            self._matrix_valid = False
        if self.global_constant:
            self.concentration = db_class.add_class_attribute(f"{self.name}",
                    initial_concentration,
                    valid_range=(0.0, np.inf),
                    units="millimolar")
        else:
            self.concentration = db_class.add_attribute(f"{self.name}",
                    initial_concentration,
                    valid_range=(0.0, np.inf),
                    units="millimolar")
            self.delta = db_class.add_attribute(f"{self.name}_delta",
                    initial_value=0.0,
                    units="millimolar / timestep")

    def __repr__(self):
        return f'<Species: {self.name} @ {self.db_class.get_name()}>'

    def _zero_input_accumulators(self):
        if not self.global_constant:
            self.delta.get_data().fill(0.0)

    def _advance(self):
        """ Update the chemical concentrations with local changes and diffusion. """
        if self.global_constant:
            return
        c  = self.concentration.get_data()
        rr = self.delta.get_data()
        xp = self.concentration.get_database().get_memory_space().get_array_module()
        xp.maximum(0, c + rr * 0.5, out=c)
        if self.diffusivity == 0.0:
            if self.decay_period < math.inf:
                c *= math.exp(-self.time_step / self.decay_period)
            return
        if not self._matrix_valid:
            self._compute_matrix()
        m = self.diffusion_matrix.get_data()
        c = m.dot(c)
        self.concentration.set_data(c)

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
    def __init__(self, name, factory, *,
                charge              = 0,
                reversal_potential  = np.nan,
                diffusivity         = 0.0,
                decay_period        = np.inf,
                inside_initial_concentration    = 0.0,
                inside_global_constant          = True,
                outside_initial_concentration   = 0.0,
                outside_global_constant         = True,):
        self.name       = str(name)
        self.factory    = factory
        self.charge     = int(charge)
        try:
            self.reversal_potential = float(reversal_potential)
        except ValueError:
            self.reversal_potential = str(reversal_potential).lower()
            assert self.reversal_potential in ("nerst", "ghk")
        self.electric = (self.charge != 0)
        if self.electric:
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

        db_class = self.factory.database.get_class("Segment")
        self.inside = SpeciesInstance(self.factory.time_step, db_class, self.name,
                initial_concentration   = inside_initial_concentration,
                global_constant         = inside_global_constant,
                decay_period            = decay_period,
                diffusivity             = diffusivity,
                geometry_component      = None)

        db_class = self.factory.database.get_class("Extracellular")
        self.outside = SpeciesInstance(self.factory.time_step, db_class, self.name,
                initial_concentration   = outside_initial_concentration,
                global_constant         = outside_global_constant,
                decay_period            = decay_period,
                diffusivity             = diffusivity,
                geometry_component      = None)

    def get_name(self) -> str:
        return self.name

    def __repr__(self):
        return f'<Species: {self.name}>'

    def _zero_input_accumulators(self):
        if self.electric:
            self.conductance.get_data().fill(0.0)
        self.inside ._zero_input_accumulators()
        self.outside._zero_input_accumulators()

    def _accumulate_conductance(self):
        database            = self.factory.database
        sum_conductance     = database.get_data("Segment.sum_conductance")
        driving_voltage     = database.get_data("Segment.driving_voltage")
        species_conductance = self.conductance.get_data()
        reversal_potential  = self._compute_reversal_potential()
        sum_conductance += species_conductance
        driving_voltage += species_conductance * reversal_potential

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
        elif self.reversal_potential == "ghk":
            voltages = access("membrane/voltages")
            x[:] = self._goldman_hodgkin_katz(self.charge, T, inside, outside, voltages)
        else:
            raise NotImplementedError(self.reversal_potential)
        return x

    def _advance(self):
        if False:
            # This function is not needed yet.

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

        if self.inside:  self.inside._advance()
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
            (inside_concentration * _efun(-z) - outside_concentration * _efun(z)))

@cp.fuse()
def _efun(z):
    if abs(z) < 1e-4:
        return 1 - z / 2
    else:
        return z / (math.exp(z) - 1)

class SpeciesFactory(dict):
    def __init__(self, parameters: dict, database, input_hook, temperature):
        super().__init__()
        self.database       = database
        self.input_hook     = input_hook
        self.time_step      = input_hook.get_time_step()
        self.temperature    = temperature
        self.add_parameters(parameters)

    def add_parameters(self, parameters: dict):
        for name, species_kwargs in parameters.items():
            self.add_species(name, species_kwargs)

    def add_species(self, name, species_kwargs) -> SpeciesType:
        if name in self:
            species = self[name]
        elif isinstance(species_kwargs, SpeciesType):
            self[name] = species = species_kwargs
        else:
            self[name] = species = SpeciesType(name, self, **species_kwargs)
        return species

    def _zero_input_accumulators(self):
        """ Zero all data buffers which the mechanisms can write to. """
        for species in self.values():
            species._zero_input_accumulators()

    def _advance(self):
        """ """
        for species in self.values():
            species._advance()

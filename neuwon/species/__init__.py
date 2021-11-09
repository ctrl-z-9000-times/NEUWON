from collections.abc import Callable, Iterable, Mapping
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
from neuwon.database import Clock
import cupy as cp
import math
import numba.cuda
import numpy as np

F = 96485.3321233100184 # Faraday's constant, Coulombs per Mole of electrons
R = 8.31446261815324 # Universal gas constant
zero_c = 273.15 # Temperature, in Kelvins.

"""
Parameters
* concentration: initial value.
* reversal_potential: is one of: number, "nerst", "goldman_hodgkin_katz"

If diffusivity is not given, then the concentration is a global constant.
"""


class SpeciesFactory(dict):
    def __init__(self, parameters: dict, database, time_step, celsius):
        super().__init__()
        self.database   = database
        self.time_step  = time_step
        self.celsius    = celsius
        self.input_clock = Clock(time_step)
        self.add_parameters(parameters)

    def add_parameters(self, parameters:dict):
        for name, species in parameters.items():
            self.add_species(name, species)

    def add_species(self, name, species) -> 'Species':
        if name in self:
            return self[name]
        if not isinstance(species, Species):
            species = Species(name, species, self)
        self.species[species.name] = species
        return species

default_species_type_parameters = {
    'charge': 0,
    'reversal_potential': None,
}

class SpeciesType:
    def __init__(self, name, parameters, factory):
        parameters.update_with_defaults(default_species_type_parameters)
        self.name       = str(name)
        self.celsius    = factory.celsius
        self.charge     = int(parameters['charge'])
        self.reversal_potential = parameters['reversal_potential']
        if self.reversal_potential is not None:
            try:
                self.reversal_potential = float(reversal_potential)
            except ValueError:
                self.reversal_potential = str(reversal_potential)
                assert self.reversal_potential in ("nerst", "goldman_hodgkin_katz")
        self.electric = (self.charge != 0) or (self.reversal_potential is not None)
        if self.electric:
            assert self.reversal_potential is not None
            segment_cls = factory.database.get("Segment")
            self.conductance = segment_cls.add_attribute(f"{self.name}_conductance",
                    initial_value=0.0,
                    valid_range=(0, np.inf),
                    units="Siemens")
            if isinstance(self.reversal_potential, float):
                self.reversal_potential_data = segment_cls.add_class_attribute(
                        f"{self.name}_reversal_potential",
                        initial_value=self.reversal_potential,
                        units="mV")
            else:
                self.reversal_potential_data = segment_cls.add_attribute(
                        f"{self.name}_reversal_potential",
                        initial_value=0.0,
                        units="mV")
            factory.input_clock.register_callback(self._accumulate_conductance)

        self.inside = None
        self.outside = None
        if 'inside' in parameters:
            1/0
            self.use_shells         = bool(use_shells)
            self.inside_archetype   = "Inside" if self.use_shells else "Segment"
            self.inside = SpeciesInstance(**parameters['inside'])
        if 'outside' in parameters:
            1/0
            self.outside_grid       = tuple(float(x) for x in outside_grid) if outside_grid is not None else None
            self.outside = SpeciesInstance(**parameters['outside'])

    def get_name(self) -> str:
        return self.name

    def __repr__(self):
        return f"<{type(self).__name__}: {self.name}>"

    def _zero_accumulators(self):
        if self.electric:
            self.conductance.get_data().fill(0.0)
        for instance in (self.inside, self.outside):
            instance._zero_accumulators()

    def _accumulate_conductance(self):
        sum_conductance     = database.get_data("Segment.sum_conductance")
        driving_voltage     = database.get_data("Segment.driving_voltage")
        species_conductance = database.get_data(f"Segment.{self.name}_conductance")
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

class SpeciesInstance:
    def __init__(self, db_class,
            geometry_component=None,
            concentration=None,
            diffusivity=None,
            decay_period=float('inf')):
        """ Argument geometry_component refers to the ratio of: (length / x-area)??? """
        self.global_const   = diffusivity is None
        self.diffusivity    = None if self.global_const else float(diffusivity)
        self.decay_period   = float(decay_period)
        if concentration is None:
            self.concentration = None
            assert self.diffusivity is None
        else:
            self.concentration = float(concentration)
            assert self.concentration >= 0.0
            add_attr = db_class.add_class_attribute if self.global_const else db_class.add_attribute
            add_attr(f"{self.name}_concentration",
                    self.concentration,
                    units="millimolar")
            if not self.global_const:
                self.delta = db_class.add_attribute(f"{self.name}_delta_concentration",
                        initial_value=0.0,
                        units="millimolar / timestep")

        assert self.diffusivity >= 0
        assert self.decay_period > 0.0
        if self.global_const: assert self.decay_period == float('inf')
        1/0

    def _zero_accumulators(self):
        self.delta.get_data().fill(0.0)

    def _advance(self):
        """ Update the chemical concentrations with local changes and diffusion. """
        1/0
        x    = access("concentrations/"+self.name)
        rr   = access("delta_concentrations/"+self.name)
        irm  = access("diffusions/"+self.name)
        x[:] = irm.dot(cp.maximum(0, x + rr * 0.5))

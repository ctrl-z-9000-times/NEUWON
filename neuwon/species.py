import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
import cupy as cp
import cupyx.scipy.sparse
import math
import copy
from collections.abc import Callable, Iterable, Mapping
from neuwon.common import Real, epsilon, F, R

library = {
    "na": {
        "charge": 1,
        "transmembrane": True,
        "reversal_potential": "nerst",
        "intra_concentration":  15e-3,
        "extra_concentration": 145e-3,
    },
    "k": {
        "charge": 1,
        "transmembrane": True,
        "reversal_potential": "nerst",
        "intra_concentration": 150e-3,
        "extra_concentration":   4e-3,
    },
    "ca": {
        "charge": 2,
        "transmembrane": True,
        "reversal_potential": "goldman_hodgkin_katz",
        "intra_concentration": 70e-9,
        "extra_concentration": 2e-3,
    },
    "cl": {
        "charge": -1,
        "transmembrane": True,
        "reversal_potential": "nerst",
        "intra_concentration":  10e-3,
        "extra_concentration": 110e-3,
    },
    "glu": {
        # "extra_concentration": 1/0, # TODO!
        "extra_diffusivity": 1e-6, # TODO!
        # "extra_decay_period": 1/0, # TODO!
    },
}

class Species:
    """ """
    def __init__(self, name,
            charge = 0,
            transmembrane = False,
            reversal_potential = "nerst",
            intra_concentration = 0.0,
            extra_concentration = 0.0,
            intra_diffusivity = None,
            extra_diffusivity = None,
            intra_decay_period = float("inf"),
            extra_decay_period = float("inf")):
        """
        If diffusivity is not given, then the concentration is constant.
        Argument reversal_potential is one of: number, "nerst", "goldman_hodgkin_katz"
        """
        self.name = str(name)
        self.charge = int(charge)
        self.transmembrane = bool(transmembrane)
        self.intra_concentration = float(intra_concentration)
        self.extra_concentration = float(extra_concentration)
        self.intra_diffusivity = float(intra_diffusivity) if intra_diffusivity is not None else None
        self.extra_diffusivity = float(extra_diffusivity) if extra_diffusivity is not None else None
        self.intra_decay_period = float(intra_decay_period)
        self.extra_decay_period = float(extra_decay_period)
        assert(self.intra_concentration >= 0.0)
        assert(self.extra_concentration >= 0.0)
        assert(self.intra_diffusivity is None or self.intra_diffusivity >= 0)
        assert(self.extra_diffusivity is None or self.extra_diffusivity >= 0)
        assert(self.intra_decay_period > 0.0)
        assert(self.extra_decay_period > 0.0)
        if reversal_potential == "nerst":
            self.reversal_potential = str(reversal_potential)
            self._reversal_potential_method = lambda T, i, o, v: nerst_potential(self.charge, T, i, o)
        elif reversal_potential == "goldman_hodgkin_katz":
            self.reversal_potential = str(reversal_potential)
            self._reversal_potential_method = self.goldman_hodgkin_katz
        else:
            self.reversal_potential = float(reversal_potential)
            self._reversal_potential_method = lambda T, i, o, v: self.reversal_potential
        # These attributes are initialized on a copy of this object:
        self.intra = None # _Diffusion instance
        self.extra = None # _Diffusion instance
        self.conductances = None # Numpy array

class _AllSpecies(dict):
    """ A dictionary containing all species. """
    def __init__(self, species_argument, time_step, geometry, all_pointers):
        # The given argument species take priority, add them first.
        self._add_species(species_argument)
        # Pull in all species which are required for the reactions.
        for ptr in all_pointers:
            if ptr.species: self._add_species(ptr.species)
        # Fill in unspecified species from the standard library.
        for name, species in self.items():
            if species is None:
                if name in library:
                    self._add_species(Species(name, **library[name]))
                else: raise ValueError("Unresolved species: %s."%name)
        # Initialize the species internal data.
        for species in self.values():
            if species.intra_diffusivity is not None:
                species.intra = _Diffusion(time_step, geometry, species, "intracellular")
            if species.extra_diffusivity is not None:
                species.extra = _Diffusion(time_step, geometry, species, "extracellular")
            if species.transmembrane:
                species.conductances = cp.zeros(len(geometry), dtype=Real)

    def _add_species(self, new_species):
        """ Add a new species to the dictionary if its name is new/unique.

        Argument new_species must be one of:
          * An instance of the Species class,
          * A dictionary of arguments for initializing a new instance of the Species class,
          * The species name as a placeholder string,
          * A list of one of the above.
        """
        if isinstance(new_species, Species):
            if new_species.name not in self or self[new_species.name] is None:
                self[new_species.name] = copy.copy(new_species)
        elif isinstance(new_species, Mapping):
            self._add_species(Species(**new_species))
        elif isinstance(new_species, str):
            if new_species not in self:
                self[new_species] = None
        elif isinstance(new_species, Iterable):
            for x in new_species:
                self._add_species(x)
        else: raise TypeError("Invalid species: %s."%repr(new_species))

    @staticmethod
    def advance(model):
        """ Note: Each call to this method integrates over half a time step. """
        dt = model._electrics.time_step
        # Save prior state.
        model._electrics.previous_voltages = cp.array(model._electrics.voltages, copy=True)
        for s in model._species.values():
            for x in (s.intra, s.extra):
                if x is not None:
                    x.previous_concentrations = cp.array(x.concentrations, copy=True)
        # Accumulate the net conductances and driving voltages from the chemical data.
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

    def check_data(self):
        for s in self.values():
            if s.transmembrane:
                assert cp.all(cp.isfinite(s.conductances)), s.name
            if s.intra is not None:
                assert cp.all(cp.isfinite(s.intra.concentrations)), s.name
                assert cp.all(cp.isfinite(s.intra.previous_concentrations)), s.name
                assert cp.all(cp.isfinite(s.intra.release_rates)), s.name
            if s.extra is not None:
                assert cp.all(cp.isfinite(s.extra.concentrations)), s.name
                assert cp.all(cp.isfinite(s.extra.previous_concentrations)), s.name
                assert cp.all(cp.isfinite(s.extra.release_rates)), s.name

class _Diffusion:
    def __init__(self, time_step, geometry, species, where):
        self.time_step                  = time_step
        self.concentrations             = cp.zeros(len(geometry), dtype=Real)
        self.previous_concentrations    = cp.zeros(len(geometry), dtype=Real) # TODO: Remove this variable!
        self.release_rates              = cp.zeros(len(geometry), dtype=Real)
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
        self.previous_voltages  = cp.full(len(geometry), initial_voltage, dtype=Real) # TODO: Remove this variable!
        self.driving_voltages   = cp.zeros(len(geometry), dtype=Real)
        self.conductances       = cp.zeros(len(geometry), dtype=Real)
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
        self.axial_resistances  = cp.array(self.axial_resistances)
        self.capacitances       = cp.array(self.capacitances)

    def check_data(self):
        assert(cp.all(cp.isfinite(self.voltages)))
        assert(cp.all(cp.isfinite(self.previous_voltages)))
        assert(cp.all(cp.isfinite(self.driving_voltages)))
        assert(cp.all(cp.isfinite(self.conductances)))

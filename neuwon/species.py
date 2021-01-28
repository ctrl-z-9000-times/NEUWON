import numpy as np
import cupy as cp
import math
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
import cupyx.scipy.sparse
from neuwon import Real, epsilon, F, R, T

class Species:
    """ """
    def __init__(self, name,
            charge = 0,
            transmembrane = False,
            reversal_potential = "nerst",
            intra_concentration = 0.0,
            extra_concentration = 0.0,
            intra_diffusivity = None,
            extra_diffusivity = None,):
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
        assert(self.intra_concentration >= 0.0)
        assert(self.extra_concentration >= 0.0)
        assert(self.intra_diffusivity is None or self.intra_diffusivity >= 0)
        assert(self.extra_diffusivity is None or self.extra_diffusivity >= 0)
        if reversal_potential == "nerst":
            self.reversal_potential = str(reversal_potential)
            # Compute the reversal potential in advance if able.
            if self.intra_diffusivity is None and self.extra_diffusivity is None:
                x = self.nerst_potential(self.intra_concentration, self.extra_concentration)
                self._reversal_potential_method = lambda i, o, v: x
            else:
                self._reversal_potential_method = lambda i, o, v: self.nerst_potential(i, o)
        elif reversal_potential == "goldman_hodgkin_katz":
            self.reversal_potential = str(reversal_potential)
            self._reversal_potential_method = self.goldman_hodgkin_katz
        else:
            self.reversal_potential = float(reversal_potential)
            self._reversal_potential_method = lambda i, o, v: self.reversal_potential
        # These attributes are initialized on a copy of this object:
        self.intra = None # Diffusion instance
        self.extra = None # Diffusion instance
        self.conductances = None # Numpy array

    def _initialize(self, time_step, geometry):
        if self.intra_diffusivity is not None:
            self.intra = Diffusion(time_step, geometry, self, "intracellular")
        if self.extra_diffusivity is not None:
            self.extra = Diffusion(time_step, geometry, self, "extracellular")
        if self.transmembrane:
            self.conductances = cp.zeros(len(geometry), dtype=Real)

class Diffusion:
    def __init__(self, time_step, geometry, species, where):
        self.time_step                  = time_step
        self.concentrations             = cp.zeros(len(geometry), dtype=Real)
        self.previous_concentrations    = cp.zeros(len(geometry), dtype=Real)
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
        elif where == "extracellular":
            for location in range(len(geometry)):
                for neighbor in geometry.neighbors[location]:
                    flux = species.extra_diffusivity * neighbor.border_surface_area / neighbor.distance
                    cols.append(location)
                    rows.append(neighbor.location)
                    data.append(+1 * flux / geometry.extra_volumes[neighbor.location])
                    cols.append(location)
                    rows.append(location)
                    data.append(-1 * flux / geometry.extra_volumes[location])
        # Note: always use double precision floating point for building the impulse response matrix.
        coefficients = csc_matrix((data, (rows, cols)), shape=(len(geometry), len(geometry)), dtype=float)
        coefficients.data *= self.time_step
        self.irm = expm(coefficients)
        # Prune the impulse response matrix at epsilon nanomolar (mol/L).
        self.irm.data[np.abs(self.irm.data) < epsilon * 1e-6] = 0
        self.irm = csr_matrix(self.irm, dtype=Real)
        self.irm = cupyx.scipy.sparse.csr_matrix(self.irm)

def nerst_potential(charge, intra_concentration, extra_concentration):
    """ Returns the reversal voltage of this ionic species. """
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

def goldman_hodgkin_katz(charge, intra_concentration, extra_concentration, voltages):
    """ Returns the reversal voltage of this ionic species. """
    xp = cp.get_array_module(intra_concentration)
    if charge == 0: return xp.full_like(intra_concentration, np.nan)
    z = charge * F / (R * T) * voltages
    return charge * F * (intra_concentration * _efun(-z) - extra_concentration * _efun(z))

class Electrics:
    def __init__(self, time_step, geometry,
            intracellular_resistance = 1,
            membrane_capacitance = 1e-2,):
        # Save and check the arguments.
        self.time_step                  = time_step
        self.intracellular_resistance   = float(intracellular_resistance)
        self.membrane_capacitance       = float(membrane_capacitance)
        assert(self.intracellular_resistance > 0)
        assert(self.membrane_capacitance > 0)
        # Initialize data buffers.
        self.voltages           = cp.zeros(len(geometry), dtype=Real)
        self.previous_voltages  = cp.zeros(len(geometry), dtype=Real)
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
        coefficients = csc_matrix((data, (rows, cols)), shape=(len(geometry), len(geometry)), dtype=float)
        coefficients.data *= self.time_step
        self.irm = expm(coefficients)
        # Prune the impulse response matrix at epsilon millivolts.
        self.irm.data[np.abs(self.irm.data) < epsilon * 1e-3] = 0
        self.irm = csr_matrix(self.irm, dtype=Real)
        self.irm = cupyx.scipy.sparse.csr_matrix(self.irm)
        # Move this data to the GPU now that the CPU is done with it.
        self.axial_resistances  = cp.array(self.axial_resistances)
        self.capacitances       = cp.array(self.capacitances)

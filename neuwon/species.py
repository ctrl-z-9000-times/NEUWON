import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
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
        # The Model initializes the following attributes in a copy of this object:
        self.intra = None # Diffusion instance
        self.extra = None # Diffusion instance
        self.conductances = None # Numpy array

    def nerst_potential(self, intra_concentration, extra_concentration):
        """ Returns the reversal voltage of this ionic species. """
        z = self.charge
        if z == 0: return np.full_like(intra_concentration, np.nan)
        ratio = np.divide(extra_concentration, intra_concentration)
        return np.nan_to_num(R * T / F / z * np.log(ratio))

    def goldman_hodgkin_katz(self, intra_concentration, extra_concentration, voltages):
        """ Returns the reversal voltage of this ionic species. """
        if self.charge == 0: return np.full_like(intra_concentration, np.nan)
        def efun(z):
            if abs(z) < 1e-4:
                return 1 - z / 2
            else:
                return z / (np.exp(z) - 1)
        z = self.charge * F / (R * T) * voltages
        return self.charge * F * (intra_concentration * efun(-z) - extra_concentration * efun(z))

class Diffusion:
    def __init__(self, time_step, geometry, species, where):
        self.time_step                  = time_step
        self.concentrations             = np.zeros(len(geometry), dtype=Real)
        self.previous_concentrations    = np.zeros(len(geometry), dtype=Real)
        self.release_rates              = np.zeros(len(geometry), dtype=Real)
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

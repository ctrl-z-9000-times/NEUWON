"""


### Units

All units are prefix-less.
* Meters, Grams, Seconds
* Volts, Amperes, Ohms, Siemens, Farads

"""

__all__ = "Model Geometry Neighbor Mechanism Species".split()

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
from math import pi, sqrt, exp
from abc import ABC, abstractmethod

Real = np.dtype('f4')
Location = np.dtype('u4')
ROOT = 2**32 - 1

class Model:
    def __init__(self, time_step,
            coordinates,
            parents,
            diameters,
            insertions,
            reactions,
            species,
            stagger=True):
        self.time_step = float(time_step)
        self.stagger = bool(stagger)
        self.geometry = Geometry(coordinates, parents, diameters)
        self.reactions = tuple(reactions)
        self.mechanisms = MechanismsContainer(insertions, self.time_step, self.geometry)
        # Each call to advance the diffusions & electrics integrates over half of the time step.
        self.species = SpeciesContainer(species, self.time_step / 2, self.geometry)
        self.electrics = Electrics(self.time_step / 2, self.geometry)

    def __len__(self):
        return len(self.geometry)

    def advance(self):
        if self.stagger:
            self._advance_staggered()
        else:
            self._advance_lockstep()

    def _advance_lockstep(self):
        """ Naive integration strategy, here for reference only. """
        # Update diffusions & electrics for the whole time step using the
        # state of the reactions at the start of the time step.
        self.electrics._update_membrane_circuit(self.species)
        self.species._advance()
        self.electrics._advance()
        self.species._advance()
        self.electrics._advance()
        # Update the reactions for the whole time step using the
        # concentrations & voltages from halfway through the time step.
        r_in, r_out = self.species._setup_reaction_io(self.electrics._previous_voltages)
        self._reactions_advance(r_in, r_out)
        self.mechanisms._advance(r_in, r_out)

    def _advance_staggered(self):
        """
        All systems (reactions, diffusions & electrics) are integrated using
        input values from halfway through their time step. Tracing through the
        exact sequence of operations is difficult because both systems see the
        other system as staggered halfway through their time step.

        For more information see: The NEURON Book, 2003.
        Chapter 4, Section: Efficient handling of nonlinearity.
        """
        self.species._advance()
        self.electrics._advance()
        r_in, r_out = self.species._setup_reaction_io(self.electrics._previous_voltages)
        self._reactions_advance(r_in, r_out)
        self.mechanisms._advance(r_in, r_out)
        self.electrics._update_membrane_circuit(self.species)
        self.species._advance()
        self.electrics._advance()

    def _reactions_advance(self, reaction_inputs, reaction_outputs):
        for reaction in self.reactions:
            for location in range(len(self)):
                reaction(location, reaction_inputs, reaction_outputs)

    def inject_current(self, location, value):
        location = int(location)
        value = float(value)
        dv = value * self.time_step / self.electrics.capacitances[location]
        self.electrics.voltages[location] += dv

class Geometry:
    def __init__(self, coordinates, parents, diameters):
        # Check and save the arguments.
        self.coordinates = np.array(coordinates, dtype=Real)
        self.parents = np.array([ROOT if p is None else p for p in parents], dtype=Location)
        self.diameters = np.array(diameters, dtype=Real)
        assert(len(self.coordinates) == len(self))
        assert(len(self.parents) == len(self))
        assert(len(self.diameters) == len(self))
        assert(all(all(np.isfinite(c)) for c in self.coordinates))
        assert(all(p < len(self) or p == ROOT for p in self.parents))
        assert(all(d >= 0 for d in self.diameters))
        # Compute the children lists.
        self.children = [[] for _ in range(len(self))]
        for location, parent in enumerate(self.parents):
            if not self.is_root(location):
                self.children[parent].append(location)
        assert(all(len(self.children[x]) >= 1 for x in range(len(self)) if self.is_root(x)))
        # The child with the largest diameter is special and is always kept at
        # the start of the children list.
        for siblings in self.children:
            siblings.sort()
        # Compute basic cellular properties.
        self.lengths                = np.empty(len(self), dtype=Real)
        self.surface_areas          = np.empty(len(self), dtype=Real)
        self.cross_sectional_areas  = np.empty(len(self), dtype=Real)
        self.intra_volumes          = np.empty(len(self), dtype=Real)
        for location, parent in enumerate(self.parents):
            coords = self.coordinates[location]
            radius = self.diameters[location] / 2
            if self.is_root(location):
                # Root of new tree. The body of this segment is half of the
                # frustum spanning between this node and its first child.
                eldest = self.children[location][0]
                other_point = (coords + self.coordinates[eldest]) / 2
                other_radius = (radius + (self.diameters[eldest] / 2)) / 2
            elif self.is_root(parent) and self.children[parent][0] == location:
                other_point = (coords + self.coordinates[parent]) / 2
                other_radius = (radius + (self.diameters[parent] / 2)) / 2
            else:
                other_point = self.coordinates[parent]
                other_radius = self.diameters[parent] / 2
            self.lengths[location] = length = np.linalg.norm(coords - other_point)
            self.surface_areas[location] = Geometry._surface_area_frustum(other_radius, radius, length)
            # TODO: there are actually a bunch of special cases on the surface area.
            if len(self.children[location]) == 0:
                pass
            avg_radius = (radius + other_radius) / 2.0
            self.cross_sectional_areas[location] = pi * avg_radius ** 2
            self.intra_volumes[location] = Geometry._volume_of_frustum(other_radius, radius, length)
        # TODO: Compute extracellular properties.
        self.extra_volumes = np.zeros(len(self), dtype=Real)
        self.neighbors = [[] for _ in range(len(self))]

    def __len__(self):
        return len(self.coordinates)

    def is_root(self, location):
        return self.parents[location] == ROOT

    def _surface_area_frustum(radius_1, radius_2, length):
        """ Lateral surface area, does not include the base or top. """
        s = sqrt((radius_1 - radius_2) ** 2 + length ** 2)
        return pi * (radius_1 + radius_2) * s

    def _volume_of_frustum(radius_1, radius_2, length):
        return pi / 3.0 * length * (radius_1 * radius_1 + radius_1 * radius_2 + radius_2 * radius_2)

    def nearest_neighbors(self, k, coordinates, max_distance):
        1/0

class Neighbor:
    # location
    # distance
    # border_surface_area
    pass

class Mechanism(ABC):
    @abstractmethod
    def __init__(self, time_step, location, geometry, *args):
        pass
    @abstractmethod
    def advance(self, reaction_inputs, reaction_outputs):
        pass
    @property
    @abstractmethod
    def species(self):
        """ A list of Species required by this mechanism. """
        return []

class MechanismsContainer:
    def __init__(self, insertions, time_step, geometry):
        assert(len(insertions) == len(geometry))
        self.insertions = [[] for _ in range(len(geometry))]
        self.instances = {}
        for location, mechs_list in enumerate(insertions):
            for mech in mechs_list:
                mech_type = mech[0]
                mech_args = mech[1:]
                assert(issubclass(mech_type, Mechanism))
                inst = mech_type(time_step, location, geometry, *mech_args)
                self.insertions[location].append(inst)
                if mech_type not in self.instances:
                    self.instances[mech_type] = []
                self.instances[mech_type].append(inst)

    def _advance(self, r_in, r_out):
        for mech_type, instances_list in self.instances.items():
            for inst in instances_list:
                inst.advance(r_in, r_out)

class Species:
    def __init__(self,
            name,
            charge = 0,
            conductance = False,
            reversal_potential = None,
            intra_initial_concentration = 0.0,
            extra_initial_concentration = 0.0,
            intra_diffusivity = None,
            extra_diffusivity = None,
        ):
        self.name = str(name)
        self.charge = int(charge)
        if reversal_potential is None:
            self.reversal_potential = None
        else:
            self.reversal_potential = float(reversal_potential)
        self.conductance = bool(conductance)
        self.intra_initial_concentration = float(intra_initial_concentration)
        self.extra_initial_concentration = float(extra_initial_concentration)
        assert(self.intra_initial_concentration >= 0.0)
        assert(self.extra_initial_concentration >= 0.0)
        if intra_diffusivity is None:
            self.intra_diffusivity = None
        else:
            self.intra_diffusivity = float(intra_diffusivity)
            assert(self.intra_diffusivity >= 0)
        if extra_diffusivity is None:
            self.extra_diffusivity = None
        else:
            self.extra_diffusivity = float(extra_diffusivity)
            assert(self.extra_diffusivity >= 0)

    def is_extracellular(self):
        return self.extra_diffusivity is not None
    def is_intracellular(self):
        return self.intra_diffusivity is not None

class SpeciesContainer:
    def __init__(self, species, time_step, geometry):
        self.species = tuple(species)
        assert(all(isinstance(s, Species) for s in self.species))
        assert(len(set(s.name for s in self.species)) == len(self.species))
        self.time_step = time_step
        self.conductances = {}
        self.intracellular = {}
        self.extracellular = {}
        for s in self.species:
            if s.is_intracellular():
                self.intracellular[s.name] = IntracellularDiffusion()
            if s.is_extracellular():
                self.extracellular[s.name] = ExtracellularDiffusion()
            if s.conductance:
                self.conductances[s.name] = np.zeros(len(geometry), dtype=Real)

    def _setup_reaction_io(self, voltages):
        r_in = {}
        for name in "v voltage membrane_potential electric_potential".split():
            r_in[name] = voltages
        r_out = {}
        for s, c in self.conductances.items():
            r_out[s + "_g"] = c
        for s, d in self.intracellular:
            r_out[s + "_intra"] = 1/0
        for s, d in self.extracellular:
            r_out[s + "_extra"] = 1/0
        for data in r_out.values():
            data.fill(0)
        return r_in, r_out

    def _advance(self):
        for diffusion in self.intracellular.values():
            diffusion._advance()
        for diffusion in self.extracellular.values():
            diffusion._advance()

class IntracellularDiffusion:
    def __init__(self, geometry):
        self.diffusivity = 1/0
        self.initial_concentration = 1/0
        self.irm = 1/0
        self.concentrations = 1/0
        self._previous_concentrations = 1/0
        self.release_rates = 1/0

    def _advance(self):
        1/0

class ExtracellularDiffusion:
    def __init__(self, geometry):
        self.diffusivity = 1/0
        self.initial_concentration = 1/0
        self.irm = 1/0
        self.concentrations = 1/0
        self._previous_concentrations = 1/0
        self.release_rates = 1/0

    def _advance(self):
        1/0

class Electrics:
    intracellular_resistance = 1
    membrane_capacitance = 1e-2
    def __init__(self, time_step, geometry):
        self.time_step          = time_step
        self.voltages           = np.zeros(len(geometry), dtype=Real)
        self._previous_voltages = np.zeros(len(geometry), dtype=Real)
        self.driving_voltages   = np.zeros(len(geometry), dtype=Real)
        self.conductances       = np.zeros(len(geometry), dtype=Real)
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
        rows = []; cols = []; data = []
        for location in range(len(geometry)):
            if geometry.is_root(location):
                continue
            parent = geometry.parents[location]
            r = self.axial_resistances[location]
            if geometry.is_root(parent) and geometry.children[parent][0] == location:
                r += self.axial_resistances[parent]
            rows.append(location)
            cols.append(parent)
            data.append(+1 / r / self.capacitances[location])
            rows.append(location)
            cols.append(location)
            data.append(-1 / r / self.capacitances[location])
            rows.append(parent)
            cols.append(location)
            data.append(+1 / r / self.capacitances[parent])
            rows.append(parent)
            cols.append(parent)
            data.append(-1 / r / self.capacitances[parent])
        coefficients = csc_matrix((data, (rows, cols)), shape=(len(geometry), len(geometry)), dtype=Real)
        coefficients.data *= self.time_step
        self.irm = expm(coefficients)
        self.irm = csr_matrix(self.irm, dtype=Real)
        # TODO: Prune the matrix at EPSILON in milivolts.

    def _update_membrane_circuit(self, species_container):
        """ Update the net conductances and driving voltages from the chemical data. """
        self.conductances.fill(0)
        self.driving_voltages.fill(0)
        for species in species_container.species:
            if not species.conductance: continue
            g = species_container.conductances[species.name]
            erev = species.reversal_potential
            self.conductances += g
            self.driving_voltages += g * erev
        self.driving_voltages /= self.conductances
        np.nan_to_num(self.driving_voltages, copy=False)

    def _advance(self):
        self._previous_voltages = np.array(self.voltages, copy=True)
        # Calculate the trans-membrane currents.
        for location in range(len(self.voltages)):
            delta_v = self.driving_voltages[location] - self.voltages[location]
            recip_rc = self.conductances[location] / self.capacitances[location]
            self.voltages[location] += (delta_v * (1.0 - exp(-self.time_step * recip_rc)))
        # Calculate the lateral currents throughout the neurons.
        self.voltages = self.irm.dot(self.voltages)

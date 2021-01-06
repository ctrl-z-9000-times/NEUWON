"""
NEUWON is a simulation framework for neuroscience and artificial intelligence
specializing in conductance based models. This software is a modern remake of
the NEURON simulator. It is accurate, efficient, and easy to use.

All units are prefix-less.
* Meters, Grams, Seconds
* Volts, Amperes, Ohms, Siemens, Farads
* Concentration are in units of Moles / Meter^3
"""
# Public API Entry Points:
__all__ = """Segment Mechanism Reaction Species Model Geometry Neighbor
GrowSomata Growth make_kinetic_table KineticModelTable""".split()
# Numeric/Scientific Library Imports.
import numpy as np
import scipy
import scipy.spatial
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse.linalg
# Standard Library Imports.
import math
import random
import itertools
import functools
import copy
import subprocess
import tempfile
import os
from collections.abc import Callable, Iterable
from collections import namedtuple
# Third Party Imports.
from graph_algorithms import depth_first_traversal as dft

F = 96485.3321233100184 # Faraday's constant, Coulumbs per Mole of electrons
R = 8.31446261815324 # Universal gas constant
celsius = 37 # Human body temperature
T = celsius + 273.15 # Human body temperature in Kelvins

Real = np.dtype('f4')
epsilon = np.finfo(Real).eps
Location = np.dtype('u4')

class Reaction:
    """ Abstract class for specifying reactions between omnipresent species. """
    def required_species(self):
        """ Optional, Returns the Species required by this mechanism.
        Allowed return types: Species, names of species, and lists either. """
        return []
    def advance_reaction(self, time_step, location, reaction_inputs, reaction_outputs):
        """ """
        pass

class Mechanism:
    """ Abstract class for specifying mechanisms which are localized and stateful. """
    def required_species(self):
        """ Optional, Returns the Species required by this mechanism.
        Allowed return types: Species, names of species, and lists either. """
        return []
    def instance_dtype(self):
        """ Returns the numpy data type for a structured array. """
        raise TypeError("Abstract method called!")
    def new_instance(self, time_step, location, geometry, *args):
        """ """
        raise TypeError("Abstract method called!")
    def advance_instance(self, instance, time_step, location, reaction_inputs, reaction_outputs):
        """ """
        raise TypeError("Abstract method called!")

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

def _docstring_wrapper(property_name, docstring):
        def get_prop(self):
            return self.__dict__[property_name]
        def set_prop(self, value):
            self.__dict__[property_name] = value
        return property(get_prop, set_prop, None, docstring)

class Segment:
    parent      = _docstring_wrapper("parent", "Segment or None")
    children    = _docstring_wrapper("children", "List of Segment's")
    coordinates = _docstring_wrapper("coordinates", "Tuple of 3 floats")
    diameter    = _docstring_wrapper("diameter", "Float (positive)")
    insertions  = _docstring_wrapper("insertions", "List of pairs of (mechanisms, mechanisms_arguments_tuple)")
    def __init__(self, coordinates, diameter, parent=None):
        self.model = None
        self.location = None
        self.parent = parent
        assert(isinstance(self.parent, Segment) or self.parent is None)
        self.children = []
        self.coordinates = tuple(float(x) for x in coordinates)
        assert(len(self.coordinates) == 3)
        self.diameter = float(diameter)
        assert(diameter >= 0)
        self.insertions = []
        if self.parent is None:
            self.path_length = 0
        else:
            parent.children.append(self)
            segment_length = np.linalg.norm(np.subtract(parent.coordinates, self.coordinates))
            self.path_length = parent.path_length + segment_length

    def add_segment(self, coordinates, diameter, maximum_segment_length=np.inf):
        coordinates = tuple(float(x) for x in coordinates)
        diameter = float(diameter)
        maximum_segment_length = float(maximum_segment_length)
        assert(maximum_segment_length > 0)
        parent = self
        parent_diameter = self.diameter
        parent_coordinates = self.coordinates
        length = np.linalg.norm(np.subtract(parent_coordinates, coordinates))
        divisions = max(1, math.ceil(length / maximum_segment_length))
        segments = []
        for i in range(divisions):
            x = (i + 1) / divisions
            _x = 1 - x
            coords = (  coordinates[0] * x + parent_coordinates[0] * _x,
                        coordinates[1] * x + parent_coordinates[1] * _x,
                        coordinates[2] * x + parent_coordinates[2] * _x)
            diam = diameter * x + parent_diameter * _x
            child = Segment(coords, diam, parent)
            segments.append(child)
            parent = child
        return segments

    def insert_mechanism(self, mechanism, *args, **kwargs):
        self.insertions.append((mechanism, args, kwargs))

    def set_diameter(self, P = (0, .01, 1e-6)):
        paths = {}
        def mark_paths(node):
            if not node.children:
                paths[node] = [0]
            else:
                paths[node] = []
                for c in node.children:
                    l = np.linalg.norm(np.subtract(node.coordinates, c.coordinates))
                    paths[node].extend(x + l for x in paths[c])
        for _ in dft(self, lambda n: n.children, postorder=mark_paths): pass
        for n in dft(self, lambda n: n.children):
            diameters = []
            for length in paths[n]:
                diameters.append(P[0] * length ** 2 + P[1] * length + P[2])
            n.diameter = np.mean(diameters)

    def get_voltage(self):
        assert(self.model is not None)
        return self.model.electrics.voltages[self.location]

    def inject_current(self, value):
        assert(self.model is not None)
        self.model.inject_current(self.location, value)

from neuwon.growth import Growth, GrowSomata

class Model:
    def __init__(self, time_step,
            neurons,
            reactions,
            species,
            stagger=True):
        self.time_step = float(time_step)
        self.stagger = bool(stagger)
        coordinates, parents, diameters, insertions = self._build_model(neurons)
        assert(len(coordinates) > 0)
        self.geometry = Geometry(coordinates, parents, diameters)
        self.reactions = tuple(reactions)
        assert(all(issubclass(r, Reaction) for r in self.reactions))
        self._init_mechansisms(insertions)
        self._init_species(species)
        self.electrics = Electrics(self.time_step / 2, self.geometry)

    def _build_model(self, neurons):
        roots = set()
        for n in neurons:
            while n.parent:
                n = n.parent
            roots.add(n)
        segments = []
        for r in roots:
            segments.extend(dft(r, lambda x: x.children))
        coordinates = [x.coordinates for x in segments]
        for location, x in enumerate(segments):
            if x.model is not None:
                raise ValueError("Segment included in multiple models!")
            x.location = location
            x.model = self
        parents = [getattr(x.parent, "location", None) for x in segments]
        diameters = [x.diameter for x in segments]
        insertions = [x.insertions for x in segments]
        return (coordinates, parents, diameters, insertions)

    def _init_mechansisms(self, insertions):
        self.mechanisms = {}
        for location, insertions_list in enumerate(insertions):
            for mech_type, args, kwargs in insertions_list:
                if mech_type not in self.mechanisms:
                    self.mechanisms[mech_type] = ([], [])
                instance = mech_type.new_instance(
                        self.time_step, location, self.geometry, *args, **kwargs)
                self.mechanisms[mech_type][0].append(location)
                self.mechanisms[mech_type][1].append(instance)
        for mech_type, (locations, instances) in self.mechanisms.items():
            self.mechanisms[mech_type] = (
                    np.array(locations, dtype=Location),
                    np.array(instances, dtype=mech_type.instance_dtype()))

    def _add_species(self, s):
        """ Add a species if its name is unique. Accepts placeholder strings. """
        if isinstance(s, Species):
            if self.species.get(s.name, None) is None:
                self.species[s.name] = copy.copy(s)
        elif isinstance(s, str):
            if s not in self.species:
                self.species[s] = None
        elif isinstance(s, Iterable):
            for x in s:
                self._add_species(x)
        else:
            raise TypeError("Invalid species: %s."%repr(s))

    def _init_species(self, species):
        # Compile a list of all species. Add the given argument species first,
        # then pull in any species required by the reactions & mechanisms, and
        # finaly pull in any remaining unspecified species from the standard
        # library.
        self.species = {}
        for s in species:
            self._add_species(s)
        for reaction in self.reactions:
            if hasattr(reaction, "required_species"):
                self._add_species(reaction.required_species())
        for mechanism in self.mechanisms:
            if hasattr(mechanism, "required_species"):
                self._add_species(mechanism.required_species())
        # TODO: Make a standard library for species.
        # for s in species_library:
        #     if s in self.species:
        #         self._add_species(s)
        # Make sure that all species are fully specified.
        for name, species in self.species.items():
            if species is None:
                raise ValueError("Unresolved species: "+str(name))
        # Initialize the species internal data.
        for s in self.species.values():
            if s.intra_diffusivity is not None:
                s.intra = Diffusion(self.time_step / 2, self.geometry, s, "intracellular")
            if s.extra_diffusivity is not None:
                s.extra = Diffusion(self.time_step / 2, self.geometry, s, "extracellular")
            if s.transmembrane:
                s.conductances = np.zeros(len(self), dtype=Real)
        # Setup the reaction input & output structures.
        self.ReactionInputs = namedtuple("ReactionInputs", "v intra extra")
        self.ReactionOutputs = namedtuple("ReactionOutputs", "conductances intra extra")
        self.IntraSpecies = namedtuple("IntraSpecies",
                [n for n, s in self.species.items() if s.intra_diffusivity is not None])
        self.ExtraSpecies = namedtuple("ExtraSpecies",
                [n for n, s in self.species.items() if s.extra_diffusivity is not None])
        self.Conductances = namedtuple("Conductances",
                [n for n, s in self.species.items() if s.transmembrane])

    def _setup_reaction_io(self):
        r_in = self.ReactionInputs(
            v = self.electrics.previous_voltages,
            intra = self.IntraSpecies(**{
                n: s.intra.previous_concentrations
                    for n, s in self.species.items() if s.intra is not None}),
            extra = self.ExtraSpecies(**{
                n: s.extra.previous_concentrations
                    for n, s in self.species.items() if s.extra is not None}))
        r_out = self.ReactionOutputs(
            conductances = self.Conductances(**{
                n: s.conductances
                    for n, s in self.species.items() if s.transmembrane}),
            intra = self.IntraSpecies(**{
                n: s.intra.release_rates
                    for n, s in self.species.items() if s.intra is not None}),
            extra = self.ExtraSpecies(**{
                n: s.extra.release_rates
                    for n, s in self.species.items() if s.extra is not None}))
        for outter in r_out:
            for inner in outter:
                inner.fill(0)
        return r_in, r_out

    def __len__(self):
        return len(self.geometry)

    def advance(self):
        if self.stagger:
            """
            All systems (reactions & mechanisms, diffusions & electrics) are
            integrated using input values from halfway through their time step.
            Tracing through the exact sequence of operations is difficult because
            both systems see the other system as staggered halfway through their
            time step.

            For more information see: The NEURON Book, 2003.
            Chapter 4, Section: Efficient handling of nonlinearity.
            """
            self._diffusions_advance()
            self._reactions_advance()
            self._diffusions_advance()
        else:
            """
            Naive integration strategy, for reference only.
            """
            # Update diffusions & electrics for the whole time step using the
            # state of the reactions at the start of the time step.
            self._diffusions_advance()
            self._diffusions_advance()
            # Update the reactions for the whole time step using the
            # concentrations & voltages from halfway through the time step.
            self._reactions_advance()

    def _reactions_advance(self):
        reaction_inputs, reaction_outputs = self._setup_reaction_io()
        for reaction in self.reactions:
            for location in range(len(self)):
                reaction.advance_reaction(
                        self.time_step, location, reaction_inputs, reaction_outputs)
        for mechanisms, (locations, instances) in self.mechanisms.items():
            for location, instance in zip(locations, instances):
                mechanisms.advance_instance(instance,
                        self.time_step, location, reaction_inputs, reaction_outputs)

    def _diffusions_advance(self):
        """ Note: Each call to this method integrates over half a time step. """
        dt = self.electrics.time_step
        # Save prior state.
        self.electrics.previous_voltages = np.array(self.electrics.voltages, copy=True)
        for s in self.species.values():
            for x in (s.intra, s.extra):
                if x is not None:
                    x.previous_concentrations = np.array(x.concentrations, copy=True)
        # Accumulate the net conductances and driving voltages from the chemical data.
        self.electrics.conductances.fill(0)     # Zero accumulator.
        self.electrics.driving_voltages.fill(0) # Zero accumulator.
        for s in self.species.values():
            if not s.transmembrane: continue
            s.reversal_potential = s._reversal_potential_method(
                s.intra_concentration if s.intra is None else s.intra.concentrations,
                s.extra_concentration if s.extra is None else s.extra.concentrations,
                self.electrics.voltages)
            self.electrics.conductances += s.conductances
            self.electrics.driving_voltages += s.conductances * s.reversal_potential
        self.electrics.driving_voltages /= self.electrics.conductances
        np.nan_to_num(self.electrics.driving_voltages, copy=False)
        # Calculate the transmembrane currents.
        diff_v = self.electrics.driving_voltages - self.electrics.voltages
        rc = self.electrics.capacitances / self.electrics.conductances
        alpha = np.exp(-dt / rc)
        self.electrics.voltages += diff_v * (1.0 - alpha)
        # Calculate the lateral currents throughout the neurons.
        self.electrics.voltages = self.electrics.irm.dot(self.electrics.voltages)
        # Calculate the transmembrane ion flows.
        for s in self.species.values():
            if not s.transmembrane: continue
            if s.intra is None and s.extra is None: continue
            integral_v = dt * (s.reversal_potential - self.electrics.driving_voltages)
            integral_v += rc * diff_v * alpha
            moles = s.conductances * integral_v / (s.charge * F)
            if s.intra is not None:
                s.intra.concentrations += moles / self.geometry.intra_volumes
            if s.extra is not None:
                s.extra.concentrations -= moles / self.geometry.extra_volumes
        # Calculate the local release / removal of chemicals.
        for s in self.species.values():
            for x in (s.intra, s.extra):
                if x is None: continue
                x.concentrations = np.maximum(0, x.concentrations + x.release_rates * dt)
                # Calculate the lateral diffusion throughout the space.
                x.concentrations = x.irm.dot(x.concentrations)

    def inject_current(self, location, value):
        location = int(location)
        value = float(value)
        dv = value * self.time_step / self.electrics.capacitances[location]
        self.electrics.voltages[location] += dv

    def draw_image(self,
            output_filename,
            resolution,
            camera_coordinates,
            camera_loot_at,
            segment_colors,
            fog_color=(1,1,1),
            fog_distance=np.inf):
        """ Use POVRAY to render an image of the model. """
        pov = ""
        pov += "camera { location <%s> look_at  <%s> }\n"%(
            ", ".join(str(x) for x in camera_coordinates),
            ", ".join(str(x) for x in camera_loot_at))
        pov += "global_settings { ambient_light rgb<1, 1, 1> }\n"
        pov += "light_source { <1, 0, 0> color rgb<1, 1, 1>}\n"
        pov += "light_source { <-1, 0, 0> color rgb<1, 1, 1>}\n"
        pov += "light_source { <0, 1, 0> color rgb<1, 1, 1>}\n"
        pov += "light_source { <0, -1, 0> color rgb<1, 1, 1>}\n"
        pov += "light_source { <0, 0, 1> color rgb<1, 1, 1>}\n"
        pov += "light_source { <0, 0, -1> color rgb<1, 1, 1>}\n"
        if fog_distance == np.inf:
            pov += "background { color rgb<%s> }\n"%", ".join(str(x) for x in fog_color)
        else:
            pov += "fog { distance %s color rgb<%s>}\n"%(str(fog_distance),
            ", ".join(str(x) for x in fog_color))
        for location in range(len(self)):
            parent = self.geometry.parents[location]
            coords = self.geometry.coordinates[location]
            # Special cases for root of tree, whos segment body is split between
            # it and its eldest child.
            if self.geometry.is_root(location):
                eldest = self.geometry.children[location][0]
                other_coords = (coords + self.geometry.coordinates[eldest]) / 2
            elif self.geometry.is_root(parent) and self.geometry.children[parent][0] == location:
                other_coords = (coords + self.geometry.coordinates[parent]) / 2
            else:
                other_coords = self.geometry.coordinates[parent]
            pov += "cylinder { <%s>, <%s>, %s "%(
                ", ".join(str(x) for x in coords),
                ", ".join(str(x) for x in other_coords),
                str(self.geometry.diameters[location] / 2))
            pov += "texture { pigment { rgb <%s> } } }\n"%", ".join(str(x) for x in segment_colors[location])
        pov_file = tempfile.NamedTemporaryFile(suffix=".pov", mode='w+t', delete=False)
        pov_file.write(pov)
        pov_file.close()
        subprocess.run(["povray",
            "-D", # Disables immediate graphical output, save to file instead.
            "+O" + output_filename,
            "+W" + str(resolution[0]),
            "+H" + str(resolution[1]),
            pov_file.name,],
            stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL,
            check=True,)
        os.remove(pov_file.name)

from neuwon.geometry import Neighbor, Geometry

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
        self.irm = scipy.sparse.linalg.expm(coefficients)
        # Prune the impulse response matrix at epsilon nanomolar (mol/L).
        self.irm.data[np.abs(self.irm.data) < epsilon * 1e-6] = 0
        self.irm = csr_matrix(self.irm, dtype=Real)

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
        self.voltages           = np.zeros(len(geometry), dtype=Real)
        self.previous_voltages  = np.zeros(len(geometry), dtype=Real)
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
        self.irm = scipy.sparse.linalg.expm(coefficients)
        # Prune the impulse response matrix at epsilon millivolts.
        self.irm.data[np.abs(self.irm.data) < epsilon * 1e-3] = 0
        self.irm = csr_matrix(self.irm, dtype=Real)

# I don't know how to orgranize this code in an intuitive way...

_tables = {} # tables[name][time_step] = KineticModelTable
def make_kinetic_table(name, time_step, *args, **kwargs):
        name = str(name)
        time_step = float(time_step)
        if name in _tables:
            if time_step in _tables[name]:
                return _tables[name][time_step]
        else:
            _tables[name] = {}
        _tables[name][time_step] = KineticModelTable(name=name, time_step=time_step, *args, **kwargs)
        return _tables[name][time_step]

class KineticModelTable:
    def __init__(self, time_step, inputs, states, kinetics,
        name="",
        initial_state=None,
        conserve_sum=False,
        atol=1e-6):
        """ """
        # Save and check the arguments.
        self.time_step = float(time_step)
        self.name = name = str(name)
        if self.name:
            if self.name[-1].isnumeric() or self.name[-1].isupper():
                name = self.name + "_"
        if isinstance(inputs, int):
            assert(inputs >= 0)
            self.inputs = namedtuple(name+"Inputs", ("input%i"%i for i in range(inputs)))
        else:
            if isinstance(inputs, str):
                inputs = [x.strip(",") for x in inputs.split()]
            self.inputs = namedtuple(name+"Inputs", (str(i) for i in inputs))
        self.states = namedtuple(name+"States", (str(s) for s in states))
        self.conserve_sum = float(conserve_sum) if conserve_sum else None
        if initial_state is not None:
            initial_state = str(initial_state)
            assert(initial_state in self.states._fields)
            assert(self.conserve_sum is not None)
            zeros = self.states(*[0]*len(self.states._fields))
            self.initial_state = zeros._replace(**{initial_state: self.conserve_sum})
        # List of non-zero elements of the coefficients matrix in the derivative function:
        #       dX/dt = C * X, where C is Coefficients matrix and X is state vector.
        # Stored as tuples of (src, dst, coef, func)
        #       Where "src" and "dst" are indexes into the state vector.
        #       Where "coef" is constant rate mulitplier.
        #       Where "func" is optional function: func(*inputs) -> coefficient
        self.kinetics = []
        for reactant, product, forward, reverse in kinetics:
            r_idx = self.states._fields.index(str(reactant))
            p_idx = self.states._fields.index(str(product))
            for src, dst, rate in ((r_idx, p_idx, forward), (p_idx, r_idx, reverse)):
                if isinstance(rate, str):
                    coef = 1
                    inp_idx = self.inputs.index(rate)
                    func = functools.partial(lambda inp_idx, *args: args[inp_idx], inp_idx)
                elif isinstance(rate, Callable):
                    coef = 1
                    func = rate
                else:
                    coef = float(rate)
                    func = None
                self.kinetics.append((src, dst, +coef, func))
                self.kinetics.append((src, src, -coef, func))
        # Initialize the interpolation grid.
        self.lower = np.full(len(self.inputs._fields), +np.inf, dtype=Real)
        self.upper = np.full(len(self.inputs._fields), -np.inf, dtype=Real)

    def advance(self, inputs, states):
        assert(len(inputs) == len(self.inputs._fields))
        assert(len(states) == len(self.states._fields))
        # Bounds check the inputs, resize interpolation grid if necessary.
        if any(self.lower > inputs) or any(self.upper < inputs):
            self._compute_interpolation_grid(inputs)
        # Determine which grid box the inputs are inside of.
        inputs = self.grid_factor * np.subtract(inputs, self.lower)
        lower_idx = np.array(np.floor(inputs), dtype=int)
        upper_idx = np.array(np.ceil(inputs), dtype=int)
        upper_idx = np.minimum(upper_idx, self.grid_size - 1) # Protect against floating point error.
        # Prepare to find the interpolation weights, by finding the distance
        # from the input point to each corner of its grid box.
        inputs -= lower_idx
        corner_weights = [np.subtract(1, inputs), inputs]
        # Visit each corner of the grid box and accumulate the results.
        results = np.zeros(len(self.states._fields), dtype=Real)
        for corner in itertools.product(*([(0,1)] * len(self.inputs._fields))):
            idx = np.choose(corner, [lower_idx, upper_idx])
            weight = np.product(np.choose(corner, corner_weights))
            results += weight * self.data[idx].dot(states).flat
        # Enforce the invariant sum of states.
        if self.conserve_sum is not None:
            results *= self.conserve_sum / sum(results)
        return self.states._make(results)

    def _compute_interpolation_grid(self, inputs):
        # Find the min & max of the input domain.
        old_range  = self.upper - self.lower
        self.lower = np.minimum(self.lower, inputs)
        self.upper = np.maximum(self.upper, inputs)
        new_range  = self.upper - self.lower
        grid_range = self.upper - self.lower
        grid_range[grid_range == 0] = 1
        if not all(old_range > 0):
            self.grid_size = np.full(len(self.inputs._fields), 2, dtype=int)
        else:
            # Expand the interpolation grid in proportion to the increase in the input domain.
            # pct_change = new_range / old_range
            # self.grid_size = np.array(np.round(self.grid_size * pct_change), dtype=int)
            self.grid_size = np.array([100])
        self.grid_factor = (self.grid_size - 1) / grid_range
        self.data = np.empty(list(self.grid_size) + [len(self.states._fields)]*2, dtype=Real)
        # Visit every location on the new interpolation grid.
        grid_axes = [list(enumerate(np.linspace(*args, dtype=float)))
                    for (args) in zip(self.lower, self.upper, self.grid_size)]
        for inputs in itertools.product(*grid_axes):
            index, inputs = zip(*inputs)
            self.data[index] = self._compute_impulse_response_matrix(inputs)
        # TODO: Determine if interpolation accuracy is sufficient or if it needs
        # more grid points. This will require an additional "accuracy" parameter.
        print(self.name, "atol", self._compute_min_accuracy())

    def _compute_min_accuracy(self, num_test_points=100):
        atol = 0
        for _ in range(num_test_points):
            inputs = np.random.uniform(self.lower, self.upper)
            state = np.random.uniform(size=len(self.states._fields))
            if self.conserve_sum is not None:
                state *= self.conserve_sum / sum(state)
            exact = self._compute_impulse_response_matrix(inputs).dot(state)
            if self.conserve_sum is not None:
                exact *= self.conserve_sum / sum(exact)
            interp = np.array(self.advance(inputs, state))
            atol = max(atol, np.max(np.abs(exact - interp)))
        return atol

    def _compute_impulse_response_matrix(self, inputs):
        A = np.zeros([len(self.states._fields)] * 2, dtype=float)
        for src, dst, coef, func in self.kinetics:
            if func is not None:
                A[dst, src] += coef * func(*inputs)
            else:
                A[dst, src] += coef
        return scipy.linalg.expm(A * self.time_step)

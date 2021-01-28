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
__all__ = """Segment Mechanism Reaction Species Model Geometry Neighbor""".split()
# Numeric/Scientific Library Imports.
import numpy as np
import cupy as cp
import numba.cuda
# Standard Library Imports.
import math
import random
import itertools
import functools
import copy
import subprocess
import tempfile
import os
from collections.abc import Callable, Iterable, Mapping
from collections import namedtuple
# Third Party Imports.
from graph_algorithms import depth_first_traversal as dft
from htm.bindings.sdr import SDR

F = 96485.3321233100184 # Faraday's constant, Coulumbs per Mole of electrons
R = 8.31446261815324 # Universal gas constant
celsius = 37 # Human body temperature
T = celsius + 273.15 # Human body temperature in Kelvins

Real = np.dtype('f8')
epsilon = np.finfo(Real).eps
Location = np.dtype('u4')

from neuwon.geometry import Neighbor, Geometry
from neuwon.species import Species, Diffusion, Electrics
from neuwon.mechanisms import Mechanism
import neuwon.mechanisms

class Reaction:
    """ Abstract class for specifying reactions between omnipresent species. """
    def required_species(self):
        """ Optional, Returns the Species required by this mechanism.
        Allowed return types: Species, names of species, and lists either. """
        return []
    def advance_reaction(self, time_step, location, reaction_inputs, reaction_outputs):
        """ """
        pass

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

    def inject_current(self, current=None, duration=1e-3):
        assert(self.model is not None)
        self.model.inject_current(self.location, current, duration)


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
        neuwon.mechanisms._init_mechansisms(self, insertions)
        self._init_species(species)
        self.electrics = Electrics(self.time_step / 2, self.geometry)
        self._injected_currents = Model._InjectedCurrents()
        self._ap_detector = Model._AP_Detector()
        numba.cuda.synchronize()

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
            s._initialize(self.time_step / 2, self.geometry)
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
        numba.cuda.synchronize()
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
            self._check_data()
            self._diffusions_advance()
            self._check_data()
            self._reactions_advance()
            self._check_data()
            self._diffusions_advance()
            self._check_data()
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

    def _check_data(self):
        for mech_type, (locations, instances) in self.mechanisms.items():
            if isinstance(instances, Mapping):
                for key, array in instances.items():
                    assert cp.all(cp.isfinite(array)), (mech_type, key)
            elif instances.dtype.kind in "fc":
                assert cp.all(cp.isfinite(instances)), mech_type
            elif instances.dtype.fields is not None:
                instances = instances.copy_to_host()
                for name in instances.dtype.fields:
                    assert np.all(np.isfinite(instances[name])), (mech_type, name)
        for s in self.species.values():
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
        assert(cp.all(cp.isfinite(self.electrics.voltages)))
        assert(cp.all(cp.isfinite(self.electrics.previous_voltages)))
        assert(cp.all(cp.isfinite(self.electrics.driving_voltages)))
        assert(cp.all(cp.isfinite(self.electrics.conductances)))

    def _reactions_advance(self):
        dt = self.time_step
        reaction_inputs, reaction_outputs = self._setup_reaction_io()
        for reaction in self.reactions:
            f = reaction.advance_reaction
            for location in range(len(self)):
                f(dt, location, reaction_inputs, reaction_outputs)
            numba.cuda.synchronize()
        for mechanisms, (locations, instances) in self.mechanisms.items():
            mechanisms.advance(locations, instances, dt, reaction_inputs, reaction_outputs)
            numba.cuda.synchronize()

    def _diffusions_advance(self):
        """ Note: Each call to this method integrates over half a time step. """
        dt = self.electrics.time_step
        # Save prior state.
        self.electrics.previous_voltages = cp.array(self.electrics.voltages, copy=True)
        for s in self.species.values():
            for x in (s.intra, s.extra):
                if x is not None:
                    x.previous_concentrations = cp.array(x.concentrations, copy=True)
        numba.cuda.synchronize()
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
            numba.cuda.synchronize()
        self.electrics.driving_voltages /= self.electrics.conductances
        numba.cuda.synchronize()
        self.electrics.driving_voltages = cp.nan_to_num(self.electrics.driving_voltages)
        numba.cuda.synchronize()
        # Calculate the transmembrane currents.
        diff_v = self.electrics.driving_voltages - self.electrics.voltages
        rc = self.electrics.capacitances / self.electrics.conductances
        alpha = cp.exp(-dt / rc)
        self.electrics.voltages += diff_v * (1.0 - alpha)
        numba.cuda.synchronize()
        # Calculate the externally applied currents.
        self._injected_currents.advance(dt, self.electrics)
        # Calculate the lateral currents throughout the neurons.
        self.electrics.voltages = self.electrics.irm.dot(self.electrics.voltages)
        # self._ap_detector.advance(dt, self.electrics.voltages)
        # Calculate the transmembrane ion flows.
        numba.cuda.synchronize()
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
                x.concentrations = cp.maximum(0, x.concentrations + x.release_rates * dt)
                # Calculate the lateral diffusion throughout the space.
                x.concentrations = x.irm.dot(x.concentrations)
                numba.cuda.synchronize()

    class _InjectedCurrents:
        def __init__(self):
            self.currents = []
            self.locations = []
            self.remaining = []

        def advance(self, time_step, electrics):
            for idx, (amps, location, t) in enumerate(
                    zip(self.currents, self.locations, self.remaining)):
                dv = amps * min(time_step, t) / electrics.capacitances[location]
                electrics.voltages[location] += dv
                self.remaining[idx] -= time_step
            keep = [t > 0 for t in self.remaining]
            self.currents  = [x for k, x in zip(keep, self.currents) if k]
            self.locations = [x for k, x in zip(keep, self.locations) if k]
            self.remaining = [x for k, x in zip(keep, self.remaining) if k]

    def inject_current(self, location, current = None, duration = 1.4e-3):
        location = int(location)
        assert(location < len(self))
        duration = float(duration)
        assert(duration >= 0)
        if current is None:
            target_voltage = 200e-3
            current = target_voltage * self.electrics.capacitances[location] / duration
        else:
            current = float(current)
        self._injected_currents.currents.append(current)
        self._injected_currents.locations.append(location)
        self._injected_currents.remaining.append(duration)

    class _AP_Detector:
        """ Detect Action Potentials at strategic locations throughout the model.
        This uses a simple rising edge threshold detector. """
        threshold = 20e-3
        def __init__(self):
            self.locations = np.empty(0, dtype=Location)
            self.triggered = np.empty(0, dtype=Real)
            self.detected  = np.empty(0, dtype=Location)
            self.elapsed   = np.empty(0, dtype=Real)

        def add_location(self, location):
            self.locations = np.append(self.locations, [location])
            self.triggered = np.append(self.triggered, [False])

        def advance(self, time_step, voltages):
            self.elapsed += time_step
            high_voltage = voltages[self.locations] >= self.threshold
            events = np.nonzero(np.logical_and(high_voltage, self.triggered))
            self.triggered = high_voltage
            self.detected  = np.append(self.detected, events)
            self.elapsed   = np.append(self.elapsed, np.zeros(len(events), dtype=Real))

    def detect_APs(self, location):
        """ """
        if isinstance(location, Segment):
            location = location.location
        location = int(location)
        assert(location < len(self))
        self._ap_detector.add_location(location)

    def detected_APs(self):
        """ """
        retval = (self._ap_detector.detected, self._ap_detector.elapsed)
        self._ap_detector.detected = np.empty(0, dtype=Location)
        self._ap_detector.elapsed  = np.empty(0, dtype=Real)
        return retval

    def activity_SDR(self):
        locations, elapsed = self.detected_APs()
        sdr = SDR(dimensions = (len(self._ap_detector.locations),))
        sdr.sparse = locations
        return sdr

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

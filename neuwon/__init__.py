"""
NEUWON is a simulation framework for neuroscience and artificial intelligence
specializing in conductance based models. This software is a modern remake of
the NEURON simulator. It is accurate, efficient, and easy to use.

### Units

All units are prefix-less.
* Meters, Grams, Seconds
* Volts, Amperes, Ohms, Siemens, Farads

"""

__all__ = """
Reaction Mechanism Species Conductance
Region Intersection Union Rectangle Sphere Cone Cylinder
GrowSomata Growth
Segment
Model Geometry Neighbor
""".split()

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
import scipy.spatial
from math import exp, ceil
from abc import ABC, abstractmethod
import random
import heapq as q
import itertools
import copy
from subprocess import run, STDOUT, DEVNULL
import tempfile
import os
from graph_algorithms import depth_first_traversal as dft

Real = np.dtype('f4')
epsilon = np.finfo(Real).eps
Location = np.dtype('u4')
ROOT = np.iinfo(Location).max

class Reaction(ABC):
    """ """
    @abstractmethod
    def __call__(location, reaction_inputs, reaction_outputs):
        """ """
        pass
    @property
    @abstractmethod
    def species(self):
        """ A list of Species required by this reaction. """
        return []

class Mechanism(ABC):
    """ """
    @abstractmethod
    def __init__(self, time_step, location, geometry, *args):
        """ """
        pass
    @abstractmethod
    def advance(self, reaction_inputs, reaction_outputs):
        """ """
        pass
    @property
    @abstractmethod
    def species(self):
        """ A list of Species required by this mechanism. """
        return []
    @property
    @abstractmethod
    def conductances(self):
        """ A list of membrane conductances required by this mechanism. """
        return []

class Species:
    def __init__(self, name, where, initial_concentration, diffusivity):
        self.name = str(name)
        self.where = str(where)
        self.initial_concentration = float(initial_concentration)
        self.diffusivity = float(diffusivity)
        assert(self.where in ("intracellular", "extracellular"))
        assert(self.initial_concentration >= 0)
        assert(self.diffusivity >= 0)

class Conductance:
    def __init__(self, name, reversal_potential = None,):
        self.name = str(name)
        # TODO: Make this accept intracellular & extracellular species and
        # compute the reversal potential given the charge & concentrations.
        if reversal_potential is None:
            self.reversal_potential = None
        else:
            self.reversal_potential = float(reversal_potential)

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
        divisions = max(1, ceil(length / maximum_segment_length))
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

    def insert_mechanism(self, mechanism, *args):
        assert(issubclass(mechanism, Mechanism))
        self.insertions.append((mechanism, args))

    def set_diameter(self, P = (4e-6, 3e-6, 1e-6)):
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

class Region(ABC):
    """ """
    @abstractmethod
    def contains(self, coordinates):
        """ Returns bool: does this region contain the given coordinates? """
    @abstractmethod
    def aabb(self):
        """ Returns pair (lower_corner, upper_corner) of an axis aligned
        bounding box which entirely contains this region. """
    def sample_point(self):
        """ Returns a random point from within the region. """
        lower, upper = self.aabb()
        if not (all(np.isfinite(lower)) and all(np.isfinite(upper))):
            raise TypeError("Region is infinite!")
        while True:
            x = np.add(lower, np.random.uniform(size=(3)) * np.subtract(upper, lower))
            if self.contains(x):
                return x
    def sample_points(self, density):
        """
        Returns a list of points from within the region. Density is approximate
        and is specified as points per unit of length cubed.
        """
        lower, upper = self.aabb()
        if not (all(np.isfinite(lower)) and all(np.isfinite(upper))):
            raise TypeError("Region is infinite!")
        aabb_volume = np.product(np.subtract(upper, lower))
        num_points = int(round(density * aabb_volume))
        points = np.add(lower, np.random.uniform(size=(num_points, 3)) * np.subtract(upper, lower))
        return points[[self.contains(x) for x in points]]

class Intersection(Region):
    """ Intersection of regions """
    def __init__(self, regions):
        """ Argument regions is an iterable of Region instances. """
        super().__init__()
        self.regions = tuple(regions)
        assert(all(isinstance(rgn, Region) for rgn in self.regions))
        low, high = zip(*(rgn.aabb() for rgn in self.regions))
        self.lower_corner = np.max(low, axis=0)
        self.upper_corner = np.min(high, axis=0)
    def aabb(self):
        return (self.lower_corner, self.upper_corner)
    def contains(self, coordinates):
        return all(rgn.contains(coordinates) for rgn in self.regions)

class Union(Region):
    """ Union of regions """
    def __init__(self, regions):
        """ Argument regions is an iterable of Region instances. """
        super().__init__()
        self.regions = tuple(regions)
        assert(all(isinstance(rgn, Region) for rgn in self.regions))
        low, high = zip(*(rgn.aabb() for rgn in self.regions))
        self.lower_corner = np.min(low, axis=0)
        self.upper_corner = np.max(high, axis=0)
    def aabb(self):
        return (self.lower_corner, self.upper_corner)
    def contains(self, coordinates):
        return any(rgn.contains(coordinates) for rgn in self.regions)

class Not(Region):
    """ Region covering everywhere *except* for the given region. """
    def __init__(self, region):
        """ Argument region is an instance of a subclass of Region. """
        super().__init__()
        self.region = region
        assert(isinstance(self.region, Region))
    def aabb(self):
        return ([-np.inf]*3, [+np.inf]*3)
    def contains(self, coordinates):
        return not self.region.contains(coordinates)

class Rectangle(Region):
    """ Axis Aligned Rectangular Prism """
    def __init__(self, corner1, corner2):
        """ Arguments corner1 and corner2 are the coordinates of any
        opposing corners of the box. """
        super().__init__()
        self.a = np.minimum(corner1, corner2)
        self.b = np.maximum(corner1, corner2)
        assert(len(self.a) == 3 and len(self.b) == 3)
    def aabb(self):
        return (self.a, self.b)
    def contains(self, coordinates):
        return all(self.a <= coordinates) and all(self.b > coordinates)

class Sphere(Region):
    def __init__(self, center, radius):
        """ Argument center is the 3D coordinates of the center of the sphere.
        Argument radius is distance from the center to include inside the sphere. """
        super().__init__()
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        assert(len(self.center) == 3)
        assert(self.radius >= 0)
    def aabb(self):
        return (self.center - self.radius, self.center + self.radius)
    def contains(self, coordinates):
        return np.linalg.norm(self.center - coordinates) <= self.radius

class Cone(Region):
    """ Right circular cone """
    def __init__(self, point, cap, radius):
        """ """
        super().__init__()
        self.point  = np.array(point, dtype=float)
        self.cap    = np.array(cap, dtype=float)
        self.radius = float(radius)
        assert(len(self.point) == 3)
        assert(len(self.cap) == 3)
        assert(self.radius >= 0)
    def aabb(self):
        return 1/0 # TODO
    def contains(self, coordinates):
        return 1/0 # TODO

class Cylinder(Region):
    """ Right circular cylinder """
    def __init__(self, point1, point2, radius):
        super().__init__()
        self.point1 = np.array(point1, dtype=float)
        self.point2 = np.array(point2, dtype=float)
        self.radius = float(radius)
        assert(len(self.point1) == 3)
        assert(len(self.point2) == 3)
        assert(self.radius >= 0)
        self.lower = np.minimum(self.point1, self.point2)
        self.upper = np.maximum(self.point1, self.point2)
        sqr = (self.point1 - self.point2) ** 2
        for dim in range(3):
            k = np.sqrt(sum(x for d, x in enumerate(sqr) if d != dim) / sum(sqr))
            self.lower[dim] -= k * self.radius
            self.upper[dim] += k * self.radius
        self.axis = self.point2 - self.point1
        self.length_sqr = sum(self.axis ** 2)
        self.radius_sqr = self.radius ** 2
    def aabb(self):
        return (self.lower, self.upper)
    def contains(self, coordinates):
        # https://flipcode.com/archives/Fast_Point-In-Cylinder_Test.shtml
        displacement = np.subtract(coordinates, self.point1)
        dot = self.axis.dot(displacement)
        if dot < 0 or dot > self.length_sqr:
            return False
        dist_sqr = displacement.dot(displacement) - dot*dot/self.length_sqr;
        return dist_sqr <= self.radius_sqr

class GrowSomata:
    def __init__(self, region, density, diameter):
        self.region = region
        self.density = float(density)
        self.diameter = float(diameter)
        self.segments = []
        assert(isinstance(self.region, Region))
        assert(self.density >= 0)
        assert(self.diameter > 0)
        self._compute()

    def _compute(self):
        for coordinates in self.region.sample_points(self.density):
            self.segments.extend(self.single(coordinates, self.diameter))

    @classmethod
    def single(cls, coordinates, diameter):
        """ This approximates a sphere using a cylinder with an equivalent
        surface area and volume. """
        cylinder_diameter = (2 / 3) * diameter
        cylinder_length = (3 / 2) * diameter
        x = list(coordinates)
        x[1] -= cylinder_length / 2
        s1 = Segment(x, cylinder_diameter, None)
        x = list(coordinates)
        x[1] += cylinder_length / 2
        s2 = Segment(x, cylinder_diameter, s1)
        return [s1, s2]

class Growth:
    """ Grow dendrites or axons

    This implements the TREES algorithm with the morphological constraints used
    by the ROOTS algorithm.

    TREES:
        Cuntz H, Forstner F, Borst A, Hausser M (2010) One Rule to Grow Them
        All: A General Theory of Neuronal Branching and Its Practical
        Application. PLoS Comput Biol 6(8): e1000877.
        doi:10.1371/journal.pcbi.1000877

    ROOTS:
        Bingham CS, Mergenthal A, Bouteiller J-MC, Song D, Lazzi G and Berger TW
        (2020) ROOTS: An Algorithm to Generate Biologically Realistic Cortical
        Axons and an Application to Electroceutical Modeling. Front. Comput.
        Neurosci. 14:13. doi: 10.3389/fncom.2020.00013
    """
    def __init__(self,
            seeds,
            region,
            carrier_point_density,
            balancing_factor=0.5,
            extension_angle=np.pi,
            extension_distance=np.inf,
            bifurcation_angle=np.pi,
            bifurcation_distance=np.inf,
            extend_before_bifurcate=False,
            only_bifurcate=False,
            maximum_segment_length=np.inf,):
        if isinstance(seeds, Segment):
            self.seeds = [seeds]
        else:
            self.seeds = list(seeds)
        self.region = region
        self.carrier_point_density  = float(carrier_point_density)
        self.balancing_factor       = float(balancing_factor)
        self.extension_angle        = float(extension_angle)
        self.extension_distance     = float(extension_distance)
        self.bifurcation_angle      = float(bifurcation_angle)
        self.bifurcation_distance   = float(bifurcation_distance)
        self.extend_before_bifurcate = bool(extend_before_bifurcate)
        self.only_bifurcate         = bool(only_bifurcate)
        self.maximum_segment_length = float(maximum_segment_length)
        assert(all(isinstance(s, Segment) for s in self.seeds))
        assert(isinstance(region, Region))
        assert(0 <= self.carrier_point_density)
        assert(0 <= self.balancing_factor)
        assert(0 <= self.extension_angle <= np.pi)
        assert(0 <= self.extension_distance)
        assert(0 <= self.bifurcation_angle <= np.pi)
        assert(0 <= self.bifurcation_distance)
        self._compute()

    def _compute(self):
        self.segments = [] # List of all newly created segments, user facing output.
        carrier_points = self.region.sample_points(self.carrier_point_density)
        free_points = np.ones(len(carrier_points), dtype=bool)
        tree = scipy.spatial.cKDTree(carrier_points)
        max_distance = max(self.extension_distance, self.bifurcation_distance)
        # Use Min Heap priority queue to find the lowest cost segments.
        # Heap datum format is (cost, parent_segment, carrier_point_index)
        costs = [] # Primary queue for all potential new segments.
        bifurcations = [] # Secondary queue for potential bifurcations, used when extend_before_bifurcate=True.
        def compute_costs_to_all_carriers(parent):
            for index in tree.query_ball_point(parent.coordinates, max_distance):
                if free_points[index]:
                    c = self.cost_function(parent, carrier_points[index])
                    q.heappush(costs, (c, parent, index))
        for seed in self.seeds:
            compute_costs_to_all_carriers(seed)
        while costs or bifurcations:
            if costs:
                c, parent, index = q.heappop(costs)
                if not free_points[index]: continue
                if self.extend_before_bifurcate and parent.children:
                    q.heappush(bifurcations, (c, parent, index))
                    continue
            elif bifurcations:
                c, parent, index = q.heappop(bifurcations)
                if not free_points[index]: continue
            coordinates = carrier_points[index]
            if not self.morphological_constraints_satisfied(parent, coordinates):
                continue
            free_points[index] = False
            s = parent.add_segment(coordinates, 3e-6, self.maximum_segment_length)
            self.segments.extend(s)
            compute_costs_to_all_carriers(s[-1])

    def cost_function(self, parent, child_coordinates):
        distance = np.linalg.norm(np.subtract(parent.coordinates, child_coordinates))
        path_length = parent.path_length + distance
        return distance + self.balancing_factor * path_length

    def morphological_constraints_satisfied(self, parent, child_coordinates):
        if self.only_bifurcate and len(parent.children) >= 2:
            return False
        distance = np.linalg.norm(np.subtract(parent.coordinates, child_coordinates))
        def angle(a, b):
            return np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
        if parent.parent:
            parent_vector = np.subtract(parent.coordinates, parent.parent.coordinates)
            child_vector = np.subtract(child_coordinates, parent.coordinates)
            angle = angle(parent_vector, child_vector)
        else:
            angle = None
        # Check extension distance & angle.
        if not parent.children:
            if distance > self.extension_distance:
                return False
            if angle and angle > self.extension_angle:
                return False
        else:
            # Check bifurcation distance & angle.
            if distance > self.bifurcation_distance:
                return False
            if angle and angle > self.bifurcation_angle:
                return False
        return True

class Model:
    def __init__(self, time_step,
            neurons,
            reactions,
            species,
            conductances,
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
        self._init_conductances(conductances)
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
        self.mechanism_insertions = np.zeros(len(self), dtype=object)
        self.mechanism_instances = {}
        for location, insertions_list in enumerate(insertions):
            self.mechanism_insertions[location] = []
            for mechanism_specification in insertions_list:
                mech_type = mechanism_specification[0]
                mech_args = mechanism_specification[1:]
                assert(issubclass(mech_type, Mechanism))
                instance = mech_type(self.time_step, location, self.geometry, *mech_args)
                self.mechanism_insertions[location].append(instance)
                if mech_type not in self.mechanism_instances:
                    self.mechanism_instances[mech_type] = []
                self.mechanism_instances[mech_type].append(instance)

    def _init_species(self, species):
        index = {}
        def add_species(s):
            if isinstance(s, Species):
                if index.get(s.name, None) is None:
                    index[s.name] = s
            else:
                s = str(s)
                if s not in index:
                    index[s] = None
        for s in species:
            add_species(s)
        for r in itertools.chain(self.reactions, self.mechanism_instances.keys()):
            for s in r.species:
                add_species(s)
        # TODO: Fill in missing species from a standard library of common ones.
        for name, species in index.items():
            if species is None:
                raise ValueError("Unresolved species: "+str(name))
        self.species = tuple(Diffusion(self.time_step / 2, self.geometry, species)
                for species in sorted(index.values()))

    def _init_conductances(self, conductances):
        index = {}
        def add_conductance(x):
            if isinstance(x, Conductance):
                if index.get(x.name, None) is None:
                    index[x.name] = x
            else:
                x = str(x)
                if x not in index:
                    index[x] = None
        for x in conductances:
            add_conductance(x)
        for m in self.mechanism_instances.keys():
            for x in m.conductances:
                add_conductance(x)
        # TODO: Fill in missing conductances as able.
        for name, conductance in index.items():
            if conductance is None:
                raise ValueError("Unresolved conductance: "+str(name))
        self.conductances = tuple(copy.copy(c) for c in sorted(index.values(), key = lambda c: c.name))
        for c in self.conductances:
            c.data = np.zeros(len(self), dtype=Real)

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
        self._diffusions_advance()
        self._diffusions_advance()
        # Update the reactions for the whole time step using the
        # concentrations & voltages from halfway through the time step.
        self._reactions_advance()

    def _advance_staggered(self):
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

    def _setup_reaction_io(self):
        r_in = {}
        r_in["v"] = self.electrics.previous_voltages
        for x in self.species:
            r_in[x.name] = x.previous_concentrations
        r_out = {}
        for x in self.species:
            r_out[x.name] = x.release_rates
        for x in self.conductances:
            r_out[x.name] = x.data
        for data in r_out.values():
            data.fill(0)
        return r_in, r_out

    def _reactions_advance(self):
        reaction_inputs, reaction_outputs = self._setup_reaction_io()
        for reaction in self.reactions:
            for location in range(len(self)):
                reaction(location, reaction_inputs, reaction_outputs)
        for mech_type, instances_list in self.mechanism_instances.items():
            for inst in instances_list:
                inst.advance(reaction_inputs, reaction_outputs)
        self._conductances_advance()

    def _conductances_advance(self):
        """ Update the net conductances and driving voltages from the chemical data. """
        self.electrics.conductances.fill(0)
        self.electrics.driving_voltages.fill(0)
        for x in self.conductances:
            self.electrics.conductances += x.data
            self.electrics.driving_voltages += x.data * x.reversal_potential
        self.electrics.driving_voltages /= self.electrics.conductances
        np.nan_to_num(self.electrics.driving_voltages, copy=False)

    def _diffusions_advance(self):
        # Each call to advance the diffusions & electrics integrates over half of the time step.
        for x in self.species:
            x._advance()
        self.electrics._advance()

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
        run(["povray",
            "-D", # Disables immediate graphical output, save to file instead.
            "+O" + output_filename,
            "+W" + str(resolution[0]),
            "+H" + str(resolution[1]),
            pov_file.name,],
            stderr=STDOUT, stdout=DEVNULL,
            check=True,)
        os.remove(pov_file.name)

class Geometry:
    """ Physical shapes & structures of neurons """
    def __init__(self, coordinates, parents, diameters,
            maximum_extracellular_radius=100e-6,):
        # Check and save the arguments.
        self.coordinates = np.array(coordinates, dtype=Real)
        self.parents = np.array([ROOT if p is None else p for p in parents], dtype=Location)
        self.diameters = np.array(diameters, dtype=Real)
        self.maximum_extracellular_radius = float(maximum_extracellular_radius)
        assert(len(self.coordinates) == len(self))
        assert(len(self.parents)     == len(self))
        assert(len(self.diameters)   == len(self))
        assert(all(all(np.isfinite(c)) for c in self.coordinates))
        assert(all(p < len(self) or p == ROOT for p in self.parents))
        assert(all(d >= 0 for d in self.diameters))
        assert(self.maximum_extracellular_radius > epsilon * 1e-6)
        self._init_tree_properties()
        self._init_cellular_properties()
        self._init_extracellular_properties()

    def _init_tree_properties(self):
        # Compute the children lists.
        self.children = np.empty(len(self), dtype=object)
        for location in range(len(self)):
            self.children[location] = []
        for location, parent in enumerate(self.parents):
            if not self.is_root(location):
                self.children[parent].append(location)
        # Root must have at least one child, because cylinder is defined as between two points.
        assert(all(len(self.children[x]) >= 1 for x in range(len(self)) if self.is_root(x)))
        # The child with the largest diameter is special and is always kept at
        # the start of the children list.
        for siblings in self.children:
            siblings.sort(reverse=True, key=lambda x: self.diameters[x])
        # Compute lengths, which are the distances between each node and its
        # parent node. All root node lengths are NAN.
        self.lengths = np.empty(len(self), dtype=Real)
        for location in range(len(self)):
            if self.is_root(location):
                self.lengths[location] = np.nan
            else:
                self.lengths[location] = np.linalg.norm(
                    self.coordinates[location] - self.coordinates[self.parents[location]])
        assert(all(l >= epsilon * (1e-6)**1 or self.is_root(idx) for idx, l in enumerate(self.lengths)))

    def _init_cellular_properties(self):
        self.cross_sectional_areas = np.array([np.pi * (d / 2) ** 2 for d in self.diameters], dtype=Real)
        self.surface_areas = np.empty(len(self), dtype=Real)
        self.intra_volumes = np.empty(len(self), dtype=Real)
        for location, parent in enumerate(self.parents):
            radius = self.diameters[location] / 2
            if self.is_root(location):
                # Root of new tree. The body of this segment is half of the
                # cylinder spanning between this node and its first child.
                eldest = self.children[location][0]
                length = self.diameters[eldest] / 2
            elif self.is_root(parent) and self.children[parent][0] == location:
                length = self.lengths[location] / 2
            else:
                length = self.lengths[location]
            # Primary segments are straightforward extensions of the parent
            # branch. Non-primary segments are lateral branchs off to the side
            # of the parent branch. Subtract the parent's radius from the
            # secondary nodes length, to avoid excessive overlap between
            # segments.
            if self.is_root(location):
                primary = True
            else:
                siblings = self.children[parent]
                if siblings[0] == location or (self.is_root(parent) and siblings[1] == location):
                    primary = True
                else:
                    primary = False
            if not primary:
                parent_radius = self.diameters[parent] / 2
                if length > parent_radius + epsilon * 1e-6:
                    length -= parent_radius
                else:
                    # This segment is entirely enveloped within its parent. In
                    # this corner case allow the segment to protrude directly
                    # from the center of the parent instead of the surface.
                    pass
            self.surface_areas[location] = 2 * np.pi * radius * length
            self.intra_volumes[location] = np.pi * radius ** 2 * length
            # Account for the surface area on the tips of terminal/leaf segments.
            num_children = len(self.children[location])
            if num_children == 0 or (self.is_root(location) and num_children == 1):
                self.surface_areas[location] += np.pi * radius ** 2
        assert(all(x  >= epsilon * (1e-6)**2 for x in self.cross_sectional_areas))
        assert(all(sa >= epsilon * (1e-6)**2 for sa in self.surface_areas))
        assert(all(v  >= epsilon * (1e-6)**3 for v in self.intra_volumes))

    def _init_extracellular_properties(self):
        self._tree = scipy.spatial.cKDTree(self.coordinates)
        self.extra_volumes = np.empty(len(self), dtype=Real)
        self.neighbors = np.zeros(len(self), dtype=object)
        bounding_sphere = np.array([
                [ 1, 0,  0, -self.maximum_extracellular_radius],
                [-1, 0,  0, -self.maximum_extracellular_radius],
                [0,  1,  0, -self.maximum_extracellular_radius],
                [0, -1,  0, -self.maximum_extracellular_radius],
                [0,  0,  1, -self.maximum_extracellular_radius],
                [0,  0, -1, -self.maximum_extracellular_radius],
            ])
        origin = np.zeros(3)
        for location in range(len(self)):
            coords = self.coordinates[location]
            neighbors = self._tree.query_ball_point(coords, 2 * self.maximum_extracellular_radius)
            neighbors.remove(location)
            midpoints = np.array((self.coordinates[neighbors] - coords) / 2, dtype=float)
            normals = midpoints / np.linalg.norm(midpoints)
            offsets = np.sum(normals * midpoints, axis=1).reshape(-1,1)
            planes = np.vstack((np.hstack((normals, -offsets)), bounding_sphere))
            halfspace_hull = scipy.spatial.HalfspaceIntersection(planes, origin, qhull_options='t')
            convex_hull = scipy.spatial.ConvexHull(halfspace_hull.intersections)
            self.extra_volumes[location] = convex_hull.volume
            self.neighbors[location] = []
            for v in halfspace_hull.dual_vertices:
                if v not in range(len(neighbors)): continue # Adjacency to the bounding sphere.
                n = Neighbor()
                n.location = neighbors[v]
                n.distance = np.linalg.norm(coords - self.coordinates[n.location])
                # The scipy bindings to QHull don't appear expose the area of
                # each facet. Instead find the vertexes of this facet, project
                # the vertexes onto the facets plane, and recompute the convex
                # hull in 2D to get the surface area.
                basis1 = scipy.spatial.transform.Rotation.from_rotvec([np.pi/2, 0, 0]).apply(planes[v,:3])
                basis2 = scipy.spatial.transform.Rotation.from_rotvec([0, np.pi/2, 0]).apply(planes[v,:3])
                projection = []
                for x in halfspace_hull.intersections:
                    if abs(np.dot(x, planes[v,:3]) + planes[v,3]) <= epsilon:
                        projection.append([basis1.dot(x), basis2.dot(x)])
                facet_hull = scipy.spatial.ConvexHull(projection, qhull_options='t')
                n.border_surface_area = facet_hull.volume
                self.neighbors[location].append(n)

    def __len__(self):
        return len(self.coordinates)

    def is_root(self, location):
        return self.parents[location] == ROOT

    def nearest_neighbors(self, coordinates, k, maximum_distance=np.inf):
        coordinates = np.array(coordinates, dtype=Real)
        assert(coordinates.shape == (3,))
        assert(all(np.isfinite(x) for x in coordinates))
        k = int(k)
        assert(k >= 1)
        d, i = self._tree.query(coordinates, k, distance_upper_bound=maximum_distance)
        return i

class Neighbor:
    __slots__ = ("location", "distance", "border_surface_area")

class Diffusion:
    def __init__(self, time_step, geometry, species):
        self.time_step                  = time_step
        self.species                    = species
        self.concentrations             = np.zeros(len(geometry), dtype=Real)
        self.previous_concentrations    = np.zeros(len(geometry), dtype=Real)
        self.release_rates              = np.zeros(len(geometry), dtype=Real)
        # Compute the coefficients of the derivative function:
        # dX/dt = C * X, where C is Coefficients matrix and X is state vector.
        rows = []; cols = []; data = []
        if species.where == "intracellular":
            for location in len(geometry):
                if geometry.is_root(location):
                    continue
                parent = geometry.parents[location]
                l = geometry.lengths[location]
                if geometry.is_root(parent) and geometry.children[parent][0] == location:
                    l += geometry.lengths[parent]
                flux = diffusivity * geometry.cross_sectional_areas[location] / l
                rows.append(location)
                cols.append(parent)
                data.append(+1 * flux / geometry.intra_volumes[parent])
                rows.append(location)
                cols.append(location)
                data.append(-1 * flux / geometry.intra_volumes[location])
                rows.append(parent)
                cols.append(location)
                data.append(+1 * flux / geometry.intra_volumes[location])
                rows.append(parent)
                cols.append(parent)
                data.append(-1 * flux / geometry.intra_volumes[parent])
        elif species.where == "extracellular":
            for location in len(geometry):
                for neighbor in geometry.neighbors[location]:
                    flux = diffusivity * neighbor.border_surface_area / neighbor.distance
                    rows.append(location)
                    cols.append(neighbor.location)
                    data.append(+1 * flux / geometry.extra_volumes[neighbor.location])
                    rows.append(location)
                    cols.append(location)
                    data.append(-1 * flux / geometry.extra_volumes[location])
        # Note: always use double precision floating point for building the impulse response matrix.
        coefficients = csc_matrix((data, (rows, cols)), shape=(len(geometry), len(geometry)), dtype=float)
        coefficients.data *= self.time_step
        self.irm = expm(coefficients)
        # Prune the impulse response matrix at epsilon.
        self.irm.data[np.abs(self.irm.data) < epsilon] = 0
        self.irm = csr_matrix(self.irm, dtype=Real)

    def _advance(self):
        self.previous_concentrations = np.array(self.concentrations, copy=True)
        # Calculate the local release / removal.
        for location in range(len(self.concentrations)):
            self.concentrations[location] = max(0,
                self.concentrations[location] + self.release_rates * self.time_step)
        # Calculate the lateral diffusion.
        self.concentrations = self.irm.dot(self.concentrations)

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
        rows = []; cols = []; data = []
        for location in range(len(geometry)):
            if geometry.is_root(location):
                continue
            parent = geometry.parents[location]
            r = self.axial_resistances[location]
            rows.append(location)
            cols.append(parent)
            data.append(+1 / r / self.capacitances[location]) # TODO: Is this the correct capacitance?
            rows.append(location)
            cols.append(location)
            data.append(-1 / r / self.capacitances[location])
            rows.append(parent)
            cols.append(location)
            data.append(+1 / r / self.capacitances[parent]) # TODO: Is this the correct capacitance?
            rows.append(parent)
            cols.append(parent)
            data.append(-1 / r / self.capacitances[parent])
        # Note: always use double precision floating point for building the impulse response matrix.
        coefficients = csc_matrix((data, (rows, cols)), shape=(len(geometry), len(geometry)), dtype=float)
        coefficients.data *= self.time_step
        self.irm = expm(coefficients)
        # Prune the impulse response matrix at epsilon millivolts.
        self.irm.data[np.abs(self.irm.data) < epsilon * 1e-3] = 0
        self.irm = csr_matrix(self.irm, dtype=Real)

    def _advance(self):
        self.previous_voltages = np.array(self.voltages, copy=True)
        # Calculate the trans-membrane currents.
        for location in range(len(self.voltages)):
            delta_v = self.driving_voltages[location] - self.voltages[location]
            recip_rc = self.conductances[location] / self.capacitances[location]
            self.voltages[location] += (delta_v * (1.0 - exp(-self.time_step * recip_rc)))
        # Calculate the lateral currents throughout the neurons.
        self.voltages = self.irm.dot(self.voltages)

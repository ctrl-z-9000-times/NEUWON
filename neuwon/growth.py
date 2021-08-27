import numpy as np
import scipy.spatial
import heapq
import itertools
import random
from collections.abc import Callable, Iterable, Mapping
from graph_algorithms import depth_first_traversal as dft
from neuwon.regions import Region
from neuwon.database import epsilon, DB_Class

"""
API design notes:
-> Growth Routine API/concept.
        A class that the user can call to incrementially grow neurites.
        growth routines will contain:
            * At least one Region,
            * Dozens of parameters,
            * growth_routine.grow(*args) -> list of Segments
            * References to other growth-routines (for example as tips to grow off of)
            * an API getting out segments (for other growth routines to grow off of)

User will create an instance of GrowthRoutine for each group of neurons they
want to create. Then they will incrementially (or all at once) create segments.

"""

class _Distribution:
    def __init__(self, arg):
        if isinstance(arg, Iterable):
            mean, std_dev = arg
            self.mean    = float(mean)
            self.std_dev = float(std_dev)
        else:
            self.mean    = float(arg)
            self.std_dev = 0.0
    def __call__(self, size=1):
        return np.random.normal(self.mean, self.std_dev, size=size)

class GrowthRoutine:
    def grow(self, *args, **kwargs) -> "list of Segments":
        """ """
        raise NotImplementedError(type(self))

    def get_segments(self):
        return self.segments # Default implementation.

class Soma(GrowthRoutine):
    def __init__(self, Segment, region, diameter):
        self.Segment = Segment
        self.region = region
        self.diameter = _Distribution(diameter)
        self.segments = []
        assert(isinstance(self.region, Region))
        assert(self.diameter.mean - 2 * self.diameter.std_dev > 0)

    def grow(self, num_cells):
        new_segments = []
        for _ in range(num_cells):
            coordinates = self.region.sample_point()
            diameter = self.diameter()
            while diameter <= epsilon:
                diameter = self.diameter()
            x = self.Segment(None, coordinates, diameter)
            new_segments.append(x)
        self.segments.extend(new_segments)
        return new_segments

class Tree(GrowthRoutine):
    """ Grow dendrites or axons

    This implements the TREES algorithm combined with the morphological
    constraints used by the ROOTS algorithm.

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
    def __init__(self, roots,
            region,
            carrier_point_density,
            balancing_factor=0.5,
            extension_angle=np.pi,
            extension_distance=np.inf,
            bifurcation_angle=np.pi,
            bifurcation_distance=np.inf,
            extend_before_bifurcate=False,
            only_bifurcate=False,
            maximum_segment_length=np.inf,
            diameter=.3e-6,):
        if isinstance(roots, GrowthRoutine):
            self.roots = list(roots.get_segments())
        elif isinstance(roots, Iterable):
            self.roots = list(roots)
        else:
            self.roots = [roots]
        self.Segment = type(self.roots[0])
        assert all(type(r) == self.Segment for r in self.roots)
        self.region = region
        assert(isinstance(region, Region))
        self.carrier_point_density  = float(carrier_point_density)
        self.balancing_factor       = float(balancing_factor)
        self.extension_angle        = float(extension_angle)
        self.extension_distance     = float(extension_distance)
        self.bifurcation_angle      = float(bifurcation_angle)
        self.bifurcation_distance   = float(bifurcation_distance)
        self.extend_before_bifurcate = bool(extend_before_bifurcate)
        self.only_bifurcate         = bool(only_bifurcate)
        self.maximum_segment_length = float(maximum_segment_length)
        self.diameter               = float(diameter)
        assert(all(isinstance(s, self.Segment) for s in self.roots))
        assert(0 <= self.carrier_point_density)
        assert(0 <= self.balancing_factor)
        assert(0 <= self.extension_angle <= np.pi)
        assert(0 <= self.extension_distance)
        assert(0 <= self.bifurcation_angle <= np.pi)
        assert(0 <= self.bifurcation_distance)

    def grow(self):
        self.path_lengths = PathLengthCache()
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
                    heapq.heappush(costs, (c, parent, index))
        for seed in self.roots:
            compute_costs_to_all_carriers(seed)
        while costs or bifurcations:
            if costs:
                c, parent, index = heapq.heappop(costs)
                if not free_points[index]: continue
                if self.extend_before_bifurcate and parent.children:
                    heapq.heappush(bifurcations, (c, parent, index))
                    continue
            elif bifurcations:
                c, parent, index = heapq.heappop(bifurcations)
                if not free_points[index]: continue
            coordinates = carrier_points[index]
            if not self.morphological_constraints_satisfied(parent, coordinates):
                continue
            free_points[index] = False
            s = self.Segment(parent, coordinates, self.diameter)
            self.segments.append(s)
            compute_costs_to_all_carriers(s)
        del self.path_lengths


    def cost_function(self, parent, child_coordinates):
        distance = np.linalg.norm(np.subtract(parent.coordinates, child_coordinates))
        path_length = self.path_lengths[parent] + distance
        return distance + self.balancing_factor * path_length

    def morphological_constraints_satisfied(self, parent, child_coordinates):
        if (self.only_bifurcate and
            len(parent.children) >= 2 and
            parent not in self.roots):
            return False
        distance = np.linalg.norm(np.subtract(parent.coordinates, child_coordinates))
        def angle_function(a, b):
            return np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
        if parent.parent:
            parent_vector = np.subtract(parent.coordinates, parent.parent.coordinates)
            child_vector = np.subtract(child_coordinates, parent.coordinates)
            angle = angle_function(parent_vector, child_vector)
        else:
            angle = None
        # Check extension distance & angle.
        if not parent.children:
            if distance > self.extension_distance:
                return False
            if angle is not None and angle > self.extension_angle:
                return False
        else:
            # Check bifurcation distance & angle.
            if distance > self.bifurcation_distance:
                return False
            if angle is not None and angle > self.bifurcation_angle:
                return False
        return True

class PathLengthCache:
    def __init__(self):
        self.path_lengths = {}

    def __getitem__(self, segment):
        segment_idx = segment.get_unstable_index()
        length = self.path_lengths.get(segment_idx, None)
        if length is not None: return length
        # Walk towards the root of the tree.
        cursor = segment
        path = []
        while True:
            idx = cursor.get_unstable_index()
            if idx in self.path_lengths:
                break
            if cursor.is_root():
                self.path_lengths[idx] = 0
                break
            path.append(cursor)
            cursor = cursor.parent
        # Compute path length for all segments in the path.
        while path:
            cursor = path.pop()
            parent = cursor.parent
            cursor_idx = cursor.get_unstable_index()
            parent_idx = parent.get_unstable_index()
            self.path_lengths[cursor_idx] = cursor.length + self.path_lengths[parent_idx]

        return self.path_lengths[segment_idx]



class Synapses(GrowthRoutine):
    def __init__(self, model, axons, dendrites, pre_gap_post, diameter, num_synapses):
        self.model = model
        self.axons = list(axons)
        self.dendrites = list(dendrites)
        num_synapses = int(num_synapses)
        pre_len, gap_len, post_len = pre_gap_post
        f_pre = pre_len / sum(pre_gap_post)
        f_post = post_len / sum(pre_gap_post)
        self.presynaptic_segments = []
        self.postsynaptic_segments = []
        # Find all possible synapses.
        pre = scipy.spatial.cKDTree([x.coordinates for x in self.axons])
        post = scipy.spatial.cKDTree([x.coordinates for x in self.dendrites])
        results = pre.query_ball_tree(post, sum(pre_gap_post))
        results = list(itertools.chain.from_iterable(
            ((pre, post) for post in inner) for pre, inner in enumerate(results)))
        # Select some synapses and make them.
        random.shuffle(results)
        for pre, post in results:
            if num_synapses <= 0:
                break
            pre = self.axons[pre]
            post = self.dendrites[post]
            if pre_len and len(pre.children) > 1: continue
            if post_len and len(post.children) > 1: continue
            if pre_len == 0:
                self.presynaptic_segments.append(pre)
            else:
                x = (1 - f_pre) * np.array(pre.coordinates) + f_pre * np.array(post.coordinates)
                self.presynaptic_segments.append(model.create_segment(pre, x, diameter)[0])
            if post_len == 0:
                self.postsynaptic_segments.append(post)
            else:
                x = (1 - f_post) * np.array(post.coordinates) + f_post * np.array(pre.coordinates)
                self.postsynaptic_segments.append(model.create_segment(post, x, diameter)[0])
            num_synapses -= 1
        self.presynaptic_segments = list(set(self.presynaptic_segments))
        self.segments = self.presynaptic_segments + self.postsynaptic_segments



# TODO: This code got ripped out of the Segment module...
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

import numpy as np
import scipy.spatial
import heapq
import itertools
import random
from collections.abc import Callable, Iterable
from graph_algorithms import depth_first_traversal as dft
from neuwon.regions import Region
from neuwon.model import Segment

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

class GrowSynapses:
    def __init__(self, axons, dendrites, pre_gap_post, diameter, num_synapses):
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
                self.presynaptic_segments.append(Segment(x, diameter, pre))
            if post_len == 0:
                self.postsynaptic_segments.append(post)
            else:
                x = (1 - f_post) * np.array(post.coordinates) + f_post * np.array(pre.coordinates)
                self.postsynaptic_segments.append(Segment(x, diameter, post))
            num_synapses -= 1
        self.presynaptic_segments = list(set(self.presynaptic_segments))
        self.segments = self.presynaptic_segments + self.postsynaptic_segments

class Growth:
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
    def __init__(self,
            model,
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
            maximum_segment_length=np.inf,
            diameter=.3e-6,):
        self.model = model
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
        self.diameter               = float(diameter)
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
        self.path_lengths = {}
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
        for seed in self.seeds:
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
            s = self.model.create_segment(parent, coordinates, self.diameter,
                    maximum_segment_length=self.maximum_segment_length)
            self.segments.extend(s)
            compute_costs_to_all_carriers(s[-1])

    def cost_function(self, parent, child_coordinates):
        if parent not in self.path_lengths:
            self.path_lengths[parent] = 0
            cursor = parent
            while cursor.parent is not None:
                self.path_lengths[parent] += np.linalg.norm(cursor.coordinates - cursor.parent.coordinates)
                cursor = cursor.parent
                if cursor in self.path_lengths:
                    self.path_lengths[parent] += self.path_lengths[cursor]
                    break
        distance = np.linalg.norm(np.subtract(parent.coordinates, child_coordinates))
        path_length = self.path_lengths[parent] + distance
        return distance + self.balancing_factor * path_length

    def morphological_constraints_satisfied(self, parent, child_coordinates):
        if (self.only_bifurcate and
            len(parent.children) >= 2 and
            parent not in self.seeds):
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

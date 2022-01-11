from graph_algorithms import depth_first_traversal as dft
from neuwon.brains.regions import Region
import heapq
import math
import numpy as np
import scipy.spatial


def growth_algorithm(roots, global_region, carrier_point_density, path_length_cache=None, *,
            balancing_factor=0.5,
            extension_angle=math.pi,
            extension_distance=math.inf,
            bifurcation_angle=math.pi,
            bifurcation_distance=math.inf,
            extend_before_bifurcate=False,
            only_bifurcate=False,
            maximum_segment_length=math.inf,
            neuron_region=None,
            segment_parameters,):
    """ Grow dendrites and axons

    This implements the TREES algorithm combined with the morphological
    constraints of the ROOTS algorithm.

    TREES:
        Cuntz H, Forstner F, Borst A, Hausser M (2010) One Rule to Grow Them
        All: A General Theory of Neuronal Branching and Its Practical
        Application. PLoS Comput Biol 6(8): e1000877.
        doi:10.1371/journal.pcbi.1000877

    ROOTS:
        Bingham CS, Mergenthal A, Bouteiller J-MC, Song D, Lazzi G and Berger TW
        (2020) ROOTS: An Algorithm to Generate Biologically Realistic Cortical
        Axons and an Application to Electroceutical Modeling. Front. Comput.
        Neurosci. 14:13. doi:10.3389/fncom.2020.00013
    """

    # Gather and check parameters.
    if path_length_cache is None: path_length_cache = PathLengthCache()
    carrier_point_density   = float(carrier_point_density)
    balancing_factor        = float(balancing_factor)
    extension_angle         = float(extension_angle)
    extension_distance      = float(extension_distance)
    bifurcation_angle       = float(bifurcation_angle)
    bifurcation_distance    = float(bifurcation_distance)
    max_distance            = max(extension_distance, bifurcation_distance)
    extend_before_bifurcate = bool(extend_before_bifurcate)
    only_bifurcate          = bool(only_bifurcate)
    maximum_segment_length  = float(maximum_segment_length)
    assert isinstance(global_region, Region)
    assert isinstance(neuron_region, Region) or (neuron_region is None)
    assert 0 <= carrier_point_density
    assert 0 <= balancing_factor
    assert 0 <= extension_angle <= math.pi
    assert 0 <= extension_distance
    assert 0 <= bifurcation_angle <= math.pi
    assert 0 <= bifurcation_distance

    def cost_function(parent, child_coordinates):
        distance = np.linalg.norm(np.subtract(parent.coordinates, child_coordinates))
        path_length = path_length_cache[parent] + distance
        return distance + balancing_factor * path_length

    def morphological_constraints_satisfied(parent, child_coordinates):
        if (only_bifurcate and
            len(parent.children) >= 2 and
            not parent.is_root()):
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
            if distance > extension_distance:
                return False
            if angle is not None and angle > extension_angle:
                return False
        else:
            # Check bifurcation distance & angle.
            if distance > bifurcation_distance:
                return False
            if angle is not None and angle > bifurcation_angle:
                return False
        return True

    # Generate the carrier points and associated data structures.
    carrier_points  = global_region.sample_points(carrier_point_density)
    free_points     = np.ones(len(carrier_points), dtype=bool)
    tree = scipy.spatial.cKDTree(carrier_points)

    # Use min-heap priority queue to find the lowest cost segments.
    # Heap datum format is (cost, parent_segment, carrier_point_index)
    costs = [] # Primary queue for all potential new segments.
    bifurcations = [] # Secondary queue for potential bifurcations, used when extend_before_bifurcate=True.
    def compute_costs_to_all_carriers(parent):
        for index in tree.query_ball_point(parent.coordinates, max_distance):
            if not free_points[index]:
                continue
            coordinates = carrier_points[index]
            if neuron_region is not None:
                soma_coordinates = parent.neuron.root.coordinates
                relative_coordinates = coordinates - soma_coordinates
                if not neuron_region.contains(relative_coordinates):
                    continue
            cost = cost_function(parent, coordinates)
            heapq.heappush(costs, (cost, parent, index))

    # Initialize the potential connections from the starting seeds to all carrier points.
    for seed in roots:
        compute_costs_to_all_carriers(seed)

    # Run the modified Prim's algorithm.
    new_segments = []
    while costs or bifurcations:
        if costs:
            c, parent, index = heapq.heappop(costs)
            if not free_points[index]: continue
            if extend_before_bifurcate and parent.children:
                heapq.heappush(bifurcations, (c, parent, index))
                continue
        elif bifurcations:
            c, parent, index = heapq.heappop(bifurcations)
            if not free_points[index]: continue
        coordinates = carrier_points[index]
        if not morphological_constraints_satisfied(parent, coordinates):
            continue
        free_points[index] = False
        s = parent.add_segment(coordinates, **segment_parameters)
        new_segments.append(s)
        compute_costs_to_all_carriers(s)

    return new_segments


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



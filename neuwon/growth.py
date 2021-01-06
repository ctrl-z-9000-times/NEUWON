import numpy as np
import scipy.spatial
import heapq
from graph_algorithms import depth_first_traversal as dft
from neuwon.regions import Region

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
            diameter=None,):
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
        self.diameter               = diameter
        assert(all(isinstance(s, Segment) for s in self.seeds))
        assert(isinstance(region, regions.Region))
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
            s = parent.add_segment(coordinates, .3e-6, self.maximum_segment_length)
            self.segments.extend(s)
            compute_costs_to_all_carriers(s[-1])
        for s in self.segments:
            if s.parent in self.seeds:
                if self.diameter is None:
                    s.set_diameter()
                elif isinstance(self.diameter, Iterable):
                    s.set_diameter(self.diameter)
                else:
                    for x in dft(s, lambda x: x.children):
                        x.diameter = self.diameter

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

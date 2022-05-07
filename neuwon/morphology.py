from . import regions
from .regions import Region
from collections.abc import Iterable, Callable, Mapping
from graph_algorithms import depth_first_traversal as dft
from neuwon.database import epsilon
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

class NeuronGrowthProgram:
    def __init__(self, model, neuron_type, program):
        self.model = model
        self.neuron_type = str(neuron_type)
        self.neurons = []
        self.segments = []
        self.path_length_cache = PathLengthCache()
        self._run_program(*program)

    def _run_program(self, soma_parameters, *instructions_list):
        self._grow_soma(**soma_parameters)
        for parameters in instructions_list:
            self._run_growth_algorithm(**parameters)

    def _grow_soma(self, *,
                segment_type,
                region,
                diameter,
                number=None,
                density=None,
                mechanisms={},):
        # Clean the inputs.
        segment_type = str(segment_type)
        region = self.model.regions.make_region(region)
        assert (number is not None) or (density is not None), 'missing argument "number" or "density".'
        assert (number is None) != (density is None), "'number' and 'density' are mutually exclusive."
        if number is not None:
            coordinates = [region.sample_point() for _ in range(number)]
        else:
            coordinates = region.sample_points(density)
        diameter = _Distribution(diameter)

        new_segments = []
        for c in coordinates:
            d = diameter()
            while d <= epsilon:
                d = diameter()
            n = self.model.Neuron(c, d, segment_type=segment_type)
            self.neurons.append(n)
            new_segments.append(n.root)
        self.segments.extend(new_segments)
        for segment in new_segments:
            segment.insert(mechanisms)

    def _run_growth_algorithm(self, *,
                segment_type,
                region,
                diameter,
                grow_from=None,
                exclude_from=None,
                morphology={},
                mechanisms={},):
        # Clean the inputs.
        segment_type = str(segment_type)
        region = self.model.regions.make_region(region)

        if grow_from is None:
            roots = self.segments
        else:
            # grow_from is a segment type (or a list of them?)
            # It's a reference to a segment-type made in this program
            #       Raise an error if its not present / not grown yet.
            #       Only applies to the neurons made by *this* NeuronGrowthProgram.
            1/0 # TODO!

        neuron_region = morphology.pop('neuron_region', None)
        if neuron_region is not None:
            neuron_region = self.model.regions.make_region(neuron_region)

        if exclude_from:
            1/0 # TODO!
            # exclude_from is a segment type (or list of them)
            #   The current growth step will not grow in the same neuron_region as the
            #   steps which produce those segment types.

        competitive = bool(morphology.pop('competitive', True))

        # Run the growth algorithm.
        if competitive:
            segments = growth_algorithm(roots, region,
                    path_length_cache=self.path_length_cache,
                    segment_parameters={
                            'segment_type': segment_type,
                            'diameter':     float(diameter),},
                    neuron_region=neuron_region,
                    **morphology)
        else:
            roots_by_neuron = {}
            for segment in roots:
                roots_by_neuron.setdefault(segment.neuron, []).append(segment)
            segments = []
            for roots in roots_by_neuron.values():
                segments.extend(growth_algorithm(roots, region,
                        path_length_cache=self.path_length_cache,
                        segment_parameters={
                                'segment_type': segment_type,
                                'diameter':     float(diameter),},
                        neuron_region=neuron_region,
                        **morphology))

        self.segments.extend(segments)
        for segment in segments:
            segment.insert(mechanisms)

class NeuronFactory(dict):
    def __init__(self, rxd_model, neuron_parameters:dict, segment_parameters:dict={}):
        super().__init__()
        self.rxd_model = rxd_model
        for neuron_type, program in neuron_parameters.items():
            program = self._merge_segment_type_defaults(program, segment_parameters)
            self.add_neuron_type(neuron_type, program)

    def _merge_segment_type_defaults(self, program, segment_parameters):
        merged = []
        for instruction in program:
            segment_type = instruction.get("segment_type", None)
            if segment_type is None:
                merged.append(instruction)
                continue
            defaults = segment_parameters.get(segment_type, None)
            if defaults is None:
                merged.append(instruction)
                continue
            instruction = _recursive_merge(defaults, instruction)
            merged.append(instruction)
        return merged

    def add_neuron_type(self, neuron_type: str, program: list):
        neuron_type = str(neuron_type)
        assert neuron_type not in self
        self[neuron_type] = NeuronGrowthProgram(self.rxd_model, neuron_type, program).neurons

def _recursive_merge(defaults, updates):
    output = dict(defaults)
    for k,v in updates.items():
        if isinstance(v, Mapping):
            output[k] = _recursive_merge(output.get(k, {}), v)
        else:
            output[k] = v
    return output

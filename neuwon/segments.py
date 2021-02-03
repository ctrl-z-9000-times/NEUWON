import numpy as np
import math
from neuwon import docstring_wrapper
from graph_algorithms import depth_first_traversal as dft

class Segment:
    parent      = docstring_wrapper("parent", "Segment or None")
    children    = docstring_wrapper("children", "List of Segment's")
    coordinates = docstring_wrapper("coordinates", "Tuple of 3 floats")
    diameter    = docstring_wrapper("diameter", "Float (positive)")
    insertions  = docstring_wrapper("insertions", "List of pairs of (mechanisms, mechanisms_arguments_tuple)")

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

def _serialize_segments(model, segments):
    roots = set()
    for x in segments:
        while x.parent:
            x = x.parent
        roots.add(x)
    segments = []
    for r in roots:
        segments.extend(dft(r, lambda x: x.children))
    segment_location = {x: location for location, x in enumerate(segments)}
    coordinates = [x.coordinates for x in segments]
    parents     = [segment_location.get(x.parent, None) for x in segments]
    diameters   = [x.diameter for x in segments]
    insertions  = [x.insertions for x in segments]
    for location, x in enumerate(segments):
        if x.model is not None:
            raise ValueError("Segment included in multiple models!")
        x.location = location
        x.model = model
    return (coordinates, parents, diameters, insertions)

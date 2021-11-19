from neuwon.database import epsilon
from .electric import Electric
from .geometry import Tree, Geometry
import numpy as np
import re

# THOUGHT: I'm going to reorg everthing to call the neuron as the public entry
# point. and then the neuron init will make all of the bookkeeping, and then
# pass through to segs init.

class Neuron:
    __slots__ = ()
    @staticmethod
    def _intialize(database, **electric_arguments):
        neuron_data  = database.add_class(Neuron)
        segment_data = database.add_class(Segment)
        Segment._initialize(database, **electric_arguments)

        neuron_data .add_attribute('root', dtype='Segment')
        segment_data.add_attribute('neuron', dtype='Neuron')

        neuron_cls          = neuron_data.get_instance_type()
        neuron_cls._Segment = Segment_data.get_instance_type()
        return neuron_cls

    def __init__(self, coordinates, diameter):
        Segment = type(self)._Segment
        self.root = Segment(parent=None, coordinates=coordinates, diameter=diameter)

    def add_segment(self, coordinates, diameter):
        Segment = type(self)._Segment
        return Segment(self.root, coordinates, diameter)

class Segment(Tree, Geometry, Electric):
    """ """
    __slots__ = ()
    @staticmethod
    def _initialize(database, **electric_arguments):
        """ Do not directly call this method! Call Neuron._initialize() instead. """
        Tree._initialize(database)
        Geometry._initialize(database)
        Electric._initialize(database, **electric_arguments)

    def __init__(self, parent, coordinates, diameter):
        Tree.__init__(self, parent)
        Geometry.__init__(self, coordinates, diameter)
        Electric.__init__(self)
        if self.parent is not None:
            self.neuron = self.parent.neuron

    @classmethod
    def load_swc(cls, swc_data):
        # TODO: Arguments for coordinate offsets and rotations.
        swc_data = str(swc_data)
        if swc_data.endswith(".swc"):
            with open(swc_data, 'rt') as f:
                swc_data = f.read()
        swc_data = re.sub(r"#.*", "", swc_data) # Remove comments.
        swc_data = [x.strip() for x in swc_data.split("\n") if x.strip()]
        entries = dict()
        for line in swc_data:
            cursor = iter(line.split())
            sample_number = int(next(cursor))
            structure_id = int(next(cursor))
            coords = (float(next(cursor)), float(next(cursor)), float(next(cursor)))
            radius = float(next(cursor))
            parent = int(next(cursor))
            entries[sample_number] = cls(entries.get(parent, None), coords, 2 * radius)

    def add_segment(self, coordinates, diameter):
        return Segment(self, coordinates, diameter)

    def add_section(self, coordinates, diameter, maximum_segment_length=np.inf):
        """
        Returns a list of Segments.
        """
        cls = type(self)
        maximum_segment_length = float(maximum_segment_length)
        assert maximum_segment_length > 0
        coords      = [float(x) for x in coordinates]
        diameter    = float(diameter)
        displace    = np.subtract(self.coordinates, coordinates)
        length      = np.linalg.norm(displace)
        p_coords    = np.array(self.coordinates)
        if self.is_sphere() or len(self.children) > 0:
            self_r  = 0.5 * self.diameter
            if self_r < length - epsilon:
                p_coords -= displace * (self_r / length)
                length   -= self_r
        divisions = max(1, int(np.ceil(length / maximum_segment_length)))
        x = np.linspace(p_coords[0], coordinates[0], divisions+1)
        y = np.linspace(p_coords[1], coordinates[1], divisions+1)
        z = np.linspace(p_coords[2], coordinates[2], divisions+1)
        d = np.linspace(self.diameter, diameter,   divisions+1)
        args = zip(x,y,z,d)
        next(args)
        section = []
        for (x,y,z,d) in args:
            self = cls(self, (x,y,z), d)
            section.append(self)
        return section

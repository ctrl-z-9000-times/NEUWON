from neuwon.database import epsilon
from .electric import Electric
from .geometry import Tree, Geometry
import numpy as np
import re

class Segment(Tree, Geometry, Electric):
    """ """
    __slots__ = ()
    @classmethod
    def _initialize(cls, database, **electric_arguments):
        db_cls = database.add_class("Segment", cls)
        Tree._initialize(database)
        Geometry._initialize(database)
        Electric._initialize(database, **electric_arguments)
        return db_cls.get_instance_type()

    def __init__(self, parent, coordinates, diameter):
        Tree.__init__(self, parent)
        Geometry.__init__(self, coordinates, diameter)
        Electric.__init__(self)
        # if self.parent is not None:
        #     self.neuron = self.parent.neuron

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

    @classmethod
    def make_section(cls, parent, coordinates, diameter, maximum_segment_length=np.inf):
        """
        Argument parents:
        Argument coordinates:
        Argument diameters:
        Argument maximum_segment_length

        Returns a list of Segments.
        """
        maximum_segment_length = float(maximum_segment_length)
        assert maximum_segment_length > 0
        if maximum_segment_length == np.inf or parent is None:
            return [cls(parent, coordinates, diameter)]
        coords      = [float(x) for x in coordinates]
        diameter    = float(diameter)
        displace    = np.subtract(parent.coordinates, coordinates)
        length      = np.linalg.norm(displace)
        p_coords    = np.array(parent.coordinates)
        if parent.is_sphere() or len(parent.children) > 0:
            parent_r  = 0.5 * parent.diameter
            if parent_r < length - epsilon:
                p_coords -= displace * (parent_r / length)
                length   -= parent_r
        divisions   = max(1, int(np.ceil(length / maximum_segment_length)))
        section     = []
        x = np.linspace(p_coords[0], coordinates[0], divisions+1)
        y = np.linspace(p_coords[1], coordinates[1], divisions+1)
        z = np.linspace(p_coords[2], coordinates[2], divisions+1)
        d = np.linspace(parent.diameter, diameter,   divisions+1)
        args = zip(x,y,z,d)
        next(args)
        for (x,y,z,d) in args:
            parent = cls(parent, (x,y,z), d)
            section.append(parent)
        return section

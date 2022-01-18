from collections.abc import Hashable
from .electric import Electric
from .geometry import Geometry
from .tree     import Tree
from neuwon.database import Real, epsilon, Pointer, NULL, Compute
import numpy as np

class Segment(Tree, Geometry, Electric):
    """ """
    __slots__ = ()
    @staticmethod
    def _initialize(database, **electric_arguments):
        Tree._initialize(database)
        Geometry._initialize(database)
        Electric._initialize(database, **electric_arguments)
        # Add type information.
        segment_data = database.get_class('Segment')
        segment_data.add_attribute('segment_type_id', NULL, dtype=Pointer,
                doc="Read-only attribute.")
        segment_cls = segment_data.get_instance_type()
        segment_cls.segment_types_list = []
        return segment_cls

    def __init__(self, parent, coordinates, diameter, segment_type=None):
        self.segment_type = segment_type
        Tree.__init__(self, parent)
        Geometry.__init__(self, coordinates, diameter)
        Electric.__init__(self)
        if self.parent is not None:
            self.neuron = self.parent.neuron

    @property
    def segment_type(self):
        if self.segment_type_id == NULL:
            return None
        else:
            return type(self).segment_types_list[self.segment_type_id]
    @segment_type.setter
    def segment_type(self, segment_type):
        if self.segment_type_id != NULL:
            raise ValueError(f'{self} already has a segment_type!')
        if segment_type is None:
            return
        types_list = type(self).segment_types_list
        try:
            self.segment_type_id = types_list.index(segment_type)
        except ValueError:
            assert isinstance(segment_type, Hashable)
            self.segment_type_id = len(types_list)
            types_list.append(segment_type)

    def add_segment(self, coordinates, diameter, segment_type=None):
        return type(self)(self, coordinates, diameter, segment_type=segment_type)

    def add_section(self, coordinates, diameter,
            maximum_segment_length = np.inf,
            segment_type = None):
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
            self = cls(self, (x,y,z), d, segment_type=segment_type)
            section.append(self)
        return section

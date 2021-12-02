from collections.abc import Hashable
from .electric import Electric
from .geometry import Geometry
from .tree     import Tree
from neuwon.database import Real, epsilon, Pointer, NULL, Compute
import numpy as np
import re

class Neuron:
    __slots__ = ()
    @staticmethod
    def _initialize(database, **electric_arguments):
        neuron_data  = database.add_class(Neuron)
        segment_data = database.add_class(Segment)
        neuron_cls   = neuron_data .get_instance_type()
        segment_cls  = segment_data.get_instance_type()
        neuron_cls._Segment = segment_cls # Save the segment class constructor.
        Segment._initialize(database, **electric_arguments)
        Neuron._initialize_AP_detector(neuron_data)
        # Link neurons and segments together.
        neuron_data .add_attribute('root', dtype='Segment', allow_invalid=True)
        segment_data.add_attribute('neuron', dtype='Neuron')
        # Add type information.
        neuron_cls. _neuron_types_list  = []
        segment_cls._segment_types_list = []
        neuron_data .add_attribute('neuron_type_id',  NULL, dtype=Pointer,
                doc="Read-only attribute.")
        segment_data.add_attribute('segment_type_id', NULL, dtype=Pointer,
                doc="Read-only attribute.")
        return neuron_cls # Return the entry point to the public API.

    @staticmethod
    def _initialize_AP_detector(neuron_data):
        neuron_data.add_class_attribute('AP_detector_threshold', 20)
        neuron_data.add_attribute('AP_detector_segment', dtype='Segment', allow_invalid=True)
        neuron_data.add_attribute('AP_detected', False, dtype=bool)
        neuron_data.add_attribute('_AP_true_state', False, dtype=bool)

    def __init__(self, coordinates, diameter,
                neuron_type=None,
                segment_type=None):
        self.neuron_type = neuron_type
        Segment = type(self)._Segment
        self.root = Segment(parent=None, coordinates=coordinates, diameter=diameter,
                            segment_type=segment_type)
        self.root.neuron = self
        self.AP_detector_segment = self.root

    @property
    def neuron_type(self):
        if self.neuron_type_id == NULL:
            return None
        else:
            return type(self)._neuron_types_list[self.neuron_type_id]
    @neuron_type.setter
    def neuron_type(self, neuron_type):
        if self.neuron_type_id != NULL:
            raise ValueError(f'{self} already has a neuron_type!')
        if neuron_type is None:
            return
        types_list = type(self)._neuron_types_list
        try:
            self.neuron_type_id = types_list.index(neuron_type)
        except ValueError:
            assert isinstance(neuron_type, Hashable)
            self.neuron_type_id = len(types_list)
            types_list.append(neuron_type)

    def set_AP_detector(self, segment=None, threshold=None):
        if segment is not None:
            self.AP_detector_segment = segment
        if threshold is not None:
            self.AP_detector_threshold = threshold

    @Compute
    def _advance_AP_detector(self):
        if self.AP_detector_segment == NULL:
            over_threshold = False
        else:
            over_threshold = self.AP_detector_segment.voltage >= self.AP_detector_threshold
        self.AP_detected = over_threshold and not self._AP_true_state
        self._AP_true_state = over_threshold

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
            1/0 # TODO: this needs to be rewritten!
            entries[sample_number] = cls(entries.get(parent, None), coords, 2 * radius)

class Segment(Tree, Geometry, Electric):
    """ """
    __slots__ = ()
    @staticmethod
    def _initialize(database, **electric_arguments):
        """ Do not directly call this method! Call Neuron._initialize() instead. """
        Tree._initialize(database)
        Geometry._initialize(database)
        Electric._initialize(database, **electric_arguments)

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
            return type(self)._segment_types_list[self.segment_type_id]
    @segment_type.setter
    def segment_type(self, segment_type):
        if self.segment_type_id != NULL:
            raise ValueError(f'{self} already has a segment_type!')
        if segment_type is None:
            return
        types_list = type(self)._segment_types_list
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

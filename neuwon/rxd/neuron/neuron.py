from collections.abc import Hashable
from .segment import Segment
from neuwon.database import Real, epsilon, Pointer, NULL, Compute
import re

class Neuron:
    __slots__ = ()
    @staticmethod
    def _initialize(database, **electric_arguments):
        """ This also initializes the Segment class. """
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
        neuron_cls.neuron_types_list  = []
        neuron_data.add_attribute('neuron_type_id',  NULL, dtype=Pointer) # TODO: Document what this is and how to use it.
        return neuron_cls # Return the entry point to the public API.

    @staticmethod
    def _initialize_AP_detector(neuron_data):
        neuron_data.add_class_attribute('AP_detector_threshold', 20)
        neuron_data.add_attribute('AP_detector_segment', dtype='Segment',
                # This will never be NULL, but don't enforce it as a constraint on the database.
                allow_invalid=True)
        neuron_data.add_attribute('AP_detected', False, dtype=bool)
        neuron_data.add_attribute('_AP_true_state', False, dtype=bool)

    def __init__(self, coordinates, diameter, *,
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
            return type(self).neuron_types_list[self.neuron_type_id]
    @neuron_type.setter
    def neuron_type(self, neuron_type):
        if self.neuron_type_id != NULL:
            raise ValueError(f'{self} already has a neuron_type!')
        if neuron_type is None:
            return
        types_list = type(self).neuron_types_list
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
        neuron = None
        segments = dict()
        for line in swc_data.split('\n'):
            line = re.sub(r"#.*", "", line).strip() # Remove comments and excess white space.
            if not line:
                continue
            cursor = iter(line.split())
            sample_number = int(next(cursor))
            structure_id = int(next(cursor))
            coords = (float(next(cursor)), float(next(cursor)), float(next(cursor)))
            radius = float(next(cursor))
            parent = int(next(cursor))
            parent = segments.get(parent, None)
            if parent is not None:
                segments[sample_number] = parent.add_segment(coords, 2 * radius)
            else:
                assert neuron is None
                neuron = cls(coords, 2 * radius)
                segments[sample_number] = neuron.root

    @Compute
    def _filter_by_type(self, neuron_mask) -> bool:
        return neuron_mask[self.neuron_type_id]

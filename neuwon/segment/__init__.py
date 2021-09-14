from neuwon.database import epsilon
from neuwon.segment.electric import ElectricProperties
from neuwon.segment.geometry import SegmentGeometry
import numpy as np
import re

def _custom_super(target_type, obj):
    """ Special wrapper for built-in method "super".

    Returns a bound super proxy object, where the MRO starts *at* the given
    target_type, as opposed to super's default behavior of starting the
    MRO *after* the given type.
    """
    if isinstance(obj, type):
        mro = obj.mro()
    else:
        mro = type(obj).mro()
    idx = mro.index(target_type) - 1
    return super(mro[idx], obj)

class SegmentMethods(SegmentGeometry, ElectricProperties):
    """ """
    __slots__ = ()
    @classmethod
    def _initialize(cls, database, *,
                initial_voltage,
                cytoplasmic_resistance,
                membrane_capacitance,):
        db_cls = database.add_class("Segment", cls)
        cls = db_cls.get_instance_type()
        _custom_super(SegmentGeometry, cls)._initialize(db_cls)
        _custom_super(ElectricProperties, cls)._initialize(db_cls,
                initial_voltage=initial_voltage,
                cytoplasmic_resistance=cytoplasmic_resistance,
                membrane_capacitance=membrane_capacitance,)
        return cls

    def __init__(self, parent, coordinates, diameter):
        SegmentGeometry.__init__(self, parent, coordinates, diameter)
        ElectricProperties.__init__(self)

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

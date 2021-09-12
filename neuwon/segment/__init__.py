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
    def _initialize(cls, database,
                initial_voltage = -70,
                cytoplasmic_resistance = 1e6,
                membrane_capacitance = 1e-14,):
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

import numpy as np
import re
from neuwon.segment.geometry import Tree, Geometry
from neuwon.segment.electric import ElectricProperties

__all__ = ["SegmentMethods"]

class SegmentMethods(Tree, Geometry, ElectricProperties):
    """ """
    __slots__ = ()
    @classmethod
    def _initialize(cls, database,
                initial_voltage = -70,
                cytoplasmic_resistance = 1,
                membrane_capacitance = .01,):
        db_cls = database.add_class("Segment", cls)
        Tree._initialize(db_cls)
        Geometry._initialize(db_cls)
        ElectricProperties._initialize(db_cls,
                initial_voltage=initial_voltage,
                cytoplasmic_resistance=cytoplasmic_resistance,
                membrane_capacitance=membrane_capacitance,)

    def __init__(self, parent, coordinates, diameter):
        Tree.__init__(self, parent)
        Geometry.__init__(self, coordinates, diameter)
        ElectricProperties.__init__(self, )

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

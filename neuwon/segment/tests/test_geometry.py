from neuwon.database import Database
from neuwon.segment.geometry import SegmentGeometry
import pytest

def mk_seg(db=None):
    if db is None:
        db  = Database()
    db_cls  = db.add_class("Segment", SegmentGeometry)
    SegmentGeometry._initialize(db_cls)
    return db_cls.get_instance_type()

def test_ball_and_stick():
    db      = Database()
    Segment = mk_seg(db)
    ball  = Segment(parent=None, coordinates=[0,0,0], diameter=42)
    stick = []
    tip   = ball
    for i in range(10):
        tip = Segment(parent=tip, coordinates=[i+22,0,0], diameter=3)
        stick.append(tip)
    for x in db.get("Segment").get_all_instances():
        print(x.volume)
    db.check()

@pytest.mark.skip()
def test_swc():
    db = Database()
    Geometry._initialize(db)
    Segment = db.get("Segment").get_instance_type()
    # http://neuromorpho.org/neuron_info.jsp?neuron_name=109-6_5_6_L1_CA1_N2_CG
    my_neuron = Segment.load_swc("swc_files/109-6_5_6_L1_CA1_N2_CG.CNG.swc")

    segs = [x for x in db.get("Segment").get_all_instances() if x.is_cylinder()]
    length  = sum(x.length for x in segs)
    surface = sum(x.surface_area for x in segs)
    volume  = sum(x.volume for x in segs)

    pct_diff = lambda a,b: abs(a-b) * 2 / (a + b)

    print("length", length)
    print("surface", surface)
    print("volume", volume)

    assert pct_diff(length,  1315.5) <= .05
    assert pct_diff(surface, 6621.44) <= .05
    assert pct_diff(volume,  3500.58) <= .05
    db.check()

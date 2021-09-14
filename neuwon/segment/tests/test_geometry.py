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

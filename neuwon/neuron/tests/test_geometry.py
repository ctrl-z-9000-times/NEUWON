from neuwon.database import Database
from neuwon.neuron.segment import Segment
import pytest

def test_ball_and_stick():
    db    = Database()
    Seg   = Segment._initialize(db)
    ball  = Seg(parent=None, coordinates=[0,0,0], diameter=42)
    stick = []
    tip   = ball
    assert ball.is_root()
    for i in range(10):
        tip = Seg(parent=tip, coordinates=[i+22,0,0], diameter=3)
        stick.append(tip)
    assert not tip.is_root()
    for x in db.get("Segment").get_all_instances():
        print(f'l: {x.length}, v: {x.volume}')
    db.check()

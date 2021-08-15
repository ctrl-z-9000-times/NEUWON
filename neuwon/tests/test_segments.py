from neuwon.database import *
from neuwon.segment import *

def test_ball_and_stick():
    db = Database()
    Electrics._initialize(db)
    Segment = db.get("Segment").get_instance_type()
    ball = Segment(parent=None, coordinates=[0,0,0], diameter=42)
    stick = []
    tip = ball
    for i in range(10):
        tip = Segment(parent=tip, coordinates=[i+43,0,0], diameter=3)
        stick.append(tip)
    print(tip)

if __name__ == '__main__': test_ball_and_stick()

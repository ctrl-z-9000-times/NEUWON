from neuwon.database import *
from neuwon.database.time import *
from pytest import approx
import random

def test_clock():
    db = Database()
    c = Clock(db, 1)
    
    db = Database()
    c = Clock(db, .1)
    
    db = Database()
    c = Clock(db, 1/40, 'ms')
    assert c.get_time() == 0

    for x in range(40): c.tick()
    assert c() == 1
    c.reset()
    assert c() == 0

    c.set_time(99)
    for x in range(40): c.tick()
    assert c.get_time() == 100

class Model:
    def __init__(self):
        self.db = Database()
        self.Foo = self.db.add_class("Foo")
        self.bar = self.Foo.add_attribute("bar", 0, units='my_units')
        self.Foo = self.Foo.get_instance_type()
        self.clock = self.db.add_clock(.1, 'ms')

def test_time_series_buffers():
    m = Model()
    b = TimeSeriesBuffer()

    f = m.Foo()
    b.record(f, "bar")
    for i in range(1, 51):
        f.bar = (i / 10) % 2
        m.clock.tick()
    b.stop()
    assert len(b) == 50
    # b.plot()

    f2 = m.Foo()
    b.play(f2, "bar")
    for i in range(5):
        m.clock.tick()
    assert f2.bar == .5

def test_object_traces():
    mean = 12
    std = .25

    m = Model()
    f = m.Foo()
    t = Trace((f, "bar"), 100)

    while m.clock() < 1000:
        f.bar = np.random.normal(mean, std)
        m.clock.tick()

    assert t.mean       == approx(mean, .1)
    assert t.var ** .5  == approx(std,  .1)

def test_attribute_traces():
    # TODO: test the code paths for taking traces of entire components.

    mean = 12
    std = .25

    m = Model()
    f = m.Foo()
    t = Trace(m.bar, 100)
    m.clock.tick()

    1/0

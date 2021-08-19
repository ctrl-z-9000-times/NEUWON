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

def test_traces():
    m = Model()
    f = m.Foo()
    trace_obj  = Trace((f, "bar"), 100)
    trace_attr = Trace(m.db.get("Foo.bar"), 200)

    m.Foo()
    m.Foo()
    num_foo = len(m.db.get("Foo"))

    mean = 12
    std = .5

    m.clock.reset()
    while m.clock() < 2000:
        m.db.set_data("Foo.bar", np.random.normal(mean, std, num_foo))
        m.clock.tick()

    assert trace_obj.mean       == approx(mean, .1)
    assert trace_obj.var ** .5  == approx(std,  .5)
    
    trace_mean = trace_attr.mean.get_data()
    trace_var  = trace_attr.var.get_data()
    for idx in range(num_foo):
        assert trace_mean[idx]       == approx(mean, .1)
        assert trace_var[idx] ** .5  == approx(std,  .5)

import random
from neuwon.database import *
from neuwon.database.time import *

def test_clock():
    c = Clock(1)
    c = Clock(.1)
    c = Clock(1/40, 'ms')
    assert c.time() == 0

    for x in range(40): c.tick()
    assert c() == 1
    c.reset()
    assert c() == 0

    c.set_time(99)
    for x in range(40): c.tick()
    assert c.clock() == 100

class Model:
    def __init__(self):
        self.db = Database()
        self.Foo = self.db.add_class("Foo")
        self.bar = self.Foo.add_attribute("bar", 0, units='my_units')
        self.Foo = self.Foo.get_instance_type()
        self.c = Clock(.1, 'ms')

def test_time_series_buffers():
    m = Model()
    b = TimeSeriesBuffer(m.c)

    f = m.Foo()
    b.record(f, "bar")
    for i in range(1, 51):
        f.bar = (i / 10) % 2
        m.c.tick()
    b.stop()
    # b.plot()

    f2 = Foo()
    b.play(f2, "bar")
    for i in range(5):
        m.c.tick()
    assert f2.bar == .5

def test_traces():
    m = Model()
    t = Trace(m.c, m.bar, 10)

    num_foo = 32
    for _ in range(num_foo):
        m.Foo()

    for x in range(1000):
        m.bar.set(np.random.normal(mean, std, num_foo))
        c.tick()

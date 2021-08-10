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

def test_time_series_buffers():
    db = Database()
    Foo = db.add_class("Foo")
    Foo.add_attribute("bar", 0, units='my_units')
    Foo = Foo.get_instance_type()
    c = Clock(.1, 'ms')
    b = TimeSeriesBuffer(c)

    f = Foo()
    b.record(f, "bar")
    for i in range(1, 51):
        f.bar = (i / 10) % 2
        c.tick()
    b.stop()
    # b.plot()

    f2 = Foo()
    b.play(f2, "bar")
    for i in range(5):
        c.tick()
    assert f2.bar == .5

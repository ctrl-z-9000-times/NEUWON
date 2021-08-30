from neuwon.database import *
from neuwon.database.time import *
from pytest import approx, mark
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
    for i in range(50):
        f.bar = (i / 10) % 2
        m.clock.tick()
    b.stop()
    assert len(b) == 51
    # b.plot()

    f2 = m.Foo()
    b.play(f2, "bar", mode='=')
    for i in range(6):
        m.clock.tick()
        correct_value = (i / 10) % 2
        assert f2.bar == approx(correct_value, 1e-9)

    b.stop()
    b.clear()
    assert not len(b)

def test_waveforms():
    m = Model()
    f = m.Foo()

    sqr = TimeSeriesBuffer().square_wave(0, 1, 1)
    sqr.play(f, "bar", mode="=")

    assert f.bar == 1
    for _ in range(7): m.clock.tick()
    assert f.bar == 0

    assert sum(sqr.play_data) == 5
    assert len(sqr.play_data) == 10

    tri = TimeSeriesBuffer().triangle_wave(0, 1, 2)
    tri.play(f, "bar", mode="=", loop=True)
    assert max(tri.play_data) == approx(1, .01)

    saw = TimeSeriesBuffer().sawtooth_wave(0, 1, 100)
    f2 = m.Foo()
    saw.play(f2, "bar", mode="=", loop=True)
    assert max(saw.play_data) == approx(1, .01)
    for _ in range(int(99.9 / m.clock.dt)): m.clock.tick()
    assert f2.bar == approx(1, .01)

    sin = TimeSeriesBuffer().sine_wave(2, 5, 10)
    # TODO: CHeck that when looping, the phase does not drift!
    1/0

def test_traces():
    # Test no samples / very few samples.
    for samples in range(4):
        inner_test_traces(
            period      = 100,
            samples     = samples,
            start       = True,
            mean        = 42,
            std         = 0,
            tolerance   = 1e-9,)

    inner_test_traces(
        period      =   50,
        samples     = 1000,
        start       = False, # Test no-start
        mean        = 12,
        std         = 0.5,
        tolerance   = 0.1,)

    for device in [False, True]: # Test on GPU.
        # Test num-samples == period
        inner_test_traces(
            period      = 100,
            samples     = 100,
            start       = True,
            mean        = 4000,
            std         =  600,
            tolerance   = 0.1,
            device      = device,)

def test_trace_averages():
    """ period >> num-samples. """
    inner_test_traces(
        period      = 1e9,
        samples     = 2e3,
        start       = True,
        mean        = -120,
        std         = 0.3,
        tolerance   = 0.02,)

def inner_test_traces(period, samples, start, mean, std, tolerance, device=False):
    m = Model()
    if device: m.db.to_device()
    f = m.Foo()
    for _ in range(32):
        m.Foo()
    num_foo = len(m.db.get("Foo"))
    m.db.set_data("Foo.bar", np.random.normal(mean, std, num_foo))

    trace_obj  = Trace((f, "bar"), period)
    trace_attr = Trace(m.db.get("Foo.bar"), period, start=start)
    while m.db.clock() < samples:
        m.db.set_data("Foo.bar", np.random.normal(mean, std, num_foo))
        m.clock.tick()

    assert trace_obj.get_mean()                 == approx(mean, tolerance)
    assert trace_obj.get_standard_deviation()   == approx(std,  tolerance)

    trace_mean = trace_attr.get_mean()
    trace_std  = trace_attr.get_standard_deviation()
    if device:
        trace_mean = trace_mean.get()
        trace_std  = trace_std.get()
    for idx in range(num_foo):
        assert trace_mean[idx]  == approx(mean, tolerance)
        assert trace_std[idx]   == approx(std,  tolerance)
    m.db.check()
    trace_obj.reset()
    trace_attr.reset()

from neuwon.database import *
from neuwon.database.time import *
from pytest import approx, mark
import matplotlib.pyplot as plt
import random

def test_clock():
    c = Clock(1)
    c = Clock('3.4')
    c = Clock(.1)
    c = Clock(1/40, 'ms')
    assert c.get_time() == 0

    for x in range(40): c.tick()
    assert c() == 1
    c.reset()
    assert c() == 0

    c.set_time(99)
    for x in range(40): c.tick()
    assert c.get_time() == 100

class Model:
    def __init__(self, dt=0.1):
        self.db = Database()
        db_Foo = self.db.add_class("Foo")
        self.bar = db_Foo.add_attribute("bar", 0, units='my_units')
        self.Foo = db_Foo.get_instance_type()
        self.clock = self.db.add_clock(dt, 'ms')

def test_time_series_buffers():
    m = Model()
    b = TimeSeries()

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
    m.clock.tick()

def test_waveforms_basic():
    m = Model()
    f = m.Foo()

    sqr = TimeSeries().square_wave(0, 1, 1)
    sqr.play(f, "bar", mode="=")

    assert f.bar == 1
    for _ in range(7): m.clock.tick()
    assert f.bar == 0

    assert sum(sqr.play_data) == 5
    assert len(sqr.play_data) == 10

    tri = TimeSeries().triangle_wave(0, 1, 2)
    tri.play(f, "bar", mode="=", loop=True)
    assert max(tri.play_data) == approx(1, .01)

    saw = TimeSeries().sawtooth_wave(0, 1, 100)
    f2 = m.Foo()
    saw.play(f2, "bar", mode="=", loop=True)
    assert max(saw.play_data) == approx(1, .01)
    for _ in range(int(99.9 / m.clock.dt)): m.clock.tick()
    assert f2.bar == approx(1, .01)

    sin = TimeSeries().sine_wave(2, 5, 10)

def test_waveform_frequency():
    """
    Generate some waveforms, play them back into a recording buffewr and, and
    run it for a long time. Then check that it has the correct frequency and
    that it did not drift.
    """
    m = Model(dt=1)
    p = 73
    n_cycles = 500
    f_sqr = m.Foo()
    f_tri = m.Foo()
    f_saw = m.Foo()
    f_sin = m.Foo()
    TimeSeries().square_wave(  -1, 1, p).play(f_sqr, "bar", mode="=", loop=True)
    TimeSeries().triangle_wave(-1, 1, p).play(f_tri, "bar", mode="=", loop=True)
    TimeSeries().sawtooth_wave(-1, 1, p).play(f_saw, "bar", mode="=", loop=True)
    TimeSeries().sine_wave(    -1, 1, p).play(f_sin, "bar", mode="=", loop=True)
    buf_sqr = TimeSeries().record(f_sqr, "bar")
    buf_tri = TimeSeries().record(f_tri, "bar")
    buf_saw = TimeSeries().record(f_saw, "bar")
    buf_sin = TimeSeries().record(f_sin, "bar")
    for _ in range(round(n_cycles * p / m.clock.dt)):
        m.clock.tick()
    # Check for correct phase / no drifting.
    assert f_sqr.bar == 1
    assert f_tri.bar == -1
    assert f_saw.bar == -1
    assert f_sin.bar == 0
    # Check frequency.
    for buf in (buf_sqr, buf_tri, buf_saw, buf_sin):
        data = buf.get_data()
        ts   = buf.get_timestamps()
        fft  = np.abs(np.fft.fft(data))
        freq = np.fft.fftfreq(len(fft), m.clock.dt)
        maxf = next(f for f in freq[np.argsort(-fft)] if f >= 0)
        assert maxf == approx(1.0 / p, 1e-4)
        # Optional, plot the waveform and its & Fourier transform.
        if False:
            plt.subplot(2, 1, 1)
            plt.plot(ts, data)
            plt.subplot(2, 1, 2)
            plt.plot(freq, fft)
            plt.xlim(0, max(freq))
            plt.show()

def test_obj_trace():
    m = Model()
    f = m.Foo()
    # Test short run behavior, on startup.
    f.bar = 42
    t = Trace(f, "bar", 100)
    for _ in range(3):
        f.bar = 42
        m.clock.tick()
    assert t.get_mean()                 == approx(42)
    assert t.get_standard_deviation()   == approx(0)
    # Test short run behavior, immediately after reset.
    t.reset()
    f.bar = 64
    m.clock.tick()
    assert t.get_mean()                 == approx(64)
    assert t.get_standard_deviation()   == approx(0)
    # Test long run behavior.
    for _ in range(2000):
        f.bar = np.random.normal(12, 5)
        m.clock.tick()
    assert t.get_mean()                 == approx(12, rel=.05)
    assert t.get_standard_deviation()   == approx(5,  rel=.05)
    m.db.check()
    t.reset()

def test_attr_trace():
    mean = -120
    std  = 0.3
    m = Model()
    for _ in range(32):
        m.Foo()
    num_foo = len(m.db.get("Foo"))
    m.db.set_data("Foo.bar", np.random.normal(mean, std, num_foo))
    t = TraceAll(m.db.get("Foo.bar"), 100)
    for i in range(5000):
        m.db.set_data("Foo.bar", np.random.normal(mean, std, num_foo))
        if i % 2 == 1:
            with m.db.using_memory_space('cuda'):
                gpu_array = m.db.get_data("Foo.bar")
                m.clock.tick()
        else:
            m.clock.tick()

    trace_mean = t.get_mean()
    trace_std  = t.get_standard_deviation()
    for idx in range(num_foo):
        assert trace_mean[idx]  == approx(mean, rel=.05)
        assert trace_std[idx]   == approx(std,  rel=.10)
    m.db.check()
    t.reset()

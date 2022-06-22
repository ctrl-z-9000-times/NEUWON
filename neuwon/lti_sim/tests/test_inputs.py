from lti_sim.inputs import *
import pytest
import random

def test_linear():
    a = LinearInput('foobar', 4, 7)
    a.set_num_buckets(17)

    for v in a.sample_space(55):
        assert 4 <= v <= 7

    b, l = a.get_bucket_location(4)
    assert b == 0
    assert l == 0.0
    b, l = a.get_bucket_location(7 - 1e-12)
    assert b == 17-1
    assert l == pytest.approx(1.0)

    assert a.bisect_inputs(5, 6) == pytest.approx(5.5)

def test_logarithmic():
    a = LogarithmicInput('foobar', 0, 2e3)
    a.set_num_buckets(37, scale=1e-3)

    for v in a.sample_space(55):
        assert 0 <= v <= 2e3

    b, l = a.get_bucket_location(0)
    assert b == 0
    assert l == 0.0
    b, l = a.get_bucket_location(2e3 - 1e-12)
    assert b == 37-1
    assert l == pytest.approx(1.0)

    assert 5.25 < a.bisect_inputs(5, 6) < 5.49

def test_linear_inverse():
    a = LinearInput('foobar', 11, 57)
    a.set_num_buckets(107)

    trials = [random.uniform(a.minimum, a.maximum) for _ in range(100)]
    trials.append(a.minimum)
    trials.append(a.maximum)

    for v in trials:
        l  = a.get_bucket_value(v)
        vv = a.get_input_value(l)
        assert v == pytest.approx(vv, abs=1e-12)
        assert 0.0 <= l <= a.num_buckets
        assert a.minimum <= v <= a.maximum

def test_logarithmic_inverse():
    a = LogarithmicInput('foobar', 0, 57)
    a.set_num_buckets(107, scale=1e-3)

    trials = [random.uniform(a.minimum, a.maximum) for _ in range(100)]
    trials.append(a.minimum)
    trials.append(a.maximum)
    trials.extend(7 ** -x for x in range(100)) # Test very small values.

    for v in trials:
        l  = a.get_bucket_value(v)
        vv = a.get_input_value(l)
        assert v == pytest.approx(vv, abs=1e-12)
        assert 0.0 <= l <= a.num_buckets
        assert a.minimum <= v <= a.maximum

def test_random():
    a = LogarithmicInput('foobar', 0, 2e3)
    a.set_num_buckets(37, scale=1e-3)

    float(a.random())
    assert a.random([3,3]).shape == (3,3)
    assert a.random().dtype == np.float64

    for trial in range(100):
        assert a.minimum <= a.random() <= a.maximum
    rnd = a.random(100, dtype=np.float32)
    assert np.all(rnd >= a.minimum)
    assert np.all(rnd <= a.maximum)
    assert rnd.dtype == np.float32
    assert np.mean(rnd) < a.maximum / 4 # Check for non-uniform distribution.

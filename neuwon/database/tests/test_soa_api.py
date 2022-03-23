from neuwon.database import *
from neuwon.database.memory_spaces import *
import cupy
import numpy as np
import pytest

def test_repr():
    print(repr(host))
    print(repr(cuda))

def test_arrays():
    db = Database()
    Foo_data = db.add_class("Foo")
    bar_data = Foo_data.add_attribute("bar", 42)
    Foo = Foo_data.get_instance_type()

    for _ in range(3):
        f = Foo()

    bar_data.set_data([2,3,4])
    with db.using_memory_space('cuda'):
        data = bar_data.get_data()
        assert isinstance(data, cupy.ndarray)
        data = data.get()
        assert np.all(data == [2,3,4])
    assert bar_data.get_memory_space() is cuda

    # Test free'ing attribute buffers.
    assert not bar_data.is_free()
    bar_data.free()
    assert bar_data.is_free()
    assert f.bar == 42
    assert bar_data.is_free()
    assert np.all(bar_data.get_data() == 42)
    assert not bar_data.is_free()

    bar_data.set_data([4,3,2])
    with db.using_memory_space(host):
        assert np.all(bar_data.get_data() == [4,3,2])
    assert bar_data.get_memory_space() is host

    with db.using_memory_space('cuda'):
        Foo()

def test_create_on_device():
    db = Database()
    with db.using_memory_space('cuda'):
        Foo_data = db.add_class("Foo")
        Foo = Foo_data.get_instance_type()

    for i in range(3):
        f = Foo()
    with db.using_memory_space('cuda'):
        bar_data = Foo_data.add_attribute("bar", 42)
        sp_data  = Foo_data.add_sparse_matrix('sp', Foo)
        ptr_data = Foo_data.add_connectivity_matrix('ptr', Foo_data)
        f.ptr = [f]
        assert isinstance(sp_data.get_data(), cuda.matrix_module.spmatrix)
        f.sp = ([f,f], [3.4, 5.5])
        assert isinstance(sp_data.get_data(), cuda.matrix_module.spmatrix)
    assert f in f.ptr
    cols, vals = f.sp
    assert cols == [f, f]
    assert vals == [3.4, 5.5]

def test_initial_distribution():
    db = Database()
    Foo_data = db.add_class("Foo")
    Foo     = Foo_data.get_instance_type()
    u1_data = Foo_data.add_attribute('u1',
            initial_distribution=('uniform', 2, 5),
            valid_range = (2, 5))
    n1_data = Foo_data.add_attribute('n1',
            initial_distribution=('normal', 10, 2),
            valid_range = (6, 14),)
    c1_data = Foo_data.add_class_attribute('c1',
            initial_distribution=('uniform', -500, 500),)

    for i in range(10000):
        Foo()

    db.check()

    # Verify that free'ing the data causes it to generate new values.
    u1_orig = list(u1_data.get_data())
    u1_data.free()
    assert u1_orig != list(u1_data.get_data())
    # Verify that free'ing class attributes also causes it to generate new values.
    c1_orig = Foo.c1
    c1_data.free()
    assert c1_orig != Foo.c1

    # Verify that outliers of the normal distribution are always inside the valid_range.
    assert all((6 <= x <= 14) for x in n1_data.get_data())

    # Verify that the random distributions are correct.
    rtol = 0.05
    assert all((2 <= x <= 5) for x in u1_data.get_data())
    u1_mean = np.mean(u1_data.get_data())
    u1_std  = np.std( u1_data.get_data())
    u1_std_correct = (5 - 2) / (12 ** .5)
    assert u1_mean == pytest.approx(3.5, rtol)
    assert u1_std  == pytest.approx(u1_std_correct, rtol)

    assert -500 <= Foo.c1 <= 500

    n1_mean = np.mean(n1_data.get_data())
    n1_std  = np.std( n1_data.get_data())
    assert n1_mean == pytest.approx(10, rtol)
    assert n1_std  == pytest.approx( 2, rtol * 2)

    # Check that it raises errors when the distribution is outside of the valid_range.
    with pytest.raises(Exception):
        Foo_data.add_attribute('qq',
                initial_distribution=('uniform', 2, 5),
                valid_range = (2, 4.5),)
    with pytest.raises(Exception):
        Foo_data.add_attribute('zz',
                initial_distribution=('normal', 10, 2),
                valid_range = (-10, 11.5),)

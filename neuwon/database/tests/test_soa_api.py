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

@pytest.mark.skip
def test_matrixes():
    1/0 # TODO
    # IDEA: Make a matrix of all configurations: (host/device, data-fmt).
    # Then convert between the formats (at random or exhaustively), and check
    # that data is preserved and not crashes.


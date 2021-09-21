from neuwon.database import *
from neuwon.database.memory_spaces import *
import pytest
import cupy

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

    bar_data.set_data([4,3,2])
    with db.using_memory_space(host):
        assert np.all(bar_data.get_data() == [4,3,2])
    assert bar_data.get_memory_space() is host

    with db.using_memory_space('cuda'):
        Foo()

@pytest.mark.skip
def test_matrixes():
    1/0 # TODO
    # IDEA: Make a matrix of all configurations: (host/device, data-fmt).
    # Then convert between the formats (at random or exhaustively), and check
    # that data is preserved and not crashes.


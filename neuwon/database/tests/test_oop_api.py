from neuwon.database import *
import pytest

class Foo:
    """ Equivalent OOP class definition. """
    def __init__(self):
        self.bar = 4

def test_OOP_API():
    class Foo(DB_Object):
        def my_helper_method(self):
            self.bar

    db = Database()
    Foo = db.add_class("Foo", instance_type=Foo)
    Foo.add_attribute("bar")
    FooFactory = Foo.get_instance_type()

    x = FooFactory()
    x.bar = 4
    assert x.bar == 4

    x.my_helper_method()

    Foo.add_attribute("ptr", dtype="Foo")
    y = FooFactory()
    x.ptr = y
    x.ptr.bar = 5
    assert y.bar == 5
    Foo.add_class_attribute("shared_value", 77)
    assert x.shared_value == 77
    y.shared_value = 55
    assert x.shared_value == 55

    Foo.add_sparse_matrix("connections", "Foo").to_lil()
    print((x.connections))
    x.connections = ([y], [2.])
    print((x.connections))

    Foo.add_list_attribute("friends", dtype=Foo)
    x.friends.append(y)

    if False:
        help(Foo)
        help(x)


def test_custom_classes():

    class _Section_obj:
        def __init__(self, nsegs):
            for i in range(nsegs):
                self.segments.append(Segment(section=self))

        @property
        def nsegs(self):
            return len(self.segments)

    class _Segment_obj(DB_Object):
        __slots__ = ()
        def get_location(self):
            if self.section is None: return None
            index = self.section.segments.index(self)
            return (index + .5) / self.section.nsegs

    _db = Database()
    _Section_cls = _db.add_class("Section", _Section_obj)
    _Segment_cls = _db.add_class("Segment", _Segment_obj)
    _Section_cls.add_list_attribute("segments", dtype="Segment")
    _Segment_cls.add_attribute("section", dtype="Section", allow_invalid=True)
    Section = _Section_cls.get_instance_type()
    Segment = _Segment_cls.get_instance_type()

    my_secs = [Section(n) for n in range(20)]
    absdif = lambda a,b: abs(a-b)
    assert my_secs[7].nsegs == 7
    assert my_secs[3].segments[0].get_location() == pytest.approx(1/6)
    assert my_secs[3].segments[1].get_location() == pytest.approx(3/6)
    assert my_secs[3].segments[2].get_location() == pytest.approx(5/6)

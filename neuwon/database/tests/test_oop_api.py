from neuwon.database import *
import pytest

class Foo:
    """ Equivalent OOP class definition. """
    def __init__(self):
        self.bar = 4

def test_OOP_API():
    class FooBaseClass:
        "Single line docstring."
        __slots__ = ()
        def my_helper_method(self):
            self.bar

    db = Database()
    Foo = db.add_class("Foo", FooBaseClass)
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

    Foo.add_connectivity_matrix("friends", Foo)
    x.friends.append(y)

    print(db)
    if False:
        help(Foo)
        help(x)


def test_custom_classes():

    class _SectionBaseClass:
        """
        My
        multi
        line
        documentation
        """
        __slots__ = ()
        def __init__(self, nsegs):
            self.segments = [Segment(section=self) for _ in range(nsegs)]

        @property
        def nsegs(self):
            return len(self.segments)

    class _SegmentBaseClass:
        # no docstring.
        __slots__ = ()
        def get_location(self):
            if self.section is None: return None
            index = self.section.segments.index(self)
            return (index + .5) / self.section.nsegs

    db = Database()
    Section_cls = db.add_class("Section", _SectionBaseClass)
    Segment_cls = db.add_class("Segment", _SegmentBaseClass)
    Section_cls.add_connectivity_matrix("segments", "Segment")
    Segment_cls.add_attribute("section", dtype="Section", allow_invalid=True)
    Section = Section_cls.get_instance_type()
    Segment = Segment_cls.get_instance_type()

    assert Section.__doc__ == _SectionBaseClass.__doc__ # Check for inherit docs.
    assert Segment.get_database_class() is Segment_cls # Check for classmethod.

    my_secs = [Section(n) for n in range(20)]
    absdif = lambda a,b: abs(a-b)
    assert my_secs[7].nsegs == 7
    assert my_secs[3].segments[0].get_location() == pytest.approx(1/6)
    assert my_secs[3].segments[1].get_location() == pytest.approx(3/6)
    assert my_secs[3].segments[2].get_location() == pytest.approx(5/6)

    print(db)

def test_requires_slots():
    class NoSlots:
        pass
    db = Database()
    with pytest.raises(Exception):
        db.add_class("NoSlots", NoSlots)

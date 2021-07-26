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
    Foo = db.add_class("Foo", instance_class=Foo)
    Foo.add_attribute("bar")

    x = Foo()
    x.bar = 4
    assert x.bar == 4

    x.my_helper_method()

    Foo.add_attribute("ptr", dtype="Foo")
    y = Foo()
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

    class _Section(DB_Object):
        @staticmethod
        def _initialize_db(db):
            db.get_class("Section").add_list_attribute("segments", dtype="Segment")

        def __init__(self, index, nsegs=None):
            # How can the programmer tell if this was called to create a handle
            # to an existing object or to create a new object? SOLUTION: I
            # could just not call their init when I'm not creating a new
            # object? I could steal this function and call Instance.__init__
            # instead?
            DB_Object.__init__(self, index)
            if nsegs is None: return
            for i in range(nsegs):
                self.segments.append(Segment(section=self))

        @property
        def nsegs(self):
            return len(self.segments)

    class _Segment(DB_Object):
        @staticmethod
        def _initialize_db(db):
            db.get_class("Segment").add_attribute("section", dtype="Section", allow_invalid=True)

        def get_location(self):
            if self.section is None: return None
            index = self.section.segments.index(self)
            return (index + .5) / self.section.nsegs

    db = Database()
    Section = db.add_class("Section", _Section)
    Segment = db.add_class("Segment", _Segment)
    _Section._initialize_db(db)
    _Segment._initialize_db(db)

    my_secs = [Section(n) for n in range(20)]
    absdif = lambda a,b: abs(a-b)
    assert my_secs[3].segments[0].get_location() == pytest.approx(1/6)
    assert my_secs[3].segments[1].get_location() == pytest.approx(3/6)
    assert my_secs[3].segments[2].get_location() == pytest.approx(5/6)

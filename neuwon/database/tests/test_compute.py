from neuwon.database import Database
from neuwon.database.callable import Function
import math


def test_basic_function():
    @Function
    def foo(x):
        """ Foo's help message! """
        return x + 3

    help(foo)
    assert foo(3) == 6

    qq = Function(lambda: 42)
    assert qq() == 42


def test_basic_method():
    class Seg:
        __slots__ = ()
        @Function
        def bar(self):
            self.v -= 4

    db = Database()
    Seg_data = db.add_class("Seg", Seg)
    Seg_data.add_attribute("v", -70)
    Seg = Seg_data.get_instance_type()
    my_seg = Seg()
    my_seg.bar()
    Seg.bar()
    assert my_seg.v == -78



class Segment:
    def area(r):
        return math.pi * (r**2)

    __slots__ = ()
    @classmethod
    def initialize(cls, db):
        seg_data = db.add_class("Segment", cls)
        seg_data.add_attribute("r", 33.3)

    def _compute_area(self):
        self.area = area(self.r)


def test_compile():
    db = Database()
    Segment.initialize(db)



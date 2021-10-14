from neuwon.database import Database, Function, Method
import math
import pytest
import random


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
    assert my_seg.v == -74
    Seg.bar()
    assert my_seg.v == -78


def test_calling_methods():
    class Seg:
        __slots__ = ()
        @Function
        def foo(self):
            self.v += 4
            self.bar()
        @Function
        def bar(self):
            self.v -= 4

    db = Database()
    Seg_data = db.add_class("Seg", Seg)
    Seg_data.add_attribute("v", -70)
    Seg = Seg_data.get_instance_type()
    for _ in range(6): Seg()
    my_seg = Seg()
    my_seg.foo()
    assert my_seg.v == -70
    Seg.foo()
    assert my_seg.v == -70
    my_seg.bar()
    assert my_seg.v == -74


@Function
def area_eq(r):
    return math.pi * (r**2)

class Segment:
    __slots__ = ()
    @classmethod
    def initialize(cls, db):
        seg_data = db.add_class("Segment", cls)
        seg_data.add_attribute("r", 33.3)
        seg_data.add_attribute("area")
        return seg_data.get_instance_type()

    @Function
    def _compute_area(self):
        self.area = area_eq(self.r)

@pytest.mark.skip
def test_compile():
    db = Database()
    Seg = Segment.initialize(db)
    for _ in range(99): Seg()
    Seg._compute_area()


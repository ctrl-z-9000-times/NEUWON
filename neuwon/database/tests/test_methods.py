from neuwon.database import Database, Function, Method
import math
import pytest


def test_basic_function():
    @Function
    def foo(x):
        """ Foo's help message! """
        return x + 3

    help(foo)
    assert foo(3) == 6

    qq = Function(lambda: 42)
    assert qq() == 42


def test_functions_calling_functions():
    @Function
    def foo(x):
        return bar(x) + 3
    @Function
    def bar(x):
        return x - 3
    assert foo(4) == 4
    assert bar(4) == 1


# @pytest.mark.skip
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


# @pytest.mark.skip
def test_calling_methods():
    class Seg:
        __slots__ = ()
        @Method
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


x = 5
# @pytest.mark.skip
def test_calling_functions():
    # y = 0
    @Function
    def area_eq(r):
        return math.pi * (r**2)
    class Segment:
        __slots__ = ()
        @classmethod
        def initialize(cls, db):
            seg_data = db.add_class("Segment", cls)
            seg_data.add_attribute("d", 33.3)
            seg_data.add_attribute("area")
            return seg_data.get_instance_type()
        @Method
        def _compute_area(self):
            self.area = area_eq(self.d / 2 + x + y)

    db = Database()
    Seg = Segment.initialize(db)
    y = -5 # Test late initialize/JIT.
    for _ in range(9): Seg()
    my_seg = Seg()
    my_seg.d = 12
    for _ in range(9): Seg()
    Seg._compute_area()
    assert my_seg.area == pytest.approx(math.pi * 36)


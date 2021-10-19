from neuwon.database import Database, Function, Method, TimeSeries
import math
import random
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


def test_basic_method():
    class Seg:
        __slots__ = ()
        @Function
        def bar(self):
            self.v -= 4
        def args(self, x):
            return self.v * x

    db = Database()
    Seg_data = db.add_class("Seg", Seg)
    Seg_data.add_attribute("v", -70)
    Seg = Seg_data.get_instance_type()
    my_seg = Seg()
    my_seg.bar()
    assert my_seg.v == -74
    Seg.bar()
    assert my_seg.v == -78

    assert my_seg.args(0) == 0
    assert my_seg.args(1) == my_seg.v


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
            return self.v

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
    assert my_seg.bar() == my_seg.v


x = 5
def test_calling_functions():
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


leak_tau = 10 # Test reading Global in method-in-method closure.
def test_pointer_chains():
    db = Database()
    dt = 0.1 # Test reading Nonlocal in method-in-method closure.
    class Neuron:
        __slots__ = ()
        @Method
        def postsyn_psp(self, x):
            # Called methods must retain their I/O & Closure.
            self.v += dt * x # Integrate inputs.
        @Method
        def advance(self):
            self.v -= (self.v + 70) * math.exp(-dt/leak_tau) # Leak.
    Neuron_data = db.add_class("Neuron", Neuron)
    Neuron_data.add_attribute("v", -70, valid_range=[-100,0])
    Neuron_data.add_class_attribute("thresh", -30)
    Neuron = Neuron_data.get_instance_type()
    state_decay = math.exp(-dt / 4.0)
    class Syn:
        __slots__ = ()
        @Method
        def advance(self):
            # Test reading through pointer indirection:
            qq = int(self.pre.v >= self.pre.thresh)
            xx = self.post.v
            state += qq
            state *= state_decay
            # Test calling methods on pointers:
            self.post.postsyn_psp(state * strength)
        @Method
        def neuron_ap_reset(self):
            # Test writing data through pointer indirection:
            if self.post.v >= self.post.thresh:
                self.post.v = -100
    Syn_data = db.add_class("Syn", Syn)
    Syn_data.add_attribute("state", 0.0)
    Syn_data.add_attribute("strength", 0.01)
    Syn_data.add_attribute("pre", dtype=Neuron)
    Syn_data.add_attribute("post", dtype=Neuron)
    Syn = Syn_data.get_instance_type()
    def advance():
        Neuron.advance()
        Syn.advance()
        Syn.neuron_ap_reset()
    # Isolated synapse testcase.
    presyn = Neuron()
    postsyn = Neuron()
    syn = Syn(pre=presyn,post=postsyn)
    for _ in range(round(1/dt)): advance()
    assert postsyn.v == -70
    presyn.v = -10
    for _ in range(round(1/dt)): advance()
    assert postsyn.v > -69.99
    # Construct a neural network.
    n = [Neuron() for _ in range(1000)]
    s = []
    nsyn = 100 * len(n)
    for pre, post in zip(random.choices(n, k=nsyn), random.choices(n, k=nsyn)):
        s.append(Syn(pre=pre, post=post))
    db.check()
    # TODO: Record from several of the neurons.
    1/0
    # TODO: Setup network inputs.
    1/0

    # TODO: Run it through three periods: init-off, inputs-turn-it-on, sustained/recurrent activity.
    #       And check that it does the expected thing at each step.

    for _ in range(round(100 / dt)):
        advance()

    # TODO: Check for sane recurrent activity.
    1/0


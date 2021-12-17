from neuwon.database import Database, Compute, TimeSeries
import math
import pytest
import random
import numpy as np
import cupy


def test_basic_method():
    class Seg:
        __slots__ = ()
        @Compute
        def bar(self):
            """ Hello Seg.Bar's Docs! """
            self.v -= 4
        @Compute
        def args(self, x) -> float:
            return self.v * x

    db = Database()
    Seg_data = db.add_class("Seg", Seg)
    Seg_data.add_attribute("v", -70)
    Seg = Seg_data.get_instance_type()
    Seg.bar()
    my_seg = Seg()
    help(my_seg.bar)
    my_seg.bar()
    assert my_seg.v == -74
    Seg.bar()
    assert my_seg.v == -78

    assert my_seg.args(0) == 0
    assert my_seg.args(1) == my_seg.v


def test_calling_methods():
    class Seg:
        __slots__ = ()
        @Compute
        def foo(self, method_arg) -> float:
            self.v += 4
            self.bar()
            return method_arg
        @Compute
        def bar(self) -> float:
            self.v -= 4
            return self.v

    db = Database()
    Seg_data = db.add_class("Seg", Seg)
    Seg_data.add_attribute("v", -70)
    Seg = Seg_data.get_instance_type()
    for _ in range(6): Seg()
    my_seg = Seg()
    assert my_seg.foo(42) == 42
    assert my_seg.v == -70
    assert my_seg.v == -70
    my_seg.bar()
    assert my_seg.v == -74
    assert my_seg.bar() == my_seg.v
    Seg.foo(None, 33.33) # Instance-arg required if there are further args.
    Seg.bar() # No args so instance-arg is not required.


x = 5
def test_calling_functions():
    @Compute
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
        @Compute
        def _compute_area(self):
            self.area = area_eq(self.d / 2 + x + y)

    assert area_eq(1) == pytest.approx(math.pi)
    db = Database()
    Seg = Segment.initialize(db)
    y = -5 # Test late initialize/JIT.
    for _ in range(9): Seg()
    my_seg = Seg()
    my_seg.d = 12
    for _ in range(9): Seg()
    Seg._compute_area()
    assert my_seg.area == pytest.approx(math.pi * 36)


def test_annotations():
    class Foo:
        __slots__ = ()
        @classmethod
        def initialize(cls, db):
            foo_data = db.add_class(cls)
            foo_data.add_attribute('bar', 0)
            foo_data.add_attribute('ref', dtype=Foo, allow_invalid=True)
            return foo_data.get_instance_type()
        @Compute
        def do(self):
            # Rename the variables to test type annotations.
            my_self: 'Foo' = self
            my_ref: 'Foo' = my_self.ref
            my_ref.bar
            add(self, my_ref)
        def do2(self, other: 'Foo'):
            add(self, other)
    @Compute
    def add(q: 'Foo', qq: 'Foo'):
        q.bar += qq.bar
    db = Database()
    Foo = Foo.initialize(db)
    thing1 = Foo()
    thing2 = Foo()
    thing1.ref = thing1
    thing2.ref = thing1
    thing1.do()
    assert thing1.bar == 0
    thing1.bar = 1
    thing1.do()
    assert thing1.bar == 2
    thing1.do2(thing2)
    assert thing1.bar == 2
    thing2.do2(thing1)
    assert thing2.bar == 2
    add(thing1, thing2) # No compute acceleration, identical semantics.
    assert thing1.bar == 4


def test_return_value():
    class Foo:
        __slots__ = ()
        @Compute
        def bar(self) -> int:
            self.red_herring()
            return self.data
        @Compute
        def red_herring(self): # Return type annotation should not be required here.
            return 1234
    db = Database()
    foo_data = db.add_class(Foo)
    foo_data.add_attribute('data', 1234)
    Foo = foo_data.get_instance_type()
    for _ in range(77): Foo()
    assert Foo().bar() == 1234
    host_data = Foo.bar()
    print('Host array type:', type(host_data))
    assert all(x == 1234 for x in host_data)
    # Note, the exact return type is not important as long as its an efficient array.
    assert isinstance(host_data, np.ndarray)
    with db.using_memory_space('cuda'):
        cuda_data = Foo.bar()
    print('CUDA array type:', type(cuda_data))
    assert all(x == 1234 for x in cuda_data)
    assert isinstance(cuda_data, cupy.ndarray) # Exact type can change, as long as its a GPU array.


@pytest.mark.skip
def test_compute_init():
    class MyClass:
        __slots__ = ()
        @classmethod
        def initialize(cls, db):
            myclass_data = db.add_class(cls)
            myclass_data.add_attribute("x", float('nan'))
            return myclass_data.get_instance_type()
        @Compute
        def __init__(self):
            self.x = 3

    db = Database()
    MyClass = MyClass.initialize(db)
    assert MyClass().x == 3


def test_compute_on_memory_space():
    class Foo:
        __slots__ = ()
        @Compute
        def bar(self):
            self.x += 1

    db = Database()
    foo_data = db.add_class(Foo)
    foo_data.add_attribute('x', 0)
    Foo = foo_data.get_instance_type()
    Foo.bar() # Test calling with no instances, on host.
    with db.using_memory_space('cuda'):
        Foo.bar() # Test calling with no instances, on CUDA.
        Foo() # Test making a new instance inside in CUDA's context.
        Foo.bar() # Test "normal" Compute on CUDA.
    Foo.bar()
    assert all(x == 2 for x in db.get_data("Foo.x"))


leak_tau = 5 # Test reading Global in method-in-method closure.
def test_pointer_chains():
    """
    This testcase constructs a network of spiking neurons.
    However the code has been twisted to test various corner cases.
    """
    db = Database()
    dt = 0.1 # Test reading Nonlocal in method-in-method closure.
    db.add_clock(dt)
    class Neuron:
        __slots__ = ()
        @Compute
        def postsyn_psp(self, x):
            # Called methods must retain their I/O & Closure.
            self.v += dt * x # Integrate inputs.
        @Compute
        def advance(self):
            self.v -= (self.v + 70) * (1 - math.exp(-dt/leak_tau)) # Leak.
    Neuron_data = db.add_class("Neuron", Neuron)
    Neuron_data.add_attribute("v", -70, valid_range=[-100,0])
    Neuron_data.add_class_attribute("thresh", -30)
    Neuron = Neuron_data.get_instance_type()
    state_decay = math.exp(-dt / 3.0)
    class Syn:
        __slots__ = ()
        @Compute
        def advance(self):
            # Test reading through pointer indirection:
            xx = self.post.v # Read from the correct instance, not this one.
            qq = int(self.pre.v >= self.pre.thresh)
            xy = self.post.v # Read from the correct instance, not this one.
            self.state += qq
            self.state *= state_decay
            # Test calling methods on pointers:
            self.post.postsyn_psp(self.state * self.strength)
        @Compute
        def neuron_ap_reset(self):
            # Test writing data through pointer indirection:
            if self.post.v >= self.post.thresh:
                self.post.v = -100
    Syn_data = db.add_class("Syn", Syn)
    Syn_data.add_attribute("state", 0.0)
    Syn_data.add_attribute("strength", 0.77)
    Syn_data.add_attribute("pre", dtype=Neuron)
    Syn_data.add_attribute("post", dtype=Neuron)
    Syn = Syn_data.get_instance_type()
    def advance():
        Neuron.advance()
        Syn.advance()
        Syn.neuron_ap_reset()
        db.check()
        db.clock.tick()
    # Isolated synapse testcase.
    presyn = Neuron()
    postsyn = Neuron()
    syn = Syn(pre=presyn,post=postsyn)
    for _ in range(round(1/dt)):
        advance()
    assert postsyn.v == -70
    presyn.v = -10
    for _ in range(round(1/dt)):
        print('exp decay', presyn.v)
        advance()
    assert postsyn.v > -69.99
    # Construct a neural network.
    n = [Neuron() for _ in range(50)]
    s = []
    nsyn = 50 * len(n)
    for pre, post in zip(random.choices(n, k=nsyn), random.choices(n, k=nsyn)):
        s.append(Syn(pre=pre, post=post))
    db.check()
    # Record from several of the neurons.
    outputs = random.sample(n, 20)
    probes  = [TimeSeries().record(x, 'v') for x in outputs]
    # Run without inputs.
    for _ in range(round(3/dt)):
        advance()
    # Check that network is silent.
    for p in probes:
        for x in p.get_data():
            assert x == pytest.approx(-70)
    # Setup and start the network inputs.
    inputs  = random.sample(n, len(n) // 3)
    inputs  = [n for n in inputs if n not in outputs]
    def burst_of_inputs():
        for x in inputs:
            TimeSeries().square_wave(0, 60, period=1, duty_cycle=1).play(x, 'v')
    burst_of_inputs()
    for _ in range(round(20 / dt)):
        advance()
    # Check for received spikes.
    for p in probes:
        assert any(x > -69.99 for x in p.get_data())
    # Check for silent network with no inputs.
    probes = [TimeSeries().record(x, 'v') for x in outputs]
    for _ in range(round(10 / dt)):
        advance()
    for p in probes:
        assert all(x < -50 for x in p.get_data())
    # Increase synapse strength and check for sustained recurrent activity.
    stength_data = Syn_data.get_data('strength')
    stength_data *= 3.33
    burst_of_inputs()
    for _ in range(round(10 / dt)):
        advance()
    probes = [TimeSeries().record(x, 'v') for x in outputs]
    for _ in range(round(5 / dt)):
        advance()
    for p in probes:
        assert any(x > -60 for x in p.get_data())


from neuwon.database import Database
from neuwon.rxd.neurons.neurons import Neuron as NeuronSuperclass
import pytest

def test_neuron_type():
    db = Database()
    Neuron, Segment = NeuronSuperclass._initialize(db)
    # Type system must be oiptional: no types given.
    n1 = Neuron([0,0,0], 10)
    assert n1.neuron_type == None
    # Add neuron type later.
    n1.neuron_type = 'x'
    assert n1.neuron_type == 'x'
    # Can only add type once.
    with pytest.raises(ValueError):
        n1.neuron_type = 'q'
    # Check that types are correctly converted to unique integer ID's.
    n2 = Neuron([10,10,10], 5, neuron_type='x')
    assert n1.neuron_type_id == n2.neuron_type_id
    n3 = Neuron([10,10,10], 5, neuron_type='y')
    assert n1.neuron_type_id != n3.neuron_type_id

def test_segment_type():
    db = Database()
    Neuron, Segment = NeuronSuperclass._initialize(db)
    # Type system must be optional: no types given.
    s1  = Neuron([0,0,0], 10).root
    s10 = s1.add_segment([2,3,4], 1)
    s11 = s1.add_section([2,3,4], 1)
    s12 = Segment(s1, [2,3,4], 1)
    assert s1.segment_type == None
    assert s10.segment_type == None
    assert all(s.segment_type == None for s in s11)
    assert s12.segment_type == None
    # Add segment type later.
    s1.segment_type = 'x'
    assert s1.segment_type == 'x'
    # Can only add type once.
    with pytest.raises(ValueError):
        s1.segment_type = 'q'
    # Set segment type for root in init.
    assert Neuron([0,0,0], 10, segment_type=12345).root.segment_type == 12345
    # Check that types are correctly converted to unique integer ID's.
    s2 = s1.add_segment([10,0,0], 3, segment_type='x')
    assert s1.segment_type_id == s2.segment_type_id
    s3 = Segment(s1, [10,10,10], 5, segment_type='y')
    assert s1.segment_type_id != s3.segment_type_id

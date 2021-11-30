from neuwon.database import Database
from neuwon.neuron.neuron import Neuron as NeuronSuperclass
import pytest

def test_neuron_type():
    db = Database()
    Neuron = NeuronSuperclass._initialize(db)
    # Type system must be oiptional: no types given.
    n1 = Neuron([0,0,0], 10)
    assert n1.neuron_type == None
    # Add neuron type later.
    n1.neuron_type = 'x'
    assert n1.neuron_type == 'x'
    # Can only add type once.
    with pytest.raises(Exception):
        n1.neuron_type = 'q'
    # Check that types are correctly converted to unique integer ID's.
    n2 = Neuron([10,10,10], 5, neuron_type='x')
    assert n1.neuron_type_id == n2.neuron_type_id
    n3 = Neuron([10,10,10], 5, neuron_type='y')
    assert n1.neuron_type_id != n3.neuron_type_id

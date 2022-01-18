from neuwon.rxd import RxD_Model
from neuwon.synapse.constraints import Constraints

def test_basic():
    rxd = RxD_Model(.1)
    Neuron = rxd.get_Neuron()
    n1 = Neuron([0,0,0], 1, neuron_type='nt1', segment_type='soma')
    n2 = Neuron([10,0,0], 1, neuron_type='nt2', segment_type='soma')
    n3 = Neuron([10,10,0], 1, neuron_type='nt3', segment_type='soma')

    n1.root.add_section([0,20,0], .1, 2, segment_type='st1')
    n3.root.add_section([10,-10,0], .1, 2, segment_type='st3')

    c = Constraints(rxd.database,
            presynapse_neuron_types = ['nt1'],
            postsynapse_segment_types = ['st3'],
            maximum_distance = 2,)

    candidates = c.find_all_candidates()

    print(candidates)
    1/0



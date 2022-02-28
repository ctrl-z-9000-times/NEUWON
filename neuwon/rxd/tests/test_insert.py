from neuwon.rxd import RxD_Model
import pytest

def test_basic_insertion():
    my_model = RxD_Model(
        mechanisms= {
            'hh': './nmodl_library/hh.mod',
        },)
    my_neuron   = my_model.Neuron([0,0,0], 10)
    my_root     = my_neuron.root
    my_segments = my_root.add_section([100,0,0], 3, maximum_segment_length=10)

    my_root.insert({})
    instances = my_root.insert({'hh': 1})

    assert instances['hh'].get_database_class().get_name() == 'hh'
    my_model.check()

@pytest.mark.skip
def test_other_mechanisms():
    # TODO: Test inserting mechanisms that reference each other.
    1/0

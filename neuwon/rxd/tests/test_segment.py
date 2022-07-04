from neuwon.database import Database, TimeSeries
from neuwon.rxd import RxD_Model
from neuwon.rxd.neuron.neuron import Neuron as NeuronSuperclass
import pytest

def test_sections():
    dt = .1
    db = Database()
    Neuron = NeuronSuperclass._initialize(db,
            initial_voltage = -70,
            cytoplasmic_resistance = 1e6,
            membrane_capacitance = 1e-14,)
    root = Neuron([0, 0,0], 4).root
    sec1 = root.add_section([10,0,0], 1, 3)
    sec2 = root.add_section([10,10,0], 1, 3)

    for sec in [sec1, sec2]:
        for x in sec:
            print(x.coordinates)

    assert root.is_root()
    assert len(sec1) == 3 # math.ceil((10 - 4/2) / 3)
    assert len(sec2) == 5 # math.ceil((math.sqrt(10**2 + 10*2) - 2) / 3)
    assert not sec2[-1].is_root()
    db.check()

@pytest.mark.skip()
def test_swc():
    db = Database()
    Geometry._initialize(db)
    Seg = db.get("Segment").get_instance_type()
    # http://neuromorpho.org/neuron_info.jsp?neuron_name=109-6_5_6_L1_CA1_N2_CG
    my_neuron = Seg.load_swc("swc_files/109-6_5_6_L1_CA1_N2_CG.CNG.swc")

    segs    = [x for x in Seg.get_all_instances() if x.is_cylinder()]
    length  = sum(x.length for x in segs)
    surface = sum(x.surface_area for x in segs)
    volume  = sum(x.volume for x in segs)

    pct_diff = lambda a,b: abs(a-b) * 2 / (a + b)

    print("length", length)
    print("surface", surface)
    print("volume", volume)

    assert pct_diff(length,  1315.5) <= .05
    assert pct_diff(surface, 6621.44) <= .05
    assert pct_diff(volume,  3500.58) <= .05
    db.check()

def test_insert_smoke_test():
    my_model = RxD_Model(
            mechanisms= ['./nmodl_library/artificial/hh.mod',])
    my_neuron   = my_model.Neuron([0,0,0], 10)
    my_root     = my_neuron.root
    my_segments = my_root.add_section([100,0,0], 3, maximum_segment_length=10)

    my_root.insert({})
    instances = my_root.insert({'hh': 1}) # Insert mechanism with associated magnitude.
    my_segments[2].insert('hh') # Insert by mechanism name.
    my_segments[3].insert(['hh']) # Insert by list of names.
    # Duplicate insert should not raise an error.
    my_segments[4].insert({'hh': 1})
    my_segments[4].insert({'hh': 1})

    assert instances['hh'].get_database_class().get_name() == 'hh'
    my_model.check()

def test_insert_interacting_mechanisms():
    my_model = RxD_Model(
        time_step = 0.1,
        temperature = 6.3,
        mechanisms = [
                './nmodl_library/artificial/hh.mod',
                './neuwon/rxd/tests/local.mod',
        ],
        species = [
                {'name': 'na', 'reversal_potential': +60,},
                {'name': 'k',  'reversal_potential': -88,},
                {'name': 'l',  'reversal_potential': -54.3,},
        ],)
    my_neuron   = my_model.Neuron([0,0,0], 10)
    my_root     = my_neuron.root
    instances   = my_root.insert({'hh': 1, 'local': 1})
    probe       = TimeSeries().record(my_root, 'voltage', discard_after=30)

    print(my_model.mechanisms['hh']._advance_pycode)
    print(my_model.mechanisms['local']._advance_pycode)

    # Check for no spontaneous AP at start up.
    while my_model.clock() < 20:
        my_model.advance()
    assert all(v < -20 for v in probe.get_data())

    # Check that AP works.
    my_model.clock.reset()
    my_root.inject_current(.1, .5)
    while my_model.clock() < 20:
        my_model.advance()
    assert any(v > 20 for v in probe.get_data())

    # Wait for "local" mechanism to cause hh magitude to decay.
    while my_model.clock() < 500:
        my_model.advance()

    # Check that AP is gone.
    my_model.clock.reset()
    my_root.inject_current(.1, .5)
    while my_model.clock() < 20:
        my_model.advance()
    assert all(v < -20 for v in probe.get_data())

    my_model.check()

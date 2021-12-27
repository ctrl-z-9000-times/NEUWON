from neuwon.database import Database
from neuwon.rxd.neurons.neurons import Neuron as NeuronSuperclass
import pytest

def test_sections():
    dt = .1
    db = Database()
    Neuron, Segment = NeuronSuperclass._initialize(db,
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

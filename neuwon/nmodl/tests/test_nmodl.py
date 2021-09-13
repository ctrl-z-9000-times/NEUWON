from neuwon.segment import SegmentMethods
from neuwon.database import Database
from neuwon.nmodl import NmodlMechanism
import pytest

def test_hh_smoke_test():
    dt = .1
    db = Database()
    hh = NmodlMechanism("./nmodl_library/hh.mod", use_cache=False)
    Segment = SegmentMethods._initialize(db,
                initial_voltage = -70,
                cytoplasmic_resistance = 1e6,
                membrane_capacitance = 1e-14,)
    db_cls = Segment.get_database_class()
    db_cls.add_attribute("na_conductance")
    db_cls.add_attribute("k_conductance")
    db_cls.add_attribute("l_conductance")

    hh = hh.initialize(db, time_step=dt, celsius=6.3)

    my_seg = Segment(None, [0,0,0], 12)
    my_hh  = hh(my_seg, scale=.2)
    hh.advance()
    hh.advance()
    hh(Segment(None, [40,0,0], 12), scale=.2)
    for _ in range(40):
        hh.advance()
    db.check()

@pytest.mark.skip()
def test_kinetic_model():
    nav11 = NmodlMechanism("./nmodl_library/Balbi2017/Nav11_a.mod", use_cache=False)

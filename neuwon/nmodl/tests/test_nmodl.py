from neuwon.segment import SegmentMethods
from neuwon.database import Database
from neuwon.nmodl import NmodlMechanism
import pytest

def test_hh():
    hh = NmodlMechanism("./nmodl_library/hh.mod", use_cache=False)
    db = Database()
    Segment = SegmentMethods._initialize(db)
    db_cls = Segment.get_database_class()
    db_cls.add_attribute("conductances_na")
    db_cls.add_attribute("conductances_k")
    db_cls.add_attribute("conductances_l")

    hh = hh.initialize(db, time_step=.1, celsius=6.3)

    my_seg = Segment(None, [0,0,0], 12)
    my_hh  = hh(my_seg, scale=.2)

    hh.advance()

@pytest.mark.skip()
def test_kinetic_model():
    nav11 = NmodlMechanism("./nmodl_library/Balbi2017/Nav11_a.mod", use_cache=False)

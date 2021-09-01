from neuwon.segment import SegmentMethods
from neuwon.database import Database
from neuwon.nmodl import NmodlMechanism
import pytest


def mk_db():
    db = Database()
    SegmentMethods._initialize(db)
    return db

@pytest.mark.skip()
def test_hh():
    hh = NmodlMechanism("./nmodl_library/hh.mod", use_cache=False)
    db = mk_db()
    hh.initialize(db, time_step=.1, celsius=6.3)

@pytest.mark.skip()
def test_kinetic_model():
    nav11 = NmodlMechanism("./nmodl_library/Balbi2017/Nav11_a.mod", use_cache=False)

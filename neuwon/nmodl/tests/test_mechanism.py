from neuwon.nmodl import NmodlMechanism
from neuwon.database import Database

def test_mechanism():
    hh = NmodlMechanism("./nmodl_library/hh.mod", use_cache=False)
    db = Database()
    hh.initialize(db)

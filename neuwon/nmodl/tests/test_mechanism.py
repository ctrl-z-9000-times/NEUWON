from neuwon.database import Database
from neuwon.nmodl import NmodlMechanism
import pytest

def test_hh():
    hh = NmodlMechanism("./nmodl_library/hh.mod", use_cache=False)

@pytest.mark.skip()
def test_kinetic_model():
    nav11 = NmodlMechanism("./nmodl_library/Balbi2017/Nav11_a.mod", use_cache=False)

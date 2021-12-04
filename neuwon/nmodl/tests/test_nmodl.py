from neuwon.model import Model
from neuwon.database import Database
from neuwon.nmodl import NMODL
import pytest

def test_hh_smoke_test():
    m = Model({'time_step': .1,},
        mechanisms={
            'hh': NMODL("./nmodl_library/hh.mod", use_cache=False)
    })
    m.database.check()

    hh = m.mechanisms['hh']
    hh.advance()

    my_seg = m.Neuron([0,0,0], 12).root
    my_hh  = hh(my_seg, scale=.2)
    hh.advance()
    hh(m.Neuron([40,0,0], 12), scale=.2)
    for _ in range(40):
        hh.advance()
    m.database.check()

@pytest.mark.skip()
def test_kinetic_model():
    nav11 = NMODL("./nmodl_library/Balbi2017/Nav11_a.mod", use_cache=False)

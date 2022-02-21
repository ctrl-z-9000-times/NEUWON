from neuwon.rxd.rxd_model import RxD_Model
from neuwon.rxd.nmodl import NMODL
import pytest

def test_hh_smoke_test():
    m = RxD_Model(time_step = .1,
        mechanisms={
            'hh': NMODL("./nmodl_library/hh.mod", use_cache=False)},
        species={
            'na': {'reversal_potential': +60,},
            'k': {'reversal_potential': -88,},
            'l': {'reversal_potential': -54.3,},
        },
    )
    m.database.check()

    hh = m.mechanisms['hh']
    help(hh)
    print('PARAMETERS:', hh._parameters)
    print('ADVANCE PYCODE:')
    print(hh._advance_pycode)

    hh.advance()

    my_seg = m.Neuron([0,0,0], 12).root
    my_hh  = hh(my_seg, magnitude=.24)
    assert my_seg.l_conductance == 0
    hh.advance()
    assert my_seg.l_conductance > 0
    hh(m.Neuron([40,0,0], 12).root, 0.42)
    for _ in range(40):
        hh.advance()
    m.database.check()
    assert my_seg.na_conductance > 0


@pytest.mark.skip()
def test_ampa5_smoke_test():
    m = RxD_Model(time_step = .1,
        mechanisms={
            'ampa': NMODL("./nmodl_library/Destexhe1994/ampa5.mod", use_cache=False)},
        species={
            'glu': {'outside': {'diffusivity': 0.005},},
        },
    )
    m.database.check()

    ampa = m.mechanisms['ampa']
    help(ampa)
    print('PARAMETERS:', ampa._parameters)
    print('ADVANCE PYCODE:')
    print(ampa._advance_pycode)

    ampa.advance()

    my_seg = m.Neuron([0,0,0], 12).root
    my_ampa = ampa(my_seg, .2)
    # assert my_seg.l_conductance == 0
    # ampa.advance()
    # assert my_seg.l_conductance > 0
    # ampa(m.Neuron([40,0,0], 12).root, scale=.2)
    # for _ in range(40):
    #     ampa.advance()
    # m.database.check()
    # assert my_seg.na_conductance > 0


@pytest.mark.skip()
def test_kinetic_model():
    nav11 = NMODL("./nmodl_library/Balbi2017/Nav11_a.mod", use_cache=False)

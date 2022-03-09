from neuwon.rxd.rxd_model import RxD_Model
from neuwon.rxd.nmodl import NMODL
import pytest


def test_hh():
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
    print('ADVANCE PYCODE:\n' + hh._advance_pycode)

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


def test_ampa5():
    m = RxD_Model(time_step = .1,
        mechanisms={
            'ampa': NMODL("./nmodl_library/Destexhe1994/ampa5.mod", use_cache=False),
        },
        species={
            'glu': {
                'outside': {
                    'initial_concentration': 0,
                    'decay_period': 5,
            }},
            'na': {'reversal_potential': +60,},
            'k': {'reversal_potential': -88,},
        },
    )
    ampa = m.mechanisms['ampa']
    help(ampa)
    print('ADVANCE PYCODE:\n' + ampa._advance_pycode)
    ampa.advance()
    ECS = m.database.get_class("Extracellular").get_instance_type()

    my_seg = m.Neuron([0,0,0], 12).root
    my_ecs = ECS([0,0,0], .001)
    my_ampa = ampa(my_seg, .12345, outside=my_ecs)
    m.check()

    # Check silent conditions.
    while m.clock() < 2:
        m.advance()
    assert my_ampa.O < 0.000001
    assert my_seg.voltage < -50

    # Check active conditions.
    my_ecs.glu += 1
    m.clock.reset()
    while m.clock() < 2:
        m.advance()

    assert my_ampa.O > .01
    assert my_seg.voltage > -50
    m.check()


@pytest.mark.skip()
def test_kinetic_model():
    nav11 = NMODL("./nmodl_library/Balbi2017/Nav11_a.mod", use_cache=False)


from neuwon.rxd.rxd_model import RxD_Model
from neuwon.rxd.nmodl import NMODL
import os.path
import pytest

dirname = os.path.dirname(__file__)


def test_hh():
    m = RxD_Model(time_step = .1,
        mechanisms = [NMODL(dirname + "/mod/hh.mod", use_cache=False)],
        species = [
            {'name': 'na', 'reversal_potential': +60,},
            {'name': 'k',  'reversal_potential': -88,},
            {'name': 'l',  'reversal_potential': -54.3,},
        ],
    )
    m.check()

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
    m.check()
    assert my_seg.na_conductance > 0


def test_sparse_solver():
    m = RxD_Model(
        time_step = 0.1,
        mechanisms = [
            NMODL(dirname + "/mod/Nav11.mod", use_cache=False),
            NMODL(dirname + "/mod/AMPA.mod",  use_cache=False),
        ],
        species = [
            {'name': 'na', 'reversal_potential': +60,},
            {'name': 'k',  'reversal_potential': -88,},
        ],
    )
    m.check()

    Nav11 = m.mechanisms['na11a']
    help(Nav11)
    print('ADVANCE PYCODE Nav11:\n' + Nav11._advance_pycode)

    AMPA  = m.mechanisms['AMPA13']
    help(AMPA)
    print('ADVANCE PYCODE AMPA:\n' + AMPA._advance_pycode)

    my_seg = m.Neuron([0,0,0], 12).root
    Nav11(my_seg)
    AMPA(my_seg)

    m.advance()
    m.check()


# TODO: Merge this testcase into the sparse test case?
@pytest.mark.skip()
def test_ampa():
    m = RxD_Model(time_step = .1,
        mechanisms={
            'ampa': NMODL(dirname + "/mod/AMPA.mod", use_cache=False),
        },
        species={
            'glu': {
                'decay_period': 5,
                'outside_global_constant': False,
            },
            'zero': {'reversal_potential': +60,},
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
    assert my_ampa.O1 < 0.000001
    assert my_seg.voltage < -50

    # Check active conditions.
    my_ecs.glu += 1
    m.clock.reset()
    while m.clock() < 2:
        m.advance()

    assert my_ampa.O1 > .01
    assert my_seg.voltage > -50
    m.check()


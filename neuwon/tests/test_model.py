from neuwon.model import Model
from neuwon.nmodl import NmodlMechanism
from neuwon.species import Species
from neuwon.database.time import TimeSeriesBuffer
import pytest

def test_smoke_test():
    m = Model(.1)
    m.advance()
    m.Segment(None, [0,0,0], 13)
    m.advance()

def test_model_hh():
    m = Model(.01,
            cytoplasmic_resistance = 1e6,
            membrane_capacitance = 1e-14,)
    na = m.add_species(Species("na", reversal_potential=60))
    k  = m.add_species(Species("k", reversal_potential=-80))
    l  = m.add_species(Species("l", reversal_potential=-50))
    # help(m.Segment)
    hh_cls = NmodlMechanism("./nmodl_library/hh.mod", use_cache=False)
    hh = m.add_reaction(hh_cls)
    print(hh_cls._breakpoint_pycode)
    # help(hh)
    root = tip = m.Segment(None, [-1,0,7], 13)
    hh(root, 1e-9)
    for x in range(40):
        tip = m.Segment(tip, [x,0,7], 1)
        hh(tip, 1e-9)

    x = TimeSeriesBuffer().record(tip, 'voltage')

    m.advance()
    m.check()

    while m.clock() < 10:
        m.advance()
        m.check()
    root.inject_current(1e-9, 2)
    while m.clock() < 50:
        m.advance()
        m.check()

    x.plot()

if __name__ == "__main__": test_model_hh()

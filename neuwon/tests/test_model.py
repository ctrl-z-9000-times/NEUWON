from neuwon.model import Model
from neuwon.nmodl import NmodlMechanism
from neuwon.species import Species
from neuwon.database.time import TimeSeriesBuffer
import pytest

def test_model_basic():
    m = Model(.1)
    m.advance()
    m.Segment(None, [0,0,0], 13)
    m.advance()

def test_model_hh():
    m = Model(.1,
            cytoplasmic_resistance = 1e6,
            membrane_capacitance = 1e-14,)
    na = m.add_species(Species("na", reversal_potential=80))
    k  = m.add_species(Species("k", reversal_potential=-80))
    l  = m.add_species(Species("l", reversal_potential=-40))
    # help(m.Segment)
    hh = m.add_reaction(NmodlMechanism("./nmodl_library/hh.mod", use_cache=False))
    # help(hh)
    root = tip = m.Segment(None, [-1,0,7], 13)
    hh(root, .01)
    for x in range(40):
        tip = m.Segment(tip, [x,0,7], 13)
        hh(tip, .01)

    m.advance()
    m.check()

    x = TimeSeriesBuffer()
    x.record(tip, 'voltage')

    while m.clock() < 10: m.advance()
    TimeSeriesBuffer().square_wave(0, 1e-9, 4).play(root, "voltage")
    while m.clock() < 50: m.advance()

    m.check()
    x.plot()

# test_model_hh()

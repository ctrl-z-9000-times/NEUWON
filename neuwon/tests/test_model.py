from neuwon.model import Model
from neuwon.nmodl import NmodlMechanism
from neuwon.species import Species
import pytest

def test_model_basic():
    m = Model(.1)
    m.advance()
    m.Segment(None, [0,0,0], 13)
    m.advance()

def test_model_hh():
    m = Model(.1)
    tip = m.Segment(None, [-1,0,7], 13)
    for x in range(40):
        tip = m.Segment(tip, [x,0,7], 13)
    na = m.add_species(Species("na", reversal_potential=80))
    k  = m.add_species(Species("k", reversal_potential=-80))
    l  = m.add_species(Species("l", reversal_potential=-60))
    # help(m.Segment)
    hh = m.add_reaction(NmodlMechanism("./nmodl_library/hh.mod", use_cache=False))

    for _ in range(20):
        m.advance()



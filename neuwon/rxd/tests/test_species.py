from neuwon.database import Database
from neuwon.rxd.neurons import Neuron as NeuronSuperclass
from neuwon.rxd.species import SpeciesInstance, SpeciesFactory
import pytest
import math

def test_instances():
    db = Database()
    dt = .1
    Loc = db.add_class('Location')
    Loc.add_sparse_matrix('x', Loc, doc='xarea/dist')

    const = SpeciesInstance(dt, Loc, 'const', 1.11, constant=True)
    na  = SpeciesInstance(dt, Loc, 'na', 3)
    glu = SpeciesInstance(dt, Loc, 'glu', 1, decay_period=3)
    ca  = SpeciesInstance(dt, Loc, 'ca', 2, diffusivity=6, geometry_component='x')
    all_species = [const,na,glu,ca,]

    Loc = Loc.get_instance_type()
    l1  = Loc()
    l2  = Loc()
    l3  = Loc()

    help(Loc)

    l1.ca += 1
    # Connect these three in a line, the exact values don't matter.
    l1.x = ([l2],       [.1])
    l2.x = ([l1,l3],    [.2, .05])
    l3.x = ([l2],       [.04])

    l3.na_delta += 1

    assert l2.const == 1.11
    assert l2.na == 3
    assert l2.ca == 2

    assert hasattr(l1, 'na_delta')
    assert not hasattr(l1, 'const_delta')

    for s in all_species:
        print('SPECIES:', s.name)
        s._zero_accumulators()
        s._advance()
    db.check()

    assert l3.const == 1.11
    assert l2.na == 3
    assert l3.na == pytest.approx(3.5)
    assert l1.glu == pytest.approx(math.exp(-dt / 3))

    assert l1.ca < 3
    assert l2.ca > 2
    assert l3.ca > 2


test_parameters = {
    'const_e': {
        'reversal_potential': -60,
    },
    # TODO: Test all of the other configurations for computing reversal potential:
    #           Constant species values,
    #           Diffusive species values,
    #           Nerst, GHK
}

def test_species_containers():
    db = Database()
    Neuron = NeuronSuperclass._initialize(db)

    all_s = SpeciesFactory(test_parameters, db, .1, 37)
    db.check()
    for name, s in all_s.items():
        assert s.get_name() in repr(s)
    all_s._zero_accumulators()
    all_s.input_hook()
    all_s._advance()
    Neuron([1,2,3], 7)
    Neuron([4,5,6], 7)
    Neuron([7,8,9], 7)
    all_s._zero_accumulators()
    all_s.input_hook()
    all_s._advance()
    db.check()

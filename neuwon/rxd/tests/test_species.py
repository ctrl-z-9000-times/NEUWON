from neuwon.database import Database, Clock
from neuwon.rxd.neuron import Neuron as NeuronSuperclass
from neuwon.rxd.species import _SpeciesInstance
from neuwon.rxd.rxd_model import RxD_Model
import pytest
import math


def test_instances():
    db = Database()
    dt = .1
    Loc = db.add_class('Location')

    const = _SpeciesInstance(dt, Loc, 'const',
                initial_concentration   = 1.11,
                global_constant         = True,
                decay_period            = math.inf,
                diffusivity             = 0,
                geometry_component      = None,)
    na  = _SpeciesInstance(dt, Loc, 'na',
                initial_concentration   = 3,
                global_constant         = False,
                decay_period            = math.inf,
                diffusivity             = 0,
                geometry_component      = None,)
    glu = _SpeciesInstance(dt, Loc, 'glu',
                initial_concentration   = 1,
                global_constant         = False,
                decay_period            = 3,
                diffusivity             = 0,
                geometry_component      = None,)
    all_species = [const,na,glu,]

    Loc = Loc.get_instance_type()
    l1  = Loc()
    l2  = Loc()
    l3  = Loc()

    help(Loc)

    for s in all_species:
        s._zero_accumulators()

    l3.na_derivative += .5 / dt

    assert l2.const == 1.11
    assert l2.na == 3

    assert hasattr(l1, 'na_derivative')
    assert not hasattr(l1, 'const_derivative')

    for s in all_species:
        print('SPECIES:', s.name)
        s._advance()
    db.check()

    assert l3.const == 1.11
    assert l2.na == 3
    assert l3.na == pytest.approx(3.5)
    assert l1.glu == pytest.approx(math.exp(-dt / 3))


@pytest.mark.skip()
def test_diffusion_simple():
    db = Database()
    dt = .1
    Loc = db.add_class('Location')
    Loc.add_sparse_matrix('x', Loc, doc='xarea/dist')

    species = _SpeciesInstance(dt, Loc, 'species', 2, diffusivity=6, geometry_component='x')

    Loc = Loc.get_instance_type()
    l1  = Loc()
    l2  = Loc()
    l3  = Loc()
    # Connect them in a line, the exact values don't matter.
    l1.x = ([l2],       [.1])
    l2.x = ([l1,l3],    [.2, .05])
    l3.x = ([l2],       [.04])

    l1.species += 1
    assert l2.species == 2

    species._zero_accumulators()
    species._advance()

    assert l1.species < 3
    assert l2.species > 2
    assert l3.species > 2


def test_species_containers():

    test_parameters = [
        {
            'name': 'const_e',
            'reversal_potential': -60,
        },
        # TODO: Test all of the other configurations for computing reversal potential:
        #           Constant species values,
        #           Diffusive species values,
        #           Nerst, GHK
    ]
    m = RxD_Model(species=test_parameters)
    db = m.get_database()
    Neuron = m.Neuron

    m.check()
    for name, s in m.species.items():
        assert s.name in repr(s)
    # Check that it doesn't crash with no data.
    for s in m.species.values():
        s._zero_accumulators()
        s._advance()
    m.species.input_hook.tick()
    m.check()
    # Check that it doesn't crash with data.
    Neuron([1,2,3], 7)
    Neuron([4,5,6], 7)
    Neuron([7,8,9], 7)
    for s in m.species.values():
        s._zero_accumulators()
        s._advance()
    m.species.input_hook.tick()
    m.check()


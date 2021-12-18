from neuwon.database import Database
from neuwon.parameters import Parameters
from neuwon.neuron.neuron import Neuron as NeuronSuperclass
from neuwon.species import SpeciesFactory

test_parameters = Parameters({
    'only_e': {
        'reversal_potential': -60,
    },
    # 'global_const_species': {
    #     'inside': {
    #         'concentration': 11,
    #         'global_constant': True,
    #     },
    # },
    # 'non_diffusive_species': {
    #     'inside': {
    #         'concentration': 22.2,
    #         'diffusivity': 0.0,
    #     },
    # },
    # 'full_species': {
    #     'inside': {
    #         'concentration': 33,
    #         'diffusivity': 0.01,
    #         'decay_period': 77.7,
    #     },
    # },
})

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

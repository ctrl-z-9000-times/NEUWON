from neuwon.database import Database
from neuwon.database.time import Clock
from neuwon.segment import SegmentMethods
from neuwon.species import *

test_parameters = {
    'only_e': {
        'reversal_potential': -60,
    },
    'global_const_species': {
        'inside': {
            'concentration': 11,
            'global_constant': True,
        },
    },
    'non_diffusive_species': {
        'inside': {
            'concentration': 22.2,
            'diffusivity': 0.0,
        },
    },
    'full_species': {
        'inside': {
            'concentration': 33,
            'diffusivity': 0.01,
            'decay_period': 77.7,
        },
    },
}

def test_init():
    db = Database()
    SegmentMethods._initialize(db,
            initial_voltage = -70,
            cytoplasmic_resistance = 100,
            membrane_capacitance = 1,)

    s = Species("leak", reversal_potential = -60)
    s._initialize(db, .5, 8, inclock)
    inclock.tick()
    s._advance(db)

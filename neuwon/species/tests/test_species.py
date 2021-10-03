from neuwon.database import Database
from neuwon.database.time import Clock
from neuwon.segment import SegmentMethods
from neuwon.species import *

def test_init():
    db = Database()
    SegmentMethods._initialize(db,
            initial_voltage = -70,
            cytoplasmic_resistance = 1e6,
            membrane_capacitance = 1e-14,)

    s = Species("leak", reversal_potential = -60)
    inclock = Clock(.5)
    s._initialize(db, .5, 8, inclock)
    inclock.tick()
    s._advance(db)

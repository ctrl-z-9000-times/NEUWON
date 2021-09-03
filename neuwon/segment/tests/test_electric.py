from neuwon.database import Database
from neuwon.segment import SegmentMethods
import numpy as np
import pytest

dt = .1

def test_advance():
    db = Database()
    Segment = SegmentMethods._initialize(db)
    Segment._electric_advance(dt)
    Segment._electric_advance(dt)
    Segment._electric_advance(dt)
    for i in range(17):
        root = Segment(None, [0,0,0], 13)
    Segment._electric_advance(dt)
    Segment._electric_advance(dt)
    Segment._electric_advance(dt)

def test_time_constant():
    db = Database()
    Segment = SegmentMethods._initialize(db)
    root = Segment(None, [0,0,0], 20)
    root.voltage = 0
    root.driving_voltage = 1
    root.sum_conductance = 1e-10
    tau = root.capacitance / root.sum_conductance
    nstep = 1000
    dt = tau / nstep
    for _ in range(nstep):
        Segment._electric_advance(dt)
    assert root.voltage == pytest.approx(1 - (1 / np.e))

@pytest.mark.skip
def test_length_constant():
    db = Database()
    Segment = SegmentMethods._initialize(db,
            cytoplasmic_resistance = 1e-12,
            membrane_capacitance = 1e-14)
    root = Segment(None, [0,0,0], 3)
    section = [root]
    tip = root
    for x in np.linspace(1.5, 100, 50):
        tip = Segment(tip, [x, 0, 0], 2)
        section.append(tip)
    for seg in section:
        seg.driving_voltage = -70
        seg.sum_conductance = 1e-15 * seg.surface_area
    root.driving_voltage = 0
    root.sum_conductance = 1e-14 * root.surface_area

    for i in range(10):
        Segment._electric_advance(dt)

    v = [seg.voltage for seg in section]
    x = [seg.coordinates[0] for seg in section]
    import matplotlib.pyplot as plt
    plt.plot(x, v)
    plt.show()
test_length_constant()

@pytest.mark.skip("Can't do without the model.")
def test_inject_current():
    db = Database()
    Segment = SegmentMethods._initialize(db)
    root = Segment(None, [0,0,0], 13)
    root.inject_current(1e-9)
    for i in range(73):
        Segment._electric_advance(dt)

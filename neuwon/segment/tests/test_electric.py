from neuwon.database import Database
from neuwon.segment import SegmentMethods
import bisect
import numpy as np
import pytest

def test_advance_smoke_test():
    dt = .1
    db = Database()
    Segment = SegmentMethods._initialize(db,
            initial_voltage = -70,
            cytoplasmic_resistance = 1e6,
            membrane_capacitance = 1e-14,)
    Segment._electric_advance(dt)
    Segment(None, [0,0,0], 13)
    Segment._electric_advance(dt)
    # Make disconnected segments.
    for i in range(17):
        root = Segment(None, [0,0,0], 13)
    # Make connected segments.
    tip = root
    for i in range(1, 111):
        tip = Segment(tip, [i,0,0], 1)
    Segment._electric_advance(dt)
    # db.check() # Don't check because some data is still uninitialized.

def test_time_constant():
    db = Database()
    Segment = SegmentMethods._initialize(db,
            initial_voltage = -70,
            cytoplasmic_resistance = 1e6,
            membrane_capacitance = 1e-14,)
    root = Segment(None, [0,0,0], 20)
    root.voltage = 0
    root.driving_voltage = 1
    root.sum_conductance = 1e-10
    tau = root.capacitance / root.sum_conductance
    nstep = 1000
    dt = tau / nstep
    dt = dt * 1e3 # Convert to milliseconds.
    for _ in range(nstep):
        Segment._electric_advance(dt)
    assert root.voltage == pytest.approx(1 - (1 / np.e))
    db.check()

@pytest.mark.skip
def test_length_constant():
    diam = 1
    rm = 1e9
    ri = 1e6
    db = Database()
    Segment = SegmentMethods._initialize(db,
            initial_voltage = -70,
            cytoplasmic_resistance = 1e6,
            membrane_capacitance = 1e-14,)
    root = Segment(None, [0,0,0], diam)
    section = [root]
    tip = root
    for x in np.linspace(1, 100, 300):
        tip = Segment(tip, [x, 0, 0], diam)
        section.append(tip)
    x_coords = [seg.coordinates[0] for seg in section]
    for seg in section:
        seg.voltage             = 0
        seg.driving_voltage     = 0
        seg.sum_conductance     = seg.surface_area / rm
        seg.capacitance         = 1e-9
        seg.axial_resistance    = ri * seg.length
    probe   = section[len(section) // 2]
    probe_x = probe.coordinates[0]

    length_constant = ((rm / np.pi / diam) / ri) ** .5
    print("length_constant:", length_constant, 'um')

    dt = .01
    for i in range(round(10/dt)):
        probe.voltage = 1
        Segment._electric_advance(dt)
    db.check()
    irm = db.get_data("Segment.electric_propagator_matrix")
    assert np.sum(irm.data) / irm.shape[0] == pytest.approx(1), "Check that charge is conserved."

    import matplotlib.pyplot as plt
    plt.plot(x_coords, [seg.voltage for seg in section])
    plt.show()

    test = section[bisect.bisect_left(x_coords, probe_x + length_constant)]
    assert (test.coordinates[0] - probe_x) == pytest.approx(length_constant, .05)
    ratio = test.voltage / probe.voltage
    print("lambda ratio:", ratio)
    assert ratio == pytest.approx(1 / np.e, .25)

# test_length_constant()

def test_inject_current():
    from neuwon.model import Model
    m = Model(.01)
    root = m.Segment(None, [0,0,0], 10)

    init_v = root.voltage
    while m.clock() < 3:
        m.advance()
    assert init_v == pytest.approx(root.voltage)

    init_Q = 1e-3 * root.voltage * root.capacitance
    root.inject_current(1e-9, 2)
    delta_Q = 1e-9 * 2e-3
    while m.clock() < 6:
        m.advance()
    m.check()
    new_Q = 1e-3 * root.voltage * root.capacitance
    assert init_Q + delta_Q == pytest.approx(new_Q)

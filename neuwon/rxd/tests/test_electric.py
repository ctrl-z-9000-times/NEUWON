from neuwon.database import Database
from neuwon.rxd.neuron.neuron import Neuron as NeuronSuperclass
import bisect
import numpy as np
import pytest

def test_advance_smoke_test():
    dt = .1
    db = Database()
    Neuron = NeuronSuperclass._initialize(db,
            initial_voltage = -70,
            cytoplasmic_resistance = 100,
            membrane_capacitance = 1,)
    Segment = db.get('Segment').get_instance_type()
    Segment._advance_electric(dt)
    Neuron([0,0,0], 13)
    Segment._advance_electric(dt)
    # Make disconnected segments.
    for i in range(17):
        root = Neuron([0,0,0], 13).root
    # Make connected segments.
    tip = root
    for i in range(1, 111):
        tip = tip.add_segment([i,0,0], 1)
    Segment._advance_electric(dt)
    # db.check() # Don't check because some data is still uninitialized.

def test_time_constant():
    db = Database()
    Neuron = NeuronSuperclass._initialize(db,
            initial_voltage = -70,
            cytoplasmic_resistance = 100,
            membrane_capacitance = 1,)
    Segment = db.get('Segment').get_instance_type()
    root = Neuron([0,0,0], 20).root
    root.voltage = 0
    root.driving_voltage = 1
    root.sum_conductance = 1e-10
    tau = root.get_time_constant()
    nstep = 1000
    dt = tau / nstep
    dt = dt * 1e3 # Convert to milliseconds.
    for _ in range(nstep):
        Segment._advance_electric(dt)
    assert root.voltage == pytest.approx(1 - (1 / np.e))
    db.check()

def measure_length_constant(diam, rm, max_len, dt, ptol, plot):
    db = Database()
    Neuron = NeuronSuperclass._initialize(db,
            initial_voltage = -70,
            cytoplasmic_resistance = 100,
            membrane_capacitance = 1,)
    Segment = db.get('Segment').get_instance_type()
    root = Neuron([0,0,0], diam).root
    section = [root]
    section.extend(root.add_section([3000, 0, 0], diam, max_len))
    for seg in section:
        seg.voltage             = 0
        seg.driving_voltage     = 0
        seg.sum_conductance     = seg.surface_area / rm
    probe   = section[len(section) // 2]
    probe_x = probe.coordinates[0]

    length_constant = probe.get_length_constant()
    print("Lambda:", length_constant, 'um')

    for i in range(round(10/dt)):
        probe.voltage = 1
        Segment._advance_electric(dt)
    db.check()

    x_coords = [seg.coordinates[0] for seg in section]
    test = section[bisect.bisect_left(x_coords, probe_x + length_constant)]
    ratio = test.voltage / probe.voltage
    print("Attenuation at lambda:", ratio)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(x_coords, [seg.voltage for seg in section])
        plt.show()
    assert ratio == pytest.approx(1 / np.e, ptol)
    assert (test.coordinates[0] - probe_x) == pytest.approx(length_constant, .1)

    irm = db.get("Segment.electric_propagator_matrix").get_data()
    assert np.sum(irm.data) / irm.shape[0] == pytest.approx(1), "Check that charge is conserved."

def test_length_constant_1():
    measure_length_constant(
        diam = 1,
        rm = 4e11,
        max_len = 2,
        dt = .01,
        ptol = .05,
        plot = False)

def test_length_constant_2():
    measure_length_constant(
        diam = .9,
        rm = 2e11,
        max_len = 10,
        dt = .025,
        ptol = .15,
        plot = False)

def test_length_constant_3():
    measure_length_constant(
        diam = 5, # large neurite.
        rm = 9e11, # low conductance.
        max_len = 5,
        dt = .025,
        ptol = .20,
        plot = False)

def test_length_constant_4():
    measure_length_constant(
        diam = 5, # large neurite.
        rm = 5e10, # high conductance.
        max_len = 5,
        dt = .025,
        ptol = .20,
        plot = False)

def test_length_constant_5():
    measure_length_constant(
        diam = 2,
        rm = 3e11,
        max_len = 20, # Low resolution
        dt = .1, # Low resolution
        ptol = .25,
        plot = False)

def test_inject_current():
    from neuwon.rxd.rxd_model import RxD_Model
    m = RxD_Model(time_step = .01)
    root = m.Neuron([0,0,0], 10).root

    init_v = root.voltage
    while m.clock() < 3:
        m.advance()
    assert init_v == pytest.approx(root.voltage)

    init_Q = 1e-3 * root.voltage * root.capacitance
    root.inject_current(1, 2)
    delta_Q = 1e-9 * 2e-3
    while m.clock() < 6:
        m.advance()
        print(root.voltage)
    m.check()
    new_Q = 1e-3 * root.voltage * root.capacitance
    assert init_Q + delta_Q == pytest.approx(new_Q)

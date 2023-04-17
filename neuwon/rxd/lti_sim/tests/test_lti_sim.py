from neuwon.rxd.lti_sim import LinearInput, LogarithmicInput
from neuwon.rxd.lti_sim.lti_model import LTI_Model
import os
import pytest

test_dir = os.path.dirname(__file__)

def test_Nav11():
    nmodl_file = os.path.join(test_dir, "Nav11.mod")
    v = LinearInput('v', -200, 200)
    x = LTI_Model(nmodl_file, [v], 0.1, 37.0)
    assert x.name == 'na11a'
    assert x.num_inputs == 1
    assert x.inputs == [v]
    assert x.input1 is v
    assert x.num_states == 6
    assert x.state_names == sorted(x.state_names)
    assert x.conserve_sum == 1.0
    assert x.parameters['C1C2b2'] == 18
    initial_state = x.get_initial_state()
    assert initial_state.pop('C1') == pytest.approx(1)
    for value in initial_state.values():
        assert value == pytest.approx(0, abs=1e-6)

def test_AMPA():
    nmodl_file = os.path.join(test_dir, "ampa13.mod")
    C = LogarithmicInput('C', 0, 1000)
    x = LTI_Model(nmodl_file, [C], 0.1, 37.0)
    assert x.name == 'AMPA13'
    assert x.num_inputs == 1
    assert x.inputs == [C]
    assert x.input1 is C
    assert x.num_states == 13
    assert x.state_names == sorted(x.state_names)
    assert x.conserve_sum == 1.0
    assert x.parameters['Rb1'] == 800
    initial_state = x.get_initial_state()
    assert initial_state.pop('C0') == pytest.approx(1)
    for value in initial_state.values():
        assert value == pytest.approx(0, abs=1e-6)

def test_NMDA():
    nmodl_file = os.path.join(test_dir, "NMDA.mod")
    C = LogarithmicInput('C', 0, 1000)
    v = LinearInput('v', -200, 200)
    x = LTI_Model(nmodl_file, [v, C], 0.1, 37.0)
    assert x.name == 'NMDA_Mg'
    assert x.num_inputs == 2
    assert x.inputs == [C, v]
    assert x.input1 is C
    assert x.input2 is v
    assert x.num_states == 10
    assert x.state_names == sorted(x.state_names)
    assert x.conserve_sum == 1.0
    assert x.parameters['Rb'] == 10e-3
    initial_state = x.get_initial_state()
    assert initial_state.pop('UMg') == pytest.approx(1)
    for value in initial_state.values():
        assert value == pytest.approx(0, abs=1e-6)

def test_nonlinear():
    nmodl_file = os.path.join(test_dir, "nonlinear.mod")
    with pytest.raises(AssertionError):
        LTI_Model(nmodl_file, [], 0.1, 68)

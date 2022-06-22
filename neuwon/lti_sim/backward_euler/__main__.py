import argparse
import ctypes
import lti_sim
import matplotlib.pyplot as plt
import numpy as np
import os.path
import random
import subprocess
import tempfile

voltage_input = lti_sim.LinearInput('v', -100, 100)
glu_input     = lti_sim.LogarithmicInput('C', 0, 1e3)
voltage_input.set_num_buckets(1)
glu_input.set_num_buckets(1, scale=0.01)

py_dir    = os.path.dirname(lti_sim.__file__)
nav11_mod = os.path.join(py_dir, "tests", "Nav11.mod")
ampa_mod  = os.path.join(py_dir, "tests", "ampa13.mod")
nmda_mod  = os.path.join(py_dir, "tests", "NMDA.mod")

lti_kwargs = {'temperature': 37.0, 'float_dtype': np.float64, 'target': 'host'}
nav11_lti_sim = lambda ts, err: lti_sim.main(nav11_mod, [voltage_input], ts, error=err, **lti_kwargs)[1]
ampa_lti_sim = lambda ts, err: lti_sim.main(ampa_mod, [glu_input], ts, error=err, **lti_kwargs)[1]
nmda_lti_sim = lambda ts, err: lti_sim.main(nmda_mod, [glu_input, voltage_input], ts, error=err, **lti_kwargs)[1]

def load_cpp(filename, num_inputs, num_states, TIME_STEP, opt_level=1):
    """ Compile one of the Backward Euler C++ files and link it into python. """
    dirname  = os.path.abspath(os.path.dirname(__file__))
    src_file = os.path.join(dirname, filename)
    so_file  = tempfile.NamedTemporaryFile(suffix='.so', delete=False)
    so_file.close()
    eigen = os.path.join(dirname, "eigen")
    subprocess.run(["g++", src_file, "-o", so_file.name,
                    f"-DTIME_STEP={TIME_STEP}",
                    "-I"+eigen, "-shared", "-fPIC",
                    "-ffast-math", f"-O{opt_level}"],
                    check=True)
    fn = ctypes.CDLL(so_file.name).advance_state
    argtypes = [ctypes.c_int]
    for _ in range(num_inputs):
        argtypes.append(np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C'))
        argtypes.append(np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=1, flags='C'))
    for _ in range(num_states):
        argtypes.append(np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C'))
    fn.argtypes = argtypes
    return fn

def measure_accuracy(inputs, num_states, functions, time_steps,
                    num_instances = 1000,
                    run_time = 10e6,):
    """
    Measure the accuracy at different time-steps.
    Returns the RMS Error of the states at every time step.
    """
    num_functions = len(functions)
    assert len(time_steps) == num_functions
    assert all(isinstance(ts, int) and ts > 0 for ts in time_steps)
    assert all(ts % time_steps[0] == 0 for ts in time_steps)
    assert all(ts <= run_time for ts in time_steps)
    # Generate valid initial state data.
    initial_state = [np.random.uniform(size=num_instances) for _ in range(num_states)]
    sum_x = np.zeros(num_instances)
    for x in initial_state:
        sum_x += x
    for x in initial_state:
        x /= sum_x
    # Duplicate the initial_state data for every time step.
    state = [initial_state]
    for _ in range(num_functions - 1):
        state.append([np.array(x, copy=True) for x in initial_state])
    # Generate random inputs. Will hold the input constant for the duration of the test.
    input_args = []
    for inp in inputs:
        input_args.append(inp.random(num_instances))
        input_args.append(np.array(np.arange(num_instances), dtype=np.int32))
    # 
    t = [0] * num_functions
    def advance(idx):
        functions[idx](num_instances, *input_args, *state[idx])
        t[idx] += time_steps[idx]
    # 
    sqr_err_accum  = [0.0] * num_functions
    num_points     = [0] * num_functions
    num_points[0] += 1 # Don't div zero.
    def compare(idx):
        for a, b in zip(state[0], state[idx]):
            sqr_err_accum[idx] += np.sum((a - b) ** 2)
            num_points[idx] += num_instances
    # 
    while t[0] < run_time:
        advance(0)
        for idx in range(1, num_functions):
            if t[idx] <  t[0]: advance(idx)
            if t[idx] == t[0]: compare(idx)
    # 
    rms_err = [(sum_sqr / num) ** 0.5 for sum_sqr, num in zip(sqr_err_accum, num_points)]
    return rms_err[1:]

def plot_accuracy_vs_timestep():
    time_steps_ns = [100, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    time_steps_ms = [ts/1000/1000 for ts in time_steps_ns]
    print("Measuring RMS-Error at time steps:", time_steps_ms)
    exact_accuracy  = 1e-9
    target_accuracy = 1e-4
    nav  = True
    ampa = True
    nmda = True
    if nav:
        nav11_exact = nav11_lti_sim(time_steps_ms[0], exact_accuracy)
        # Nav11 Backward Euler
        nav11_be_fn = [load_cpp("Nav11.cpp", 1, 6, ts, 1) for ts in time_steps_ms[1:]]
        nav11_be_fn.insert(0, nav11_exact)
        nav11_be_err = measure_accuracy([voltage_input], 6, nav11_be_fn, time_steps_ns)
        print("Nav11 Backward Euler:", nav11_be_err)
        # Nav11 Matrix Exponential
        nav11_me_fn = [nav11_lti_sim(ts, target_accuracy) for ts in time_steps_ms[1:]]
        nav11_me_fn.insert(0, nav11_exact)
        nav11_me_err = measure_accuracy([voltage_input], 6, nav11_me_fn, time_steps_ns)
        print("Nav11 Matrix Exponential:", nav11_me_err)
    if ampa:
        ampa_exact = ampa_lti_sim(time_steps_ms[0], exact_accuracy)
        # AMPA Receptor Backward Euler
        ampa_be_fn = [load_cpp("ampa13.cpp", 1, 13, ts, 1) for ts in time_steps_ms[1:]]
        ampa_be_fn.insert(0, ampa_exact)
        ampa_be_err = measure_accuracy([glu_input], 13, ampa_be_fn, time_steps_ns)
        print("AMPA Backward Euler:", ampa_be_err)
        # AMPA Receptor Matrix Exponential
        ampa_me_fn = [ampa_lti_sim(ts, target_accuracy) for ts in time_steps_ms[1:]]
        ampa_me_fn.insert(0, ampa_exact)
        ampa_me_err = measure_accuracy([glu_input], 13, ampa_me_fn, time_steps_ns)
        print("AMPA Matrix Exponential:", ampa_me_err)
    if nmda:
        nmda_exact = nmda_lti_sim(time_steps_ms[0], 1e-6) # NOTE: 2D models struggle to achieve exact accuracy.
        # NMDA Receptor Backward Euler
        nmda_be_fn = [load_cpp("NMDA.cpp", 2, 10, ts, 1) for ts in time_steps_ms[1:]]
        nmda_be_fn.insert(0, nmda_exact)
        nmda_be_err = measure_accuracy([glu_input, voltage_input], 10, nmda_be_fn, time_steps_ns)
        print("NMDA Backward Euler:", nmda_be_err)
        # NMDA Receptor Matrix Exponential
        nmda_me_fn = [nmda_lti_sim(ts, target_accuracy) for ts in time_steps_ms[1:]]
        nmda_me_fn.insert(0, nmda_exact)
        nmda_me_err = measure_accuracy([glu_input, voltage_input], 10, nmda_me_fn, time_steps_ns)
        print("NMDA Matrix Exponential:", nmda_me_err)
    plt.figure('Accuracy Comparison')
    plt.title('Accuracy vs Time Step')
    plt.ylabel('RMS Error')
    plt.xlabel('Time Step, Milliseconds')
    if nav:  plt.loglog(time_steps_ms[1:], nav11_be_err, 'r',    label='Nav11,\nBackward Euler')
    if ampa: plt.loglog(time_steps_ms[1:], ampa_be_err,  'b',    label='AMPA Receptor,\nBackward Euler')
    if nmda: plt.loglog(time_steps_ms[1:], nmda_be_err,  'lime', label='NMDA Receptor,\nBackward Euler')
    if nav:  plt.loglog(time_steps_ms[1:], nav11_me_err, 'firebrick',  marker='s', label='Nav11,\nMatrix Exponential')
    if ampa: plt.loglog(time_steps_ms[1:], ampa_me_err,  'mediumblue', marker='s', label='AMPA Receptor,\nMatrix Exponential')
    if nmda: plt.loglog(time_steps_ms[1:], nmda_me_err,  'green',      marker='s', label='NMDA Receptor,\nMatrix Exponential')
    plt.grid(axis='y')
    plt.legend()
    plt.show()

def measure_speed(fn, num_states, inputs):
    return lti_sim._measure_speed(fn, num_states, inputs, conserve_sum = 1.0,
                                  float_dtype = np.float64, target = 'host')

def plot_speed():
    err = 1e-4
    # Nav11 Backward Euler
    print("Nav11 Speed Comparison:")
    fn = load_cpp("Nav11.cpp", 1, 6, 0.1, opt_level=1)
    nav11_be = measure_speed(fn, 6, [voltage_input])
    print('BE', nav11_be)
    # Nav11 Matrix Exponential
    fn = nav11_lti_sim(0.1, err)
    nav11_me = measure_speed(fn, 6, [voltage_input])
    print('ME', nav11_me)
    # AMPA Receptor Backward Euler
    print("AMPA Receptor Speed Comparison:")
    fn = load_cpp("ampa13.cpp", 1, 13, 0.1, opt_level=1)
    ampa_be = measure_speed(fn, 13, [glu_input])
    print('BE', ampa_be)
    # AMPA Receptor Matrix Exponential
    fn = ampa_lti_sim(0.1, err)
    ampa_me = measure_speed(fn, 13, [glu_input])
    print('ME', ampa_me)
    # NMDA Receptor Backward Euler
    print("NMDA Receptor Speed Comparison:")
    fn = load_cpp("NMDA.cpp", 2, 10, 0.1, opt_level=1)
    nmda_be = measure_speed(fn, 10, [glu_input, voltage_input])
    print('BE', nmda_be)
    # NMDA Receptor Matrix Exponential
    fn = nmda_lti_sim(0.1, err)
    nmda_me = measure_speed(fn, 10, [glu_input, voltage_input])
    print('ME', nmda_me)
    print()
    # 
    plt.figure('Speed Comparison')
    plt.title('Real Time to Integrate, per Instance per Time Step')
    x = np.arange(3)
    width = 1/3
    plt.bar(x-width/2, [nav11_be, nmda_be, ampa_be,],
        width=width,
        label='Backward Euler')
    plt.bar(x+width/2, [nav11_me, nmda_me, ampa_me,],
        width=width,
        label='Matrix Exponential,\nMaximum Error: %g'%err)
    plt.ylabel('Nanoseconds')
    plt.xticks(x, ["Nav11 Channel\n6 States\n1 Input",
                   "NMDA Receptor\n10 States\n2 Inputs",
                   "AMPA Receptor\n13 States\n1 Input",])
    plt.legend()
    plt.show()

def plot_speed_vs_accuracy():
    max_err = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    print("Target accuracies", max_err)
    nav11_speed = []
    for x in max_err:
        fn = nav11_lti_sim(0.1, x)
        nav11_speed.append(measure_speed(fn, 6, [voltage_input]))
    print("Nav11 speeds", nav11_speed)
    ampa_speed = []
    for x in max_err:
        fn = ampa_lti_sim(0.1, x)
        ampa_speed.append(measure_speed(fn, 13, [glu_input]))
    print("AMPA speeds", ampa_speed)
    nmda_max_err = [1e-2, 1e-3, 1e-4]
    nmda_speed = []
    for x in nmda_max_err:
        fn = nmda_lti_sim(0.1, x)
        nmda_speed.append(measure_speed(fn, 10, [glu_input, voltage_input]))
    print("NMDA speeds", nmda_speed)
    print()
    # 
    plt.figure('Speed/Accuracy Trade-off')
    plt.title('Simulation Speed vs Accuracy')
    plt.ylabel('Real Time to Integrate, per Instance per Time Step\nNanoseconds')
    plt.xlabel('Maximum Absolute Error')
    plt.semilogx(max_err, nav11_speed, label='Nav11, 6 States, 1 Input')
    plt.semilogx(max_err, ampa_speed, label='AMPA Receptor, 13 States, 1 Input')
    plt.semilogx(nmda_max_err, nmda_speed, label='NMDA Receptor, 10 States, 2 Inputs')
    plt.ylim(bottom=0.0)
    plt.legend()
    plt.show()

parser = argparse.ArgumentParser(prog='backward_euler', description=
        """Compare the Backward Euler method of integration with the Matrix Exponential method. """)
parser.add_argument('--accuracy', action='store_true')
parser.add_argument('--speed', action='store_true')
parser.add_argument('--tradeoff', action='store_true')
args = parser.parse_args()

if args.accuracy:   plot_accuracy_vs_timestep()
if args.speed:      plot_speed()
if args.tradeoff:   plot_speed_vs_accuracy()

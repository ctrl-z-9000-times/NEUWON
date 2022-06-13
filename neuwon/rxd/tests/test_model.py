from neuwon.rxd.rxd_model import RxD_Model
from neuwon.rxd.nmodl import NMODL
from neuwon.database.time import TimeSeries
import numpy as np
import pytest

def test_smoke_test():
    m = RxD_Model()
    m.advance()
    m.Neuron([0,0,0], 13)
    m.advance()
    m.check()

    # Test manually creating segments w/o an attached neuron.
    tip = m.Segment(None, [-1,0,7], 5.7)
    for x in range(10):
        tip = m.Segment(tip, [x,0,7], 1)
    m.advance()
    # m.check() # Will fail BC neuron NULL and not allowed to be invalid.

def test_filter_by_type():
    m = RxD_Model()
    n1 = m.Neuron([0,0,0], 13, neuron_type='n1', segment_type='s1')
    n2 = m.Neuron([0,2,0], 13, neuron_type='n2', segment_type='s2')
    m.advance()
    assert m.filter_segments_by_type(['n2'], ['s2']) == [n2.root]
    assert m.filter_segments_by_type(['n2'], ['s1', 's2']) == [n2.root]
    assert m.filter_segments_by_type(['n2'], ['s1']) == []
    assert m.filter_segments_by_type(['n1', 'n2'], ['s1']) == [n1.root]
    assert m.filter_segments_by_type(['n1', 'n2'], ['s1', 's2']) == [n1.root, n2.root]
    assert m.filter_neurons_by_type(['n1']) == [n1]
    assert m.filter_neurons_by_type(['n2']) == [n2]
    assert m.filter_neurons_by_type(['n1', 'n2']) == [n1, n2]

def test_model_hh(debug=False, measure_correct=False):
    dt = .01
    if not measure_correct:
        dt *= 10 # Run faster with larger `dt`.
    m = RxD_Model(time_step = dt, temperature = 6.3,
            mechanisms = {
                'hh': NMODL("./nmodl_library/artificial/hh.mod", use_cache=False)
            })
    hh = m.mechanisms['hh']
    print(hh._advance_pycode)
    print('Initial Values:')
    for comp in m.database.get('hh').get_all_components():
        print(comp.get_name(), '\t', comp.get_initial_value())
    root = tip = m.Neuron([-1,0,7], 5.7).root
    hh_instance = hh(root)
    for x in range(10):
        tip = m.Segment(tip, [x,0,7], 1)
        hh(tip)

    x = TimeSeries().record(tip, 'voltage')
    debug_probe_1 = TimeSeries().record(root, 'driving_voltage')
    debug_probe_2 = TimeSeries().record(root, 'sum_conductance')

    m.advance()
    m.check()

    while m.clock() < 20:
        for t in [5, 15]:
            if m.clock.is_now(t):
                root.inject_current(.1, 1)
        m.advance()
        m.check()

    x.stop()
    if measure_correct:
        skip = 10
        data = list(x.get_data())[::skip]
        ts   = list(x.get_timestamps())[::skip]
        data = [round(x, 2) for x in data]
        ts   = [round(x, 2) for x in ts]
        print("correct = TimeSeries(", data, ",", ts, ")")

    correct = TimeSeries(
    [-70.0, -69.75, -69.5, -69.26, -69.02, -68.8, -68.58, -68.37, -68.16, -67.97, -67.78,
    -67.59, -67.42, -67.25, -67.08, -66.92, -66.77, -66.63, -66.48, -66.35, -66.22,
    -66.09, -65.98, -65.86, -65.75, -65.65, -65.55, -65.46, -65.37, -65.29, -65.21,
    -65.14, -65.07, -65.01, -64.96, -64.9, -64.86, -64.82, -64.79, -64.76, -64.73, -64.72,
    -64.7, -64.7, -64.7, -64.7, -64.71, -64.73, -64.75, -64.77, -64.8, -57.95, -51.25,
    -44.26, -35.5, -20.62, 10.85, 48.59, 55.34, 53.32, 50.03, 44.2, 39.32, 33.97,
    28.32, 22.52, 16.68, 10.88,
    5.19, -0.39, -5.83, -11.15, -16.38, -21.56, -26.77, -32.15, -37.95, -44.6, -52.76,
    -63.07, -74.33, -82.23, -85.48, -86.52, -86.82, -86.89, -86.88, -86.83, -86.77,
    -86.7, -86.63, -86.55, -86.47, -86.38, -86.3, -86.2, -86.1, -86.0, -85.9, -85.78,
    -85.67, -85.55, -85.43, -85.3, -85.16, -85.03, -84.88, -84.74, -84.58, -84.43, -84.27,
    -84.1, -83.93, -83.76, -83.58, -83.4, -83.21, -83.02, -82.82, -82.62, -82.42, -82.21,
    -82.0, -81.79, -81.57, -81.36, -81.13, -80.91, -80.68, -80.45, -80.22, -79.99, -79.76,
    -79.52, -79.28, -79.05, -78.81, -78.57, -78.33, -78.09, -77.85, -77.61, -77.37,
    -77.13, -76.89, -76.66, -76.42, -76.19, -75.95, -75.72, -75.49, -68.42, -61.74,
    -55.41, -49.28, -42.78, -34.19, -18.72, 14.02, 49.11, 54.24, 50.38, 46.96,
    42.66, 37.71, 32.32, 26.67, 20.9, 15.11, 9.39,
    3.76, -1.74, -7.12, -12.39, -17.58, -22.74, -27.97, -33.41, -39.35, -46.28, -54.92,
    -65.71, -76.65, -83.36, -85.87, -86.65, -86.87, -86.9, -86.88, -86.83, -86.77, -86.7,
    -86.62, -86.54, -86.46, -86.38, -86.29, -86.19, -86.09, -85.99, -85.88] ,
     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
     1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8,
     2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3,
     4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8,
     5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3,
     7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8,
     8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2,
     10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4,
     11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6,
     12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8,
     13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0,
     15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16.0, 16.1, 16.2,
     16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17.0, 17.1, 17.2, 17.3, 17.4,
     17.5, 17.6, 17.7, 17.8, 17.9, 18.0, 18.1, 18.2, 18.3, 18.4, 18.5, 18.6,
     18.7, 18.8, 18.9, 19.0, 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 19.8,
     19.9, 20.0] )

    correct.interpolate(x)
    abs_diff = np.abs(np.subtract(x.get_data(), correct.get_data()))
    if debug:
        debug_probe_1.plot(show=False)
        debug_probe_2.plot(show=False)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x.get_timestamps(), x.get_data(), 'k', label='current run')
        plt.plot(x.get_timestamps(), correct.get_data(), 'y', label='saved data')
        plt.plot(x.get_timestamps(), abs_diff, 'r', label='absolute difference')
        x.label_axes()
        plt.legend()
        plt.show()
    assert max(abs_diff) < 30

if __name__ == "__main__":
    test_model_hh(debug=True)
    # test_model_hh(measure_correct=True)

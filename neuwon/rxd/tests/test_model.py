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

def test_model_hh(debug=False):
    dt = .01
    dt *= 4.56789 # Run faster with larger `dt`.
    m = RxD_Model(time_step = dt, celsius = 6.3,
            species = {
                'na': {'reversal_potential': 40},
                'k':  {'reversal_potential': -80},
                'l':  {'reversal_potential': -50},
            },
            mechanisms = {
                'hh': NMODL("./nmodl_library/hh.mod", use_cache=False)
            })
    hh = m.mechanisms['hh']
    print(hh._advance_pycode)
    print('Initial Values:')
    for comp in m.database.get('hh').get_all_components():
        print(comp.get_name(), '\t', comp.get_initial_value())
    root = tip = m.Segment(None, [-1,0,7], 5.7)
    hh_instance = hh(root)
    for x in range(10):
        tip = m.Segment(tip, [x,0,7], 1)
        hh(tip)

    x = TimeSeries().record(tip, 'voltage')
    debug_probe_1 = TimeSeries().record(root, 'driving_voltage')
    debug_probe_2 = TimeSeries().record(root, 'sum_conductance')

    m.advance()
    m.check()

    while m.clock() < 15:
        m.advance()
        m.check()
    root.inject_current(.1e-9, 1)
    while m.clock() < 20:
        m.advance()
        m.check()

    x.stop()
    skip = 10
    data = list(x.get_data())[::skip]
    ts   = list(x.get_timestamps())[::skip]
    data = [round(x, 2) for x in data]
    ts   = [round(x, 2) for x in ts]
    print("TimeSeries(", data, ",", ts, ")")
    correct = TimeSeries(
        [-70.0, -69.54, -69.07, -68.62, -68.19, -67.77, -67.36, -66.97, -66.59,
        -66.22, -65.86, -65.51, -65.16, -64.83, -64.5, -64.18, -63.86, -63.54, -63.23,
        -62.92, -62.61, -62.3, -61.99, -61.67, -61.35, -61.02, -60.69, -60.34, -59.98,
        -59.6, -59.2, -58.77, -58.31, -57.8, -57.24, -56.6, -55.86, -55.0, -53.95,
        -52.65, -50.96, -48.66, -45.35, -40.19, -31.39, -15.26, 10.49, 29.51,
        32.67, 31.39, 29.01, 25.95, 22.36, 18.36, 14.08, 9.62, 5.05,
        0.45, -4.13, -8.67, -13.16, -17.59, -21.98, -26.37, -30.8, -35.4, -40.34,
        -45.85, -52.21, -59.49, -66.97, -72.97, -76.43, -77.97, -78.57, -78.78,
        -78.84, -78.83, -78.79, -78.74, -78.69, -78.62, -78.56, -78.49, -78.42,
        -78.34, -78.26, -78.18, -78.1, -78.01, -77.93, -77.83, -77.74, -77.64, -77.54,
        -77.44, -77.33, -77.23, -77.12, -77.0, -76.88, -76.77, -76.64, -76.52, -76.39,
        -76.26, -76.13, -75.99, -75.86, -75.72, -75.58, -75.43, -75.29, -75.14,
        -74.99, -74.84, -74.69, -74.54, -74.38, -74.23, -74.07, -73.91, -73.75,
        -73.59, -73.43, -73.27, -73.11, -72.95, -72.78, -72.62, -72.46, -72.3, -72.14,
        -71.97, -71.81, -71.65, -71.49, -71.33, -71.17, -71.02, -70.86, -70.7, -70.55,
        -70.39, -70.24, -70.09, -69.94, -69.79, -69.65, -69.5, -69.36, -62.1, -55.62,
        -49.47, -43.25, -35.96, -25.11, -6.16, 19.8, 32.63, 32.81, 29.37,
        26.53, 23.09, 19.18, 14.92, 10.45, 5.86,
        1.21, -3.42, -8.02, -12.55, -17.03, -21.45, -25.86, -30.3, -34.89, -39.77,
        -45.2, -51.43, -58.57, -66.05, -72.28, -76.05, -77.79, -78.48, -78.73, -78.81,
        -78.81, -78.78, -78.73, -78.67, -78.61, -78.54, -78.48, -78.4, -78.33, -78.25,
        -78.17, -78.09, -78.0] , [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2,
        2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6,
        3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0,
        5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4,
        6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8,
        7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2,
        9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5,
        10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7,
        11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9,
        13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1,
        14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0, 15.1, 15.2, 15.3,
        15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16.0, 16.1, 16.2, 16.3, 16.4, 16.5,
        16.6, 16.7, 16.8, 16.9, 17.0, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7,
        17.8, 17.9, 18.0, 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7, 18.8, 18.9,
        19.0, 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 19.9, 20.0] )
    correct.interpolate(x)
    abs_diff = np.abs(np.subtract(x.get_data(), correct.get_data()))
    if debug:
        # debug_probe_1.plot(show=False)
        # debug_probe_2.plot(show=False)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x.get_timestamps(), x.get_data(), 'k', label='current run')
        plt.plot(x.get_timestamps(), correct.get_data(), 'y', label='saved data')
        plt.plot(x.get_timestamps(), abs_diff, 'r', label='absolute difference')
        x.label_axes()
        plt.legend()
        plt.show()
    assert max(abs_diff) < 30

if __name__ == "__main__": test_model_hh(True)

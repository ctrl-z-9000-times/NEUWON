from lti_sim import LinearInput, LogarithmicInput
from lti_sim.approx import MatrixSamples, Approx1D, Approx2D
from lti_sim.lti_model import LTI_Model
import math
import numpy as np
import os
import pytest

test_dir = os.path.dirname(__file__)

maximum_under_estimation_factor = 4
maximum_over_estimation_factor  = 1.5

def _resample(model, approx, subdivide):
    print('default num samples', len(approx.samples.samples))
    original_num_buckets = [inp.num_buckets for inp in model.inputs]
    # Subdivide the buckets, to increase num-samples and also to ensure uniform distribution.
    for inp in model.inputs:
        inp.set_num_buckets(inp.num_buckets * subdivide)
    # Get all new samples, don't reuse any of the original samples.
    x = MatrixSamples(model)
    approx.samples = x
    approx._ensure_enough_exact_samples()
    # Restore original bucket layout.
    for inp, num_buckets in zip(model.inputs, original_num_buckets):
        inp.set_num_buckets(num_buckets)
    print('testing num samples', len(approx.samples.samples))

def _check_error(est_err, true_err):
    print("Train Error          \t| Test Error           \t\t| Percent Underestimation")
    for e1, e2 in zip(est_err, true_err):
        print(e1, '\t|', str(e2).ljust(25), '\t| ', round(100 * (e1 - e2) / e2, 2))
    print()
    assert all(est_err <= true_err * maximum_over_estimation_factor)
    assert all(est_err >= true_err / maximum_under_estimation_factor)

def test_1D():
    nmodl_file = os.path.join(test_dir, "Nav11.mod")
    v = LinearInput('v', -100, 100)
    v.set_num_buckets(50)
    m = LTI_Model(nmodl_file, [v], 0.1, 37.0)
    x = MatrixSamples(m)
    a = Approx1D(x, 2)
    est_err  = a.measure_error()
    est_rmse = a.rmse.copy()
    recalc_rmse = a.measure_error(rmse=True)
    assert np.all(np.abs(est_rmse - recalc_rmse) < 1e-15)
    _resample(m, a, 10)
    true_err  = a.measure_error(rmse=False)
    true_rmse = a.measure_error(rmse=True)
    print("Max Abs Error")
    _check_error(est_err, true_err)
    print("Root Mean Square Error")
    _check_error(est_rmse, true_rmse)

    print('TEST MAE\t\t TRAIN RMSE\t\t PCT UNDER-EST')
    for mae, rmse in zip(true_err, est_rmse):
        print(mae, '\t', rmse, '   \t', round(100 * (rmse - mae) / mae), '%')

def test_2D():
    nmodl_file = os.path.join(test_dir, "NMDA.mod")
    v = LinearInput('v', -100, 100)
    v.set_num_buckets(10)
    C = LogarithmicInput('C', 0, 100)
    C.set_num_buckets(10, scale=.001)
    m = LTI_Model(nmodl_file, [C, v], 0.1, 37.0)
    x = MatrixSamples(m)
    a = Approx2D(x, [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
    est_err = a.measure_error().reshape(-1)
    est_rmse = a.rmse.copy().reshape(-1)
    recalc_rmse = a.measure_error(rmse=True).reshape(-1)
    assert np.all(np.abs(est_rmse - recalc_rmse) < 1e-15)
    _resample(m, a, 2)
    true_err  = a.measure_error(rmse=False).reshape(-1)
    true_rmse = a.measure_error(rmse=True).reshape(-1)
    print("Max Abs Error")
    _check_error(est_err, true_err)
    print("Root Mean Square Error")
    _check_error(est_rmse, true_rmse)

    print('TEST MAE\t\t TRAIN RMSE\t\t PCT UNDER-EST')
    for mae, rmse in zip(true_err, est_rmse):
        print(mae, '\t', rmse, '   \t', round(100 * (rmse - mae) / mae), '%')

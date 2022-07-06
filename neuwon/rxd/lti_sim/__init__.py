"""
Simulator for Linear Time-Invariant Kinetic Models using the NMODL file format.

For more information see:
    Exact digital simulation of time-invariant linear systems with applications
    to neuronal modeling. Rotter S, Diesmann M (1999).
    https://doi.org/10.1007/s004220050570
"""

# Written by David McDougall, 2022

from .inputs import LinearInput, LogarithmicInput
from .lti_model import LTI_Model
from .optimizer import Optimize1D, Optimize2D
from neuwon.database import Real
import numpy as np
import time

__all__ = ('main', 'LinearInput', 'LogarithmicInput')

def main(nmodl_filename, inputs, time_step, temperature,
         error, target,
         outfile=False, verbose=False, plot=False,):
    # Read, parse, and preprocess the input file.
    model = LTI_Model(nmodl_filename, inputs, time_step, temperature)
    # 
    if   model.num_inputs == 1: OptimizerClass = Optimize1D
    elif model.num_inputs == 2: OptimizerClass = Optimize2D
    else: raise NotImplementedError('too many inputs.')
    optimized = OptimizerClass(model, error, target, (verbose >= 2)).best
    # 
    if outfile:
        optimized.backend.write(outfile)
        if outfile != optimized.backend.filename:
            print(f'Output written to: "{optimized.backend.filename}"')
    if verbose or plot:
        print(str(optimized.approx) +
              f"Run speed:    {round(optimized.runtime)} ns/Δt")
    if plot:
        optimized.approx.plot(model.name)
    return (model.get_initial_state(), optimized.backend.load())

def _measure_speed(f, num_states, inputs, conserve_sum, target):
    num_instances = 10 * 1000
    num_repetions = 200
    # 
    if target == 'host':
        xp = np
    elif target == 'cuda':
        import cupy
        xp = cupy
        start_event = cupy.cuda.Event()
        end_event   = cupy.cuda.Event()
    # Generate valid initial states.
    state = [xp.array(xp.random.uniform(size=num_instances), dtype=Real)
                for x in range(num_states)]
    if conserve_sum is not None:
        conserve_sum = float(conserve_sum)
        sum_states = xp.zeros(num_instances)
        for data in state:
            sum_states = sum_states + data
        correction_factor = conserve_sum / sum_states
        for data in state:
            data *= correction_factor
    # 
    input_indicies = xp.arange(num_instances, dtype=np.int32)
    elapsed_times = np.empty(num_repetions)
    for trial in range(num_repetions):
        input_arrays = []
        for inp in inputs:
            input_arrays.append(inp.random(num_instances, Real, xp))
            input_arrays.append(input_indicies)
        _clear_cache(xp)
        time.sleep(0) # Try to avoid task switching while running.
        if target == 'cuda':
            start_event.record()
            f(num_instances, *input_arrays, *state)
            end_event.record()
            end_event.synchronize()
            elapsed_times[trial] = 1e6 * cupy.cuda.get_elapsed_time(start_event, end_event)
        elif target == 'host':
            start_time = time.thread_time_ns()
            f(num_instances, *input_arrays, *state)
            elapsed_times[trial] = time.thread_time_ns() - start_time
    return np.min(elapsed_times) / num_instances

def _clear_cache(array_module):
    # Read and then write back 32MB of data. Assuming that the CPU is using a
    # least-recently-used replacement policy, touching every piece of data once
    # should be sufficient to put it into the cache.
    big_data = array_module.empty(int(32e6 / 8), dtype=np.int64)
    big_data += 1

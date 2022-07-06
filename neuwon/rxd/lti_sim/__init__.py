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

__all__ = ('main', 'LinearInput', 'LogarithmicInput')

def main(nmodl_filename, inputs, time_step, temperature,
         error, target,
         verbose=False, plot=False,):
    # Read, parse, and preprocess the input file.
    model = LTI_Model(nmodl_filename, inputs, time_step, temperature)
    # 
    if   model.num_inputs == 1: OptimizerClass = Optimize1D
    elif model.num_inputs == 2: OptimizerClass = Optimize2D
    else: raise NotImplementedError('too many inputs.')
    best = OptimizerClass(model, error, target, (verbose >= 2)).best
    # 
    if verbose or plot:
        print(str(best.approx) +
              f"Run speed:    {round(best.runtime)} ns/Δt")
    if plot:
        best.approx.plot(model.name)
    return (model.get_initial_state(), best.backend.load())


# DEBUGGING:
if __name__ == '__main__':

    nmodl_filename = "nmodl_library/kinetic/Nav11.mod"
    inputs = []

    main(nmodl_filename, inputs, 0.1, 37,
         error=1e-4, target='host',
         verbose=2, plot=False,)

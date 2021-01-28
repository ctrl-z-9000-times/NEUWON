import numpy as np
import scipy.linalg
import cupy as cp
import numba
import math
import copy
import itertools
from collections.abc import Callable, Iterable, Mapping

from neuwon import Real, Location

class Mechanism:
    """ Abstract class for specifying mechanisms which are localized and stateful. """
    def required_species(self):
        """ Optional, Returns the Species required by this mechanism.
        Allowed return types: Species, names of species, and lists either. """
        return []
    @classmethod
    def set_time_step(self, time_step):
        """ Optional, This method is called on a deep copy of each mechanism type."""
    def instance_dtype(self):
        """ Returns the numpy data type for an instance of this mechanism.

        The returned value is one of:
            * numpy.dtype
            * Pair of (numpy.dtype, shape) to to make an array.
            * Dictionary where the values are one of the above.

        Examples
            Real
            (Real, 7)
            (Location, (Real, 3))
            {"a": Location, "b": (Real, 3)}
        """
        raise TypeError("Abstract method called!")
    def new_instance(self, time_step, location, geometry, *args):
        """ Returns initial state as or compatible with an instance_dtype. """
        raise TypeError("Abstract method called!")
    def advance(self, locations, instances, time_step, reaction_inputs, reaction_outputs):
        """ Advance all instances of this mechanism. """
        raise TypeError("Abstract method called!")

def _add_mechanism_type(self, mech_type):
    if hasattr(mech_type, "set_time_step"):
        original_type = mech_type
        mech_type = copy.deepcopy(mech_type)
        mech_type.set_time_step(self.time_step)
        self.original_mechanisms[original_type] = mech_type
    else:
        self.original_mechanisms[mech_type] = mech_type
    dtype = mech_type.instance_dtype()
    if isinstance(dtype, Mapping):
        instances = {k: [] for k in dtype}
    else:
        instances = []
    locations = []
    self.mechanisms[mech_type] = (locations, instances)

def _to_cuda_device(dtype, instances):
    """ Copy a python list of new instances to a cuda device array. """
    if isinstance(dtype, Mapping):
        assert(not any(isinstance(dt, Mapping) for dt in dtype.values())) # Nested structures are not allowed.
        structure_of_arrays = {}
        for k, dt in dtype.items():
            structure_of_arrays[k] = to_cuda_device(dt, instances[k])
        return structure_of_arrays
    else:
        if isinstance(dtype, Iterable):
            dt, shape = dtype
            if not isinstance(shape, Iterable):
                shape = (shape,)
            cuda_array = np.array(instances, dtype=dt).reshape([-1] + list(shape))
        else:
            cuda_array = np.array(instances, dtype=dtype).flatten()
        return numba.cuda.to_device(cuda_array)

def _init_mechansisms(self, insertions):
    self.original_mechanisms = {}
    self.mechanisms = {}
    for location, insertions_list in enumerate(insertions):
        for mech_type, args, kwargs in insertions_list:
            if mech_type not in self.original_mechanisms:
                _add_mechanism_type(self, mech_type)
            mech_type = self.original_mechanisms[mech_type]
            self.mechanisms[mech_type][0].append(location)
            instance = mech_type.new_instance(
                    self.time_step, location, self.geometry, *args, **kwargs)
            if isinstance(instance, Mapping):
                dtype = mech_type.instance_dtype()
                assert(isinstance(dtype, Mapping))
                assert(len(dtype) == len(instance))
                for k in mech_type.instance_dtype().keys():
                    self.mechanisms[mech_type][1][k].append(instance[k])
            else:
                self.mechanisms[mech_type][1].append(instance)
    # Construct final data arrays.
    for mech_type, (locations, instances) in self.mechanisms.items():
        dtype = mech_type.instance_dtype()
        self.mechanisms[mech_type] = (
                cp.array(locations, dtype=Location),
                _to_cuda_device(dtype, instances))

# TODO: What are the units on atol? How does the timestep factor into it?

class KineticModel:
    def __init__(self, time_step, input_ranges, states, kinetics,
        initial_state=None,
        conserve_sum=False,
        atol=1e-3):
        """ """
        # Save and check the arguments.
        self.time_step = float(time_step)
        self.input_ranges = np.array(input_ranges, dtype=Real)
        assert(len(self.input_ranges.shape) == 2 and self.input_ranges.shape[1] == 2)
        self.input_ranges.sort(axis=1)
        self.lower, self.upper = zip(*self.input_ranges)
        self.num_inputs = len(self.input_ranges)
        assert(self.num_inputs > 0)
        if isinstance(states, int):
            self.states = tuple(str(s) for s in range(states))
        else:
            if isinstance(states, str):
                states = [x.strip(",") for x in states.split()]
            self.states = tuple(str(s) for s in states)
        self.num_states = len(self.states)
        assert(self.num_states > 0)
        self.conserve_sum = float(conserve_sum) if conserve_sum else None
        self.initial_state = None
        if initial_state is not None:
            assert(self.conserve_sum is not None)
            self.initial_state = np.zeros(self.num_states, dtype=Real)
            self.initial_state[self.states.index(initial_state)] = self.conserve_sum
        self.atol = float(atol)
        assert(self.atol > 0)
        # self.kinetics is a list of non-zero elements of the coefficients
        # matrix in the derivative function:
        #       dX/dt = C * X, where C is Coefficients matrix and X is state vector.
        # Stored as tuples of (src, dst, coef, func)
        #       Where "src" and "dst" are indexes into the state vector.
        #       Where "coef" is constant rate mulitplier.
        #       Where "func" is optional function: func(*inputs) -> coefficient
        self.kinetics = []
        for reactant, product, forward, reverse in kinetics:
            r_idx = self.states.index(str(reactant))
            p_idx = self.states.index(str(product))
            for src, dst, rate in ((r_idx, p_idx, forward), (p_idx, r_idx, reverse)):
                if isinstance(rate, Callable):
                    coef = 1
                    func = rate
                else:
                    coef = float(rate)
                    func = None
                self.kinetics.append((src, dst, +coef, func))
                self.kinetics.append((src, src, -coef, func))
        # Determine how many interpolation points to use.
        self.grid_size = np.full((self.num_inputs,), 2)
        self._compute_interpolation_grid()
        while self._estimate_min_accuracy() >= self.atol:
            self.grid_size += 1
            self.grid_size *= 2
            self._compute_interpolation_grid()
        self.data = cp.array(self.data)

    def _compute_impulse_response_matrix(self, inputs):
        A = np.zeros([self.num_states] * 2, dtype=float)
        for src, dst, coef, func in self.kinetics:
            if func is not None:
                A[dst, src] += coef * func(*inputs)
            else:
                A[dst, src] += coef
        return scipy.linalg.expm(A * self.time_step)

    def _compute_interpolation_grid(self):
        """ Assumes self.grid_size is already set. """
        grid_range = np.subtract(self.upper, self.lower)
        grid_range[grid_range == 0] = 1
        self.grid_factor = np.subtract(self.grid_size, 1) / grid_range
        self.data = np.empty(list(self.grid_size) + [self.num_states]*2, dtype=Real)
        # Visit every location on the new interpolation grid.
        grid_axes = [list(enumerate(np.linspace(*args, dtype=float)))
                    for args in zip(self.lower, self.upper, self.grid_size)]
        for inputs in itertools.product(*grid_axes):
            index, inputs = zip(*inputs)
            self.data[index] = self._compute_impulse_response_matrix(inputs)

    def _estimate_min_accuracy(self):
        atol = 0
        num_points = np.product(self.grid_size)
        num_test_points = max(int(round(num_points / 10)), 100)
        for _ in range(num_test_points):
            inputs = np.random.uniform(self.lower, self.upper)
            exact = self._compute_impulse_response_matrix(inputs)
            interp = self._interpolate_impulse_response_matrix(inputs)
            atol = max(atol, np.max(np.abs(exact - interp)))
        return atol

    def _interpolate_impulse_response_matrix(self, inputs):
        assert(len(inputs) == self.num_inputs)
        inputs = np.array(inputs, dtype=Real)
        assert(all(inputs >= self.lower) and all(inputs <= self.upper)) # Bounds check the inputs.
        # Determine which grid box the inputs are inside of.
        inputs = self.grid_factor * np.subtract(inputs, self.lower)
        lower_idx = np.array(np.floor(inputs), dtype=int)
        upper_idx = np.array(np.ceil(inputs), dtype=int)
        upper_idx = np.minimum(upper_idx, self.grid_size - 1) # Protect against floating point error.
        # Prepare to find the interpolation weights, by finding the distance
        # from the input point to each corner of its grid box.
        inputs -= lower_idx
        corner_weights = [np.subtract(1, inputs), inputs]
        # Visit each corner of the grid box and accumulate the results.
        irm = np.zeros([self.num_states]*2, dtype=Real)
        for corner in itertools.product(*([(0,1)] * self.num_inputs)):
            idx = np.choose(corner, [lower_idx, upper_idx])
            weight = np.product(np.choose(corner, corner_weights))
            irm += weight * np.squeeze(self.data[idx])
        return irm

    def advance(self, inputs, states):
        numba.cuda.synchronize()
        assert(len(inputs) == self.num_inputs)
        assert(len(states.shape) == 2 and states.shape[1] == self.num_states)
        assert(states.dtype == Real)
        for l, u, x in zip(self.lower, self.upper, inputs):
            assert(x.shape[0] == states.shape[0])
            assert(cp.all(cp.logical_and(x >= l, x <= u))) # Bounds check the inputs.
        if self.num_inputs == 1:
            scratch = cp.zeros(states.shape, dtype=Real)
            threads = 64
            blocks = (states.shape[0] + (threads - 1)) // threads
            _1d[blocks,threads](inputs[0], states, scratch,
                self.lower[0], self.grid_size[0], self.grid_factor[0], self.data)
        else:
            raise TypeError("KineticModel is unimplemented for more than 1 input dimension.")
        numba.cuda.synchronize()
        # Enforce the invariant sum of states.
        if self.conserve_sum is not None:
            threads = 64
            blocks = (states.shape[0] + (threads - 1)) // threads
            _conserve_sum[blocks,threads](states, self.conserve_sum)
            numba.cuda.synchronize()

@numba.cuda.jit()
def _1d(inputs, states, scratch, input_lower_bound, grid_size, grid_factor, data):
    index = numba.cuda.grid(1)
    if index >= states.shape[0]:
        return
    inpt = inputs[index]
    state = states[index]
    scrtch = scratch[index]
    # Determine which grid box the inputs are inside of.
    inpt = (inpt - input_lower_bound) * grid_factor
    lower_idx = int(math.floor(inpt))
    upper_idx = int(math.ceil(inpt))
    upper_idx = min(upper_idx, grid_size - 1) # Protect against floating point error.
    inpt -= lower_idx
    # Visit each corner of the grid box and accumulate the results.
    _weighted_matrix_vector_multiplication(
        1 - inpt, data[lower_idx], state, scrtch)
    _weighted_matrix_vector_multiplication(
        inpt, data[upper_idx], state, scrtch)
    for i in range(len(state)):
        state[i] = scrtch[i]

@numba.cuda.jit(device=True)
def _weighted_matrix_vector_multiplication(w, m, v, results):
    """ Computes: results += weight * (matrix * vector)

    Arguments:
        [w]eight
        [m]atrix
        [v]ector
        results - output accumulator """
    for r in range(len(v)):
        dot = 0
        for c in range(len(v)):
            dot += m[r, c] * v[c]
        results[r] += w * dot

@numba.cuda.jit()
def _conserve_sum(states, target_sum):
    index = numba.cuda.grid(1)
    if index >= states.shape[0]:
        return
    state = states[index]
    accumulator = 0.
    for i in range(len(state)):
        accumulator += state[i]
    correction_factor = target_sum / accumulator
    for i in range(len(state)):
        state[i] *= correction_factor

import numpy as np
import scipy.linalg
import cupy as cp
import numba
import math
import copy
import itertools
from collections.abc import Callable, Iterable, Mapping

from neuwon.common import Real, Location

class Reaction:
    """ Abstract class for specifying reactions and mechanisms.

    Reactions can either be local or global:
    * Local reactions are inserted into segments and can have persistent state.
    * Global reactions are omnipresent and can only use globally present species
        as state. Global reactions are added to the Model once at creation.
    """
    @classmethod
    def name(self):
        """ A unique name for this reaction and all of its instances. """
        raise TypeError("Abstract method called: %s.%s()"%(repr(self), "name"))
    @classmethod
    def pointers(self):
        """ Returns a mapping of string names to Pointer objects.
        All external data access is declared by this method.
        """
        raise TypeError("Abstract method called: %s.%s()"%(repr(self), "pointers"))
    @classmethod
    def set_time_step(self, time_step):
        """ Optional, This method is called on a deep copy of each reaction type."""
    @classmethod
    def new_instance(self, time_step, location, geometry, *args):
        """ Returns a mapping of pointer names to their initial values at this location.
        Must contain entries for all pointers to custom dtype arrays. """
        return {}
    @classmethod
    def advance(self, time_step, locations, **pointers):
        """ Advance all instances of this reaction. """
        raise TypeError("Abstract method called: %s.%s()"%(repr(self), "advance"))

class Pointer:
    """ Pointers are the connection between reactions and NEUWON.
    Created by NMODL statements: USEION and POINTER. """
    def __init__(self, species=None,
            dtype=None,
            voltage=False,
            conductance=False,
            intra_concentration=False,
            extra_concentration=False,
            intra_release_rate=False,
            extra_release_rate=False,):
        """
        Argument dtype: Specify an array of the given numpy.dtype which is
        associated with this mechanism. It can be one of:
            * numpy.dtype,
            * Pair of (numpy.dtype, shape) to to make an array.
            Examples:
                np.float32
                (np.float32, 7)
                (np.float32, [4, 4])
        """
        self.species = str(species) if species else None
        if dtype is not None:
            if isinstance(dtype, Iterable):
                dtype, shape = dtype
                if isinstance(shape, Iterable):
                    shape = list(shape)
                else:
                    shape = [shape]
            else:
                shape = []
            assert(isinstance(dtype, np.dtype))
            self.dtype = (dtype, shape)
        else:
            self.dtype = None
        self.voltage = bool(voltage)
        self.conductance = bool(conductance)
        self.intra_concentration = bool(intra_concentration)
        self.extra_concentration = bool(extra_concentration)
        self.intra_release_rate = bool(intra_release_rate)
        self.extra_release_rate = bool(extra_release_rate)
        assert(1 == bool(self.dtype) + self.voltage + self.conductance +
            self.intra_concentration + self.extra_concentration +
            self.intra_release_rate + self.extra_release_rate)
        self.read = (bool(self.dtype) or self.voltage or
                    self.intra_concentration or self.extra_concentration)
        self.write = (bool(self.dtype) or self.conductance or
                    self.intra_release_rate or self.extra_release_rate)

    def NEURON_conversion_factor(self):
        if   self.dtype:        return 1
        elif self.voltage:      return 1000 # From NEUWONs volts to NEURONs millivolts.
        elif self.conductance:  return 1
        else: raise NotImplementedError(self)

    def __repr__(self):
        name = getattr(self.species, "name", self.species)
        flags = []
        if self.dtype: flags.append(str(self.dtype))
        if self.voltage: flags.append("voltage=True")
        if self.conductance: flags.append("conductance=True")
        if self.intra_concentration: flags.append("intra_concentration=True")
        if self.extra_concentration: flags.append("extra_concentration=True")
        if self.intra_release_rate: flags.append("intra_release_rate=True")
        if self.extra_release_rate: flags.append("extra_release_rate=True")
        return "Pointer(%s, %s)"%(name, ", ".join(flags))

def _init_reactions(reactions_argument, insertions, time_step, geometry):
    reactions = {}
    # The given arguments take priority, add them first.
    for r in reactions_argument: _add_reaction(reactions, r, time_step)
    # Add all inserted reactions.
    for location, insertions_here in enumerate(insertions):
        for name, args, kwargs in insertions_here:
            container = _add_reaction(reactions, name, time_step)
            container.append_new_mechanism(time_step, location, geometry, *args, **kwargs)
    # Copy data from python objects to GPU arrays.
    for container in reactions.values(): container._to_cuda_device()
    return reactions

def _add_reaction(reactions_dict, new_reaction, time_step):
    """ Adds a new reaction to the dictionary if its name is new/unique.

    Argument new_reaction must be one of:
      * An instance or subclass of the Reaction class, or
      * The name of a reaction from the standard library.

    Returns the ReactionContainer for the new_reaction.
    """
    if isinstance(new_reaction, Reaction) or (
            isinstance(new_reaction, type) and issubclass(new_reaction, Reaction)):
        name = str(new_reaction.name())
        if name not in reactions_dict:
            reactions_dict[name] = ReactionContainer(new_reaction, time_step)
    else:
        from  neuwon.nmodl import library, NmodlMechanism
        name = str(new_reaction)
        if name not in reactions_dict:
            if name in library:
                nmodl_file_path, kw_args = library[name]
                import neuwon.nmodl
                new_reaction = NmodlMechanism(nmodl_file_path, **kw_args)
                reactions_dict[name] = ReactionContainer(new_reaction, time_step)
                assert(name == new_reaction.name())
            else:
                raise ValueError("Unrecognized Reaction: %s."%name)
    return reactions_dict[name]

class ReactionContainer:
    """ Container to hold all instances of a type of reaction. """
    def __init__(self, reaction, time_step):
        if hasattr(reaction, "set_time_step"):
            reaction = copy.deepcopy(reaction)
            reaction.set_time_step(time_step)
        self.reaction = reaction
        self.pointers = dict(self.reaction.pointers())
        self.state = {name: [] for name, ptr in self.pointers.items() if ptr.dtype}
        self.locations = [] if self.state else None

    def append_new_mechanism(self, time_step, location, geometry, *args, **kwargs):
        if not self.state:
            raise TypeError("Reaction \"%s\" is global and so it can not be inserted."%self.reaction.name())
        self.locations.append(location)
        new_instance = self.reaction.new_instance(time_step, location, geometry, *args, **kwargs)
        for name in self.state: self.state[name].append(new_instance[name])
        assert(len(new_instance) == len(self.state))

    def _to_cuda_device(self):
        """ Move locations and state data from python lists to cuda device arrays. """
        self.locations = cp.array(self.locations, dtype=Location)
        for name, data in self.state.items():
            dtype, shape = self.pointers[name].dtype
            data = np.array(data, dtype=dtype).reshape([-1] + shape)
            self.state[name] = numba.cuda.to_device(data)

# TODO: What are the units on atol? How does the timestep factor into it?

class KineticModel:
    def __init__(self, time_step, input_ranges, states, kinetics,
        initial_state=None,
        conserve_sum=False,
        atol=1e-3):
        """
        Argument kinetics is list of tuples of (reactant, product, forward, reverse)
        """
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
    accum = scratch[index]
    # Determine which grid box the inputs are inside of.
    inpt = (inpt - input_lower_bound) * grid_factor
    lower_idx = int(math.floor(inpt))
    upper_idx = int(math.ceil(inpt))
    upper_idx = min(upper_idx, grid_size - 1) # Protect against floating point error.
    inpt -= lower_idx
    # Visit each corner of the grid box and accumulate the results.
    _weighted_matrix_vector_multiplication(
        1 - inpt, data[lower_idx], state, accum)
    _weighted_matrix_vector_multiplication(
        inpt, data[upper_idx], state, accum)
    for i in range(len(state)):
        state[i] = accum[i]

@numba.cuda.jit(device=True)
def _weighted_matrix_vector_multiplication(w, m, v, results):
    """ Computes: results += weight * (matrix * vector)

    Arguments:
        [w]eight
        [m]atrix
        [v]ector
        results - output accumulator """
    l = len(v)
    for r in range(l):
        dot = 0
        for c in range(l):
            dot += m[r, c] * v[c]
        results[r] += w * dot

@numba.cuda.jit()
def _conserve_sum(states, target_sum):
    index = numba.cuda.grid(1)
    if index >= states.shape[0]:
        return
    state = states[index]
    accumulator = 0.
    num_states = len(state)
    for i in range(num_states):
        accumulator += state[i]
    correction_factor = target_sum / accumulator
    for i in range(num_states):
        state[i] *= correction_factor

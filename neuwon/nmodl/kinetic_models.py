from scipy.linalg import expm

# TODO: What are the units on atol? How does the timestep factor into it?

# TODO: Make this accept a list of pointers instead of "input_ranges"

# TODO: Convert kinetics into a function.

# TODO: Use impulse response integration method in place of sparse solver...
#       TODO: How to dispatch to it?
#       TODO: Update the kinetic model to use Derivative function instead of sparse deriv equations.

# TODO: Write function to check that derivative_functions are Linear &
# time-invariant. Put this in the KineticModel class.



def _compile_derivative_blocks(self):
    """ Replace the derivative_blocks with compiled functions in the form:
            f(state_vector, **block.arguments) ->  Δstate_vector/Δt
    """
    self.derivative_functions = {}
    solve_statements = {stmt.block: stmt
            for stmt in self.breakpoint_block if isinstance(stmt, SolveStatement)}
    for name, block in self.derivative_blocks.items():
        if name not in solve_statements: continue
        if solve_statements[name].method == "sparse":
            self.derivative_functions[name] = self._compile_derivative_block(block)

def _compile_derivative_block(self, block):
    """ Returns function in the form:
            f(state_vector, **block.arguments) -> derivative_vector """
    block = copy.deepcopy(block)
    globals_ = {}
    locals_ = {}
    py = "def derivative(%s, %s):\n"%(code_gen.mangle2("state"), ", ".join(block.arguments))
    for idx, name in enumerate(self.states):
        py += "    %s = %s[%d]\n"%(name, code_gen.mangle2("state"), idx)
    for name in self.states:
        py += "    %s = 0\n"%code_gen.mangle('d' + name)
    block.map(lambda x: [] if isinstance(x, _ConserveStatement) else [x])
    py += block.to_python(indent="    ")
    py += "    return [%s]\n"%", ".join(code_gen.mangle('d' + x) for x in self.states)
    code_gen.py_exec(py, globals_, locals_)
    return numba.njit(locals_["derivative"])

def _compute_propagator_matrix(self, block, time_step, kwargs):
    1/0
    f = self.derivative_functions[block]
    n = len(self.states)
    A = np.zeros((n,n))
    for i in range(n):
        state = np.array([0. for x in self.states])
        state[i] = 1
        A[:, i] = f(state, **kwargs)
    return expm(A * time_step)

class KineticModel:
    def __init__(self, time_step, input_pointers, num_states, kinetics,
        conserve_sum=False,
        atol=1e-3):
        # Save and check the arguments.
        self.time_step = float(time_step)
        self.kinetics = kinetics
        self.input_ranges = np.array(input_ranges, dtype=Real)
        self.input_ranges.sort(axis=1)
        self.lower, self.upper = zip(*self.input_ranges)
        self.num_inputs = len(self.input_pointers)
        self.num_states = int(num_states)
        self.conserve_sum = float(conserve_sum) if conserve_sum else None
        self.atol = float(atol)
        assert(isinstance(self.kinetics, Callable))
        assert(len(self.input_ranges.shape) == 2 and self.input_ranges.shape[1] == 2)
        assert(self.num_inputs > 0)
        assert(self.num_states > 0)
        assert(self.atol > 0)
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


import numpy as np
import scipy.linalg

class LTI_Model:
    def __init__(self, name, inputs, states, derivative, conserve_sum, time_step):
        self.name       = str(name)
        # TODO: Don't sort these? Use the given order since I'm assuming that
        # the derivative also expects them in the given order...
        self.inputs     = sorted(inputs, key=lambda inp: inp.name)
        self.num_inputs = len(inputs)
        self.states     = sorted(str(x) for x in states)
        self.num_states = len(states)
        self.derivative = derivative
        self.conserve_sum = float(conserve_sum) if conserve_sum else None
        self.time_step  = float(time_step)
        assert self.time_step > 0.0
        # Make aliases "input1", "input2", etc.
        for idx, inp in enumerate(self.inputs):
            setattr(self, f"input{idx+1}", inp)
        self._check_is_LTI()

    def _check_is_LTI(self):
        for trial in range(3):
            inputs = [np.random.uniform(inp.minimum, inp.maximum) for inp in self.inputs]
            state1 = np.random.uniform(0.0, 1.0, size=self.num_states)
            state2 = state1 * 2.0
            d1 = self.derivative(*inputs, *state1)
            d2 = self.derivative(*inputs, *state2)
            for s1, s2 in zip(d1, d2):
                assert np.isfinite(s1) and np.isfinite(s2)
                assert abs(s1 - s2 / 2.0) < 1e-12, "Non-linear system detected!"

    def make_matrix(self, inputs, time_step=None):
        inputs = [float(input_value) for input_value in inputs]
        assert len(inputs) == len(self.inputs)
        for input_value, input_data in zip(inputs, self.inputs):
            assert input_data.minimum <= input_value <= input_data.maximum
        if time_step is None:
            time_step = self.time_step
        A = np.empty([self.num_states, self.num_states])
        for col in range(self.num_states):
            state = [float(x == col) for x in range(self.num_states)]
            A[:, col] = self.derivative(*inputs, *state)
        matrix = scipy.linalg.expm(A * time_step)
        for col in range(self.num_states):
            matrix[:, col] *= 1.0 / sum(matrix[:, col].flat)
        return matrix

    def get_initial_state(self):
        if x := getattr(self, '_initial_state_cache', None): return x
        if self.conserve_sum is None:
            initial_state   = np.zeros(self.num_states)
        else:
            time_step       = 3600e3 # 1 hour in milliseconds.
            inputs          = [inp.initial for inp in self.inputs]
            matrix          = self.make_matrix(inputs, time_step)
            valid_state     = np.full(self.num_states, self.conserve_sum / self.num_states)
            initial_state   = matrix.dot(valid_state)
        self._initial_state_cache = x = dict(zip(self.state_names, initial_state))
        return x

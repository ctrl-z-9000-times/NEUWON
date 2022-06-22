from .approx import MatrixSamples, Approx1D, Approx2D
from .codegen import Codegen
from .inputs import LinearInput, LogarithmicInput
from .polynomial import PolynomialForm
import collections.abc
import math
import numpy as np

class Parameters:
    def __init__(self, optimizer, num_buckets, polynomial, verbose=False):
        self.optimizer  = optimizer
        self.model      = optimizer.model
        self.verbose    = bool(verbose)
        if isinstance(num_buckets, collections.abc.Iterable):
            self.num_buckets = tuple(round(x) for x in num_buckets)
        else:
            self.num_buckets = (round(num_buckets),)
        self.set_num_buckets()
        # Make num_buckets1 & num_buckets2 aliases.
        for inp_idx, buckets in enumerate(self.num_buckets):
            setattr(self, f'num_buckets{inp_idx+1}', buckets)
        self.polynomial = PolynomialForm(self.model.inputs, polynomial)
        self.order      = self.polynomial.degree
        if self.verbose:
            print(f'Trying polynomial ({self.polynomial}) bins {self.num_buckets}')
        if   self.model.num_inputs == 1: ApproxClass = Approx1D
        elif self.model.num_inputs == 2: ApproxClass = Approx2D
        self.approx = ApproxClass(self.optimizer.samples, self.polynomial)
        self.num_samples = len(self.optimizer.samples)
        # Measure the accuracy of these parameters.
        self.rmse       = self.approx.rmse
        self.error      = self.approx.measure_error()
        self.max_rmse   = np.max(self.rmse)
        self.max_error  = np.max(self.error)
        if self.verbose:
            if self.max_error > self.optimizer.accuracy:
                status = 'FAIL'
            else:
                status = 'PASS'
            print(f'Result: Max |Error| {self.max_error}  \t{status}\n')

    def __str__(self):
        s = str(self.approx)
        try:
            s += f"Run speed:     {round(self.runtime, 2)}  ns/Δt\n"
        except AttributeError: pass
        s += f"# Samples:    {self.num_samples}\n"
        s += f"RMS Error:    {self.max_rmse}\n"
        if self.max_error > self.optimizer.accuracy:
            status = '- failed!'
        else:
            status = ''
        s += f"Max |Error|:  {self.max_error}  {status}\n"
        return s

    def set_num_buckets(self):
        """ Fixup shared data structures. """
        for inp, buckets in zip(self.model.inputs, self.num_buckets):
            inp.set_num_buckets(buckets)

    def benchmark(self):
        if hasattr(self, 'runtime'):
            return
        if self.verbose: print(f"Benchmarking polynomial ({self.polynomial}) bins {self.num_buckets}")
        self.set_num_buckets()
        self.backend = Codegen(
                self.approx,
                self.optimizer.float_dtype,
                self.optimizer.target)
        from . import _measure_speed # Import just before using to avoid circular imports.
        self.runtime = _measure_speed(
                self.backend.load(),
                self.model.num_states,
                self.model.inputs,
                self.model.conserve_sum,
                self.optimizer.float_dtype,
                self.optimizer.target)
        self.table_size = self.approx.table.nbytes
        if self.verbose: print(f"Result: {round(self.runtime, 3)} ns/Δt\n")

class Optimizer:
    def __init__(self, model, accuracy, float_dtype, target, verbose=False):
        self.model          = model
        self.accuracy       = float(accuracy)
        model.target_error  = self.accuracy
        self.float_dtype    = float_dtype
        self.target         = target
        self.verbose        = bool(verbose)
        self.samples        = MatrixSamples(self.model, self.verbose)
        self.best           = None # This attribute will store the optimized parameters.
        assert 0.0 < self.accuracy < 1.0
        if self.verbose: print()

    def _optimize_log_scale(self, num_buckets: [int], polynomial):
        """
        Note: use a very small and simple approximation for optimizing the log
        scale, since the goal is not to solve the problem but rather the goal
        is to identify the complicated areas of the function and scale them
        away from the edge of the view.
        """
        if not any(isinstance(inp, LogarithmicInput) for inp in self.model.inputs):
            return
        if self.verbose: print('Optimizing logarithmic scale ...')
        # Initialize the input's num_buckets and scale parameters.
        for inp, buckets in zip(self.model.inputs, num_buckets):
            if isinstance(inp, LinearInput):
                inp.set_num_buckets(buckets)
            elif isinstance(inp, LogarithmicInput):
                inp.set_num_buckets(buckets, scale=1.0)
                if self.verbose: print(f'Trying log2({inp.name} + {inp.scale})')
        # Reduce the scale parameter until the buckets containing zero no longer
        # have the largest errors.
        done = False
        while not done:
            cursor = Parameters(self, num_buckets, polynomial)
            done = True
            for dim, inp in enumerate(self.model.inputs):
                if not isinstance(inp, LogarithmicInput):
                    continue
                other_axes = tuple(x for x in range(self.model.num_inputs) if x != dim)
                errors = np.max(cursor.rmse, axis=other_axes)
                if np.argmax(errors) == 0:
                    inp.set_num_buckets(inp.num_buckets, inp.scale / 10)
                    if self.verbose: print(f'Trying log2({inp.name} + {inp.scale})')
                    done = False
        if self.verbose: print('Done optimizing logarithmic scale\n')

    def _optimize_polynomial(self, num_buckets, polynomial):
        self.best = self._optimize_num_buckets(num_buckets, polynomial)
        self.best.benchmark()
        experiments = {self.best.polynomial}
        # Try removing terms from the polynomial.
        experiment_queue = self.best.polynomial.suggest_remove()
        while experiment_queue:
            polynomial = experiment_queue.pop()
            if polynomial in experiments:
                continue
            experiments.add(polynomial)
            new = self._optimize_num_buckets(self.best.num_buckets, polynomial, self.best.runtime)
            new.benchmark()
            if new.runtime < self.best.runtime:
                self.best = new
                if self.verbose: print(f'New best: polynomial ({self.best.polynomial}) bins {self.best.num_buckets}\n')
                experiment_queue = self.best.polynomial.suggest_remove()
        # Try adding more terms to the polynomial.
        experiment_queue = self.best.polynomial.suggest_add()
        while experiment_queue:
            polynomial = experiment_queue.pop()
            if polynomial in experiments:
                continue
            experiments.add(polynomial)
            new = self._optimize_num_buckets(self.best.num_buckets, polynomial, self.best.runtime)
            new.benchmark()
            if new.runtime < self.best.runtime:
                self.best = new
                if self.verbose: print(f'New best: polynomial ({self.best.polynomial}) bins {self.best.num_buckets}\n')
                experiment_queue = self.best.polynomial.suggest_add()
        self.best.set_num_buckets()

class Optimize1D(Optimizer):
    def __init__(self, model, accuracy, float_dtype, target, verbose=False):
        super().__init__(model, accuracy, float_dtype, target, verbose)
        self.input1 = self.model.input1
        # Initial parameters, starting point for iterative search.
        num_buckets = [10]
        polynomial  = 3
        # Run the optimization routines.
        self._optimize_log_scale(num_buckets, polynomial)
        self._optimize_polynomial(num_buckets, polynomial)
        # Re-make the final product using all available samples.
        if self.best.num_samples < len(self.samples):
            if self.verbose: print('Remaking best approximation with more samples ...\n')
            self.best = Parameters(self, self.best.num_buckets, self.best.polynomial)
            self.best.benchmark()

    def _optimize_num_buckets(self, num_buckets, polynomial, max_runtime=None):
        (num_buckets,) = num_buckets
        cursor = Parameters(self, num_buckets, polynomial, self.verbose)
        min_buckets = 1
        # Quickly increase the num_buckets until it exceeds the target accuracy.
        while cursor.max_error > self.accuracy:
            # Terminate early if it's slower than max_runtime.
            if max_runtime is not None and cursor.num_buckets1 > 1000:
                cursor.benchmark()
                if cursor.runtime > max_runtime:
                    if self.verbose: print(f'Aborting Polynomial ({cursor.polynomial}) runs too slow.\n')
                    return cursor # It's ok to return invalid results BC they won't be used.
            min_buckets = cursor.num_buckets1
            # Heuristics to guess new num_buckets.
            orders_of_magnitude = math.log(cursor.max_error / self.accuracy, 10)
            pct_incr = max(1.5, 1.7 ** orders_of_magnitude)
            num_buckets = num_buckets * pct_incr
            new = Parameters(self, num_buckets, polynomial, self.verbose)
            # Check that the error is decreasing monotonically.
            if new.max_error < cursor.max_error:
                cursor = new
            else:
                raise RuntimeError("Failed to reach target accuracy.")
        # Slowly reduce the num_buckets until it fails to meet the target accuracy.
        while True:
            num_buckets *= 0.9
            if num_buckets <= min_buckets:
                break
            new = Parameters(self, num_buckets, polynomial, self.verbose)
            if new.max_error > self.accuracy:
                break
            else:
                cursor = new
        return cursor

class Optimize2D(Optimizer):
    def __init__(self, model, accuracy, float_dtype, target, verbose=False):
        super().__init__(model, accuracy, float_dtype, target, verbose)
        self._optimize_log_scale([10, 10],
                [[0, 0], [1, 0], [0, 1], [2, 0], [0, 2], [3, 0], [0, 3],])
        self._optimize_polynomial([20, 20],
                [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [3, 0], [0, 3]])
        # Re-make the final product using all available samples.
        if self.best.num_samples < len(self.samples):
            if self.verbose: print('Remaking best approximation with more samples ...\n')
            self.best = Parameters(self, self.best.num_buckets, self.best.polynomial)
            self.best.benchmark()

    def _optimize_num_buckets(self, num_buckets, polynomial, max_runtime=None):
        cursor = Parameters(self, num_buckets, polynomial, self.verbose)
        # Quickly increase the num_buckets until it exceeds the target accuracy.
        increase = lambda x: x * 1.50
        while cursor.max_error > self.accuracy:
            # Terminate early if it's already slower than max_runtime.
            if max_runtime is not None and np.product(cursor.num_buckets) > 1000:
                cursor.benchmark()
                if cursor.runtime > max_runtime:
                    if self.verbose: print(f'Aborting Polynomial ({cursor.polynomial}), runs too slow.\n')
                    return cursor # It's ok to return invalid results BC they won't be used.
            # Try increasing num_buckets in both dimensions in isolation.
            if self.verbose: print(f'Increasing {self.model.input1.name} bins:')
            A = Parameters(self, [increase(cursor.num_buckets1), cursor.num_buckets2], polynomial, self.verbose)
            if self.verbose: print(f'Increasing {self.model.input2.name} bins:')
            B = Parameters(self, [cursor.num_buckets1, increase(cursor.num_buckets2)], polynomial, self.verbose)
            # Take whichever experiment yielded better results. If they both
            # performed about the same then take both modifications.
            pct_diff = 2 * (A.max_error - B.max_error) / (A.max_error + B.max_error)
            thresh   = .25
            if pct_diff < -thresh:
                new = A
                if self.verbose: print(f'Taking increased {self.model.input1.name} bins.\n')
            elif pct_diff > thresh:
                new = B
                if self.verbose: print(f'Taking increased {self.model.input2.name} bins.\n')
            else:
                if self.verbose: print('Taking the increases in both dimensions:')
                new = Parameters(self, [increase(cursor.num_buckets1), increase(cursor.num_buckets2)],
                                polynomial, self.verbose)
            # Check that the error is decreasing monotonically.
            if new.max_error < cursor.max_error:
                cursor = new
            else:
                raise RuntimeError("Failed to reach target accuracy.")
        # Slowly reduce the num_buckets until it fails to meet the target accuracy.
        decrease = lambda x: x * .90
        while True:
            # Try decreasing num_buckets in both dimensions in isolation.
            if self.verbose: print(f'Decreasing {self.model.input1.name} bins.')
            A = Parameters(self, [decrease(cursor.num_buckets1), cursor.num_buckets2], polynomial, self.verbose)
            if self.verbose: print(f'Decreasing {self.model.input2.name} bins.')
            B = Parameters(self, [cursor.num_buckets1, decrease(cursor.num_buckets2)], polynomial, self.verbose)
            new = min(A, B, key=lambda p: p.max_error)
            if new.max_error > self.accuracy:
                break
            else:
                cursor = new
        return cursor

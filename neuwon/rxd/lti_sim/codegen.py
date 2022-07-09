"""
Backend for generating the run-time program, compiling it, and loading it into python.
"""

from .inputs import LinearInput, LogarithmicInput
from neuwon.database import Real, Database, Compute
from neuwon.rxd.nmodl.code_gen import exec_string
import numpy as np
import time

class Codegen:
    def __init__(self, table, target):
        self.model          = table.model
        self.table          = table.table
        self.polynomial     = table.polynomial
        self.target         = target
        self.name           = self.model.name
        self.inputs         = self.model.inputs
        self.input_names    = self.model.input_names
        self.state_names    = self.model.state_names
        self.num_states     = self.model.num_states
        self.conserve_sum   = self.model.conserve_sum
        self.initial_state  = self.model.get_initial_state()
        assert self.target in ('host', 'cuda')
        self.table_data = self._table_data()
        self.source_code = self._kernel()
        self.runtime = self._measure_speed()

    def _table_data(self):
        # Check table size matches what's expected.
        table_size = (self.num_states ** 2) * (self.polynomial.num_terms)
        for inp in self.inputs: table_size *= inp.num_buckets
        assert self.table.size == table_size
        # Flatten the input dimensions.
        table_data = self.table.reshape(-1, self.num_states, self.num_states, self.polynomial.num_terms)
        table_data = table_data.transpose(0, 2, 1, 3) # Switch from row-major to column-major format.
        return np.array(table_data, dtype=Real, order='C').reshape(-1)

    def _kernel(self):
        c  =   "@Compute\n"
        c += (f"def {self.name}_advance(self):\n"
              "    # Locate the input within the look-up table.\n")
        for idx, inp in enumerate(self.inputs):
            if isinstance(inp, LinearInput):
                c += f"    input{idx} = ({inp.db_access} - {inp.minimum}) * {inp.bucket_frq}\n"
            elif isinstance(inp, LogarithmicInput):
                c += f"    input{idx} = log2({inp.db_access} + {inp.scale})\n"
                c += f"    input{idx} = (input{idx} - {inp.log2_minimum}) * {inp.bucket_frq}\n"
            else: raise NotImplementedError(type(inp))
            c += (f"    bucket{idx} = int(input{idx})\n"
                  f"    if(bucket{idx} > {inp.num_buckets - 1}):\n"
                  f"        bucket{idx} = {inp.num_buckets - 1}\n"
                  f"        input{idx} = 1.0\n")
            if not isinstance(inp, LogarithmicInput): # What could go wrong? It's not like a chemical concentration will ever go negative, right?
                c += (f"    elif(bucket{idx} < 0):\n"
                      f"        bucket{idx} = 0\n"
                      f"        input{idx} = 0.0\n")
            c += ("    else:\n"
                 f"        input{idx} = input{idx} - bucket{idx}\n")
        nd_index = []
        stride = 1
        for inp_idx, inp in reversed(list(enumerate(self.inputs))):
            if stride == 1: nd_index.append(f"bucket{inp_idx}")
            else:           nd_index.append(f"bucket{inp_idx} * {stride}")
            stride *= inp.num_buckets
        c += (f"    tbl_ptr = {self.num_states**2 * (self.polynomial.num_terms)} * ({' + '.join(nd_index)})\n"
               "    # Compute the basis of the polynomial.\n")
        for term_idx, powers in enumerate(self.polynomial.terms):
            factors = []
            for inp_idx, power in enumerate(powers):
                factors.extend([f"input{inp_idx}"] * power)
            if factors:
                c += f"    term{term_idx} = {' * '.join(factors)}\n"
        c += ("    # Compute the dot product into the scratch buffer.\n"
             f"    state:   'Alloc({self.num_states}, Real)'\n")
        for i, x in enumerate(self.state_names):
            c += f"    state[{i}] = self.{x}\n"
        c += f"    scratch: 'Alloc({self.num_states}, Real)'\n"
        for i in range(self.num_states):
            c += f"    scratch[{i}] = 0.0\n"
        c += (f"    for col in range({self.num_states}):\n"
               "        s = state[col]\n"
              f"        for row in range({self.num_states}):\n"
               "            # Approximate this entry of the matrix.\n")
        terms = []
        c += "            polynomial = 0.0\n"
        for term_idx, powers in enumerate(self.polynomial.terms):
            if not any(p > 0 for p in powers):
                c += "            polynomial += table[tbl_ptr]\n"
                c += "            tbl_ptr += 1\n"
            else:
                c += f"            polynomial += term{term_idx} * table[tbl_ptr]\n"
                c += "            tbl_ptr += 1\n"
        c += ("            scratch[row] += polynomial * s\n")
        if self.conserve_sum is not None:
            c += ("    # Conserve the sum of the states.\n"
                  "    sum_states = 0.0\n"
                 f"    for x in range({self.num_states}):\n"
                  "        sum_states += scratch[x]\n"
                 f"    correction_factor = {self.conserve_sum} / sum_states\n"
                 f"    for x in range({self.num_states}):\n"
                  "        scratch[x] *= correction_factor\n")
        c += "    # Move the results into the state arrays.\n"
        for i, x in enumerate(self.state_names):
            c += f"    self.{x} = scratch[{i}]\n"
        print(c)
        return c

    def load(self):
        if fn := getattr(self, "_load_cache", False): return fn
        fn_name  = self.name + "_advance"
        scope    = {"numpy": np, "Compute": Compute, "table": self.table_data}
        exec_string(self.source_code, scope)
        self._load_cache = fn = scope[fn_name]
        return fn

    def _make_mock_database(self):
        """ Make a mock-up of the database for running the benchmark. """
        database = Database()
        model_class = database.add_class(self.name)
        for x in self.state_names:
            model_class.add_attribute(x, initial_distribution=('uniform', 0, 1))
        for inp in self.inputs:
            obj, ptr, attr = inp.db_access.split('.')
            assert obj == 'self'
            inp_class = database.add_class(ptr.title())
            inp_class.add_attribute(attr)
            model_class.add_attribute(ptr, dtype=inp_class)
        model_inst = model_class.get_instance_type()
        model_inst.advance = model_class.add_method(self.load())
        return database

    def _populate_mock_database(self, database, num_instances):
        xp = database.get_array_module()
        # Make instances of each class type.
        for db_class in database.get_all_classes():
            inst_type = db_class.get_instance_type()
            for _ in range(num_instances):
                inst_type()
        # Link the model instances to their locations.
        for inp in self.inputs:
            obj, ptr, attr = inp.db_access.split('.')
            database.set_data(f'{self.name}.{ptr}', xp.arange(num_instances))
        # Generate valid initial states.
        state = [database.get_data(f'{self.name}.{x}') for x in self.state_names]
        if self.conserve_sum is not None:
            sum_states = xp.zeros(num_instances)
            for array in state:
                sum_states = sum_states + array
            correction_factor = float(self.conserve_sum) / sum_states
            for array in state:
                array *= correction_factor

    def _randomize_inputs(self, database):
        for inp in self.inputs:
            obj, ptr, attr = inp.db_access.split('.')
            inp_class = database.get_class(ptr.title())
            inp_class.set_data(attr, inp.random(len(inp_class), Real, database.get_array_module()))

    def _measure_speed(self):
        num_instances = 10 * 1000
        num_repetions = 200
        database = self._make_mock_database()
        with database.using_memory_space(self.target):
            self._populate_mock_database(database, num_instances)
            # 
            if self.target == 'cuda':
                start_event = cupy.cuda.Event()
                end_event   = cupy.cuda.Event()
            model = database.get_instance_type(self.name)
            elapsed_times = np.empty(num_repetions)
            for trial in range(num_repetions):
                self._randomize_inputs(database)
                if trial < 3: database.check()
                _clear_cache(database.get_array_module())
                time.sleep(0) # Try to avoid task switching while running.
                if self.target == 'cuda':
                    start_event.record()
                    model.advance()
                    end_event.record()
                    end_event.synchronize()
                    elapsed_times[trial] = 1e6 * cupy.cuda.get_elapsed_time(start_event, end_event)
                elif self.target == 'host':
                    start_time = time.thread_time_ns()
                    model.advance()
                    elapsed_times[trial] = time.thread_time_ns() - start_time
            return np.min(elapsed_times) / num_instances

def _clear_cache(array_module):
    # Read and then write back 32MB of data. Assuming that the CPU is using a
    # least-recently-used replacement policy, touching every piece of data once
    # should be sufficient to put it into the cache.
    big_data = array_module.empty(int(32e6 / 8), dtype=np.int64)
    big_data += 1

"""
Backend for generating the run-time program, compiling it, and loading it into python.
"""

from .inputs import LinearInput, LogarithmicInput
import ctypes
import numpy as np
import os
import subprocess
import tempfile

class Codegen:
    def __init__(self, table, float_dtype, target):
        self.model          = table.model
        self.table          = table.table
        self.polynomial     = table.polynomial
        self.float_dtype    = float_dtype
        self.target         = target
        self.name           = self.model.name
        self.inputs         = self.model.inputs
        self.input_names    = self.model.input_names
        self.state_names    = self.model.state_names
        self.num_states     = self.model.num_states
        self.conserve_sum   = self.model.conserve_sum
        self.initial_state  = self.model.get_initial_state()
        assert self.num_states >= 0
        assert self.float_dtype in (np.float32, np.float64)
        assert self.target in ('host', 'cuda')
        self.source_code = (
                self._preamble() +
                self._table_data() +
                self._kernel() +
                self._entrypoint())

    def _preamble(self):
        c = (f"/{'*'*69}\n"
             f"Model Name   : {self.name}\n"
             f"Filename     : {self.model.nmodl_filename}\n"
             f"Time Step    : {self.model.time_step} ms\n"
             f"Temperature  : {self.model.temperature} C\n"
             f"Max Error    : {getattr(self.model, 'target_error', None)}\n"
             f"Target       : {self.float_dtype.__name__} {self.target}\n"
             f"Polynomial   : {self.polynomial}\n"
             f"{'*'*69}/\n\n")
        if self.target == 'host':
            if any(isinstance(inp, LogarithmicInput) for inp in self.inputs):
                c += "#include <math.h>       /* log2 */\n\n"
        if self.float_dtype == np.float32:
            c +=  "typedef float real;\n\n"
        elif self.float_dtype == np.float64:
            c +=  "typedef double real;\n\n"
        else: raise NotImplementedError(self.float_dtype)
        return c

    def _table_data(self):
        # Check table size matches what's expected.
        table_size = (self.num_states ** 2) * (self.polynomial.num_terms)
        for inp in self.inputs: table_size *= inp.num_buckets
        assert self.table.size == table_size
        # Flatten the input dimensions.
        table_data = self.table.reshape(-1, self.num_states, self.num_states, self.polynomial.num_terms)
        table_data = table_data.transpose(0, 2, 1, 3) # Switch from row-major to column-major format.
        table_data = np.array(table_data, dtype=self.float_dtype)
        data_str   = ',\n    '.join((','.join(str(x) for x in bucket.flat)) for bucket in table_data)
        if self.target == 'cuda':
            device = '__device__ '
        elif self.target == 'host':
            device = ''
        return f"{device}const real table[{table_data.size}] = {{\n    {data_str}}};\n\n"

    def _entrypoint(self):
        c = 'extern "C" '
        if self.target == 'cuda':
            c += "__global__ "
        c += f"void {self.name}_advance(int n_inst, "
        for inp in self.input_names:
            c += f"real* {inp}, int* {inp}_indices, "
        c +=  ", ".join(f"real* {state}" for state in self.state_names)
        c +=  ") {\n"
        if self.target == 'host':
            c +=  "    for(int index = 0; index < n_inst; ++index) {\n"
        elif self.target == 'cuda':
            c += ("    const int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
                  "    if( index >= n_inst ) { return; }\n")
        access_inputs = ', '.join(f"{inp}[{inp}_indices[index]]" for inp in self.input_names)
        access_states = ', '.join(f"{state} + index" for state in self.state_names)
        c += (f"        real* state[{self.num_states}] = {{{access_states}}};\n"
              f"        {self.name}_kernel({access_inputs}, state);\n")
        if self.target == 'host':
            c +=  "    }\n"
        c +=  "}\n\n"
        return c

    def _kernel(self):
        c = ""
        if self.target == 'cuda':
            c += "__device__ "
        c += f"__inline__ void {self.name}_kernel("
        for idx in range(len(self.inputs)):
            c += f"real input{idx}, "
        c += (f"real* state[{self.num_states}]) {{\n"
               "    const real* __restrict__ tbl_ptr = table;\n"
               "    // Locate the input within the look-up table.\n")
        for idx, inp in enumerate(self.inputs):
            if isinstance(inp, LinearInput):
                c += f"    input{idx} = (input{idx} - {inp.minimum}) * {inp.bucket_frq};\n"
            elif isinstance(inp, LogarithmicInput):
                if self.target == 'host':
                    c += f"    input{idx} = log2(input{idx} + {inp.scale});\n"
                elif self.target == 'cuda':
                    if self.float_dtype == np.float32:
                        c += f"    input{idx} = log2f(input{idx} + {inp.scale});\n"
                    elif self.float_dtype == np.float64:
                        c += f"    input{idx} = log2(input{idx} + {inp.scale});\n"
                    else: raise NotImplementedError(self.float_dtype)
                else: raise NotImplementedError(self.target)
                c += f"    input{idx} = (input{idx} - {inp.log2_minimum}) * {inp.bucket_frq};\n"
            else: raise NotImplementedError(type(inp))
            c += (f"    int bucket{idx} = (int) input{idx};\n"
                  f"    if(bucket{idx} > {inp.num_buckets - 1}) {{\n"
                  f"        bucket{idx} = {inp.num_buckets - 1};\n"
                  f"        input{idx} = 1.0;\n"
                   "    }\n")
            if not isinstance(inp, LogarithmicInput): # What could go wrong? It's not like a chemical concentration will ever go negative, right?
                c += (f"    else if(bucket{idx} < 0) {{\n"
                      f"        bucket{idx} = 0;\n"
                      f"        input{idx} = 0.0;\n"
                       "    }\n")
            c += ("    else {\n"
                 f"        input{idx} = input{idx} - bucket{idx};\n"
                  "    }\n")
        nd_index = []
        stride = 1
        for inp_idx, inp in reversed(list(enumerate(self.inputs))):
            if stride == 1: nd_index.append(f"bucket{inp_idx}")
            else:           nd_index.append(f"bucket{inp_idx} * {stride}")
            stride *= inp.num_buckets
        c += (f"    const int bucket = {' + '.join(nd_index)};\n"
              f"    tbl_ptr += bucket * {self.num_states**2 * (self.polynomial.num_terms)};\n"
               "    // Compute the basis of the polynomial.\n")
        for term_idx, powers in enumerate(self.polynomial.terms):
            factors = []
            for inp_idx, power in enumerate(powers):
                factors.extend([f"input{inp_idx}"] * power)
            if factors:
                c += f"    const real term{term_idx} = {' * '.join(factors)};\n"
        c += ("\n"
             f"    real scratch[{self.num_states}] = {{0.0}};\n"
             f"    for(int col = 0; col < {self.num_states}; ++col) {{\n"
              "        const real s = *state[col];\n"
             f"        for(int row = 0; row < {self.num_states}; ++row) {{\n"
              "            // Approximate this entry of the matrix.\n")
        terms = []
        for term_idx, powers in enumerate(self.polynomial.terms):
            if any(p > 0 for p in powers):
                terms.append(f"term{term_idx} * (*tbl_ptr++)")
            else:
                terms.append("(*tbl_ptr++)")
        c += (f"            const real polynomial = {' + '.join(terms)};\n"
               "            scratch[row] += polynomial * s; // Compute the dot product. \n"
               "        }\n"
               "    }\n")
        if self.conserve_sum is not None:
            c += ("    // Conserve the sum of the states.\n"
                  "    real sum_states = 0.0;\n"
                 f"    for(int x = 0; x < {self.num_states}; ++x) {{\n"
                  "        sum_states += scratch[x];\n"
                  "    }\n"
                 f"    const real correction_factor = {self.conserve_sum} / sum_states;\n"
                 f"    for(int x = 0; x < {self.num_states}; ++x) {{\n"
                  "        scratch[x] *= correction_factor;\n"
                  "    }\n")
        c += ("    // Move the results into the state arrays.\n"
             f"    for(int x = 0; x < {self.num_states}; ++x) {{\n"
              "        *state[x] = scratch[x];\n"
              "    }\n"
              "}\n\n")
        return c

    def write(self, filename=None):
        if filename is None or filename is True:
            nmodl_filename = os.path.basename(self.model.nmodl_filename)
            nmodl_filename = os.path.splitext(nmodl_filename)[0]
            if self.target == 'cuda':
                self.filename = nmodl_filename + '.cu'
            elif self.target == 'host':
                self.filename = nmodl_filename + '.cpp'
        else:
            self.filename = str(filename)
        self.filename = os.path.abspath(self.filename)
        with open(self.filename, 'wt') as f:
            f.write(self.source_code)
            f.flush()

    def load(self):
        if fn := getattr(self, "_load_cache", False): return fn
        fn_name  = self.name + "_advance"
        real_t   = "numpy." + self.float_dtype.__name__
        index_t  = "numpy.int32"
        scope    = {"numpy": np}
        arrays = [] # Name of input
        dtypes = [] # Name of numpy dtype
        maxlen = [] # Relation to number of instance (n_inst)
        for inp_name in self.input_names:
            arrays.append(inp_name)
            dtypes.append(real_t)
            maxlen.append(">=")
            arrays.append(inp_name+"_indices")
            dtypes.append(index_t)
            maxlen.append("==")
        arrays.extend(self.state_names)
        dtypes.extend([real_t] * self.num_states)
        maxlen.extend(["=="]   * self.num_states)
        pycode = (f"def {fn_name}(n_inst, {', '.join(arrays)}):\n"
                   "    n_inst = int(n_inst)\n")
        if self.target == 'host':
            for x in arrays:
                pycode += f"    assert isinstance({x}, numpy.ndarray), 'isinstance({x}, numpy.ndarray)'\n"
        for x, dt in zip(arrays, dtypes):
            pycode += f"    assert {x}.dtype == {dt}, '{x}.dtype == {dt}'\n"
        for x, op in zip(arrays, maxlen):
            pycode += f"    assert len({x}) {op} n_inst, 'len({x}) {op} n_inst'\n"
        if self.target == 'host':
            pycode += f"    _entrypoint(n_inst, {', '.join(arrays)})\n"
            scope["_entrypoint"] = self._load_entrypoint_host()
        elif self.target == 'cuda':
            pycode += ("    threads = 32\n"
                       "    blocks = (n_inst + (threads - 1)) // threads\n"
                      f"    _entrypoint((blocks,), (threads,), (n_inst, {', '.join(arrays)}))\n")
            scope["_entrypoint"] = self._load_entrypoint_cuda()
        exec(pycode, scope)
        self._load_cache = fn = scope[fn_name]
        return fn

    def _load_entrypoint_host(self):
        src_file = tempfile.NamedTemporaryFile(prefix=self.name+'_', suffix='.cpp', delete=False)
        so_file  = tempfile.NamedTemporaryFile(prefix=self.name+'_', suffix='.so', delete=False)
        src_file.close(); so_file.close()
        self.write(src_file.name)
        subprocess.run(["g++", src_file.name, "-o", so_file.name,
                        "-shared", "-O3"],
                        check=True)
        so = ctypes.CDLL(so_file.name)
        fn = so[self.name + "_advance"]
        argtypes = [ctypes.c_int]
        for _ in self.inputs:
            argtypes.append(np.ctypeslib.ndpointer(dtype=self.float_dtype, ndim=1, flags='C'))
            argtypes.append(np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C'))
        for _ in range(self.num_states):
            argtypes.append(np.ctypeslib.ndpointer(dtype=self.float_dtype, ndim=1, flags='C'))
        fn.argtypes = argtypes
        fn.restype = None
        os.remove(src_file.name)
        os.remove(so_file.name)
        return fn

    def _load_entrypoint_cuda(self):
        import cupy
        fn_name = self.name + "_advance"
        module = cupy.RawModule(code=self.source_code,
                                name_expressions=[fn_name],
                                options=('--std=c++11',),)
        return module.get_function(fn_name)

"""
All memory spaces are global singletons.
"""

import cupy
import cupyx.scipy.sparse
import numpy
import scipy.sparse
import numba
import numba.cuda

available = dict()

class MemorySpace:
    def __init__(self, name, array_module, matrix_module, jit_wrapper):
        self.name = str(name)
        available[self.name] = self
        self.matrix_module   = matrix_module
        self.array_module    = array_module
        self.array           = array_module.array
        self.jit_wrapper     = jit_wrapper

    def __repr__(self):
        return f"<MemorySpace: {self.name}>"

    def get_name(self) -> str:  return self.name
    def get_array_module(self): return self.array_module
    def get_matrix_module(self):return self.matrix_module

host = MemorySpace("host", numpy, scipy.sparse, numba.njit)
cuda = MemorySpace("cuda", cupy, cupyx.scipy.sparse, numba.cuda.jit)

class ContextManager:
    def __init__(self, database, memory_space):
        self.database = database
        if isinstance(memory_space, MemorySpace):
            self.memory_space = memory_space
        else:
            self.memory_space = available[str(memory_space)]

    def __enter__(self):
        self.prior_memory_space    = self.database.memory_space
        self.database.memory_space = self.memory_space

    def __exit__(self, exception_type, exception, traceback):
        self.database.memory_space = self.prior_memory_space

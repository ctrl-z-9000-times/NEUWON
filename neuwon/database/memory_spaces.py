"""
All memory spaces are global singletons.
"""

import cupy
import cupyx.scipy.sparse
import numpy
import scipy.sparse

available = dict()

class MemorySpace:
    def __init__(self, name, array_module, matrix_module):
        self.name = str(name)
        self.array_module = array_module
        self.matrix_module = matrix_module
        self.array = self.array_module.array
        available[self.name] = self

    def __repr__(self):
        return f"<MemorySpace: {self.name}>"

host = MemorySpace("host", numpy, scipy.sparse)
cuda = MemorySpace("cuda", cupy, cupyx.scipy.sparse)

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

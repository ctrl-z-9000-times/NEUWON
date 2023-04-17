import enum
import numba
import numpy
import scipy.sparse
try:
    import cupy
    import cupyx.scipy.sparse as cupyx_scipy_sparse
    import numba.cuda as numba_cuda
except ModuleNotFoundError:
    cupy = None
    cupyx_scipy_sparse = None
    numba_cuda = None

class MemorySpace(enum.Enum):
    host = (numpy, scipy.sparse, numba)
    cuda = (cupy, cupyx_scipy_sparse, numba_cuda)

    def __init__(self, array_module, matrix_module, jit_module):
        self.array_module    = array_module
        self.array           = array_module.array if self.array_module else None
        self.matrix_module   = matrix_module
        self.jit_module      = jit_module

    def __repr__(self):
        return f"<MemorySpace.{self.name}>"

    def __bool__(self):
        return self.array_module is not None

    def get_name(self) -> str:  return self.name
    def get_array_module(self): return self.array_module
    def get_matrix_module(self):return self.matrix_module

host = MemorySpace.host
cuda = MemorySpace.cuda

class ContextManager:
    def __init__(self, database, memory_space):
        self.database = database
        if isinstance(memory_space, MemorySpace):
            self.memory_space = memory_space
        else:
            self.memory_space = MemorySpace[str(memory_space)]
        if self.memory_space.array_module is None:
            raise ModuleNotFoundError("cupy or cupyx not found.")

    def __enter__(self):
        self.prior_memory_space    = self.database.memory_space
        self.database.memory_space = self.memory_space

    def __exit__(self, exception_type, exception, traceback):
        self.database.memory_space = self.prior_memory_space

def get_array_module(arg):
    if 'numpy' in arg.__class__.__module__:
        return MemorySpace.host.array_module
    elif 'cupy' in arg.__class__.__module__:
        return MemorySpace.cuda.array_module


class MemorySpace:
    def __init__(self):
        self.name = None
        self.array_module = None
        self.sparse_matrix_module = None

    def __repr__(self):
        return f"<MemorySpace: {self.name}>"

host = MemorySpace()
host.name = "host"
host.array_module = numpy
host.sparse_matrix_module = scipy.sparse

cuda = MemorySpace()
cuda.name = "cuda"
cuda.array_module = cupy
cuda.sparse_matrix_module = cupyx.scipy.sparse

class ContextManager:
    def __init__(self, database, memory_space):
        memory_space = str(memory_space)
        assert memory_space in ['host', 'cuda']
        self.database     = database
        self.memory_space = memory_space

    def __enter__(self):
        self.prior_memory_space    = self.database.memory_space
        self.database.memory_space = memory_space

    def __exit__(self):
        self.database.memory_space = self.prior_memory_space


""" A custom Entity-Component-System for NEUWON.

Most data is stored in the Database class, as a pair of (name & value). The
names of the data are significant, the archetype to which a component belongs is
embeded in the name as:
    "archetype_name/component_name"
"""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
import cupy
import textwrap
from collections.abc import Callable, Iterable, Mapping

Real = np.dtype('f4')
epsilon = np.finfo(Real).eps
Index = np.dtype('u4')
NULL = np.iinfo(Index).max

separator = "/"

class Database:
    def __init__(self):
        self.archetypes = {}
        self.contents = {}

    def add_archetype(self, name: str, doc: str = ""):
        """ Create a new type of entity. """
        name = str(name)
        assert(name not in self.archetypes)
        self.archetypes[name] = _Archetype(doc)

    def add_global_constant(self, name: str, value: float, doc: str = "", check=True):
        """ Add a singular floating-point value. """
        name = str(name)
        assert(name not in self.contents)
        self.contents[name] = _Value(value, doc=doc, check=check)

    def add_function(self, name, function, doc: str = ""):
        """ Add a callable function. """
        name = str(name)
        assert(name not in self.contents)
        self.contents[name] = _Function(function, doc)

    def add_component(self, name: str, doc: str = "",
            dtype=Real, shape=(1,), initial_value=None,
            user_read=False, user_write=False, check=True):
        """ Add a new data array to an Archetype.

        Argument dtype:
            if dtype is a string then it is a reference to an entity.
            Note: Reactions are not run in a deterministic order. Dynamics which
            span between reactions via reaction_references should operate at a
            significantly slower time scale than the time step.

        Argument shape:
        Argument initial_value:
        Argument user_read, user_write:
        Argument check:
        """
        name = str(name)
        assert(name not in self.contents)
        archetype, component = name.split(separator, maxsplit=1)
        assert(archetype in self.archetypes)
        self.contents[name] = arr = _Array(self.archetypes[archetype], doc=doc,
            dtype=dtype, shape=shape, initial_value=initial_value,
            user_read=user_read, user_write=user_write, check=check)
        if arr.reference:
            assert(arr.reference in self.archetypes)
            self.archetypes[arr.reference].referenced_by.append(arr)

    def add_csr_matrix(self, name, column_archetype, doc="",
            dtype=Real, user_read=False, check=True):
        """ Add a compressed sparse row matrix. """
        name = str(name)
        assert(name not in self.contents)
        archetype, component = name.split(separator, maxsplit=1)
        assert(archetype in self.archetypes)
        assert(column_archetype in self.archetypes)
        self.contents[name] = arr = _Array(self.archetypes[archetype], doc=doc,
            dtype=dtype, initial_value=0, shape="sparse", column_archetype=column_archetype,
            user_read=user_read, user_write=False, check=check)
        if arr.reference:
            assert(arr.reference in self.archetypes)
            self.archetypes[arr.reference].referenced_by.append(arr)

    def add_linear_system(self, name: str, function, epsilon, doc: str = "", check=True):
        """ Add a system of linear & time-invariant differential equations.

        Argument function(database_access) -> coefficients

        For equations of the form: dX/dt = C * X
        Where X is a component, of the same archetype as this linear system.
        Where C is a matrix of coefficients, returned by the argument "function".

        The database computes the propagator matrix but does not apply it.
        The matrix is updated after any of the entity are created or destroyed.
        """
        name = str(name)
        assert(name not in self.contents)
        archetype, component = name.split(separator, maxsplit=1)
        assert(archetype in self.archetypes)
        self.contents[name] = sys = _LinearSystem(function, doc=doc, epsilon=epsilon, check=check)
        self.archetypes[archetype].linear_systems.append(sys)

    def create_entity(self, archetype: str, number_of_instances: int = 1) -> list:
        """ Create instances of an archetype.
        Returns their internal ID's. """
        ark = self.archetypes[str(archetype)]
        num = int(number_of_instances); assert(num >= 0)
        old_size = ark.size
        new_size = old_size + num
        ark.size = new_size
        for arr in ark.components: arr.append_entities(old_size, new_size)
        for sys in ark.linear_systems: sys.up_to_date = False
        return range(old_size, new_size)

    def destroy_entity(self, archetype: str, instances: list):
        archetype = str(archetype)
        assert(archetype in self.archetypes)
        1/0 # TODO Recursively mark all destroyed instances, make a bitmask for aliveness.
        1/0 # TODO Compress the dead entries out of all data arrays.
        1/0 # TODO Update references.

        # Note: I should give the user control over which references are
        # optional and can safely be overwritten with NULL, versus the
        # references which trigger a recursive destruction. 
        # Sparse matrixes implicitly represent NULL.
        # Add another flag? Or maybe key it off of the "check".

    def num_entity(self, archetype: str):
        return self.archetypes[str(archetype)].size

    def access(self, name: str):
        """ Returns a components value or GPU data array. """
        x = self.contents[str(name)]
        if isinstance(x, _Value): return x.value
        if isinstance(x, _Function): return x.function
        elif isinstance(x, _Array):
            if x.shape == "sparse": return x.data
            else: return x.data[:x.archetype.size]
        elif isinstance(x, _LinearSystem):
            if not x.up_to_date: x.compute(self)
            return x.data

    def write_row(self, name, rows, columns, data):
        """ Zero and overwrite rows in a sparse matrix. """
        x = self.contents[str(name)]
        assert(isinstance(x, _Array) and x.shape == "sparse")
        1/0 # TODO !!!

    def invalidate_linear_systems(self, archetype: str):
        ark = self.archetypes[str(archetype)]
        for sys in ark.linear_systems: sys.up_to_date = False

    def check(self):
        for name, x in self.contents.items():
            x.check(name, self)

    def __repr__(self) -> str:
        s = ""
        # TODO: Print function listing & summarys at the top.
        # TODO: Print the global constants which are NOT associated with a component.
        for ark_name, ark in sorted(self.archetypes.items()):
            s += "=" * 80 + "\n"
            s += "Archetype %s (%d)\n"%(ark_name, ark.size)
            if ark.doc: s += textwrap.indent(ark.doc, "    ") + "\n"
            s += "\n"
            ark_prefix = ark_name + separator
            for comp_name, comp in sorted(self.contents.items()):
                if not comp_name.startswith(ark_prefix): continue
                # TODO: Print components doc strings.
                if isinstance(comp, _Value):
                    s += "Component: " + comp_name + " = " + str(comp.value) + "\n"
                elif isinstance(comp, _Array):
                    s += "Component: " + comp_name
                    if comp.reference:
                        s += ", reference"
                    else:
                        s += ", " + str(comp.dtype)
                    # TODO: Print all of the other flags too.
                    s += " " + str(comp.shape)
                    s += "\n"
                # TODO: For sparse matrixes: print the average num-non-zero per row.
                elif isinstance(comp, _LinearSystem):
                    1/0
        return s

class EntityHandle:
    """ A persistent handle on a newly created entity.

    By default, the identifiers returned by create_entity are only valid until
    the next call to destroy_entity, which moves where the entities are located.
    This class tracks where an entity gets moved to, and provides a consistent
    API for accessing the entity data. """
    def __init__(self, database, entity, index):
        self.database = database
        1/0

    def __del__(self):
        """ Unregister this from the model. """
        1/0

    def read(self, component):
        1/0

    def write(self, component):
        1/0

class _Archetype:
    def __init__(self, doc):
        self.doc = textwrap.dedent(str(doc)).strip()
        self.size = 0
        self.components = []
        self.referenced_by = []
        self.linear_systems = []

class _Value:
    def __init__(self, value, doc, user_read, check):
        self.value = float(value)
        self.doc = textwrap.dedent(str(doc)).strip()
        self.user_read = bool(user_read)
        self.check = check

    def check(self, name, database):
        if self.check:
            if isinstance(self.check, Iterable):
                op, extreme_value = self.check
                if   op == ">":  assert self.value >  extreme_value, name
                elif op == ">=": assert self.value >= extreme_value, name
                elif op == "<":  assert self.value <  extreme_value, name
                elif op == "<=": assert self.value <= extreme_value, name
                elif op == "in": 1/0 # Check in range.
                else: 1/0 # Unrecognized operation.
            else:
                assert np.isfinite(self.value), name

class _Function:
    def __init__(self, function, doc):
        assert(isinstance(function, Callable))
        self.function = function
        self.doc = textwrap.dedent(str(doc if doc else function.__doc__)).strip()

    def clean(self): pass

class _Array:
    def __init__(self, archetype, doc, dtype, shape, column_archetype, initial_value,
                user_read, user_write, check):
        self.archetype = archetype
        archetype.components.append(self)
        self.doc = textwrap.dedent(str(doc)).strip()
        if isinstance(dtype, np.dtype):
            self.dtype = dtype
            self.initial_value = initial_value
            self.reference = False
        else:
            self.dtype = Index
            self.initial_value = NULL
            self.reference = str(dtype)
        if shape == "sparse":
            self.shape = "sparse"
        elif isinstance(shape, Iterable):
            self.shape = tuple(int(round(x)) for x in shape)
        else:
            self.shape = (int(round(shape)),)
        self.user_read = bool(user_read)
        self.user_write = bool(user_write)
        self.check = check
        if self.shape == "sparse":
            self.data = csr_matrix((archetype.size, column_archetype.size), dtype=self.dtype)
        else:
            self.data = self._alloc(archetype.size)
            self.append_entities(0, archetype.size)

    def append_entities(self, old_size, new_size):
        """ Append and initialize some new instances to the data array. """
        if self.shape == "sparse":
            1/0 # TODO: ?
        else:
            if len(self.data) < new_size:
                new_data = self._alloc(new_size)
                new_data[:old_size] = self.data[:old_size]
                self.data = new_data
            if self.initial_value is not None:
                self.data[old_size: new_size].fill(self.initial_value)

    def _alloc(self, minimum_size):
        # TODO: IIRC CuPy can not deal with numpy structured arrays...
        #       Detect this issue and revert to using numba arrays.
        #       numba.cuda.to_device(numpy.array(data, dtype=dtype))
        return cupy.empty((int(round(minimum_size * 1.25)),) + self.shape, dtype=self.dtype)

    def check(self, name, database):
        if self.check:
            data = self.data
            if self.shape == "sparse": data = data.data
            if self.reference:
                if self.check == "ALLOW_NULL": 1/0
                assert not cupy.any(data == NULL), name
                1/0 # TODO: Check that all references are "ref < len(entity)"
            else:
                if isinstance(self.check, Iterable):
                    op, extreme_value = self.check
                    if   op == ">":  1/0
                    elif op == ">=": 1/0
                    elif op == "in": 1/0 # Check a range of values
                    else: 1/0
                else:
                    kind = self.dtype.kind
                    if kind == "f" or kind == "c":
                        assert cupy.all(cupy.isfinite(data)), name

class _LinearSystem:
    def __init__(self, function, doc, check, epsilon):
        self.function   = function
        self.epsilon    = float(epsilon)
        self.doc        = textwrap.dedent(str(doc)).strip()
        self.check      = check
        self.up_to_date = False
        self.data       = None

    def compute(self, database):
        coef = self.function(database.access)
        # Note: always use double precision floating point for building the impulse response matrix.
        # TODO: Detect if the user returns f32 and auto-convert it to f64.
        matrix = expm(coefficients)
        # Prune the impulse response matrix.
        matrix.data[np.abs(matrix.data) < self.epsilon] = 0
        matrix.eliminate_zeros()
        self.data = cupyx.scipy.sparse.csr_matrix(matrix, dtype=Real)
        self.up_to_date = True

    def check(self, name, database):
        if self.check:
            if not self.up_to_date: self.compute(database)
            assert cupy.all(cupy.isfinite(self.data)), name

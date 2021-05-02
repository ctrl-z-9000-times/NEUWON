""" A custom Entity-Component-System for NEUWON. """

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
import cupy
import textwrap
from collections.abc import Callable, Iterable, Mapping

Real = np.dtype('f4')
epsilon = np.finfo(Real).eps
Location = np.dtype('u4')
ROOT = np.iinfo(Location).max

sep = "/" # TODO: Either make this private or rename it into a full english word (seperator).

# TODO: API REWORK: Allow sparse matrixes on GPU, have two methods to update them:
# 1) Overwrite matrix with given data, for data which gets entirely recomputed: IRM
# 2) Update rows, for any kind of adjacency matrix: children, neighbors.
# How to call such methods? Also, they need a different code path than the regular array append...

class Database:
    def __init__(self):
        self.archetypes = {}
        self.components = {} # TODO: Rename this BC it's shared by every type of entry, not just entity-components...

    def add_archetype(self, name: str, doc: str = ""):
        """ Create a new type of entity. """
        name = str(name)
        assert(name not in self.archetypes)
        self.archetypes[name] = _Archetype(doc)

    def add_global_constant(self, name: str, value: float, doc: str = ""):
        """ Add a singular floating-point value to the Database. """
        name = str(name)
        assert(name not in self.components)
        self.components[name] = _Value(value, doc=doc)

    def add_component(self, name: str, doc: str = "",
            dtype=Real, shape=(1,), initial_value=None,
            user_read=False, user_write=False, check=True):
        """ Add a new data array to an Archetype.

        The archetype must be specified by prefixing the component name wit hthe
        archetype name followed by a slash "/".

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
        assert(name not in self.components)
        archetype, component = name.split(sep, maxsplit=1)
        assert(archetype in self.archetypes)
        self.components[name] = arr = _Array(self.archetypes[archetype], doc=doc,
            dtype=dtype, shape=shape, initial_value=initial_value,
            user_read=user_read, user_write=user_write, check=check)
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
        assert(name not in self.components)
        archetype, component = name.split(sep, maxsplit=1)
        assert(archetype in self.archetypes)
        self.components[name] = sys = _LinearSystem(function, doc=doc, epsilon=epsilon, check=check)
        self.archetypes[archetype].linear_systems.append(sys)

    def create_entity(self, archetype: str, number_of_instances: int = 1) -> list:
        """ Create instances of an archetype.
        Returns their internal ID's. """
        ark = self.archetypes[str(archetype)]
        num = int(number_of_instances); assert(num >= 0)
        old_size = ark.size
        new_size = old_size + num
        ark.size = new_size
        for arr in ark.components: arr._append_entities(old_size, new_size)
        for sys in ark.linear_systems: sys.up_to_date = False
        return range(old_size, new_size)

    def destroy_entity(self, archetype: str, instances: list):
        archetype = str(archetype)
        assert(archetype in self.archetypes)
        1/0 # TODO Recursively mark all destroyed instances, make a bitmask for aliveness.
        1/0 # TODO Compress the dead entries out of all data arrays.

    def num_entity(self, archetype: str):
        return self.archetypes[str(archetype)].size

    def access(self, name: str):
        """ Returns a components value or GPU data array. """
        x = self.components[str(name)]
        if isinstance(x, _Value): return x.value
        elif isinstance(x, _Array):
            if x.shape == "sparse": return x.data
            else: return x.data[:x.archetype.size]
        elif isinstance(x, _LinearSystem):
            if not x.up_to_date: x.compute(self)
            return x.data

    def check(self):
        for name, x in self.components.items():
            if not x.check: continue
            if isinstance(x, _Value):
                assert np.isfinite(x), name
            elif isinstance(x, _Array):
                if x.reference:
                    assert not cupy.any(x.data == ROOT), name
                else:
                    kind = x.dtype.kind
                    if kind == "f" or kind == "c":
                        assert cupy.all(cupy.isfinite(x.data)), name
            elif isinstance(x, _LinearSystem):
            if not x.up_to_date: x.compute(self)
            assert cupy.all(cupy.isfinite(x.data)), name

    def __repr__(self) -> str:
        s = ""
        # TODO: Print the global constants which are NOT associated with a component.
        for ark_name, ark in sorted(self.archetypes.items()):
            s += "=" * 80 + "\n"
            s += "Archetype %s (%d)\n"%(ark_name, ark.size)
            if ark.doc: s += textwrap.indent(ark.doc, "    ") + "\n"
            s += "\n"
            ark_prefix = ark_name + sep
            for comp_name, comp in sorted(self.components.items()):
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
        self.check = bool(check)

class _Array:
    def __init__(self, archetype, doc, dtype, shape, initial_value, user_read, user_write, check):
        self.archetype = archetype
        archetype.components.append(self)
        self.doc = textwrap.dedent(str(doc)).strip()
        if isinstance(dtype, np.dtype):
            self.dtype = dtype
            self.initial_value = initial_value
            self.reference = False
        else:
            self.dtype = Location
            self.initial_value = ROOT
            self.reference = str(dtype)
        if shape == "sparse":
            1/0
        elif isinstance(shape, Iterable):
            self.shape = tuple(int(round(x)) for x in shape)
        else:
            self.shape = (int(round(shape)),)
        self.user_read = bool(user_read)
        self.user_write = bool(user_write)
        self.check = bool(check)
        self.data = self._alloc(archetype.size)
        self._append_entities(0, archetype.size)

    def _append_entities(self, old_size, new_size):
        """ Append and initialize some new instances to the data array. """
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

class _LinearSystem:
    def __init__(self, function, doc, check, epsilon):
        self.function   = function
        self.epsilon    = float(epsilon)
        self.doc        = textwrap.dedent(str(doc)).strip()
        self.check      = bool(check)
        self.up_to_date = False
        self.data = None

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

if __name__ == "__main__":
    db = Database()
    db.add_archetype("Membrane", """
        Insane in the
        """)
    db.add_component("Membrane/voltage", initial_value=0)
    db.add_global_constant("Membrane/capacitance", 1e-2)
    x = db.create_entity("Membrane", 10)
    x = db.create_entity("Membrane", 10)
    print(db)

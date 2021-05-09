""" A custom Entity-Component-System for NEUWON.

The Database contains Archetypes, Entities, and Components.

* An Entity in the Database represents a concrete thing in the users model.
Entities can have associated data Components.

* An Archetype is a template for constructing Entities; and an Entity is an
instance of the Archetype which constructed it.

* The user gives names to all Archetypes and Components; and they are referred
to by their names in all API calls.

* There are multiple types of Archetypes and Components for representing
different and specialized things.
"""

# OUTSTANDING TASKS:
#       Check methods
#       Destroy & Relocate Entities
#       Grid Archetypes

# TODO: Consider splitting sparse matrixes into their own private component class.

import cupy
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import textwrap
import weakref
from collections.abc import Callable, Iterable, Mapping

Real = np.dtype('f4')
epsilon = np.finfo(Real).eps
Index = np.dtype('u4')
NULL = np.iinfo(Index).max

class Database:
    def __init__(self):
        self.archetypes = {}
        self.components = {}
        self.entities = []

    def add_archetype(self, name, grid = None, doc = ""):
        """ Create a new type of entity.

        Argument grid (optional) is spacing in each dimension.
        """
        name = str(name)
        assert(name not in self.archetypes)
        self.archetypes[name] = _Archetype(name, doc, grid)

    def _split_archetype(self, component_name):
        for ark_name, ark in self.archetypes.items():
            if component_name.startswith(ark_name):
                return ark_name, component_name[len(ark_name):]
        raise ValueError("Component name must be prefixed by an archetype name.")

    def _clean_component_name(self, new_component_name):
        new_component_name = str(new_component_name)
        assert(new_component_name not in self.components)
        return new_component_name

    def add_global_constant(self, name, value: float, doc = "", check=True):
        """ Add a singular floating-point value. """
        name = self._clean_component_name(name)
        self.components[name] = _Global_Constant(name, doc, value, check=check)

    def add_function(self, name, function, doc = ""):
        """ Add a callable function. """
        name = self._clean_component_name(name)
        self.components[name] = _Function(name, doc, function)

    def add_attribute(self, name, doc = "",
            dtype=Real, shape=(1,), initial_value=None, check=True):
        """ Add a piece of data to an Archetype. Every Entity will be allocated
        one instance of this attribute.

        Argument name: must start with the name of the associated archetype.

        Argument dtype:
            if dtype is a string then it is a reference to an entity.
            Note: Reactions are not run in a deterministic order. Dynamics which
            span between reactions via references should operate at a
            significantly slower time scale than the time step.

        Argument shape:
        Argument initial_value:
        Argument check:
        """
        name = self._clean_component_name(name)
        archetype, component = self._split_archetype(name)
        assert(archetype in self.archetypes)
        self.components[name] = arr = _Array(name, doc, self.archetypes[archetype],
            dtype=dtype, shape=shape, initial_value=initial_value, check=check)
        if arr.reference:
            assert(arr.reference in self.archetypes)
            self.archetypes[arr.reference].referenced_by.append(arr)

    def add_sparse_matrix(self, name, column_archetype, dtype=Real, doc="", check=True):
        """ Add a compressed sparse row matrix which is indexed by Entities.

        Argument name: determines the archetype for the row.
        """
        name = self._clean_component_name(name)
        archetype, component = self._split_archetype(name)
        assert(archetype in self.archetypes)
        assert(column_archetype in self.archetypes)
        self.components[name] = arr = _Array(name, doc, self.archetypes[archetype],
            dtype=dtype, initial_value=0, shape="sparse", column_archetype=self.archetypes[column_archetype],
            check=check)
        if arr.reference:
            assert(arr.reference in self.archetypes)
            self.archetypes[arr.reference].referenced_by.append(arr)

    def add_kd_tree(self, name, component, doc=""):
        """ """
        name = self._clean_component_name(name)
        archetype, _ = self._split_archetype(name)
        component = self.components[str(component)]
        assert(isinstance(component, _Array) and component.shape != "sparse")
        self.components[name] = x = _KD_Tree(name, doc, component)
        self.archetypes[archetype].kd_trees.append(x)

    def add_linear_system(self, name, function, epsilon, doc = "", check=True):
        """ Add a system of linear & time-invariant differential equations.

        Argument function(database_access) -> coefficients

        For equations of the form: dX/dt = C * X
        Where X is a component, of the same archetype as this linear system.
        Where C is a matrix of coefficients, returned by the argument "function".

        The database computes the propagator matrix but does not apply it.
        The matrix is updated after any of the entity are created or destroyed.
        """
        name = self._clean_component_name(name)
        archetype, component = self._split_archetype(name)
        assert(archetype in self.archetypes)
        self.components[name] = sys = _LinearSystem(name, doc, function, epsilon=epsilon, check=check)
        self.archetypes[archetype].linear_systems.append(sys)

    # TODO: Make a flag on this method to optionally return a list of Entity handles, as a convenience.
    def create_entity(self, archetype, number_of_instances: int = 1) -> list:
        """ Create instances of an archetype.
        Returns their internal ID's. """
        ark = self.archetypes[str(archetype)]
        assert(ark.grid is None)
        num = int(number_of_instances); assert(num >= 0)
        old_size = ark.size
        new_size = old_size + num
        ark.size = new_size
        for arr in ark.attributes: arr.append_entities(old_size, new_size)
        for tree in ark.kd_trees: tree.up_to_date = False
        for sys in ark.linear_systems: sys.up_to_date = False
        return range(old_size, new_size)

    def create_grid_entity(self, archetype, coordinates):
        """ Expand the grid to cover these coordinates.

        Returns the indexes of the given coordinates' grid boxes. """
        # TODO: Consider folding this into create_entity ... just overload the arguments
        # and document the dual usage. If the user fucks it up then they'll just get an error message.
        return 1/0 # TODO

    def coordinates_to_grid(self, archetype, coordinates):
        return 1/0 # TODO

    def destroy_entity(self, archetype, instances: list):
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

    def num_entity(self, archetype):
        return self.archetypes[str(archetype)].size

    def access(self, name, sparse_matrix_write=None):
        """

        Argument sparse_matrix_write (optional) will zero and overwrite rows in
                a sparse matrix

        Returns a components value. """
        x = self.components[str(name)]
        if sparse_matrix_write is not None:
            assert(isinstance(x, _Array) and x.shape == "sparse")
            rows, columns, data = sparse_matrix_write
            x.write_row(sparse_matrix_write)
            return
        if isinstance(x, _Global_Constant): return x.value
        if isinstance(x, _Function): return x.function
        if isinstance(x, _Array):
            if x.shape == "sparse": return x.data
            else: return x.data[:x.archetype.size]
        if isinstance(x, _LinearSystem):
            if not x.up_to_date: x.compute(self)
            return x.data
        if isinstance(x, _KD_Tree):
            if not x.up_to_date: x.compute(self)
            return x.tree

    def invalidate(self, archetype):
        # TODO: This method is crude. It only works on whole archetypes. What if
        # the user wanted to invalidate only certain pieces of data?
        ark = self.archetypes[str(archetype)]
        for tree in ark.kd_trees: tree.up_to_date = False
        for sys in ark.linear_systems: sys.up_to_date = False

    def check(self):
        for name, x in self.components.items():
            x.check(name, self)

    def __repr__(self, is_str=False):
        f = str if is_str else repr
        s = ""
        for comp_name, comp in sorted(self.components.items()):
            try: self._split_archetype(comp_name)
            except ValueError:
                s += f(comp) + "\n"
        for ark_name, ark in sorted(self.archetypes.items()):
            s += "=" * 80 + "\n"
            s += f(ark) + "\n"
            for comp_name, comp in sorted(self.components.items()):
                if not comp_name.startswith(ark_name): continue
                s += f(comp) + "\n"
        return s

    def __str__(self):
        return self.__repr__(is_str=True)

class Entity:
    """ A persistent handle on an entity.

    By default, the identifiers returned by create_entity are only valid until
    the next call to destroy_entity, which moves where the entities are located.
    This class tracks where an entity gets moved to, and provides a consistent
    API for accessing the entity data. """
    def __init__(self, database, archetype, index):
        self.database = database
        self.archetype = archetype
        self.index = index
        self.database.entities.append(weakref.ref(self))

    def read(self, component):
        self.database.access(component, self.index)

    def write(self, component, value):
        self.database.access(component, self.index)

class _DocString:
    def __init__(self, name, doc):
        self.name = name
        self.doc = textwrap.dedent(str(doc)).strip()
        self._check = False

    def class_name(self):
        return type(self).__name__.replace("_", " ").strip()

    def __str__(self):
        indent = "    "
        s = "%s %s\n"%(self.class_name(), repr(self))
        if self.doc: s += textwrap.indent(self.doc, indent) + "\n"
        if self._check:
            if isinstance(self._check, Iterable):
                op, extreme_value = self._check
                s += indent + "Value %s %s\n"%(op, extreme_value)
            else:
                s += indent + "Value is finite.\n"
        return s

class _Archetype(_DocString):
    def __init__(self, name, doc, grid):
        _DocString.__init__(self, name, doc)
        self.grid = None if not grid else tuple(float(x) for x in grid)
        self.size = 0
        self.attributes = []
        self.referenced_by = []
        self.linear_systems = []
        self.kd_trees = []

    def __repr__(self):
        s = self.name
        if self.size: s += "\n%d instances."%self.size
        return s

class _Global_Constant(_DocString):
    def __init__(self, name, doc, value, check):
        _DocString.__init__(self, name, doc)
        self.value = float(value)
        self._check = check

    def check(self, name, database):
        if self._check:
            if isinstance(self._check, Iterable):
                op, extreme_value = self._check
                if   op == ">":  assert self.value >  extreme_value, name
                elif op == ">=": assert self.value >= extreme_value, name
                elif op == "<":  assert self.value <  extreme_value, name
                elif op == "<=": assert self.value <= extreme_value, name
                elif op == "in": 1/0 # Check in range.
                else: 1/0 # Unrecognized operation.
            else:
                assert np.isfinite(self.value), name

    def __repr__(self):
        return "%s = %s"%(self.name, str(self.value))

class _Function(_DocString):
    def __init__(self, name, doc, function):
        if not doc and function.__doc__: doc = function.__doc__
        _DocString.__init__(self, name, doc)
        assert(isinstance(function, Callable))
        self.function = function

    def clean(self): pass

    def __repr__(self):
        return "%s()"%(self.name)

class _Array(_DocString):
    def __init__(self, name, doc, archetype, dtype, shape, initial_value, check, column_archetype=None):
        _DocString.__init__(self, name, doc)
        self.archetype = archetype
        archetype.attributes.append(self)
        if isinstance(dtype, str):
            self.dtype = Index
            self.initial_value = NULL
            self.reference = str(dtype)
        else:
            self.dtype = dtype
            self.initial_value = initial_value
            self.reference = False
        if shape == "sparse":
            self.shape = "sparse"
        elif isinstance(shape, Iterable):
            self.shape = tuple(int(round(x)) for x in shape)
        else:
            self.shape = (int(round(shape)),)
        self._check = check
        if self.shape == "sparse":
            self.data = scipy.sparse.csr_matrix((archetype.size, column_archetype.size), dtype=self.dtype)
        else:
            self.data = self._alloc(archetype.size)
            self.append_entities(0, archetype.size)

    def append_entities(self, old_size, new_size):
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

    def write_row(self, rows, columns, data):
        lil = scipy.sparse.lil_matrix(self.data)
        for r, c, d in zip(rows, columns, data):
            lil.rows[row] = c
            lil.data[row] = d
        self.data = scipy.sparse.csr_matrix(lil)

    def check(self, name, database):
        if self._check:
            data = self.data
            if self.shape == "sparse": data = data.data
            if self.reference:
                if self._check == "ALLOW_NULL": 1/0
                assert not cupy.any(data == NULL), name
                1/0 # TODO: Check that all references are "ref < len(entity)"
            else:
                if isinstance(self._check, Iterable):
                    op, extreme_value = self._check
                    if   op == ">":  1/0
                    elif op == ">=": 1/0
                    elif op == "in": 1/0 # Check a range of values
                    else: 1/0
                else:
                    kind = self.dtype.kind
                    if kind == "f" or kind == "c":
                        assert cupy.all(cupy.isfinite(data)), name

    def __repr__(self):
        if self.shape == "sparse":
            s = "%s is a sparse matrix"%self.name
        else:
            s = "%s is an array"%self.name
            # TODO: There are a lot of flags to print here:
            #       shape
            #       dtype
            #       initial_value
            #       reference
        return s

class _KD_Tree(_DocString):
    def __init__(self, name, doc, component):
        _DocString.__init__(self, name, doc)
        self.component = component
        self.tree = None
        self.up_to_date = False

    def compute(self, database):
        data = database.access(self.component).get()
        self.tree = scipy.spatial.cKDTree(data)
        self.up_to_date = True

    def check(self): pass

    def __repr__(self):
        return "%s is a KD Tree."%self.name

class _LinearSystem(_DocString):
    def __init__(self, name, doc, function, epsilon, check):
        _DocString.__init__(self, name, doc)
        self.function   = function
        self.epsilon    = float(epsilon)
        self._check      = check
        self.up_to_date = False
        self.data       = None

    def compute(self, database):
        coef = self.function(database.access)
        # Note: always use double precision floating point for building the impulse response matrix.
        # TODO: Detect if the user returns f32 and auto-convert it to f64.
        matrix = scipy.sparse.linalg.expm(coefficients)
        # Prune the impulse response matrix.
        matrix.data[np.abs(matrix.data) < self.epsilon] = 0
        matrix.eliminate_zeros()
        self.data = cupyx.scipy.sparse.csr_matrix(matrix, dtype=Real)
        self.up_to_date = True

    def check(self, name, database):
        if self._check:
            if not self.up_to_date: self.compute(database)
            assert cupy.all(cupy.isfinite(self.data)), name

    def __repr__(self):
        # TODO: For sparse matrixes: print the average num-non-zero per row.
        s = "%s is a linear system of equations."%self.name
        return s

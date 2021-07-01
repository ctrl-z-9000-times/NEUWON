""" A custom Entity-Component-System for NEUWON.

The Database contains Archetypes, Entities, and Components.

* An Entity in the Database represents a concrete thing in the users model.
Entities can have associated data Components.

* An Archetype is a template for constructing Entities; and an Entity is an
instance of the Archetype which constructed it.

* The user gives names to all Archetypes and Components, and they are referred
to by their names in all API calls. Similar to a file system, the names reflect
a hierarchical organization where Archetypes contain Components.

* There are multiple types of Archetypes and Components for representing
different and specialized things.
"""

from collections.abc import Callable, Iterable, Mapping
import collections
import cupy
import cupyx.scipy.sparse
import matplotlib
import numpy as np
import scipy.interpolate
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import sys
import textwrap
import weakref

Real    = np.dtype('f8')
epsilon = np.finfo(Real).eps
Pointer = np.dtype('u4')
NULL    = np.iinfo(Pointer).max

def eprint(*args, **kwargs): print(*args, file=sys.stderr, **kwargs)

class Database:
    def __init__(self):
        self.archetypes = {}
        self.components = {}
        self.entities = []

    def add_archetype(self, name, doc="", grid=None):
        """ Create a new type of entity.

        Argument grid (optional) will make Entities in a uniform rectangular grid.
            The argument is a tuple of the grid spacing in each dimension.
        """
        name = str(name)
        assert(name not in self.archetypes)
        self.archetypes[name] = _Archetype(name, doc, grid)

    def _get_archetype(self, name):
        name = str(name)
        for ark_name, ark in self.archetypes.items():
            if name.startswith(ark_name):
                return ark
        raise ValueError("Name must begin with an archetype name.")

    def add_global_constant(self, name, value: float, doc="", units=None,
                allow_invalid=False, bounds=(None, None),):
        """ Add a singular floating-point value. """
        _Global_Constant(self, name, doc, units, value, allow_invalid=allow_invalid, bounds=bounds,)

    def add_function(self, name, function: Callable, doc="", units=None):
        """ Add a callable function. """
        _Function(self, name, doc, units, function)

    def add_attribute(self, name, doc="", units=None, dtype=Real, shape=(1,), initial_value=None,
                allow_invalid=False, bounds=(None, None),):
        """ Add a piece of data to an Archetype. Every Entity will be allocated
        one instance of this attribute.

        Argument name: must start with the name of the associated archetype.
        Argument dtype:
        Argument shape:
        Argument initial_value:
        Argument check:

        Reference Types:
            If dtype is a string then it is a reference to an entity.
            Note: Reactions are not run in a deterministic order. Dynamics which
            span between reactions via references should operate at a
            significantly slower time scale than the time step.
            
        TODO: Explain destroy behavior...
            If the dangling references are optional then they are replaced with NULL
            references. Otherwise the entity containing the reference is destroyed.
            Destroying entities can cause a recursive destruction of multiple other
            entities.
        """
        _Attribute(self, name, doc, units, dtype=dtype, shape=shape, initial_value=initial_value,
                allow_invalid=allow_invalid, bounds=bounds,)

    def add_sparse_matrix(self, name, column_archetype, dtype=Real, doc="", units=None,
                allow_invalid=False, bounds=(None, None),):
        """ Add a sparse matrix which is indexed by Entities.
        This is useful for implementing any-to-any connections between entities.

        Sparse matrices may contain references but they will not trigger a
        recursive destruction of entities. Instead, references to destroyed
        entities are simply removed from the sparse matrix.

        Argument name: determines the archetype for the row.
        """
        _Sparse_Matrix(self, name, doc, units=units, dtype=dtype, initial_value=0,
                column_archetype=column_archetype, allow_invalid=allow_invalid, bounds=bounds,)

    def add_kd_tree(self, name, coordinates_attribute, doc=""):
        _KD_Tree(self, name, doc, coordinates_attribute)

    def add_linear_system(self, name, function, epsilon, doc="", allow_invalid=False,):
        """ Add a system of linear & time-invariant differential equations.

        Argument function(database_access) -> coefficients

        For equations of the form: dX/dt = C * X
        Where X is a component, of the same archetype as this linear system.
        Where C is a matrix of coefficients, returned by the argument "function".

        The database computes the propagator matrix but does not apply it.
        The matrix is updated after any of the entity are created or destroyed.
        """
        _Linear_System(self, name, doc, function, epsilon=epsilon, allow_invalid=allow_invalid)

    def create_entity(self, archetype_name, number_of_instances = 1, return_entity=True) -> list:
        """ Create instances of an archetype.

        If the archetype is a grid, then this accepts the coordinates of the new
        points on the grid. Otherwise this accepts the number of new entities to
        add.

        Return value:
            By default this returns a list of the new Entities.
            If optional keyword argument return_entity is set to False,
            then this returns a list of their unstable indexes.
        """
        ark = self.archetypes[str(archetype_name)]
        if ark.grid is not None:
            coordinates = number_of_instances
            raise NotImplementedError
        num = int(number_of_instances); assert(num >= 0)
        if num == 0: return []
        ark.invalidate()
        old_size = ark.size
        new_size = old_size + num
        ark.size = new_size
        for arr in ark.attributes: arr.append_entities(old_size, new_size)
        for spm in ark.sparse_matrixes: spm.append_entities(old_size, new_size)
        if return_entity:
            return [Entity(self, ark, idx) for idx in range(old_size, new_size)]
        else:
            return np.arange(old_size, new_size)

    def destroy_entity(self, archetype_name, instances: list):
        # TODO: The "allow_invalid" flag should control whether destroying
        # referenced entities causes recursive destruction of more entities.

        # TODO: Consider how to rework this to instead of keeping lists of links
        # to archetypes which need updating, to just scan through all of the
        # archetypes (in a fixed number of passes). It would allow me to
        # simplify the spiders web of lists of references.
        if not instances: return
        ark = self.archetypes[str(archetype_name)]
        ark.invalidate()
        alive = {ark: np.ones(ark.size, dtype=np.bool)}
        alive[ark][instances] = False
        # Find all dangling references to dead instance of this archetype.
        stack = list(ark.referenced_by)
        while stack:
            ref = stack.pop()
            ark = ref.archetype
            target = ref.reference
            target_alive = alive[target]
            if isinstance(ref, _Attribute):
                if ark not in alive:
                    alive[ark] = np.ones(ark.size, dtype=np.bool)
                    ark.invalidate()
                alive[ark][np.logical_not(target_alive[ref.data])] = False
                stack.extend(ark.referenced_by)
            elif isinstance(ref, _Sparse_Matrix):
                pass # References in sparse matrices never cause recursive destruction.
        # Compress destroyed instances out of the data arrays.
        for ark, alive_mask in alive.items():
            for x in ark.attributes:
                x.data = x.data[alive_mask]
            for x in ark.sparse_matrixes:
                # Compress out dead rows, columns, and if its a reference matrix
                # then remove entries which point to dead entities.
                1/0
        # Update references.
        new_indexes = {ark: np.cumsum(alive_mask) - 1 for ark, alive_mask in alive.items()}
        for x in alive:
            if isinstance(ref, _Attribute):
                x.data = new_indexes[ark][x.data]
            elif isinstance(ref, _Sparse_Matrix):
                x.data.data = new_indexes[ark][x.data.data]
        updated_entities = []
        for x in self.entities:
            entity = x.get()
            if entity is not None:
                entity.index = entity.ark
                updated_entities.append(entity)
        self.entities = updated_entities

    def num_entity(self, archetype_name):
        return self.archetypes[str(archetype_name)].size

    def access(self, component_name, sparse_matrix_write=None):
        """ This is the primary way of getting information into and out of the database.

        Argument sparse_matrix_write (optional) will zero and overwrite rows in
                a sparse matrix. It is only valid for when accessing sparse matrices.

        Returns the components value, the type of which depends on the type of component.
        """
        if sparse_matrix_write is not None:
            return self.components[str(component_name)].access(self, sparse_matrix_write)
        else:
            return self.components[str(component_name)].access(self)

    def get_units(self, component_name):
        return self.components[str(component_name)].units

    # TODO: Consider renaming this to "get_initial_value" so that its name has a
    # verb in it.
    def initial_value(self, component_name):
        """ """
        x = self.components[str(component_name)]
        if isinstance(x, _Global_Constant):
            return x.access(self)
        elif isinstance(x, _Attribute):
            return x.initial_value
        elif isinstance(x, _Sparse_Matrix):
            return x.initial_value

    def invalidate(self, archetype_or_component_name):
        x = str(archetype_or_component_name)
        try:             x = self.archetypes[x]
        except KeyError: x = self.components[x]
        x.invalidate()

    def check(self, component_name=None):
        if component_name is not None:
            self.components[str(component_name)].check(self)
        else:
            exceptions = []
            for c in self.components.values():
                try: c.check(self)
                except Exception as x: exceptions.append(str(x))
            if exceptions: raise AssertionError(",\n\t".join(sorted(exceptions)))

    def __repr__(self):
        """ Brief summary of database contents. """
        s = ""
        case_insensitive = lambda kv_pair: kv_pair[0].lower()
        components = sorted(self.components.items(), key=case_insensitive)
        archetypes = sorted(self.archetypes.items(), key=case_insensitive)
        for comp_name, comp in components:
            try: self._get_archetype(comp_name)
            except ValueError:
                s += repr(comp) + "\n"
        for ark_name, ark in archetypes:
            s += "=" * 80 + "\n"
            s += repr(ark) + "\n"
            for comp_name, comp in components:
                if not comp_name.startswith(ark_name): continue
                s += repr(comp) + "\n"
        return s

    def __str__(self):
        """ Return documentation of the database schema in Markdown format. """
        case_insensitive = lambda kv_pair: kv_pair[0].lower()
        components = sorted(self.components.items(), key=case_insensitive)
        archetypes = sorted(self.archetypes.items(), key=case_insensitive)
        s  = "## Table of Contents\n"
        s += "* [Components Without Archetypes](#components-without-archetypes)\n"
        for name, obj in archetypes:
            s += "* [Archetype: %s](%s)\n"%(name, obj._markdown_link())
        s += "* [Index](#index)\n"
        s += "---\n## Components Without Archetypes\n"
        for name, obj in components:
            try: self._get_archetype(name)
            except ValueError:
                s += str(obj) + "\n"
        for ark_name, ark in archetypes:
            s += str(ark) + "\n"
            s += "Components:\n"
            for comp_name, comp in components:
                if not comp_name.startswith(ark_name): continue
                s += "* [%s](%s)\n"%(comp_name, comp._markdown_link())
            s += "\n"
            for comp_name, comp in components:
                if not comp_name.startswith(ark_name): continue
                s += str(comp) + "\n"
        s += "---\n"
        s += "## Index\n"
        for name, obj in sorted(archetypes + components):
            s += "* [%s](%s)\n"%(name, obj._markdown_link())
        return s

    def browse_docs(self):
        from subprocess import run, PIPE
        grip = run(["grip", "--browser", "-"], input=bytes(str(self.model), encoding='utf8'))

class Entity:
    """ A persistent handle on an entity.

    By default, the identifiers returned by create_entity are only valid until
    the next call to destroy_entity, which moves where entities are located.
    This class tracks where an entity gets moved to, and provides a consistent
    API for accessing the entity data. """
    def __init__(self, database, archetype, index):
        self.database = database
        assert(isinstance(self.database, Database))
        if isinstance(archetype, _Archetype):
            self.archetype = archetype
        else:
            self.archetype = database.archetypes[str(archetype)]
        if isinstance(index, Iterable):
            index = list(index)
            assert(len(index) == 1)
            index = index.pop()
        self.index = int(index)
        assert(self.index < self.archetype.size)
        self.database.entities.append(weakref.ref(self))

    def read(self, component_name):
        """ """
        component_name = str(component_name)
        assert(self.database._get_archetype(component_name) == self.archetype)
        component = self.database.components[component_name]
        data = component.access(self.database)
        if isinstance(data, Iterable): data = data[self.index]
        if hasattr(data, "get"): data = data.get()
        if isinstance(component, _Attribute):
            if component.reference: return int(data)
            if component.shape == (1,): return float(data)
        return data

    def write(self, component, value):
        """ """
        archetype = self.database._get_archetype(component_name)
        assert(archetype == self.archetype)
        data = self.database.access(component)
        data[self.index] = value

class _DocString:
    def __init__(self, name, doc):
        self.name = str(name)
        self.doc = textwrap.dedent(str(doc)).strip()

    def _class_name(self):
        return type(self).__name__.replace("_", " ").strip()

    def _markdown_header(self):
        return "%s: %s"%(self._class_name(), self.name)

    def _markdown_link(self):
        name = "#" + self._markdown_header()
        substitutions = (
            (":", ""),
            ("/", ""),
            (" ", "-"),
        )
        for x in substitutions: name = name.replace(*x)
        return name.lower()

    def __str__(self):
        anchor = "<a name=\"%s\"></a>"%self.name
        return "%s%s\n%s\n\n"%(self._markdown_header(), anchor, self.doc)

class _Component(_DocString):
    def __init__(self, database, name, doc, units=None, allow_invalid=True, bounds=(None, None)):
        _DocString.__init__(self, name, doc)
        assert(self.name not in database.components)
        database.components[self.name] = self
        self.units = None if units is None else str(units)
        self.allow_invalid = bool(allow_invalid)
        self.bounds = tuple((x if x is None else float(x) for x in bounds))
        if None not in self.bounds: self.bounds = tuple(sorted(self.bounds))
        assert(len(self.bounds) == 2)

    def access(self, database):
        1/0 # Abstract method, required.

    def check(self, database):
        """ Abstract method, optional """

    def check_data(self, database, data, reference=False):
        """ Helper method to interpret the check flags (allow_invalid & bounds) and dtypes. """
        xp = cupy.get_array_module(data)
        if not self.allow_invalid:
            if reference: assert xp.all(xp.less(data, reference.size)), self.name + " is NULL"
            else:
                kind = data.dtype.kind
                if kind in ("f", "c"): assert not xp.any(xp.isnan(data)), self.name + " is NaN"
        lower_bound, upper_bound = self.bounds
        if lower_bound is not None:
            assert xp.all(xp.less_equal(lower_bound, data)), self.name + " less than %g"%lower_bound
        if upper_bound is not None:
            assert xp.all(xp.greater_equal(upper_bound, data)), self.name + " greater than %g"%upper_bound

    def _dtype_name(self):
        if isinstance(self.dtype, np.dtype): x = self.dtype.name
        elif isinstance(self.dtype, type): x = self.dtype.__name__
        else: x = type(self.dtype).__name__
        if hasattr(self, "shape") and self.shape != (1,):
            x += repr(list(self.shape))
        return x

    def __str__(self):
        s = "### %s"%_DocString.__str__(self)
        if hasattr(self, "value"):
            s += "Value: %g"%(self.value)
            if self.units is not None: s += " " + self.units
            s += "\n\n"
        elif self.units is not None: s += "Units: %s\n\n"%self.units
        ref = getattr(self, "reference", False)
        if ref: s += "Reference to archetype [%s](%s).\n\n"%(ref.name, ref._markdown_link())
        if hasattr(self, "dtype") and not ref:
            s += "Data type: %s\n\n"%(self._dtype_name(),)
        lower_bound, upper_bound = self.bounds
        if lower_bound is not None and upper_bound is not None:
            s += ""
        elif lower_bound is not None:
            s += ""
        elif upper_bound is not None:
            s += ""
        if getattr(self, "initial_value", None) is not None and not ref:
            s += "Initial Value: %g"%(self.initial_value)
            if self.units is not None: s += " " + self.units
            s += "\n\n"
        if self.allow_invalid:
            if ref:
                s += "Value may be NULL.\n\n"
            else: s += "Value may be NaN.\n\n"
        return s

class _Archetype(_DocString):
    def __init__(self, name, doc, grid):
        _DocString.__init__(self, name, doc)
        self.grid = None if not grid else tuple(float(x) for x in grid)
        self.size = 0
        self.attributes = []
        self.sparse_matrixes = []
        self.referenced_by = []
        self.linear_systems = []
        self.kd_trees = []

    def invalidate(self):
        for tree in self.kd_trees: tree.invalidate()
        for lin_sys in self.linear_systems: lin_sys.invalidate()

    def __repr__(self):
        s = "Archetype " + self.name
        if self.size: s += "  %d instances"%self.size
        return s

    def __str__(self):
        s = "---\n## %s"%_DocString.__str__(self)
        return s

class _Global_Constant(_Component):
    def __init__(self, database, name, doc, units, value, allow_invalid, bounds):
        _Component.__init__(self, database, name, doc, units, allow_invalid, bounds)
        self.value = float(value)
        _Component.check_data(self, database, np.array([self.value]))

    def access(self, database):
        return self.value

    def __repr__(self):
        return "Constant  %s  = %s"%(self.name, str(self.value))

class _Function(_Component):
    def __init__(self, database, name, doc, units, function):
        if not doc and function.__doc__: doc = function.__doc__
        _Component.__init__(self, database, name, doc, units, allow_invalid=False)
        self.function = function
        assert isinstance(self.function, Callable), self.name

    def access(self, database):
        return self.function

    def __repr__(self):
        return "Function  %s()"%(self.name)

class _Attribute(_Component):
    def __init__(self, database, name, doc, units, dtype, shape, initial_value, allow_invalid, bounds):
        _Component.__init__(self, database, name, doc, units, allow_invalid, bounds)
        self.archetype = ark = database._get_archetype(self.name)
        ark.attributes.append(self)
        if isinstance(dtype, str):
            self.dtype = Pointer
            self.initial_value = NULL
            self.reference = database.archetypes[str(dtype)]
            self.reference.referenced_by.append(self)
        else:
            self.dtype = dtype
            self.initial_value = initial_value
            self.reference = False
        if isinstance(shape, Iterable):
            self.shape = tuple(int(round(x)) for x in shape)
        else:
            self.shape = (int(round(shape)),)
        self.data = self._alloc(ark.size)
        self.append_entities(0, ark.size)

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
        heuristic = minimum_size * 1.25
        alloc = (int(round(heuristic)),)
        # Special case to squeeze off the trailing dimension.
        if self.shape != (1,): alloc = alloc + self.shape
        return cupy.empty(alloc, dtype=self.dtype)

    def access(self, database):
        return self.data[:self.archetype.size]

    def check(self, database):
        _Component.check_data(self, database, self.access(database), reference=self.reference)

    def __repr__(self):
        s = "Attribute " + self.name + "  "
        if self.reference: s += "ref:" + self.reference.name
        else: s += self._dtype_name()
        if self.shape != (1,): s += repr(list(self.shape))
        if self.allow_invalid:
            if self.reference: s += " (maybe NULL)"
            else: s += " (maybe NaN)"
        return s

class _Sparse_Matrix(_Component):
    def __init__(self, database, name, doc, units, dtype, initial_value, column_archetype, allow_invalid, bounds):
        _Component.__init__(self, database, name, doc, units, allow_invalid, bounds)
        self.archetype = ark = database._get_archetype(self.name)
        ark.sparse_matrixes.append(self)
        self.column_archetype = database.archetypes[str(column_archetype)]
        self.column_archetype.sparse_matrixes.append(self)
        if isinstance(dtype, str):
            self.dtype = Pointer
            self.initial_value = NULL
            self.reference = database.archetypes[str(dtype)]
            self.reference.referenced_by.append(self)
        else:
            self.dtype = dtype
            self.initial_value = initial_value
            self.reference = False
        self.data = scipy.sparse.csr_matrix((ark.size, self.column_archetype.size), dtype=self.dtype)

    def append_entities(self, old_size, new_size):
        self.data.resize((new_size, self.column_archetype.size))

    def access(self, database, sparse_matrix_write=None):
        if sparse_matrix_write:
            rows, columns, data = sparse_matrix_write
            lil = scipy.sparse.lil_matrix(self.data)
            for r, c, d in zip(rows, columns, data):
                r = int(r)
                lil.rows[r].clear()
                lil.data[r].clear()
                cols_data = [int(x) for x in c]
                order = np.argsort(cols_data)
                lil.rows[r].extend(np.take(cols_data, order))
                lil.data[r].extend(np.take(list(d), order))
            self.data = scipy.sparse.csr_matrix(lil, shape=self.data.shape)
        return self.data

    def check(self, database):
        _Component.check_data(self, database, self.data.data, reference=self.reference)

    def __repr__(self):
        s = "Matrix    " + self.name + "  "
        if self.reference: s += "ref:" + self.reference.name
        elif isinstance(self.dtype, np.dtype): s += self.dtype.name
        elif isinstance(self.dtype, type): s += self.dtype.__name__
        else: s += type(self.dtype).__name__
        if self.allow_invalid:
            if self.reference: s += " (maybe NULL)"
            else: s += " (maybe NaN)"
        try: nnz_per_row = self.data.nnz / self.data.shape[0]
        except ZeroDivisionError: nnz_per_row = 0
        s += " nnz/row: %g"%nnz_per_row
        return s

class _KD_Tree(_Component):
    def __init__(self, database, name, doc, component):
        _Component.__init__(self, database, name, doc)
        self.component = database.components[str(component)]
        assert(isinstance(self.component, _Attribute))
        assert(not self.component.reference)
        archetype = database._get_archetype(self.name)
        archetype.kd_trees.append(self)
        assert archetype == self.component.archetype, "KD-Tree and its coordinates must have the same archetype."
        self.tree = None
        self.invalidate()

    def invalidate(self):
        self.up_to_date = False

    def access(self, database):
        if not self.up_to_date:
            data = self.component.access(database).get()
            self.tree = scipy.spatial.cKDTree(data)
            self.up_to_date = True
        return self.tree

    def __repr__(self):
        return "KD Tree   " + self.name

class _Linear_System(_Component):
    def __init__(self, database, name, doc, function, epsilon, allow_invalid):
        _Component.__init__(self, database, name, doc, allow_invalid=allow_invalid)
        self.archetype = database._get_archetype(self.name)
        self.archetype.linear_systems.append(self)
        self.function   = function
        self.epsilon    = float(epsilon)
        self.data       = None

    def invalidate(self):
        self.data = None

    def access(self, database):
        if self.data is None:
            coef = self.function(database.access)
            coef = scipy.sparse.csc_matrix(coef, shape=(self.archetype.size, self.archetype.size))
            # Note: always use double precision floating point for building the impulse response matrix.
            # TODO: Detect if the user returns f32 and auto-convert it to f64.
            matrix = scipy.sparse.linalg.expm(coef)
            # Prune the impulse response matrix.
            matrix.data[np.abs(matrix.data) < self.epsilon] = 0
            matrix.eliminate_zeros()
            self.data = cupyx.scipy.sparse.csr_matrix(matrix, dtype=Real)
        return self.data

    def check(self, database):
        _Component.check_data(self, database, self.access(database).get().data)

    def __repr__(self):
        s = "Linear    " + self.name + "  "
        if self.data is None:
            s += "invalid"
        else:
            s += "nnz/row: %g"%(self.data.nnz / self.data.shape[0])
        return s

class TimeSeriesBuffer:
    def __init__(self, entity, component, clock="clock"):
        if hasattr(entity, "entity"): entity = entity.entity
        self.entity     = entity
        self.component  = str(component)
        self.clock      = str(clock)
        assert isinstance(self.entity, Entity), self.entity
        self.db         = entity.database
        self.timeseries = collections.deque(maxlen=None)
        self.timestamps = collections.deque(maxlen=None)
        self(self.entity.database.access) # Immediately record the first data point.

    def clear(self):
        self.timeseries.clear()
        self.timestamps.clear()

    def __call__(self, database_access):
        self.timeseries.append(self.entity.read(self.component))
        self.timestamps.append(database_access(self.clock)())
        return True

    def label_axes(self, axes=None):
        if axes is None: axes = matplotlib.pyplot.gca()
        axes.set_ylabel(self.db.get_units(self.component))
        axes.set_xlabel(self.db.get_units(self.clock))
        return axes

    @property
    def y(self):
        """ Data samples """
        return self.timeseries
    @property
    def x(self):
        """ Timestamps """
        return self.timestamps

    def interpolate(self, timestamps):
        """ Interpolate the value of the timeseries at the given timestamps.
        This uses linear interpolation.
        """
        f = scipy.interpolate.interp1d(self.x, self.y)
        min_t = self.timestamps[0]
        max_t = self.timestamps[-1]
        min_v = self.timeseries[0]
        max_v = self.timeseries[-1]
        results = np.empty(len(timestamps))
        for i, t in enumerate(timestamps):
            if   t < min_t: results[i] = min_v
            elif t > max_t: results[i] = max_v
            else: results[i] = f(t)
        return results

    def plot(self, show=True):
        plt = matplotlib.pyplot
        plt.figure(self.component)
        plt.title("Time Series of: " + self.component)
        self.label_axes()
        plt.plot(self.x, self.y)
        if show: plt.show()

# TODO: Consider reworking the sparse-matrix-write arguments to access so that
# the user can do more things:
#       1) Write rows.
#       2) Add coordinates.
#       3) Overwrite the matrix? Are scipy.sparse.CSR immutable?

# TODO: Grid Archetypes
#   Design the API:
#   Consider making two new components, which would be auto-magically updated:
#   -> Attribute coordinates of entity.
#   -> Function Nearest neighbor to convert coordinates to entity index.
# >>> assert(len(grid) == 3 and all(x > 0 for x in grid))

# TODO: Add a widget for getting the data as a fraction of its range, using the
# optionally given "bounds". This would be really useful for making color maps
# of simulations. It could automatically scale my voltages into a nice viewable
# [0-1] range ready to be rendered. If data does not have bounds then raise
# exception.

# TODO: Entity.read() should make an effort to follow references, in the
# situation where the requested data is not associated with the same archetype.
# This is a major quality-of-life improvement for working with the Entity API.
# 
# I want to make this example work:
#       probe.read("hh/data/m")
# 
# Instead, Right now I have to do:
#        hh_idx = np.nonzero(self.model.access("hh/insertions") == p.entity.index)[0][0]
#        m = self.model.access("hh/data/m")[hh_idx]
# 
# This would not have to work in every conceivable situation, but 99% of the
# time it just needs to follow a basic reference or search for an index, and it
# should balk if the situation is ambiguous.

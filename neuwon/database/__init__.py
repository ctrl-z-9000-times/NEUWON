""" A database for neural simulations. """

# TODO: Add a widget for getting the data as a fraction of its range, using the
# optionally given "bounds". Use-case: making color maps. It could
# automatically scale my voltages into the viewable range [0-1] ready to be
# rendered. If data does not have bounds then raise exception.

# TODO: Grids of objects
#   -> Subclass of ?
#   -> Entity creates & controls to coordinates array.
#   -> Function: Nearest neighbor to convert coordinates to entity index.

# TODO: Consider making a "ConnectivityMatrix" subclass, instead of using
# sparse boolean matrix, to avoid storing the 1's.
# This could be part of a more general purpose "list" attribute.


# IDEAS FOR GET DATA API
# np_arr = my_attr.get()
# my_attr.get(gpu=True)
# my_attr.get(gpu=MyGpuToken)
# my_attr.get(cpu=True)
# my_attr.get(cpu=True)

# .get_gpu()
# .get_cpu()

# component.get(**kwargs)
#     gpu             | bool or gpu-related-token
#     cpu/host        | bool




from collections.abc import Callable, Iterable, Mapping
import collections
import cupy
import cupyx.scipy.sparse
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

def eprint(*args, **kwargs):
    """ Prints to standard error (sys.stderr). """
    print(*args, file=sys.stderr, **kwargs)

class Database:
    def __init__(self):
        self.class_types = {}

    def add_class(self, name, instance_class=None):
        return ClassType(self, name, instance_class=instance_class)

    def get_class(self, name):
        if isinstance(name, ClassType): return name
        return self.class_types[str(name)]

    def get_component(self, name):
        cls, component = str(name).split('.', maxsplit=1)
        return self.get_class(cls).get_component(component)

    def check(self, name=None):
        if name is None:
            exceptions = []
            for c in self.components.values():
                try: c.check(self)
                except Exception as x: exceptions.append(str(x))
            if exceptions: raise AssertionError(",\n\t".join(sorted(exceptions)))
        else:
            self.get_component(component_name).check(self)

    def __repr__(self):
        """ Table summarizing contents. """
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
        """ Markdown format documentation. """
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

    # TODO: This is gui code. I think it should live elsewhere...
    def browse_docs(self):
        from subprocess import run, PIPE
        grip = run(["grip", "--browser", "-"], input=bytes(str(self), encoding='utf8'))

class _DocString:
    def __init__(self, name, doc):
        self._name = str(name)
        self.doc = textwrap.dedent(str(doc)).strip()

    @property
    def name(self):
        return self._name

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

class ClassType(_DocString):
    def __init__(self, database, name, doc="", instance_class=None):
        if type(self) != ClassType:
            if not doc: doc = type(self).__name__
        _DocString.__init__(self, name, doc)
        if type(self) == ClassType:
            self.__class__ = type(self.name, (type(self),), {})
        assert isinstance(database, Database)
        self.database = database
        assert self.name not in self.database.class_types
        self.database.class_types[self.name] = self
        self.size = 0
        self.components = dict()
        self.referenced_by = list()
        self.instances = weakref.WeakSet()
        if instance_class is not None:
            inherit = (instance_class, Instance,)
        else:
            inherit = (Instance,)
        self.instance_class = type(
                self.name + "Instance",
                inherit, {
                "__slots__": (),
                "_cls": self,
        })

    def get_component(self, name):
        return self.components[str(name)]

    def add_attribute(self, name, dtype=Real):
        # TODO: Copy the kwargs & docs from the class definitions back up to here. allow duplication.
        return Attribute(self, name, dtype=dtype)

    def add_class_attribute(self, name, value):
        # TODO: Copy the kwargs & docs from the class definitions back up to here. allow duplication.
        return ClassAttribute(self, name, value)

    def add_sparse_matrix(self, name, column_class):
        # TODO: Copy the kwargs & docs from the class definitions back up to here. allow duplication.
        return Sparse_Matrix(self, name, column_class)

    def __call__(self, **kwargs):
        """ Construct a new instance of this class.

        Keyword arguments are assignments to the instance.
            For example:
                >>> obj = MyClass(x=3, y=4)
            Is equivalient to:
                >>> obj = MyClass()
                >>> obj.x = 3
                >>> obj.y = 4
        """
        self.invalidate()
        old_size  = self.size
        new_size  = old_size + 1
        self.size = new_size
        obj = self.instance_class(old_size)
        for x in self.components.values():
            if isinstance(x, Attribute):
                x._append(old_size)
            elif isinstance(x, ClassAttribute):
                1/0
            elif isinstance(x, Sparse_Matrix):
                x._append(old_size)
            else: raise NotImplementedError
        for attribute, value in kwargs.items():
            obj.attribute = value
        return obj

    def destroy(self):
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

    def invalidate(self):
        for x in self.components.values(): x.invalidate()

    def __len__(self):
        return self.size

    # def __repr__(self):
    #     s = "Entity".ljust(10) + self.name
    #     if len(self): s += "  %d instances"%len(self)
    #     return s

    def __str__(self):
        s = "---\n## %s"%_DocString.__str__(self)
        return s

class Instance:
    __slots__ = ("_idx", "__weakref__")

    def __init__(self, index):
        self._idx = int(index)
        self._cls.instances.add(self)
        assert self._idx in range(len(self._cls))

    def __repr__(self):
        return "%s:%d"%(type(self).__name__, self._idx)

class _Component(_DocString):
    """ Abstract class for all data components. """
    def __init__(self, class_type, name, doc, units=None,
                allow_invalid=True, valid_range=(None, None)):
        _DocString.__init__(self, name, doc)
        assert isinstance(class_type, ClassType)
        assert(self.name not in class_type.components)
        self.cls = class_type
        self.cls.components[self.name] = self
        self.units = None if units is None else str(units)
        self.allow_invalid = bool(allow_invalid)
        min_, max_ = valid_range
        if min_ is None: min_ = -float('inf')
        if max_ is None: max_ = +float('inf')
        self.valid_range = tuple(sorted((min_, max_)))

    def get(self):
        """ Abstract method, required. """
        raise NotImplementedError

    def get_units(self, component_name):
        return self.units

    def get_initial_value(self, component_name):
        """ Abstract method, optional """
        return getattr(self, "initial_value", None)

    def check(self, database):
        """ Abstract method, optional """

    def invalidate(self):
        """ Abstract method, optional """

    def _check_data(self, database, data, reference=False):
        """ Helper method to interpret the check flags (allow_invalid & bounds) and dtypes. """
        xp = cupy.get_array_module(data)
        if not self.allow_invalid:
            if reference: assert xp.all(xp.less(data, reference.size)), self.name + " is NULL"
            else:
                kind = data.dtype.kind
                if kind in ("f", "c"): assert not xp.any(xp.isnan(data)), self.name + " is NaN"
        lower_bound, upper_bound = self.valid_range
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
        lower_bound, upper_bound = self.valid_range
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

class Attribute(_Component):
    def __init__(self, class_type, name,
                doc="",
                units=None,
                dtype=Real,
                shape=(1,),
                initial_value=None,
                allow_invalid=False,
                valid_range=(None, None),):
        """
        Add an instance variable to a class definiton.

        Argument dtype is one of:
            * An instance of numpy.dtype
            * The name of a class, to make pointers to instances of that class.

        If the dangling references are allow to be NULL then they are replaced
        with NULL references. Otherwise the entity containing the reference is
        destroyed. Destroying entities can cause a chain reaction of destruction.
        """
        _Component.__init__(self, class_type, name, doc, units, allow_invalid, valid_range)
        if isinstance(dtype, str) or isinstance(dtype, ClassType):
            self.dtype = Pointer
            self.initial_value = NULL
            self.reference = self.cls.database.get_class(dtype)
            self.reference.referenced_by.append(self)
        else:
            self.dtype = dtype
            self.initial_value = initial_value
            self.reference = False
        if isinstance(shape, Iterable):
            self.shape = tuple(int(round(x)) for x in shape)
        else:
            self.shape = (int(round(shape)),)
        self.data = self._alloc(len(self.cls))
        self._append(0)
        setattr(self.cls.instance_class, self.name, property(
            self._getter,
            self._setter,
            doc=self.doc,
        ))

    def _getter(self, instance):
        value = self.data[instance._idx]
        if self.reference:
            value = self.reference.instance_class(value)
        return value

    def _setter(self, instance, value):
        if self.reference:
            if isinstance(value, self.reference.instance_class):
                value = value._idx
        self.data[instance._idx] = value

    def _append(self, old_size):
        """ Prepare space for new instances at the end of the array. """
        new_size = len(self.cls)
        if len(self.data) < new_size:
            new_data = self._alloc(new_size)
            new_data[:old_size] = self.data[:old_size]
            self.data = new_data
        if self.initial_value is not None:
            self.data[old_size: new_size].fill(self.initial_value)

    def _alloc(self, minimum_size):
        """ Returns an empty array. """
        # TODO: IIRC CuPy can not deal with numpy structured arrays...
        #       Detect this issue and revert to using numba arrays.
        #       numba.cuda.to_device(numpy.array(data, dtype=dtype))
        alloc = (int(round(2 * minimum_size)),)
        # Special case to squeeze off the empty trailing dimension (1).
        if self.shape != (1,): alloc = alloc + self.shape
        try:
            return cupy.empty(alloc, dtype=self.dtype)
        except Exception:
            eprint("ERROR on GPU: allocating %s for %s"%(repr(alloc), self.name))
            raise

    def get(self):
        return self.data[:len(self.cls)]

    def check(self, database):
        _Component._check_data(self, database, self.get(), reference=self.reference)

    def __repr__(self):
        s = "Attribute " + self.name + "  "
        if self.reference: s += "ref:" + self.reference.name
        else: s += self._dtype_name()
        if self.shape != (1,): s += repr(list(self.shape))
        if self.allow_invalid:
            if self.reference: s += " (maybe NULL)"
            else: s += " (maybe NaN)"
        return s

class ClassAttribute(_Component):
    def __init__(self, class_type, name, value, doc="", units=None,
                allow_invalid=False, valid_range=(None, None),):
        """ Add a singular floating-point value. """
        _Component.__init__(self, class_type, name, doc, units, allow_invalid, valid_range)
        self.data = float(value)
        setattr(self.cls.instance_class, self.name,
                property(self._getter, self._setter, doc=self.doc))
        self.check()

    def _getter(self, instance):
        return self.data

    def _setter(self, instance, value):
        self.data = float(value)

    def get(self):
        return self.data

    def check(self):
        _Component._check_data(self, self.cls, np.array([self.data]))

    def __repr__(self):
        return "Constant  %s  = %s"%(self.name, str(self.data))

class Sparse_Matrix(_Component):
    def __init__(self, class_type, name, column_class, dtype=Real, doc="", units=None,
                allow_invalid=False, valid_range=(None, None),):
        """
        Add a sparse matrix which is indexed by Entities. This is useful for
        implementing any-to-any connections between entities.

        Sparse matrices may contain references but they will not trigger a
        recursive destruction of entities. Instead, references to destroyed
        entities are simply removed from the sparse matrix.

        Argument name: determines the archetype for the row.
        """
        _Component.__init__(self, class_type, name, doc, units, allow_invalid, valid_range)
        if isinstance(column_class, ClassType):
            self.column_class = column_class
        else:
            self.column_class = self.cls.database.get_class(column_class)
        self.dtype = dtype
        self.data = scipy.sparse.csr_matrix((len(self.cls), len(self.column_class)), dtype=self.dtype)
        self.fmt = 'csr'
        setattr(self.cls.instance_class, self.name, property(
                self._getter, self._setter, doc=self.doc))

    def _getter(self, instance):
        if self.fmt == 'lil':
            vec = self.data[instance._idx]
            return (vec.rows, vec.data)
        else: raise NotImplementedError

    def _setter(self, instance, value):
        columns, data = value
        self.write_row(instance._idx, columns, data)

    def to_fmt(self, fmt):
        if self.fmt != fmt:
            if   fmt == 'lil': self.to_lil()
            elif fmt == 'csr': self.to_csr()
            else: raise ValueError(fmt)
            self.fmt = fmt
        return self

    def to_lil(self):
        self.data = scipy.sparse.lil_matrix(self.data, shape=self.data.shape)

    def to_csr(self):
        self.data = scipy.sparse.csr_matrix(self.data, shape=self.data.shape)

    def _append(self, old_size):
        self.data.resize((len(self.cls), len(self.column_class)))

    def access(self, sparse_matrix_write=None):
        return self.data

    def write_row(self, row, columns, values):
        self.to_fmt('lil')
        r = int(row)
        self.data.rows[r].clear()
        self.data.data[r].clear()
        columns = [x._idx if isinstance(x, Instance) else int(x) for x in columns]
        order = np.argsort(columns)
        self.data.rows[r].extend(np.take(columns, order))
        self.data.data[r].extend(np.take(values, order))

    # TODO: Consider adding more write methods:
    #       1) Write rows. (done)
    #       2) Insert coordinates.
    #       3) Overwrite the matrix?

    def check(self, database):
        _Component._check_data(self, database, self.data.data, reference=self.reference)

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

class KD_Tree(_Component):
    def __init__(self, entity, name, coordinates_attribute, doc=""):
        _Component.__init__(self, entity, name, doc)
        self.component = database.components[str(coordinates_attribute)]
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

class Linear_System(_Component):
    def __init__(self, class_type, name, function, epsilon, doc="", allow_invalid=False,):
        """ Add a system of linear & time-invariant differential equations.

        Argument function(database_access) -> coefficients

        For equations of the form: dX/dt = C * X
        Where X is a component, of the same archetype as this linear system.
        Where C is a matrix of coefficients, returned by the argument "function".

        The database computes the propagator matrix but does not apply it.
        The matrix is updated after any of the entity are created or destroyed.
        """
        _Component.__init__(self, class_type, name, doc, allow_invalid=allow_invalid)
        self.function   = function
        self.epsilon    = float(epsilon)
        self.data       = None

    def invalidate(self):
        self.data = None

    def get(self):
        if self.data is None: self._compute()
        return self.data

    def _compute(self):
        coef = self.function(self.cls.database)
        coef = scipy.sparse.csc_matrix(coef, shape=(self.archetype.size, self.archetype.size))
        # Note: always use double precision floating point for building the impulse response matrix.
        # TODO: Detect if the user returns f32 and auto-convert it to f64.
        matrix = scipy.sparse.linalg.expm(coef)
        # Prune the impulse response matrix.
        matrix.data[np.abs(matrix.data) < self.epsilon] = 0
        matrix.eliminate_zeros()
        self.data = cupyx.scipy.sparse.csr_matrix(matrix, dtype=Real)

    def check(self):
        _Component._check_data(self, database, self.get().get().data)

    def __repr__(self):
        s = "Linear    " + self.name + "  "
        if self.data is None:
            s += "invalid"
        else:
            s += "nnz/row: %g"%(self.data.nnz / self.data.shape[0])
        return s

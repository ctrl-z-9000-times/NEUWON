""" A database for neural simulations. """

# TODO: Add ability to sort entities (API stubbs only)

# TODO: Consider making a "ConnectivityMatrix" subclass, instead of using
# sparse boolean matrix, to avoid storing the 1's.
# This could be part of a more general purpose "list" attribute.

from collections.abc import Callable, Iterable, Mapping
import collections
import cupy
import cupyx.scipy.sparse
import itertools
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
        self.class_types = dict()

    def add_class(self, name, instance_class=None) -> 'ClassType':
        return ClassType(self, name, instance_class=instance_class)

    def get_class(self, name) -> 'ClassType':
        if isinstance(name, ClassType): return name
        return self.class_types[str(name)]

    def get_all_classes(self) -> tuple:
        return tuple(self.class_types.values())

    def get_component(self, name) -> '_Component':
        cls, component = str(name).split('.', maxsplit=1)
        return self.get_class(cls).get_component(component)

    def get_all_components(self) -> tuple:
        return tuple(itertools.chain.from_iterable(
                x.components.values() for x in self.class_types.values()))

    def to_host(self) -> 'Database':
        for cls in self.class_types.values():
            for comp in cls.components.values():
                comp.to_host()
        return self

    def check(self, name=None):
        if name is None:
            components = self.get_all_components()
        else:
            try:
                components = self.get_class(name).get_all_components()
            except KeyError:
                components = [self.get_component(component_name)]
        exceptions = []
        for c in components:
            try: c.check()
            except Exception as x: exceptions.append(str(x))
        if exceptions: raise AssertionError(",\n\t".join(sorted(exceptions)))

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

class _DocString:
    def __init__(self, name, doc):
        self._name = str(name)
        self.doc = textwrap.dedent(str(doc)).strip()

    @property
    def name(self):
        return self._name

class ClassType(_DocString):
    def __init__(self, database, name, doc="", instance_class=None):
        if type(self) != ClassType:
            if not doc: doc = type(self).__name__
        _DocString.__init__(self, name, doc)
        if type(self) == ClassType:
            self.__class__ = type(self.name + "Factory", (type(self),), {})
        assert isinstance(database, Database)
        self.database = database
        assert self.name not in self.database.class_types
        self.database.class_types[self.name] = self
        self.size = 0
        self.components = dict()
        self.referenced_by = list()
        self.referenced_by_sparse_matrix_columns = list()
        self.instances = weakref.WeakSet()
        if instance_class is not None:
            inherit = (instance_class, Instance,)
        else:
            inherit = (Instance,)
        self.instance_class = type(
                self.name,
                inherit, {
                "__slots__": (),
                "_cls": self,})

    def get_component(self, name):
        return self.components[str(name)]

    def get_all_components(self) -> tuple:
        return tuple(self.components.values())

    def get_all_instances(self) -> list:
        return [self.instance_class(idx) for idx in range(self.size)]

    def add_attribute(self, name, initial_value=None, dtype=Real, doc=""):
        # TODO: Copy the kwargs & docs from the class definitions back up to here. allow duplication.
        return Attribute(self, name, initial_value=initial_value, dtype=dtype, doc=doc,)

    def add_class_attribute(self, name, value):
        # TODO: Copy the kwargs & docs from the class definitions back up to here. allow duplication.
        return ClassAttribute(self, name, value)

    def add_sparse_matrix(self, name, column_class, doc=""):
        # TODO: Copy the kwargs & docs from the class definitions back up to here. allow duplication.
        return Sparse_Matrix(self, name, column_class, doc=doc,)

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
        old_size  = self.size
        new_size  = old_size + 1
        self.size = new_size
        for x in self.components.values():
            if isinstance(x, Attribute): x._append(old_size, new_size)
            elif isinstance(x, ClassAttribute): pass
            elif isinstance(x, Sparse_Matrix): x._resize()
            else: raise NotImplementedError
        for x in self.referenced_by_sparse_matrix_columns: x._resize()
        obj = self.instance_class(old_size)
        for attribute, value in kwargs.items():
            setattr(obj, attribute, value)
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

    def check(self, name=None):
        if name is None: self.db.check(self)
        else: self.get_component(name).check()

    def __len__(self):
        return self.size

    # def __repr__(self):
    #     s = "Entity".ljust(10) + self.name
    #     if len(self): s += "  %d instances"%len(self)
    #     return s

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
        self.cls = class_type # TODO: Consider renaming this to _cls for consistency?
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

    def to_host(self) -> '_Component':
        """ Abstract method, optional """
        return self

    def to_device(self, device_id=None) -> '_Component':
        """ Abstract method, optional """
        return self

    def check(self):
        """ Abstract method, optional """

    def _check_data(self, data, reference=False):
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
        self.mem = 'host'
        self.data = []
        setattr(self.cls.instance_class, self.name,
                property(self._getter, self._setter, doc=self.doc,))

    def _getter(self, instance):
        value = self.data[instance._idx]
        if hasattr(value, 'get'): value = value.get()
        if self.reference:
            value = self.reference.instance_class(value)
        return value

    def _setter(self, instance, value):
        if self.reference:
            if isinstance(value, self.reference.instance_class):
                value = value._idx
        self.data[instance._idx] = value

    def _append(self, old_size, new_size):
        """ Prepare space for new instances at the end of the array. """
        if len(self.data) < new_size:
            new_data = self._alloc(new_size)
            new_data[:old_size] = self.data[:old_size]
            if self.initial_value is not None:
                new_data[old_size:].fill(self.initial_value)
            self.data = new_data

    def _alloc(self, size):
        """ Returns an empty array. """
        # TODO: IIRC CuPy can not deal with numpy structured arrays...
        #       Detect this issue and revert to using numba arrays.
        #       numba.cuda.to_device(numpy.array(data, dtype=dtype))
        shape = (2 * size,)
        # Special case to squeeze off the empty trailing dimension (1).
        if self.shape != (1,): shape = shape + self.shape
        if self.mem == 'host':
            return np.empty(shape, dtype=self.dtype)
        elif self.mem == 'cuda':
            return cupy.empty(shape, dtype=self.dtype)
        else: raise NotImplementedError

    def get(self):
        """ Returns either "numpy.ndarray" or "cupy.ndarray" """
        return self.data[:len(self.cls)]

    def to_host(self) -> 'Attribute':
        if self.mem == 'host': pass
        elif self.mem == 'cuda': self.data = self.data.get()
        else: raise NotImplementedError
        self.mem = 'host'
        return self

    def to_device(self, device_id=None) -> 'Attribute':
        if self.mem == 'host': self.data = cupy.array(self.data)
        elif self.mem == 'cuda': pass
        else: raise NotImplementedError
        self.mem = 'cuda'
        return self

    def check(self):
        self._check_data(self.get(), reference=self.reference)

    def __repr__(self):
        s = "Attribute " + self.name + "  "
        if self.reference: s += "ref:" + self.reference.name
        else: s += self._dtype_name()
        if self.allow_invalid:
            if self.reference: s += " (maybe NULL)"
            else: s += " (maybe NaN)"
        return s

class ClassAttribute(_Component):
    def __init__(self, class_type, name, initial_value, doc="", units=None,
                allow_invalid=False, valid_range=(None, None),):
        """ Add a singular floating-point value. """
        _Component.__init__(self, class_type, name, doc, units, allow_invalid, valid_range)
        self.initial_value = float(initial_value)
        self.data = self.initial_value
        setattr(self.cls.instance_class, self.name,
                property(self._getter, self._setter, doc=self.doc))

    def _getter(self, instance):
        return self.data

    def _setter(self, instance, value):
        self.data = float(value)

    def get(self):
        return self.data

    def set(self, value):
        self.data = value

    def check(self):
        self._check_data(np.array([self.data]))

    def __repr__(self):
        return "Constant  %s  = %s"%(self.name, str(self.data))

class Sparse_Matrix(_Component):
    # TODO: Consider adding more write methods:
    #       1) Write rows. (done)
    #       2) Insert coordinates.
    #       3) Overwrite the matrix?
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
        if self.column_class != self.cls:
            self.column_class.referenced_by_sparse_matrix_columns.append(self)
        self.dtype = dtype
        self.data = scipy.sparse.csr_matrix((len(self.cls), len(self.column_class)), dtype=self.dtype)
        self.fmt = 'csr'
        setattr(self.cls.instance_class, self.name,
                property(self._getter, self._setter, doc=self.doc))

    def _getter(self, instance):
        if self.fmt == 'lil':
            vec = self.data[instance._idx]
            return (vec.rows, vec.data)
        else: raise NotImplementedError

    def _setter(self, instance, value):
        columns, data = value
        self.write_row(instance._idx, columns, data)

    def to_fmt(self, fmt):
        """ Argument fmt is one of: "csr", "lil". """
        if self.fmt != fmt:
            if   fmt == 'lil': self.to_lil()
            if   fmt == 'coo': self.to_coo()
            elif fmt == 'csr': self.to_csr()
            else: raise ValueError(fmt)
        return self

    def to_lil(self):
        self.data = scipy.sparse.lil_matrix(self.data, shape=self.data.shape)
        self.fmt = "lil"
        return self

    def to_coo(self):
        self.data = scipy.sparse.coo_matrix(self.data, shape=self.data.shape)
        self.fmt = "coo"
        return self

    def to_csr(self):
        self.data = scipy.sparse.csr_matrix(self.data, shape=self.data.shape)
        self.fmt = "csr"
        return self

    def to_host(self) -> 'Sparse_Matrix':
        1/0
        return self

    def to_device(self, device_id=None) -> 'Sparse_Matrix':
        1/0
        return self

    @property
    def shape(self):
        return (len(self.cls), len(self.column_class))

    def _resize(self):
        self.data.resize(self.shape)

    def get(self):
        return self.data

    def set(self, new_matrix):
        assert new_matrix.shape == self.shape
        self.data = new_matrix
        self.to_csr()

    def write_row(self, row, columns, values):
        self.to_fmt('lil')
        r = int(row)
        self.data.rows[r].clear()
        self.data.data[r].clear()
        columns = [x._idx if isinstance(x, Instance) else int(x) for x in columns]
        order = np.argsort(columns)
        self.data.rows[r].extend(np.take(columns, order))
        self.data.data[r].extend(np.take(values, order))

    def check(self):
        self._check_data(self.data.data)

    def __repr__(self):
        s = "Matrix    " + self.name + "  "
        if self.reference: s += "ref:" + self.reference.name
        s += self._dtype_name()
        if self.allow_invalid:
            if self.reference: s += " (maybe NULL)"
            else: s += " (maybe NaN)"
        try: nnz_per_row = self.data.nnz / self.data.shape[0]
        except ZeroDivisionError: nnz_per_row = 0
        s += " nnz/row: %g"%nnz_per_row
        return s

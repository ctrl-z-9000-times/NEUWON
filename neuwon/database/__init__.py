""" A database for neural simulations. """

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
import textwrap
import weakref

Real    = np.dtype('f8')
epsilon = np.finfo(Real).eps
Pointer = np.dtype('u4')
NULL    = np.iinfo(Pointer).max
Instance = np.dtype([('cls', np.dtype('u4')), ('idx', Pointer)]) # Naming conflicts :(

class Database:
    def __init__(self):
        self.class_types = dict()

    def add_class(self, name, instance_class=None) -> 'ClassType':
        return ClassType(self, name, instance_class=instance_class)

    def get(self, name) -> 'ClassType' or '_DataComponent':
        if isinstance(name, ClassType):
            assert name.get_database() is self
            return name
        if isinstance(name, _DataComponent):
            assert name.get_database() is self
            return name
        _cls, _, attr = str(name).partition('.')
        obj = self.class_types[_cls]
        if attr: obj = obj.components[attr]
        return obj

    def get_class(self, name):
        _cls = self.get(name)
        if isinstance(_cls, _DataComponent): _cls = _cls.get_class()
        return _cls

    def get_component(self, name):
        component = self.get(name)
        assert isinstance(component, _DataComponent)
        return component

    def get_data(self, name):
        return self.get_component(name).get_data()

    def set_data(self, name, value):
        self.get_component(name).set_data(value)

    def get_all_classes(self) -> tuple:
        return tuple(self.class_types.values())

    def get_all_components(self) -> tuple:
        return tuple(itertools.chain.from_iterable(
                x.components.values() for x in self.class_types.values()))

    def to_host(self) -> 'Database':
        for _cls in self.class_types.values():
            for comp in _cls.components.values():
                comp.to_host()
        return self

    def to_device(self):
        for _cls in self.class_types.values():
            for x in _cls.components.values():
                x.to_device()
        return self

    def sort(self):
        """
        NOTES:

        When classes use pointers to other classes as sort-keys then they create
        a dependency in their sorting-order, which the database will need to
        deal with. The database looks at each classes sorting function and
        determines its dependencies (pointers which it uses as sort keys) and
        makes a DAG of the dependencies. Then it flattens the DAG into a
        topologically sorted order to sort the data in.

        EXAMPLE OF SORT ORDER DEPENDENCIES:
              1) Sort cells by their assigned CPU thread,
              2) Sort sections by cell & topologically,
              3) Sort segment by section & index on section,
              4) Sort ion channels by segment.

        Be efficiency by updating pointers in a single batch.

        If users want to topological sort their data then they need to make an
        attribute to hold the topologically sorted order, and then use that
        attribute as a sort_key.
        """
        raise NotImplementedError

    def check(self, name=None):
        if name is None:
            components = self.get_all_components()
        else:
            x = self.get(name)
            if isinstance(x, ClassType):
                components = x.get_all_components()
            else:
                components = [x]
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

    def get_name(self): return self.name
    def get_doc(self): return self.doc

class ClassType(_DocString):
    def __init__(self, database, name, doc="", instance_class=None, sort_key=tuple()):
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
        self.sort_key = tuple(self.database.get_component(x) for x in
                (sort_key if isinstance(sort_key, Iterable) else (sort_key,)))

    def get(self, name):
        return self.components[str(name)]

    def get_data(self, name):
        return self.components[str(name)].get_data()

    def get_all_components(self) -> tuple:
        return tuple(self.components.values())

    def get_all_instances(self) -> list:
        return [self.instance_class(idx) for idx in range(self.size)]

    def get_database(self):
        return self.database

    def add_attribute(self, name, initial_value=None, dtype=Real, shape=(1,),
                doc="", units=None, allow_invalid=False, valid_range=(None, None),):
        return Attribute(self, name, initial_value=initial_value, dtype=dtype, shape=shape,
                doc=doc, units=units, allow_invalid=allow_invalid, valid_range=valid_range,)

    def add_class_attribute(self, name, initial_value, dtype=Real, shape=(1,),
                doc="", units=None, allow_invalid=False, valid_range=(None, None),):
        return ClassAttribute(self, name, initial_value, dtype=dtype, shape=shape,
                doc=doc, units=units, allow_invalid=allow_invalid, valid_range=valid_range,)

    def add_list_attribute(self, name, dtype=Real, shape=(1,),
                doc="", units=None, allow_invalid=False, valid_range=(None, None),):
        return ListAttribute(self, name, dtype=dtype, shape=shape,
                doc=doc, units=units, allow_invalid=allow_invalid, valid_range=valid_range,)

    def add_sparse_matrix(self, name, column, dtype=Real,
                doc="", units=None, allow_invalid=False, valid_range=(None, None),):
        return Sparse_Matrix(self, name, column, dtype=dtype,
                doc=doc, units=units, allow_invalid=allow_invalid, valid_range=valid_range,)

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
        else: self.get(name).check()

    def __len__(self):
        return self.size

    def __repr__(self):
        return "%s:%"%(type(self).__name__, str(len(self)))

class Instance:
    __slots__ = ("_idx", "__weakref__")

    def __init__(self, index):
        self._idx = int(index)
        self._cls.instances.add(self)
        assert self._idx in range(len(self._cls))

    def __repr__(self):
        return "%s:%d"%(type(self).__name__, self._idx)

class _DataComponent(_DocString):
    """ Abstract class for all data components. """
    def __init__(self, class_type, name,
                doc, units, shape, dtype, initial_value, allow_invalid, valid_range):
        _DocString.__init__(self, name, doc)
        assert self.name not in class_type.components
        self._cls = class_type
        self._cls.components[self.name] = self
        if shape is None: pass
        elif isinstance(shape, Iterable):
            self.shape = tuple(int(round(x)) for x in shape)
        else:
            self.shape = (int(round(shape)),)
        if isinstance(dtype, str) or isinstance(dtype, ClassType):
            self.dtype = Pointer
            self.initial_value = NULL
            self.reference = self._cls.database.get_class(dtype)
            self.reference.referenced_by.append(self)
        else:
            self.dtype = np.dtype(dtype)
            self.initial_value = self.dtype.type(initial_value)
            self.reference = False
        self.units = None if units is None else str(units)
        self.allow_invalid = bool(allow_invalid)
        self.valid_range = tuple(valid_range)
        if None not in self.valid_range: self.valid_range = tuple(sorted(self.valid_range))
        assert len(self.valid_range) == 2
        self.mem = 'host'
        setattr(self._cls.instance_class, self.name,
                property(self._getter, self._setter, doc=self.doc,))

    def _getter(self, instance):
        """ Get the instance's data value. """
        raise NotImplementedError
    def _setter(self, instance, value):
        """ Set the instance's data value. """
        raise NotImplementedError

    def get_data(self):
        """ Returns all data for this component. """
        raise NotImplementedError
    def set_data(self, value):
        """ Replace the entire data component with a new set of values. """
        raise NotImplementedError

    def get_units(self):            return self.units
    def get_dtype(self):            return self.dtype
    def get_shape(self):            return self.shape
    def get_initial_value(self):    return self.initial_value
    def get_class(self):            return self._cls
    def get_database(self):         return self._cls.database

    def get_memory_space(self):
        """
        Returns the current location of the data component:

        Returns "host" when located in python's memory space.
        Returns "cuda" when located in CUDA's memory space.
        """
        return self.mem

    def to_host(self) -> 'self':
        """ Abstract method, optional """
        return self

    def to_device(self) -> 'self':
        """ Abstract method, optional """
        return self

    def check(self):
        """
        Check data for values which are:
            NaN
            NULL
            Out of bounds
        """
        data = self.get_data()
        if isinstance(self, ClassAttribute):  data = np.array([data])
        elif isinstance(self, Sparse_Matrix): data = data.data
        reference = getattr(self, 'reference', False)
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

    def _type_info(self):
        s = ""
        if self.reference: s += "ref:" + self.reference.name
        else: s += self.dtype.name
        if self.shape != (1,): s += repr(list(self.shape))
        return s

    def __repr__(self):
        return "%s: %s.%s %s"%(type(self).__name__, self._cls.name, self.name, self._type_info())

class Attribute(_DataComponent):
    """ """
    def __init__(self, class_type, name, initial_value=None, dtype=Real, shape=(1,),
                doc="", units=None, allow_invalid=False, valid_range=(None, None),):
        """
        Add an instance variable to a class type.

        Argument dtype is one of:
            * An instance of numpy.dtype
            * The name of a class, to make pointers to instances of that class.

        If the dangling references are allow to be NULL then they are replaced
        with NULL references. Otherwise the entity containing the reference is
        destroyed. Destroying entities can cause a chain reaction of destruction.
        """
        _DataComponent.__init__(self, class_type, name,
            doc=doc, units=units, dtype=dtype, shape=shape, initial_value=initial_value,
            allow_invalid=allow_invalid, valid_range=valid_range)
        self.data = self._alloc(0)
        self._append(0, len(self._cls))

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
        if self.shape != (1,): # Don't append empty trailing dimension.
            shape += self.shape
        if self.mem == 'host':
            return np.empty(shape, dtype=self.dtype)
        elif self.mem == 'cuda':
            return cupy.empty(shape, dtype=self.dtype)
        else: raise NotImplementedError

    def get_data(self):
        """ Returns either "numpy.ndarray" or "cupy.ndarray" """
        return self.data[:len(self._cls)]

    def set_data(self, value):
        assert len(value) == len(self.get_class())
        if self.mem == "host":
            self.data = np.array(value, dtype=self.dtype)
        elif self.mem == "cuda":
            self.data = cupy.array(value, self.dtype)
        else: raise NotImplementedError
        if self.shape != (1,): assert self.data.shape[1:] == self.shape

    def to_host(self) -> 'Attribute':
        if self.mem == 'host': pass
        elif self.mem == 'cuda': self.data = self.data.get()
        else: raise NotImplementedError
        self.mem = 'host'
        return self

    def to_device(self) -> 'Attribute':
        if self.mem == 'host': self.data = cupy.array(self.data)
        elif self.mem == 'cuda': pass
        else: raise NotImplementedError
        self.mem = 'cuda'
        return self

class ClassAttribute(_DataComponent):
    """ """
    def __init__(self, class_type, name, initial_value,
                dtype=Real, shape=(1,),
                doc="", units=None,
                allow_invalid=False, valid_range=(None, None),):
        """ Add a class variable to a class type.

        All instance of the class will use a single shared value for this attribute.
        """
        _DataComponent.__init__(self, class_type, name,
                dtype=dtype, shape=shape, doc=doc, units=units, initial_value=initial_value,
                allow_invalid=allow_invalid, valid_range=valid_range)
        self.data = self.initial_value

    def _getter(self, instance):
        return self.get_data()

    def _setter(self, instance, value):
        self.set_data(value)

    def get_data(self):
        return self.data

    def set_data(self, value):
        self.data = self.dtype.type(value)

class ListAttribute(_DataComponent):
    """ """
    def __init__(self, class_type, name, dtype=Real, shape=(1,),
                doc="", units=None, allow_invalid=False, valid_range=(None, None),):
        """ Special case for Attributes holding lists. """
        _DataComponent.__init__(self, class_type, name, initial_value=tuple(),
                dtype=dtype, shape=shape, doc=doc, units=units,
                allow_invalid=allow_invalid, valid_range=valid_range)
        self.fmt = 'list'
        self.data = self._alloc(len(self._cls))
        self._append(0, len(self._cls))

    def _append(self, old_size, new_size):
        """ Prepare space for new instances at the end of the array. """
        if len(self.data) < new_size:
            new_data = self._alloc(new_size)
            new_data[:old_size] = self.data[:old_size]
            self.data = new_data
        for idx in range(old_size, new_size): self.data[idx] = []

    def _alloc(self, size):
        """ Returns an empty array. """
        shape = (2 * size,)
        if self.shape != (1,): # Don't append empty trailing dimension.
            shape += self.shape
        if self.mem == 'host':
            return np.empty(shape, dtype=object)
        elif self.mem == 'cuda':
            return cupy.empty(shape, dtype=self.dtype)
        else: raise NotImplementedError(self.mem)

    def to_list(self):
        if self.fmt == 'list': pass
        elif self.fmt == 'array':
            1/0
            self.fmt = 'list'
        else: raise NotImplementedError(self.fmt)
        return self

    def to_array(self):
        if self.fmt == 'list':
            1/0
            self.fmt = 'array'
        elif self.fmt == 'array': pass
        else: raise NotImplementedError(self.fmt)
        return self

    def to_host(self):
         # TODO!
        return self
    def to_device(self):
         # TODO!
        return self

    def _getter(self, instance):
        return self.data[instance._idx]

    def _setter(self, instance, value):
        1/0 # TODO!

    def get_data(self):
        return self.data[:len(self._cls)]

    def set_data(self, value):
        1/0 # TODO!

    def get_connectivity_matrix(self):
        """
        For lists containing pointers. Copies & returns this component as a real
        sparse matrix class, including the data buffer of 1's.

        Return a CSR matrix.
        """
        data = self.to_array()
        return 1/0

class Sparse_Matrix(_DataComponent):
    """ """
    # TODO: Consider adding more write methods:
    #       1) Write rows. (done)
    #       2) Insert coordinates.
    #       3) Overwrite the matrix?
    def __init__(self, class_type, name, column, dtype=Real, doc="", units=None,
                allow_invalid=False, valid_range=(None, None),):
        """
        Add a sparse matrix that is indexed by Entities. This is useful for
        implementing any-to-any connections between entities.

        Sparse matrices may contain references but they will not trigger a
        recursive destruction of entities. Instead, references to destroyed
        entities are simply removed from the sparse matrix.
        """
        _DataComponent.__init__(self, class_type, name,
                dtype=dtype, shape=None, doc=doc, units=units, initial_value=0,
                allow_invalid=allow_invalid, valid_range=valid_range)
        if isinstance(column, ClassType):
            self.column = column
        else:
            self.column = self._cls.database.get_class(column)
        if self.column != self._cls:
            self.column.referenced_by_sparse_matrix_columns.append(self)
        self.dtype = dtype
        self.data = scipy.sparse.csr_matrix((len(self._cls), len(self.column)), dtype=self.dtype)
        self.fmt = 'csr'
        self.sparse_module = scipy.sparse

    def _getter(self, instance):
        if self.fmt == 'lil':
            vec = self.data[instance._idx]
            return (vec.rows, vec.data)
        else: raise NotImplementedError

    def _setter(self, instance, value):
        columns, data = value
        self.write_row(instance._idx, columns, data)

    def to_lil(self):
        if self.fmt != "lil":
            self.data = self.sparse_module.lil_matrix(self.data)
            self.fmt = "lil"
        return self

    def to_coo(self):
        if self.fmt != "coo":
            self.data = self.sparse_module.coo_matrix(self.data)
            self.fmt = "coo"
        return self

    def to_csr(self):
        if self.fmt != "csr":
            self.data = self.sparse_module.csr_matrix(self.data)
            self.fmt = "csr"
        return self

    def to_host(self):
        if self.mem == 'host': pass
        elif self.mem == 'cuda':
            self.data = self.data.get()
            self.sparse_module = scipy.sparse
            self.mem = 'host'
        else: raise NotImplementedError
        return self

    def to_device(self):
        if self.mem == 'host':
            if self.fmt == 'lil': self.fmt = 'csr'
            self.sparse_module = cupyx.scipy.sparse
            if self.fmt == 'csr': self.data = self.sparse_module.csr_matrix(self.data)
            elif self.fmt == 'coo': self.data = self.sparse_module.coo_matrix(self.data)
            else: raise NotImplementedError
            self.mem = 'cuda'
        elif self.mem == 'cuda': pass
        else: raise NotImplementedError
        return self

    @property
    def shape(self):
        return (len(self._cls), len(self.column))

    def _resize(self):
        self.data.resize(self.shape)

    def get_data(self):
        return self.data

    def set_data(self, new_matrix):
        assert new_matrix.shape == self.shape
        self.data = new_matrix
        self.fmt = "unknown"
        self.to_csr()

    def write_row(self, row, columns, values):
        self.to_lil()
        r = int(row)
        self.data.rows[r].clear()
        self.data.data[r].clear()
        columns = [x._idx if isinstance(x, Instance) else int(x) for x in columns]
        order = np.argsort(columns)
        self.data.rows[r].extend(np.take(columns, order))
        self.data.data[r].extend(np.take(values, order))

    def __repr__(self):
        s = _DataComponent.__repr__(self)
        try: nnz_per_row = self.data.nnz / self.data.shape[0]
        except ZeroDivisionError: nnz_per_row = 0
        s += " nnz/row: %g"%nnz_per_row
        return s

ClassType.add_attribute.__doc__         = Attribute.__init__.__doc__
ClassType.add_class_attribute.__doc__   = ClassAttribute.__init__.__doc__
ClassType.add_list_attribute.__doc__    = ListAttribute.__init__.__doc__
ClassType.add_sparse_matrix.__doc__     = Sparse_Matrix.__init__.__doc__

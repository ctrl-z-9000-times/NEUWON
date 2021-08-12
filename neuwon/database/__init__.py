""" A database for neural simulations. """

from collections.abc import Callable, Iterable, Mapping
import collections
import cupy
import cupyx.scipy.sparse
import itertools
import numpy as np
import scipy.sparse
import textwrap
import weakref

Real    = np.dtype('f8')
epsilon = np.finfo(Real).eps
Pointer = np.dtype('u4')
NULL    = np.iinfo(Pointer).max

class Database:
    def __init__(self):
        """ Create a new empty database. """
        self.class_types = dict()

    def add_class(self, name: str, base_class:type=None) -> 'DB_Class':
        return DB_Class(self, name, base_class=base_class)

    def get(self, name: str) -> 'DB_Class' or '_DataComponent':
        """ Get the database's internal representation of the named thing.

        Argument name can refer to either:
            * A DB_Class
            * An Attibute, ClassAttribute, or Sparse_Matrix

        Example:
        >>> Foo = database.add_class("Foo")
        >>> bar = Foo.add_attribute("bar")
        >>> assert Foo == database.get("Foo")
        >>> assert bar == database.get("Foo.bar")
        """
        if isinstance(name, DB_Class):
            assert name.database is self
            return name
        elif isinstance(name, _DataComponent):
            assert name._cls.database is self
            return name
        elif isinstance(name, str): pass
        else:
            name = str(name)
        _cls, _, attr = name.partition('.')
        obj = self.class_types[_cls]
        if attr: obj = obj.components[attr]
        return obj

    def get_class(self, name: str):
        """ Get the database's internal representation of a class.

        Argument name can be anything which `self.get(name)` accepts.
        """
        _cls = self.get(name)
        if isinstance(_cls, _DataComponent): _cls = _cls.get_class()
        return _cls

    def get_component(self, name: str):
        component = self.get(name)
        assert isinstance(component, _DataComponent)
        return component

    def get_data(self, name: str):
        """ Shortcut to: self.get_component(name).get_data() """
        return self.get_component(name).get_data()

    def set_data(self, name: str, value):
        """ Shortcut to: self.get_component(name).set_data(value) """
        self.get_component(name).set_data(value)

    def get_all_classes(self) -> tuple:
        return tuple(self.class_types.values())

    def get_all_components(self) -> tuple:
        return tuple(itertools.chain.from_iterable(
                x.components.values() for x in self.class_types.values()))

    def to_host(self) -> 'Database':
        """ Move all data to this python process's memory space. """
        for _cls in self.class_types.values():
            for comp in _cls.components.values():
                comp.to_host()
        return self

    def to_device(self):
        """ Move all data components to the default CUDA device. """
        for _cls in self.class_types.values():
            for x in _cls.components.values():
                x.to_device()
        return self

    def sort(self):
        """ Sort all DB_Classes according to their "sort_key" arguments. """
        1/0
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

    def check(self, name:str=None):
        """ Run all configured checks on the database.

        Argument name refers to a class or data component and if given it limits
                the scope of what is checked.
        """
        if name is None:
            components = self.get_all_components()
        else:
            x = self.get(name)
            if isinstance(x, DB_Class):
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

class _Documentation:
    def __init__(self, name, doc):
        self.name = str(name)
        self.doc = textwrap.dedent(str(doc)).strip()

    def get_name(self) -> str: return self.name
    def get_doc(self) -> str:  return self.doc

class _DB_Object:
    """ Super class for the external representation of a class.

    Each instance of DB_Class creates a new subclass of this class, with the
    purpose of linking all of the subclass instances to the database instance.

    This class takes lower precedence in the method resolution order (MRO) than
    the user's custom base_class, so they may override any/all of these methods.
    """
    __slots__ = ()

    def __init__(self, **kwargs):
        """
        Keyword arguments are assigned to the new instance, for example:
                >>> obj = MyClass(x=3, y=4)
            Is equivalent to:
                >>> obj = MyClass()
                >>> obj.x = 3
                >>> obj.y = 4
        """
        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

    # TODO: Should the user be allowed to override this method?
    def __eq__(self, other):
        return ((type(self) is type(other)) and (self._idx == other._idx))

    def __repr__(self):
        return "<%s:%d>"%(self._cls.name, self._idx)

class DB_Class(_Documentation):
    """ This is the database's internal representation of a class type. """
    def __init__(self, database, name: str, base_class=None, sort_key=tuple(), doc="",):
        """ Create a new class which is managed by the database.

        Argument base_class:

        Argument sort_key:

        """ # TODO-DOC
        _Documentation.__init__(self, name, doc)
        assert isinstance(database, Database)
        self.database = database
        assert self.name not in self.database.class_types
        self.database.class_types[self.name] = self
        self.size = 0
        self.components = dict()
        self.referenced_by = list()
        self.referenced_by_sparse_matrix_columns = list()
        self.instances = weakref.WeakValueDictionary()
        self.sort_key = tuple(self.database.get_component(x) for x in
                (sort_key if isinstance(sort_key, Iterable) else (sort_key,)))
        # Make a new subclass to represent instances which are part of *this* database.
        bases = (_DB_Object,)
        if base_class is not None: bases = (base_class,) + bases
        __slots__ = ("_idx",)
        if not hasattr(base_class, "__weakref__"): __slots__ += ("__weakref__",)
        self.instance_type = type(self.name, bases, {
                "_cls": self,
                "__slots__": __slots__,
                "__init__": self._instance__init__,
                "get_unstable_index": self._get_unstable_index,
                "get_database_class": self._get_database_class,
                "__module__": bases[0].__module__,
        })
        self.instance_type.__init__.__doc__ = base_class.__init__.__doc__ # This modifies a shared object, which is probably a bug.

    @staticmethod
    def _instance__init__(new_obj, *args, **kwargs):
        self = new_obj._cls
        self.instances[id(new_obj)] = new_obj
        old_size  = self.size
        new_size  = old_size + 1
        self.size = new_size
        for x in self.components.values():
            if isinstance(x, Attribute): x._append(old_size, new_size)
            elif isinstance(x, ClassAttribute): pass
            elif isinstance(x, Sparse_Matrix): x._resize()
            elif isinstance(x, ListAttribute): x._append(old_size, new_size)
            else: raise NotImplementedError(type(x))
        for x in self.referenced_by_sparse_matrix_columns: x._resize()
        new_obj._idx = old_size
        super(type(new_obj), new_obj).__init__(*args, **kwargs)

    @staticmethod
    def _get_unstable_index(db_object):
        """ TODO: Explain how / when this index changes. """
        return db_object._idx

    @staticmethod
    def _get_database_class(db_object):
        """ Get the database's internal representation of this object's type. """
        return db_object._cls

    def get_instance_type(self) -> _DB_Object:
        """ Get the public / external representation of this DB_Class. """
        return self.instance_type

    def index_to_object(self, unstable_index: int) -> _DB_Object:
        """ Create a new _DB_Object """ # TODO-DOC
        idx = int(unstable_index)
        if idx == NULL: return None
        assert 0 <= idx < len(self)
        cls = self.instance_type
        obj = cls.__new__(cls)
        obj._idx = idx
        return obj

    def get(self, name: str) -> '_DataComponent':
        """
        Get the database's internal representation of a data component that is
        attached to this DB_Class.
        """
        return self.components[str(name)]

    def get_data(self, name: str):
        """ Shortcut to: self.get(name).get_data() """
        return self.get(name).get_data()

    def get_all_components(self) -> tuple:
        """
        Returns a tuple containing all data components that are attached to this
        class.
        """
        return tuple(self.components.values())

    def get_all_instances(self) -> list:
        """
        Returns a list containing every instance of this class which currently
        exists.
        """
        return [self.index_to_object(idx) for idx in range(self.size)]

    def get_database(self) -> Database:
        return self.database

    def add_attribute(self, name:str, initial_value=None, dtype=Real, shape=(1,),
                doc:str="", units:str="", allow_invalid=False, valid_range=(None, None),):
        return Attribute(self, name, initial_value=initial_value, dtype=dtype, shape=shape,
                doc=doc, units=units, allow_invalid=allow_invalid, valid_range=valid_range,)

    def add_class_attribute(self, name:str, initial_value, dtype=Real, shape=(1,),
                doc:str="", units:str="", allow_invalid=False, valid_range=(None, None),):
        return ClassAttribute(self, name, initial_value, dtype=dtype, shape=shape,
                doc=doc, units=units, allow_invalid=allow_invalid, valid_range=valid_range,)

    def add_sparse_matrix(self, name:str, column, dtype=Real,
                doc:str="", units:str="", allow_invalid=False, valid_range=(None, None),):
        return Sparse_Matrix(self, name, column, dtype=dtype,
                doc=doc, units=units, allow_invalid=allow_invalid, valid_range=valid_range,)

    def add_connectivity_matrix(self, name:str, column, doc=""):
        return Connectivity_Matrix(self, name, column, doc=doc)

    def destroy(self):
        """
        TODO: Explain how this determines which objects to destroy.

        If the dangling references are allow to be NULL then they are replaced
        with NULL references. Otherwise the entity containing the reference is
        destroyed. Destroying entities can cause a chain reaction of destruction.

        Sparse matrices may contain references but they will not trigger a
        recursive destruction of entities. Instead, references to destroyed
        entities are simply removed from the sparse matrix.
        """
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

    def check(self, name:str=None):
        """ Run all configured checks on this class or on only the given data component. """
        if name is None: self.db.check(self)
        else: self.get(name).check()

    def __len__(self):
        """ Returns how many instances of this class currently exist. """
        return self.size

    def __repr__(self):
        return "<DB_Class '%s'>"%(self.name)

class _DataComponent(_Documentation):
    """ Abstract class for all types of data storage. """
    
    # TODO: Consider renaming "class_type" to "db_class" throughout, also add
    # type annotations in arglists when appropriate
    
    def __init__(self, class_type, name,
                doc, units, shape, dtype, initial_value, allow_invalid, valid_range):
        _Documentation.__init__(self, name, doc)
        assert isinstance(class_type, DB_Class)
        assert self.name not in class_type.components
        self._cls = class_type
        self._cls.components[self.name] = self
        if shape is None: pass
        elif isinstance(shape, Iterable):
            self.shape = tuple(round(x) for x in shape)
        else:
            self.shape = (round(shape),)
        if isinstance(dtype, str) or isinstance(dtype, DB_Class):
            self.dtype = Pointer
            self.initial_value = NULL
            self.reference = self._cls.database.get_class(dtype)
            self.reference.referenced_by.append(self)
        else:
            self.dtype = np.dtype(dtype)
            if initial_value is None:
                self.initial_value = None
            else:
                self.initial_value = self.dtype.type(initial_value)
            self.reference = False
        self.units = str(units)
        self.allow_invalid = bool(allow_invalid)
        self.valid_range = tuple(valid_range)
        if None not in self.valid_range: self.valid_range = tuple(sorted(self.valid_range))
        assert len(self.valid_range) == 2
        self.mem = 'host'
        setattr(self._cls.instance_type, self.name,
                property(self._getter, self._setter, doc=self.doc,))

    def _getter(self, instance):
        """ Get the instance's data value. """
        raise NotImplementedError
    def _setter(self, instance, value):
        """ Set the instance's data value. """
        raise NotImplementedError

    def get_data(self):
        """ Returns all data for this component. """
        raise NotImplementedError(type(self))
    def set_data(self, value):
        """ Replace this entire data component with a new set of values. """
        raise NotImplementedError(type(self))

    def get_units(self) -> str:         return self.units
    def get_dtype(self) -> np.dtype:    return self.dtype
    def get_shape(self) -> tuple:       return self.shape
    def get_initial_value(self):        return self.initial_value
    def get_class(self) -> DB_Class:    return self._cls
    def get_database(self) -> Database: return self._cls.database

    def get_memory_space(self) -> str:
        """
        Returns the current location of the data component:

        Returns "host" when located in python's memory space.
        Returns "cuda" when located in CUDA's memory space.
        """
        return self.mem

    def to_host(self) -> 'self':
        """ Move the data to the CPU and into this python process's memory space. """
        # Abstract method, optional.
        return self

    def to_device(self) -> 'self':
        """ Move the data to the target device's memory space. """
        # Abstract method, optional.
        return self

    def free(self):
        """
        Release the memory used by this data component. The next time the data
        is accessed it will be reallocated and set to the default value.
        """
        # Abstract method, optional.

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
        return "<%s: %s.%s %s>"%(type(self).__name__, self._cls.name, self.name, self._type_info())

class Attribute(_DataComponent):
    """ This is the database's internal representation of an instance variable. """
    def __init__(self, class_type, name:str, initial_value=None, dtype=Real, shape=(1,),
                doc:str="", units:str="", allow_invalid=False, valid_range=(None, None),):
        """ Add an instance variable to a class type. """
        _DataComponent.__init__(self, class_type, name,
            doc=doc, units=units, dtype=dtype, shape=shape, initial_value=initial_value,
            allow_invalid=allow_invalid, valid_range=valid_range)
        self.data = self._alloc(0)
        self._append(0, len(self._cls))

    def _getter(self, instance):
        value = self.data[instance._idx]
        if hasattr(value, 'get'): value = value.get()
        if self.reference:
            return self.reference.index_to_object(value)
        return value

    def _setter(self, instance, value):
        if self.reference:
            if value is None:
                value = NULL
            else:
                assert isinstance(value, self.reference.instance_type)
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
        else: raise NotImplementedError(self.mem)

    def get_data(self):
        """ Returns either "numpy.ndarray" or "cupy.ndarray" """
        return self.data[:len(self._cls)]

    def set_data(self, value):
        size = len(self._cls)
        assert len(value) == size
        self.data = self._alloc(size)
        self.data[:size] = value
        if self.initial_value is not None:
            self.data[size:].fill(self.initial_value)

    def to_host(self) -> 'self':
        if self.mem == 'host': pass
        elif self.mem == 'cuda':
            self.data = self.data.get()
            self.mem = 'host'
        else: raise NotImplementedError(self.mem)
        return self

    def to_device(self) -> 'self':
        if self.mem == 'host':
            self.data = cupy.array(self.data)
            self.mem = 'cuda'
        elif self.mem == 'cuda': pass
        else: raise NotImplementedError(self.mem)
        return self

class ClassAttribute(_DataComponent):
    """ This is the database's internal representation of a class variable. """
    def __init__(self, class_type, name:str, initial_value,
                dtype=Real, shape=(1,),
                doc:str="", units:str="",
                allow_invalid=False, valid_range=(None, None),):
        """ Add a class variable to a class type.

        All instance of the class will use a single shared value for this attribute.
        """
        _DataComponent.__init__(self, class_type, name,
                dtype=dtype, shape=shape, doc=doc, units=units, initial_value=initial_value,
                allow_invalid=allow_invalid, valid_range=valid_range)
        self.data = self.initial_value

    def _getter(self, instance):
        if self.reference: raise NotImplementedError("todo?")
        return self.data

    def _setter(self, instance, value):
        if self.reference: raise NotImplementedError("todo?")
        self.data = self.dtype.type(value)

    def get_data(self):
        return self.data

    def set_data(self, value):
        self.data = self.dtype.type(value)

class Sparse_Matrix(_DataComponent):
    """ """ # TODO-DOC

    # TODO: Consider adding more write methods:
    #       1) Write rows. (done)
    #       2) Insert coordinates.
    #               Notes: first convert format to either lil or coo
    #       3) Overwrite the matrix. (done)

    # TODO: Figure out if/when to call mat.eliminate_zeros() and sort too.

    def __init__(self, class_type, name, column, dtype=Real, doc:str="", units:str="",
                allow_invalid=False, valid_range=(None, None),):
        """
        Add a sparse matrix that is indexed by Entities. This is useful for
        implementing any-to-any connections between entities.
        """
        _DataComponent.__init__(self, class_type, name,
                dtype=dtype, shape=None, doc=doc, units=units, initial_value=0.,
                allow_invalid=allow_invalid, valid_range=valid_range)
        self.column = self._cls.database.get_class(column)
        self.column.referenced_by_sparse_matrix_columns.append(self)
        self.shape = (len(self._cls), len(self.column))
        self.fmt = 'lil'
        self.data = self._matrix_class(self.shape, dtype=self.dtype)
        self._host_lil_mem = None

    @property
    def _sparse_module(self):
        if   self.mem == 'host': return scipy.sparse
        elif self.mem == 'cuda': return cupyx.scipy.sparse
        else: raise NotImplementedError(self.mem)

    @property
    def _matrix_class(self):
        if   self.fmt == 'lil': return self._sparse_module.lil_matrix
        elif self.fmt == 'coo': return self._sparse_module.coo_matrix
        elif self.fmt == 'csr': return self._sparse_module.csr_matrix
        else: raise NotImplementedError(self.fmt)

    def _getter(self, instance):
        if self.fmt == 'coo': self.to_lil()
        if self.fmt == 'lil':
            lil_mat = self.data
            index_to_object = self.column.index_to_object
            cols = [index_to_object(x) for x in lil_mat.rows[instance._idx]]
            data = list(lil_mat.data[instance._idx])
            if self.reference: raise NotImplementedError("Implement maybe?")
            return (cols, data)
        elif self.fmt == 'csr':
            help(self.data)
        else: raise NotImplementedError(self.fmt)

    def _setter(self, instance, value):
        columns, data = value
        self.write_row(instance._idx, columns, data)

    def to_lil(self) -> 'self':
        if self.fmt != "lil":
            self.fmt = "lil"
            self.data = self._matrix_class(self.data, dtype=self.dtype)
        return self

    def to_coo(self) -> 'self':
        if self.fmt != "coo":
            self.fmt = "coo"
            self._host_lil_mem = None
            self.data = self._matrix_class(self.data, dtype=self.dtype)
        return self

    def to_csr(self) -> 'self':
        if self.fmt != "csr":
            self.fmt = "csr"
            self._host_lil_mem = None
            self.data = self._matrix_class(self.data, dtype=self.dtype)
        return self

    def to_host(self) -> 'self':
        if self.mem == 'host': pass
        elif self.mem == 'cuda':
            self.data = self.data.get()
            self.mem = 'host'
        else: raise NotImplementedError(self.mem)
        return self

    def to_device(self) -> 'self':
        if self.mem == 'host':
            self.mem = 'cuda'
            self._host_lil_mem = None
            if self.fmt == 'lil': self.fmt = 'csr'
            if self.fmt == 'csr': self.data = self._matrix_class(self.data, dtype=self.dtype)
            elif self.fmt == 'coo': self.data = self._matrix_class(self.data, dtype=self.dtype)
            else: raise NotImplementedError(self.fmt)
        elif self.mem == 'cuda': pass
        else: raise NotImplementedError(self.mem)
        return self

    def _resize(self):
        old_shape = self.data.shape
        new_shape = self.shape = (len(self._cls), len(self.column))

        if old_shape == new_shape: return

        if self.fmt == 'csr': self.to_lil()
        if self.fmt == 'lil':
            if self._host_lil_mem is None or self._host_lil_mem.shape[0] < new_shape[0]:
                # Allocate an extra large sparse matrix.
                alloc_shape = tuple(2 * x for x in new_shape)
                self._host_lil_mem = self._matrix_class(alloc_shape, dtype=self.dtype)
                # Copy the sparse matrix data into the new matrix's internal buffer.
                self._host_lil_mem.rows[:old_shape[0]] = self.data.rows
                self._host_lil_mem.data[:old_shape[0]] = self.data.data
            # Set data to a sub-slice of the memory buffer.
            self.data._shape = new_shape # I am a trespasser.
            self.data.rows = self._host_lil_mem.rows[:new_shape[0]]
            self.data.data = self._host_lil_mem.data[:new_shape[0]]
        elif self.fmt == 'coo':
            self.data.resize(new_shape)
        else: raise NotImplementedError(self.fmt)

    def get_data(self):
        return self.data

    def set_data(self, new_matrix):
        assert new_matrix.shape == self.shape
        self.data = self._matrix_class(new_matrix, dtype=self.dtype)

    def write_row(self, row, columns, values):
        self.to_lil()
        r = int(row)
        self.data.rows[r].clear()
        self.data.data[r].clear()
        columns = [x._idx if isinstance(x, _DB_Object) else int(x) for x in columns]
        order = np.argsort(columns)
        self.data.rows[r].extend(np.take(columns, order))
        self.data.data[r].extend(np.take(values, order))

    def _type_info(self):
        s = super()._type_info()
        try: nnz_per_row = self.data.nnz / self.data.shape[0]
        except ZeroDivisionError: nnz_per_row = 0
        s += " nnz/row: %g"%nnz_per_row
        return s

class Connectivity_Matrix(Sparse_Matrix):
    """ """ # TODO-DOC
    def __init__(self, class_type, name, column, doc=""):
        """ """ # TODO-DOC
        super().__init__(class_type, name, column, doc=doc, dtype=bool,)

    def _getter(self, instance):
        return super()._getter(instance)[0]

    def _setter(self, instance, values):
        super()._setter(instance, (values, [True] * len(values)))

if True: # Append docstrings for common arguments.

        def _clean_docstr(s):
            """ Clean and check the users input for custom __docstr__'s. """
            return textwrap.dedent(str(s)).strip()

        _doc_doc = _clean_docstr("""
        Argument doc
        """) # TODO-DOC
        _units_doc = _clean_docstr("""
        Argument units
        """) # TODO-DOC
        _dtype_doc = _clean_docstr("""
        Argument dtype
        Argument dtype is one of:
            * An instance of numpy.dtype
            * A DB_Class or its name, to make pointers to instances of that class.
        """) # TODO-DOC
        _shape_doc = _clean_docstr("""
        Argument shape
        """) # TODO-DOC
        _initial_value_doc = _clean_docstr("""
        Argument initial_value
        """) # TODO-DOC
        _allow_invalid_doc = _clean_docstr("""
        Argument allow_invalid
        """) # TODO-DOC
        _valid_range_doc = _clean_docstr("""
        Argument valid_range
        """) # TODO-DOC

        DB_Class.__init__.__doc__             += _doc_doc
        Attribute.__init__.__doc__            += "\n\n".join((
                _initial_value_doc,
                _dtype_doc,
                _shape_doc,
                _allow_invalid_doc,
                _valid_range_doc,
                _doc_doc,
                _units_doc,
        ))
        ClassAttribute.__init__.__doc__       += "\n\n".join((
                _initial_value_doc,
                _dtype_doc,
                _shape_doc,
                _allow_invalid_doc,
                _valid_range_doc,
                _doc_doc,
                _units_doc,
        ))
        Sparse_Matrix.__init__.__doc__        += "\n\n".join((
                _doc_doc,
                _units_doc,
                _dtype_doc,
                _allow_invalid_doc,
                _valid_range_doc,
        ))
        Connectivity_Matrix.__init__.__doc__  += _doc_doc

        Database.add_class.__doc__                  = DB_Class.__init__.__doc__
        DB_Class.add_attribute.__doc__              = Attribute.__init__.__doc__
        DB_Class.add_class_attribute.__doc__        = ClassAttribute.__init__.__doc__
        DB_Class.add_sparse_matrix.__doc__          = Sparse_Matrix.__init__.__doc__
        DB_Class.add_connectivity_matrix.__doc__    = Connectivity_Matrix.__init__.__doc__

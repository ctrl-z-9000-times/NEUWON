""" A database for neural simulations. """

from collections.abc import Callable, Iterable, Mapping
from graph_algorithms import topological_sort
from neuwon.database import memory_spaces
import cupy
import itertools
import numpy as np
import textwrap
import weakref

Real    = np.dtype('f8')
epsilon = np.finfo(Real).eps
Pointer = np.dtype('u4')
NULL    = np.iinfo(Pointer).max

class Database:
    def __init__(self):
        """ Create a new empty database. """
        self.db_classes = dict()
        self.clock = None
        self.memory_space = memory_spaces.host
        self.sort_order = None

    def add_class(self, name: str, base_class:type=None, sort_key=tuple(), doc:str="") -> 'DB_Class':
        return DB_Class(self, name, base_class=base_class, sort_key=sort_key, doc=doc)

    def get(self, name: str) -> 'DB_Class' or '_DataComponent':
        """ Get the database's internal representation of the named thing.

        Argument name can refer to either:
            * A DB_Class
            * An Attribute, ClassAttribute, or Sparse_Matrix

        Example:
        >>> Foo = database.add_class("Foo")
        >>> bar = Foo.add_attribute("bar")
        >>> assert Foo == database.get("Foo")
        >>> assert bar == database.get("Foo.bar")
        """
        if isinstance(name, type) and issubclass(name, DB_Object):
            try:
                name = name._cls
            except AttributeError:
                1/0 # How to explain what went wrong?
        if isinstance(name, DB_Class):
            assert name.database is self
            return name
        elif isinstance(name, _DataComponent):
            assert name._cls.database is self
            return name
        elif isinstance(name, str): pass
        else: raise ValueError(f"Expected 'str' got '{type(name)}'")
        _cls, _, attr = name.partition('.')
        try:
            obj = self.db_classes[_cls]
        except KeyError:
            raise KeyError(f"No such DB_Class '{_cls}'")
        try:
            if attr: obj = obj.components[attr]
        except KeyError:
            raise AttributeError("'%s' object has no attribute '%s'"%(_cls, attr))
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
        return tuple(self.db_classes.values())

    def get_all_components(self) -> tuple:
        return tuple(itertools.chain.from_iterable(
                x.components.values() for x in self.db_classes.values()))

    def add_clock(self, tick_period:float, units:str="") -> 'neuwon.database.time.Clock':
        """ """
        from neuwon.database.time import Clock
        assert self.clock is None, "Database already has a Clock!"
        if isinstance(tick_period, Clock):
            self.clock = tick_period
        else:
            self.clock = Clock(tick_period, units=units)
        return self.clock

    def get_clock(self) -> 'neuwon.database.time.Clock':
        """ Get the default clock for this database. """
        return self.clock

    def get_memory_space(self):
        """ Returns the memory space that is currently in use. """
        return self.memory_space

    def using_memory_space(self, memory_space:str):
        """
        Request that the database use a specific memory space.

        Returns a context manager which must be used in a "with" statement.

        Example:
            >>> with database.using_memory_space('cuda'):
            >>>     database.get_data('obj.attr') -> Data in GPU memory

            * Inside of the "with" block, the database will endevour to store
              your data in the given memory space. If the memory space does not
              implement your data's type then it will simply remain in the
              host's memory space.

            * The database moves your data when you call any "get_data" method.

            * The database tracks where each data component is currently
              located, see method "get_memory_space".

        Argument memory_space is one of: "host", "cuda".

        The term "host" refers to the python process which is currently running
        and contains this database. Allocations in the "host" memory space use
        numpy arrays and scipy sparse matrixes.

        The "cuda" memory space is the default graphics card, as enumerated by
        CUDA. Allocations in the "cuda" memory space use cupy for both arrays
        and sparse matrices.

        By default all data is stored on the host.
        """
        return memory_spaces.ContextManager(self, memory_space)

    def _remove_references_to_destroyed(self):
        """ Remove all references from the living instances to the dead ones. """
        for db_class in self.db_classes.values():
            db_class._set_destroyed_mask()
        # Determine an order to scan for destroyed references so that they don't
        # need to be checked more than once.
        def destructive_references(db_class) -> [DB_Class]:
            for component in db_class.components.values():
                if component.reference and not component.allow_invalid:
                    yield component.reference
        dependencies = topological_sort(self.db_classes.values(), destructive_references)
        order = [] # First scan references according to the chain of dependencies.
        order_does_not_matter = [] # Then scan the remaining references.
        for db_class in reversed(dependencies):
            for component in db_class.components.values():
                if component.reference and not component.allow_invalid:
                    order.append(component)
                else:
                    order_does_not_matter.append(component)
        order.extend(order_does_not_matter)
        # 
        for component in order:
             component._remove_references_to_destroyed()

    def is_sorted(self):
        """ Is everything in this database sorted? """
        return all(db_class.is_sorted for db_class in self.db_classes.values())

    def sort(self):
        """ Sort everything in this database.

        Also removes any holes in the arrays which were left behind by
        destroyed instances.

        WARNING: This invalidates all unstable_index's!
        """
        self._remove_references_to_destroyed()
        # Resolve the sort keys from strings to DB_Classes.
        for db_class in self.db_classes.values():
            keys = db_class.sort_key
            if any(not isinstance(k, _DataComponent) for k in keys):
                db_class.sort_key = keys = tuple(db_class.get(k) for k in keys)
                assert all(isinstance(x, Attribute) for x in keys)
                self.sort_order = None # Sort order was invalidated by adding a new db_class.
        # Sorting by classes by references to other classes introduces a
        # dependency in the sort order.
        def sort_order_dependencies(db_class):
            """ Yields all db_classes which must be sorted before this db_class can be sorted. """
            for component in db_class.sort_key:
                if component.reference:
                    yield component.reference
        if self.sort_order is None:
            self.sort_order = topological_sort(self.db_classes.values(), sort_order_dependencies)
        # Propagate "is_sorted==False" through the dependencies.
        for db_class in reversed(self.sort_order):
            if not db_class.is_sorted: continue
            if any(not x.is_sorted for x in sort_order_dependencies(db_class)):
                db_class.is_sorted = False
        # Sort all db_classes.
        for db_class in reversed(self.sort_order):
            db_class._sort()

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
        if exceptions: raise AssertionError(",\n                ".join(sorted(exceptions))+".")

class _Documentation:
    def __init__(self, name, doc):
        self.name = str(name)
        self.doc = textwrap.dedent(str(doc)).strip()

    def get_name(self) -> str: return self.name
    def get_doc(self) -> str:  return self.doc

    _name_doc = """
        Argument name
        """

    _doc_doc = """
        Argument doc is an optional documentation string.
        """

class DB_Object:
    """ Super class for all instance types. """
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

    def __repr__(self):
        return "<%s:%d>"%(self._cls.name, self._idx)

class DB_Class(_Documentation):
    """ This is the database's internal representation of a class type. """
    def __init__(self, database, name: str, base_class=None, sort_key=tuple(), doc:str="",):
        """ Create a new class which is managed by the database.

        Argument base_class: The external representation will inherit from this
                class. The base_class can implement methods but should not store
                any data in the instance object. Docstrings attached to the
                base_class will be made publicly visible.

        Argument sort_key is [TODO]
        """
        """ TODO: Use these docs:

        If the dangling references are allow to be NULL then they are replaced
        with NULL references. Otherwise the entity containing the reference is
        destroyed. Destroying entities can cause a chain reaction of destruction.

        Sparse matrices may contain references but they will not trigger a
        recursive destruction of entities. Instead, references to destroyed
        entities are simply removed from the sparse matrix.
        """
        if not doc:
            if base_class and base_class.__doc__:
                doc = base_class.__doc__
        _Documentation.__init__(self, name, doc)
        assert isinstance(database, Database)
        self.database = database
        assert self.name not in self.database.db_classes
        self.database.db_classes[self.name] = self
        self.size = 0
        self.components = dict()
        self.referenced_by = list()
        self.referenced_by_matrix_columns = list()
        self.instances = []
        if isinstance(sort_key, str):
            self.sort_key = (sort_key,)
        else:
            self.sort_key = tuple(sort_key)
            for k in self.sort_key:
                if not isinstance(k, str):
                    raise ValueError(f"Expected 'str', got '{type(k)}'")
        self.is_sorted = True
        self._init_instance_type(base_class, doc)
        self.destroyed_list = []
        self.destroyed_mask = None

    def _init_instance_type(self, users_class, doc):
        """ Make a new subclass to represent instances which are part of *this* database. """
        # Enforce that the user's class use "__slots__".
        if users_class:
            for cls in users_class.mro()[:-1]:
                if "__slots__" not in vars(cls):
                    raise TypeError(f"Class \"{cls.__name__}\" does not define __slots__!")
        # 
        if users_class:
            super_classes = (users_class, DB_Object)
        else:
            super_classes = (DB_Object,)
        # 
        __slots__ = ("_idx",)
        if not hasattr(users_class, "__weakref__"):
            __slots__ += ("__weakref__",)
        # 
        init_doc = DB_Object.__init__.__doc__
        if users_class:
            user_init = users_class.__init__
            if user_init is not object.__init__:
                user_init_doc = user_init.__doc__
                if user_init_doc is not None:
                    init_doc = user_init_doc
        def escape(s):
            """ Escape newlines so that doc-strings can be inserted into a single line string. """
            return s.encode("unicode_escape").decode("utf-8")
        pycode = textwrap.dedent(f"""
            class {self.name}(*super_classes):
                \"\"\"{escape(doc)}\"\"\"
                # TODO: Rename "_cls" to "_db_class".
                _cls = self
                __slots__ = {__slots__}
                __module__ = super_classes[0].__module__

                def __init__(self, *args, **kwargs):
                    \"\"\"{escape(init_doc)}\"\"\"
                    self._cls._init_instance(self)
                    super().__init__(*args, **kwargs)

                def destroy(self):
                    \"\"\" \"\"\"
                    self._cls._destroy_instance(self)

                def is_destroyed(self):
                    \"\"\" \"\"\"
                    return self._idx == {NULL}

                def get_unstable_index(self):
                    \"\"\" Get the index into the database where this object is stored at.

                    WARNING: Sorting the database will invalidate this index!
                    \"\"\"
                    return self._idx

                @classmethod
                def get_database_class(cls):
                    \"\"\" Get the database's internal representation of this object's type. \"\"\"
                    return cls._cls
            """)
        if False: print(pycode)
        exec(pycode, locals())
        self.instance_type = locals()[self.name]

    __init__.__doc__ += _Documentation._doc_doc

    def _init_instance(self, new_instance):
        old_size  = self.size
        new_size  = old_size + 1
        self.size = new_size
        for x in self.components.values():
            if isinstance(x, Attribute): x._append(old_size, new_size)
            elif isinstance(x, ClassAttribute): pass
            elif isinstance(x, Sparse_Matrix): x._resize()
            else: raise NotImplementedError(type(x))
        for x in self.referenced_by_matrix_columns: x._resize()
        new_instance._idx = old_size
        self.instances.append(weakref.ref(new_instance))
        if self.sort_key: self.is_sorted = False

    def _destroy_instance(self, instance):
        idx = instance._idx
        self.destroyed_list.append(idx)
        if self.destroyed_mask is not None: self.destroyed_mask[idx] = True
        self.instances[idx] = None
        instance._idx = NULL
        self.is_sorted = False # This leaves holes in the arrays so it *always* unsorts it.

    def _set_destroyed_mask(self):
        if self.destroyed_list:
            xp = self.database.memory_space.array_module
            self.destroyed_mask = mask = xp.zeros(len(self), dtype=bool)
            mask[self.destroyed_list] = True
        else:
            self.destroyed_mask = None

    def get_instance_type(self) -> DB_Object:
        """ Get the public / external representation of this DB_Class. """
        return self.instance_type

    def index_to_object(self, unstable_index: int) -> DB_Object:
        """ Create a new DB_Object given its index. """
        if type(unstable_index) is self.instance_type: return unstable_index
        idx = int(unstable_index)
        if idx == NULL: return None
        obj = self.instances[idx]
        if obj is None: return None # Object was destroyed.
        obj = obj() # Unwrap the weakref.
        if obj is None:
            cls = self.instance_type
            obj = cls.__new__(cls)
            obj._idx = idx
            self.instances[idx] = weakref.ref(obj)
        return obj

    def get(self, name: str) -> '_DataComponent':
        """
        Get the database's internal representation of a data component that is
        attached to this DB_Class.
        """
        if isinstance(name, _DataComponent):
            assert name.get_class() is self
            return name
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
        return list(filter(None, map(self.index_to_object, range(self.size))))

    def get_database(self) -> Database:
        return self.database

    def get_database_class(self) -> 'self':
        """ Returns self, since this is the database_class. """
        return self

    def add_attribute(self, name:str, initial_value=None, dtype=Real, shape=(1,),
                doc:str="", units:str="", allow_invalid:bool=False, valid_range=(None, None),):
        return Attribute(self, name, initial_value=initial_value, dtype=dtype, shape=shape,
                doc=doc, units=units, allow_invalid=allow_invalid, valid_range=valid_range,)

    def add_class_attribute(self, name:str, initial_value, dtype=Real, shape=(1,),
                doc:str="", units:str="", allow_invalid:bool=False, valid_range=(None, None),):
        return ClassAttribute(self, name, initial_value, dtype=dtype, shape=shape,
                doc=doc, units=units, allow_invalid=allow_invalid, valid_range=valid_range,)

    def add_sparse_matrix(self, name:str, column, dtype=Real,
                doc:str="", units:str="", allow_invalid:bool=False, valid_range=(None, None),):
        return Sparse_Matrix(self, name, column, dtype=dtype,
                doc=doc, units=units, allow_invalid=allow_invalid, valid_range=valid_range,)

    def add_connectivity_matrix(self, name:str, column, doc:str=""):
        return Connectivity_Matrix(self, name, column, doc=doc)

    def _sort(self):
        if self.is_sorted: return
        xp = self.database.memory_space.array_module
        # First compress out the dead instances.
        new_to_old = xp.arange(len(self))
        if self.destroyed_list:
            new_to_old = xp.compress(xp.logical_not(self.destroyed_mask), new_to_old)
        # Sort by each key.
        for key in reversed(self.sort_key):
            sort_key_data   = self.get(key).get_data()
            rearranged      = sort_key_data[new_to_old]
            sort_order      = xp.argsort(rearranged, kind='stable')
            new_to_old      = new_to_old[sort_order]
        # Make the forward transform map.
        old_to_new = xp.empty(len(self), dtype=Pointer)
        old_to_new[new_to_old] = xp.arange(len(new_to_old))
        old_to_new[self.destroyed_list] = NULL
        # Apply the sort to each data component.
        for component in self.components.values():
            if component.is_free(): continue
            if isinstance(component, Attribute):
                component.data = component.data[new_to_old]
            elif isinstance(component, ClassAttribute):
                pass
            elif isinstance(component, Sparse_Matrix):
                component.to_coo()
                component.data.row = old_to_new[component.data.row]
            else: raise NotImplementedError(type(component))
        self.instances = list(np.take(self.instances, new_to_old))
        for ref in self.instances:
            inst = ref()
            if inst:
                inst._idx = old_to_new[inst._idx]
        # Update all references to point to their new locations.
        for component in self.referenced_by:
            assert isinstance(component, Attribute)
            assert component.reference is self
            if component.is_free(): continue
            data = component.get_data()
            null_values = (data == NULL)
            data = xp.take(old_to_new, data, mode='clip')
            data[null_values] = NULL
            component.data = data
        for matrix in self.referenced_by_matrix_columns:
            if matrix.is_free(): continue
            matrix.to_coo()
            matrix.data.col = old_to_new[matrix.data.col]
        # Bookkeeping.
        self.size = len(new_to_old)
        self.destroyed_list = []
        self.destroyed_mask = None
        self.is_sorted = True

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
    def __init__(self, db_class, name,
                doc, units, shape, dtype, initial_value, allow_invalid, valid_range):
        _Documentation.__init__(self, name, doc)
        assert isinstance(db_class, DB_Class)
        assert self.name not in db_class.components
        # TODO: Rename "_cls" to the full "db_class". Don't abbreviate it!
        self._cls = db_class
        self._cls.components[self.name] = self
        if shape is None: pass
        elif isinstance(shape, Iterable):
            self.shape = tuple(round(x) for x in shape)
        else:
            self.shape = (round(shape),)
        if isinstance(dtype, type) and issubclass(dtype, DB_Object):
            dtype = dtype.get_database_class()
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
        if self.reference is self._cls: assert self.allow_invalid
        self.valid_range = tuple(valid_range)
        if None not in self.valid_range: self.valid_range = tuple(sorted(self.valid_range))
        assert len(self.valid_range) == 2
        self.memory_space = self._cls.database.memory_space
        setattr(self._cls.instance_type, self.name,
                property(self._getter_wrapper, self._setter_wrapper, doc=self.doc,))

    def _getter_wrapper(self, instance):
        if instance._idx == NULL:
            raise ValueError("Object was destroyed!")
        return self._getter(instance)
    def _setter_wrapper(self, instance, value):
        if instance._idx == NULL:
            raise ValueError("Object was destroyed!")
        self._setter(instance, value)

    def _getter(self, instance):
        """ Get the instance's data value. """
        raise NotImplementedError(type(self))
    def _setter(self, instance, value):
        """ Set the instance's data value. """
        raise NotImplementedError(type(self))

    def get_data(self):
        """ Returns all data for this component. """
        raise NotImplementedError(type(self))
    def set_data(self, value):
        """ Replace this entire data component with a new set of values. """
        raise NotImplementedError(type(self))

    def get_units(self) -> str:             return self.units
    def get_dtype(self) -> np.dtype:        return self.dtype
    def get_shape(self) -> tuple:           return self.shape
    def get_initial_value(self) -> object:  return self.initial_value
    def get_class(self) -> DB_Class:        return self._cls
    def get_database(self) -> Database:     return self._cls.database

    def get_memory_space(self) -> str:
        """
        Returns the current location of this data component:

        Returns "host" when located in python's memory space.
        Returns "cuda" when located in CUDA's memory space.
        """
        return self.memory_space

    def free(self):
        """
        Release the memory used by this data component. The next time the data
        is accessed it will be reallocated and set to its initial_value.
        """
        raise NotImplementedError(type(self))

    def is_free(self):
        return self.data is None

    def check(self):
        """ Check data for values which are: NaN, NULL, or Out of bounds. """
        data = self.get_data()
        if isinstance(self, ClassAttribute):  data = np.array([data])
        elif isinstance(self, Sparse_Matrix): data = data.data
        xp = cupy.get_array_module(data)
        if not self.allow_invalid:
            if self.reference:
                assert xp.all(xp.less(data, self.reference.size)), self.name + " is NULL"
            else:
                if data.dtype.kind in ("f", "c"):
                    assert not xp.any(xp.isnan(data)), self.name + " is NaN"
        lower_bound, upper_bound = self.valid_range
        if lower_bound is not None:
            assert xp.all(xp.less_equal(lower_bound, data)), self.name + " less than %g"%lower_bound
        if upper_bound is not None:
            assert xp.all(xp.greater_equal(upper_bound, data)), self.name + " greater than %g"%upper_bound

    def _remove_references_to_destroyed(self):
        raise NotImplementedError(type(self))

    def _type_info(self):
        s = ""
        if self.reference: s += "ref:" + self.reference.name
        else: s += self.dtype.name
        if self.shape != (1,): s += repr(list(self.shape))
        return s

    def __repr__(self):
        return "<%s: %s.%s %s>"%(type(self).__name__, self._cls.name, self.name, self._type_info())

    _units_doc = """
        Argument units is an optional documentation string for physical units.
        """

    _dtype_doc = """
        Argument dtype is the data type for this data component. It is either:
                * An instance of numpy.dtype
                * A DB_Class or its name, to make pointers to instances of that class.
        """

    _shape_doc = """
        Argument shape is the allocation size / shape for this data component.
        """

    _allow_invalid_doc = """
        Argument allow_invalid controls whether NaN or NULL values are permissible.
        """

    _valid_range_doc = """
        Argument valid_range is pair of numbers (min, max) defining an inclusive
                range of permissible values.
        """

class Attribute(_DataComponent):
    """ This is the database's internal representation of an instance variable. """
    def __init__(self, db_class, name:str, initial_value=None, dtype=Real, shape=(1,),
                doc:str="", units:str="", allow_invalid:bool=False, valid_range=(None, None),):
        """ Add an instance variable to a class type.

        Argument initial_value is written to new instances of this attribute.
                This is applied before "base_class.__init__" is called.
                Optional, if not given then the data will not be initialized.
        """
        _DataComponent.__init__(self, db_class, name,
            doc=doc, units=units, dtype=dtype, shape=shape, initial_value=initial_value,
            allow_invalid=allow_invalid, valid_range=valid_range)
        self.data = self._alloc(len(self._cls))
        if self.initial_value is not None:
            self.data.fill(self.initial_value)

    __init__.__doc__ += "".join((
        _DataComponent._dtype_doc,
        _DataComponent._shape_doc,
        _Documentation._doc_doc,
        _DataComponent._units_doc,
        _DataComponent._allow_invalid_doc,
        _DataComponent._valid_range_doc,))

    def free(self):
        self.data = None

    def _alloc_if_free(self):
        if self.data is None:
            self.data = self._alloc(len(self._cls))
            if self.initial_value is not None:
                self.data.fill(self.initial_value)

    def _getter(self, instance):
        if self.data is None: return self.initial_value
        value = self.data[instance._idx]
        if hasattr(value, 'get'): value = value.get()
        if self.reference:
            return self.reference.index_to_object(value)
        return self.dtype.type(value)

    def _setter(self, instance, value):
        self._alloc_if_free()
        if self.reference:
            if value is None:
                value = NULL
            else:
                assert isinstance(value, self.reference.instance_type)
                value = value._idx
        self.data[instance._idx] = value

    def _append(self, old_size, new_size):
        """ Prepare space for new instances at the end of the array. """
        if self.data is None: return
        if len(self.data) < new_size:
            new_data = self._alloc(2 * new_size)
            new_data[:old_size] = self.data[:old_size]
            self.data = new_data
        if self.initial_value is not None:
            self.data[old_size:new_size].fill(self.initial_value)

    def _alloc(self, size):
        """ Returns an empty array. """
        # TODO: IIRC CuPy can not deal with numpy structured arrays...
        #       Detect this issue and revert to using numba arrays.
        #       numba.cuda.to_device(numpy.array(data, dtype=dtype))
        shape = (size,)
        if self.shape != (1,): # Don't append empty trailing dimension.
            shape += self.shape
        return self.memory_space.array_module.empty(shape, dtype=self.dtype)

    def _transfer(self, target_space):
        if self.data is None:
            self.memory_space = target_space
        elif self.memory_space is not target_space:
            if self.memory_space is memory_spaces.host:
                if target_space is memory_spaces.cuda:
                    self.data = memory_spaces.cuda.array(self.data)
                else: raise NotImplementedError(target_space)
            elif self.memory_space is memory_spaces.cuda:
                if target_space is memory_spaces.host:
                    self.data = self.data.get()
                else: raise NotImplementedError(target_space)
            else: raise NotImplementedError(self.memory_space)
            self.memory_space = target_space

    def get_data(self):
        """ Returns either "numpy.ndarray" or "cupy.ndarray" """
        self._transfer(self._cls.database.memory_space)
        self._alloc_if_free()
        return self.data[:len(self._cls)]

    def set_data(self, value):
        size = len(self._cls)
        assert len(value) == size
        if self.shape == (1,):
            shape = (size,) # Don't append empty trailing dimension.
        else:
            shape = (size,) + self.shape
        # TODO: This should accept whatever memory space it is given, and avoid
        # transfering until someone calls "get_data".
        self.data = self.memory_space.array(value, dtype=self.dtype).reshape(shape)

    def _remove_references_to_destroyed(self):
        if not self.reference: return
        if self.data is None: return
        pointer_data = self.data[:len(self._cls)]
        destroyed_mask = self.reference.destroyed_mask
        if destroyed_mask is None: return
        xp = cupy.get_array_module(destroyed_mask)
        target_is_dead = xp.take(destroyed_mask, pointer_data, axis=0, mode='clip')
        target_is_dead[pointer_data == NULL] = True
        target_is_dead = xp.nonzero(target_is_dead)[0]
        pointer_data[target_is_dead] = NULL
        if not self.allow_invalid:
            for idx in target_is_dead:
                db_obj = self._cls.index_to_object(idx)
                if db_obj is not None:
                    db_obj.destroy()
                    destroyed_mask[idx] = True

class ClassAttribute(_DataComponent):
    """ This is the database's internal representation of a class variable. """
    def __init__(self, db_class, name:str, initial_value,
                dtype=Real, shape=(1,),
                doc:str="", units:str="",
                allow_invalid:bool=False, valid_range=(None, None),):
        """ Add a class variable to a class type.

        All instance of the class will use a single shared value for this attribute.

        Argument initial_value is required.
        """
        _DataComponent.__init__(self, db_class, name,
                dtype=dtype, shape=shape, doc=doc, units=units, initial_value=initial_value,
                allow_invalid=allow_invalid, valid_range=valid_range)
        self.data = self.initial_value
        self.memory_space = memory_spaces.host
        if self.reference: raise NotImplementedError

    __init__.__doc__ += "".join((
        _DataComponent._dtype_doc,
        _DataComponent._shape_doc,
        _Documentation._doc_doc,
        _DataComponent._units_doc,
        _DataComponent._allow_invalid_doc,
        _DataComponent._valid_range_doc,
    ))

    def _getter(self, instance):
        return self.data

    def _setter(self, instance, value):
        self.data = self.dtype.type(value)

    def get_data(self):
        return self.data

    def set_data(self, value):
        self.data = self.dtype.type(value)

    def free(self):
        self.data = self.initial_value

    def _remove_references_to_destroyed(self):
        pass

class Sparse_Matrix(_DataComponent):
    """ """ # TODO-DOC

    # TODO: Consider adding more write methods:
    #       1) Write rows. (done)
    #       2) Insert coordinates.
    #               Notes: first convert format to either lil or coo
    #       3) Overwrite the matrix. (done)

    # TODO: Figure out if/when to call mat.eliminate_zeros() and sort too.

    def __init__(self, db_class, name, column, dtype=Real, doc:str="", units:str="",
                allow_invalid:bool=False, valid_range=(None, None),):
        """
        Add a sparse matrix that is indexed by DB_Objects. This is useful for
        implementing any-to-any connections between entities.

        This db_class is the index for the rows of the sparse matrix.

        Argument column refers to the db_class which is the index for the
                columns of the sparse matrix.
        """
        _DataComponent.__init__(self, db_class, name,
                dtype=dtype, shape=None, doc=doc, units=units, initial_value=0.,
                allow_invalid=allow_invalid, valid_range=valid_range)
        self.column = self._cls.database.get_class(column)
        self.column.referenced_by_matrix_columns.append(self)
        self.shape = (len(self._cls), len(self.column))
        self.fmt = 'csr'
        self.free()
        if self.reference: raise NotImplementedError

    __init__.__doc__ += "".join((
        _DataComponent._dtype_doc,
        _Documentation._doc_doc,
        _DataComponent._units_doc,
        _DataComponent._allow_invalid_doc,
        _DataComponent._valid_range_doc,))

    @property
    def _matrix_class(self):
        if   self.fmt == 'lil': return self.memory_space.matrix_module.lil_matrix
        elif self.fmt == 'coo': return self.memory_space.matrix_module.coo_matrix
        elif self.fmt == 'csr': return self.memory_space.matrix_module.csr_matrix
        else: raise NotImplementedError(self.fmt)

    def free(self):
        self.data = None
        self._host_lil_mem = None

    def _remove_references_to_destroyed(self):
        if self.data is None: return
        self.to_coo()
        # Mask off the dead entries.
        dead_rows = self._cls.destroyed_mask
        dead_cols = self.column.destroyed_mask
        masks = []
        if dead_rows is not None:
            masks.append(np.logical_not(dead_rows[self.data.row]))
        if dead_cols is not None:
            masks.append(np.logical_not(dead_cols[self.data.col]))
        # Combine all of the masks into one.
        if   len(masks) == 0: return
        elif len(masks) == 1: alive_mask = masks.pop()
        else:                 alive_mask = np.logical_and(*masks)
        # Compress out the destroyed entries.
        self.data.row  = self.data.row[alive_mask]
        self.data.col  = self.data.col[alive_mask]
        self.data.data = self.data.data[alive_mask]

    def _alloc_if_free(self):
        if self.data is None:
            self.data = self._matrix_class(self.shape, dtype=self.dtype)

    # TODO: Should this filter out zeros?
    #       Also, when should this compress out zeros?
    def _getter(self, instance):
        if self.data is None: return ([], [])
        if self.fmt == 'coo': self.to_lil()
        matrix = self.data[instance._idx]
        if self.memory_space is memory_spaces.cuda:
            matrix = matrix.get()
        if self.fmt == 'lil':
            index = matrix.rows[0]
        elif self.fmt == 'csr':
            index = matrix.indices
        else: raise NotImplementedError(self.fmt)
        index_to_object = self.column.index_to_object
        return ([index_to_object(x) for x in index], list(matrix.data))

    def _setter(self, instance, value):
        columns, data = value
        self.write_row(instance._idx, columns, data)

    def to_lil(self) -> 'self':
        self._transfer(memory_spaces.host)
        if self.data is None:
            self.fmt = "lil"
            return self
        if self.fmt != "lil":
            self.fmt = "lil"
            self.data = self._matrix_class(self.data, dtype=self.dtype)
        return self

    def to_coo(self) -> 'self':
        if self.data is None:
            self.fmt = "coo"
            return self
        if self.fmt != "coo":
            self.fmt = "coo"
            self._host_lil_mem = None
            self.data = self._matrix_class(self.data, dtype=self.dtype)
        return self

    def to_csr(self) -> 'self':
        if self.data is None:
            self.fmt = "csr"
            return self
        if self.fmt != "csr":
            self.fmt = "csr"
            self._host_lil_mem = None
            self.data = self._matrix_class(self.data, dtype=self.dtype)
        return self

    def _resize(self):
        old_shape = self.shape
        new_shape = self.shape = (len(self._cls), len(self.column))

        if old_shape == new_shape: return
        if self.data is None: return

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

    def _transfer(self, target_space):
        if self.memory_space is target_space: return
        if self.memory_space is memory_spaces.host:
            if target_space is memory_spaces.cuda:
                if self.dtype.kind not in 'fc':
                    # Cupy only supports float & complex types.
                    # Silently refuse to transfer, leave data on host.
                    return
                self.memory_space = memory_spaces.cuda
                if self.fmt == 'lil':
                    self.fmt = 'csr'
                if self.fmt == 'csr' or self.fmt == 'coo':
                    if self.data is not None:
                        self.data = self._matrix_class(self.data, dtype=self.dtype)
                else: raise NotImplementedError(self.fmt)
                self._host_lil_mem = None
            else: raise NotImplementedError(target_space)
        elif self.memory_space is memory_spaces.cuda:
            if target_space is memory_spaces.host:
                if self.data is not None:
                    self.data = self.data.get()
            else: raise NotImplementedError(target_space)
            self.memory_space = memory_spaces.host
        else: raise NotImplementedError(self.memory_space)

    def get_data(self):
        self._transfer(self._cls.database.memory_space)
        self._alloc_if_free()
        return self.data

    def set_data(self, matrix):
        for mem in (memory_spaces.host, memory_spaces.cuda):
            if isinstance(matrix, mem.matrix_module.spmatrix):
                self.memory_space = mem
                self.data         = matrix
                break
        else:
            self.memory_space = memory_spaces.host
            self.fmt = 'csr'
            self.data = self._matrix_class(matrix, dtype=self.dtype)
        assert self.data.shape == self.shape
        assert self.data.dtype == self.dtype

    def write_row(self, row, columns, values):
        self.to_lil()
        self._alloc_if_free()
        r = int(row)
        self.data.rows[r].clear()
        self.data.data[r].clear()
        columns = [x._idx if isinstance(x, DB_Object) else int(x) for x in columns]
        order = np.argsort(columns)
        self.data.rows[r].extend(np.take(columns, order))
        self.data.data[r].extend(np.take(values, order))

    def _type_info(self):
        s = super()._type_info()
        if   self.is_free():     nnz_per_row = 0
        elif self.shape[0] == 0: nnz_per_row = 0
        else:                    nnz_per_row = self.data.nnz / self.shape[0]
        s += " nnz/row: %g"%nnz_per_row
        return s

class Connectivity_Matrix(Sparse_Matrix):
    """ """ # TODO-DOC
    def __init__(self, db_class, name, column, doc:str=""):
        """ """ # TODO-DOC
        super().__init__(db_class, name, column, doc=doc, dtype=bool,)
    
    __init__.__doc__  += _Documentation._doc_doc

    def _getter(self, instance):
        return super()._getter(instance)[0]

    def _setter(self, instance, values):
        super()._setter(instance, (values, [True] * len(values)))

Database.add_class.__doc__                  = DB_Class.__init__.__doc__
DB_Class.add_attribute.__doc__              = Attribute.__init__.__doc__
DB_Class.add_class_attribute.__doc__        = ClassAttribute.__init__.__doc__
DB_Class.add_sparse_matrix.__doc__          = Sparse_Matrix.__init__.__doc__
DB_Class.add_connectivity_matrix.__doc__    = Connectivity_Matrix.__init__.__doc__

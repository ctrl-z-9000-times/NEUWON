from graph_algorithms import topological_sort
from neuwon.database import memory_spaces
from neuwon.database.doc import Documentation
from neuwon.database.dtypes import *
import itertools
import numpy as np
import textwrap
import weakref

class Database:
    """ An in-memory database for implementing simulations. """
    def __init__(self):
        """ Create a new empty database. """
        self.db_classes = dict()
        self.clock = None
        self.memory_space = memory_spaces.host
        self._sort_order = None

    def add_class(self, name: str, base_class:type=None, sort_key=tuple(), doc:str="") -> 'DB_Class':
        return DB_Class(self, name, base_class=base_class, sort_key=sort_key, doc=doc)

    def get(self, name: str) -> 'DB_Class' or 'DataComponent':
        """ Get the database's internal representation of the named thing.

        Argument name can refer to either:
            * A DB_Class
            * An Attribute, ClassAttribute, or SparseMatrix

        Example:
        >>> Foo = database.add_class("Foo")
        >>> bar = Foo.add_attribute("bar")
        >>> assert Foo == database.get("Foo")
        >>> assert bar == database.get("Foo.bar")
        """
        if isinstance(name, type):
            if issubclass(name, DB_Object):
                try:
                    name = name._db_class
                except AttributeError:
                    1/0 # TODO: How to explain what went wrong?
            else:
                if name is object: raise KeyError()
                for db_class in self.db_classes.values():
                    if issubclass(db_class.instance_type, name):
                        return db_class
        if isinstance(name, DB_Class):
            assert name.database is self
            return name
        elif isinstance(name, DataComponent):
            assert name.db_class.database is self
            return name
        elif isinstance(name, str): pass
        else: raise KeyError(f"Expected 'str' got '{type(name)}'")
        db_class, _, attr = name.partition('.')
        try:
            obj = self.db_classes[db_class]
        except KeyError:
            raise KeyError(f"No such DB_Class '{db_class}'")
        if attr:
            obj = obj.get(attr)
        return obj

    def get_class(self, name: str):
        """ Get the database's internal representation of a class.

        Argument name can be anything which `self.get(name)` accepts.
        """
        db_class = self.get(name)
        if isinstance(db_class, DataComponent): db_class = db_class.get_class()
        return db_class

    def get_component(self, name: str):
        component = self.get(name)
        assert isinstance(component, DataComponent)
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

    def add_clock(self, tick_period:float, units:str="") -> 'neuwon.database.Clock':
        """ Set the default clock for this database. """
        from neuwon.database.time import Clock
        assert self.clock is None, "Database already has a default Clock!"
        if isinstance(tick_period, Clock):
            self.clock = tick_period
            assert not units, 'Unexpected argument.'
        else:
            self.clock = Clock(tick_period, units=units)
        return self.clock

    def get_clock(self) -> 'neuwon.database.Clock':
        """ Get the default clock for this database. """
        return self.clock

    def get_memory_space(self):
        """ Returns the memory space that is currently in use. """
        return self.memory_space

    def get_array_module(self):
        """
        Returns either the "numpy" module or the "cupy" module,
        depending on which memory space is currently in use.
        """
        return self.memory_space.get_array_module()

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
        if False:
            for n in self.db_classes.values():
                print(n)
                for x in destructive_references(n):
                    print('\t->', x)
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
            if any(not isinstance(k, DataComponent) for k in keys):
                db_class.sort_key = keys = tuple(db_class.get(k) for k in keys)
                assert all(isinstance(x, Attribute) for x in keys)
                self._sort_order = None # Sort order was invalidated by adding a new db_class.
        # Sorting by classes by references to other classes introduces a
        # dependency in the sort order.
        def sort_order_dependencies(db_class):
            """ Yields all db_classes which must be sorted before this db_class can be sorted. """
            for component in db_class.sort_key:
                if component.reference:
                    yield component.reference
        if self._sort_order is None:
            self._sort_order = topological_sort(self.db_classes.values(), sort_order_dependencies)
        # Propagate "is_sorted==False" through the dependencies.
        for db_class in reversed(self._sort_order):
            if not db_class.is_sorted: continue
            if any(not x.is_sorted for x in sort_order_dependencies(db_class)):
                db_class.is_sorted = False
        # Sort all db_classes.
        for db_class in reversed(self._sort_order):
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
        return "<%s:%d>"%(self._db_class.name, self._idx)

class DB_Class(Documentation):
    """ This is the database's internal representation of a class type. """
    def __init__(self, database, name: str, base_class=None, sort_key=tuple(), doc:str="",):
        """ Create a new class which is managed by the database.

        Argument base_class is an optional superclass for the instance_type.
                All base_classes must define "__slots__ = ()".

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
        if base_class is None and isinstance(name, type):
            base_class = name
            name = base_class.__name__
        if not doc:
            if base_class and base_class.__doc__:
                doc = base_class.__doc__
        Documentation.__init__(self, name, doc)
        assert isinstance(database, Database)
        self.database = database
        assert self.name not in self.database.db_classes
        self.database.db_classes[self.name] = self
        self.size = 0
        self.components = dict()
        self.methods = dict()
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
        self._init_methods()

    def _init_instance_type(self, users_class, doc):
        """ Make a new subclass to represent instances which are part of *this* database. """
        # Enforce that the user's class use "__slots__ = ()".
        if users_class:
            assert isinstance(users_class, type)
            for cls in users_class.mro()[:-1]:
                if "__slots__" not in vars(cls):
                    raise TypeError(f"Class \"{cls.__name__}\" does not define \"__slots__ = ()\"!")
                if len(cls.__slots__) != 0:
                    raise TypeError(f"\"{cls.__name__}.__slots__\" is not empty!")
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
        self.ClassAttrMeta = ClassAttrMeta = type('ClassAttrMeta', (type,), {})
        pycode = textwrap.dedent(f"""
            class {self.name}(*super_classes, metaclass=ClassAttrMeta):
                \"\"\"{escape(doc)}\"\"\"
                _db_class = self
                __slots__ = {__slots__}
                __module__ = super_classes[0].__module__

                def __init__(self, *args, **kwargs):
                    \"\"\"{escape(init_doc)}\"\"\"
                    self._db_class._init_instance(self)
                    super().__init__(*args, **kwargs)

                def destroy(self):
                    \"\"\" \"\"\"
                    self._db_class._destroy_instance(self)

                def is_destroyed(self) -> bool:
                    \"\"\" \"\"\"
                    return self._idx == {NULL}

                def get_unstable_index(self) -> int:
                    \"\"\"
                    Get the index of this object in its DB_Class's data arrays.

                    WARNING: Sorting the database will invalidate this index!
                    \"\"\"
                    return self._idx

                @classmethod
                def get_database_class(cls) -> 'DB_Class':
                    \"\"\" Get the database's internal representation of this object's type. \"\"\"
                    return cls._db_class
            """)
        if False: print(pycode)
        exec(pycode, locals())
        self.instance_type = locals()[self.name]

    def _init_methods(self):
        """
        Find all methods attached to the instance type and add them as methods.
        """
        for attr_name in dir(self.instance_type):
            if attr_name.startswith('__') and attr_name.endswith('__'): continue
            attr = getattr(self.instance_type, attr_name)
            if isinstance(attr, Compute):
                attr._register_method(self)

    def add_method(self, compute):
        assert isinstance(compute, Compute)
        return compute._register_method(self, add_attr=False)

    def _init_instance(self, new_instance):
        old_size  = self.size
        new_size  = old_size + 1
        self.size = new_size
        for x in self.components.values():
            if isinstance(x, Attribute): x._append(old_size, new_size)
            elif isinstance(x, ClassAttribute): pass
            elif isinstance(x, SparseMatrix): x._resize()
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

    # TODO: Make this into a propper attribute so that the user can see it more
    # easily. Make it private ie '_destroyed'. Someday the user might want to
    # run a little bit of computation on unsorted data, which I should support!
    # Although it would have sub-optimal performance, sorting the data has a
    # performance cost as well. Sorting also invalidates all unstable_index's.
    # 
    # But, on the other hand, if the user ever -god forbid- wrote to the
    # destroyed mask then the DB *will* implode. The OOP instances need to be
    # updated (BC the setter/getters to not check the flag).
    #       Well, its private, so if they mess it up and die, thats really only their own fault.
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

    def get(self, name: str) -> 'DataComponent':
        """
        Get the database's internal representation of a data component that is
        attached to this DB_Class.
        """
        if isinstance(name, DataComponent):
            assert name.get_class() is self
            return name
        name = str(name)
        if name in self.components: return self.components[name]
        if name in self.methods:    return self.methods[name]
        raise AttributeError("'%s' object has no attribute '%s'"%(self.name, name))

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
        return SparseMatrix(self, name, column, dtype=dtype,
                doc=doc, units=units, allow_invalid=allow_invalid, valid_range=valid_range,)

    def add_connectivity_matrix(self, name:str, column, doc:str=""):
        return ConnectivityMatrix(self, name, column, doc=doc)

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
            elif isinstance(component, SparseMatrix):
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

Database.add_class.__doc__ = DB_Class.__init__.__doc__

from neuwon.database.data_components import (DataComponent, ClassAttribute, Attribute,
                                            SparseMatrix, ConnectivityMatrix)
from neuwon.database.compute import Compute

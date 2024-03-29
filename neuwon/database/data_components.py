from collections.abc import Callable, Iterable, Mapping
from . import memory_spaces
from .database import DB_Class, DB_Object, Database
from .doc import Documentation
from .dtypes import *
import numpy as np

class DataComponent(Documentation):
    """ Abstract class for all types of data storage. """
    def __init__(self, db_class, name, *,
                doc, units, shape, dtype, initial_value, allow_invalid, valid_range,
                initial_distribution=None, class_attribute=False):
        self.units = str(units).strip()
        if self.units:
            doc = doc + "\nunits: " + self.units
        Documentation.__init__(self, name, doc)
        assert isinstance(db_class, DB_Class)
        assert self.name not in dir(db_class.instance_type), "name conflict."
        assert self.name not in dir(db_class.ClassAttrMeta), "name conflict."
        self.db_class = db_class
        self.db_class.components[self.name] = self
        self.qualname = f'{self.db_class.name}.{self.name}'
        if shape is None: pass
        elif isinstance(shape, Iterable):
            self.shape = tuple(round(x) for x in shape)
        else:
            self.shape = (round(shape),)
        if isinstance(dtype, type) and issubclass(dtype, DB_Object):
            dtype = dtype.get_database_class()
        try:
            self.reference = self.db_class.database.get_class(dtype)
            self.reference.referenced_by.append(self)
            self.dtype = Pointer
            assert initial_value is None
            self.initial_value = NULL
        except KeyError:
            self.dtype = np.dtype(dtype)
            if initial_value is None:
                self.initial_value = None
            else:
                self.initial_value = self.dtype.type(initial_value)
            self.reference = False
        self.allow_invalid = bool(allow_invalid)
        # This restriction prevents indefinitely long chain reactions when destroying objects.
        if self.reference is self.db_class: assert self.allow_invalid
        self.valid_range = tuple(valid_range)
        if None not in self.valid_range: self.valid_range = tuple(sorted(self.valid_range))
        assert len(self.valid_range) == 2
        self._setup_initial_distribution(initial_distribution)
        self.memory_space = self.db_class.database.memory_space
        setattr(self.db_class.instance_type, self.name,
                property(self._getter_wrapper, self._setter_wrapper, doc=self.doc,))
        if class_attribute:
            # Add class_attributes to the instance_type as well as the
            # ClassAttrMeta because the meta class does not show up in the
            # documentation.
            setattr(self.db_class.ClassAttrMeta, self.name,
                    property(self._getter_wrapper, self._setter_wrapper, doc=self.doc,))

    def _setup_initial_distribution(self, initial_distribution):
        if initial_distribution is None:
            self.initial_distribution = None
            return
        if self.initial_value is not None:
            raise ValueError("initial_value and initial_distribution parameters are mutually exclusive!")
        assert not isinstance(initial_distribution, str)
        distribution_type, a, b = initial_distribution
        lower_bound, upper_bound = self.valid_range
        if distribution_type == 'uniform':
            distr_lower = float(a)
            distr_upper = float(b)
            if distr_lower > distr_upper:
                (distr_lower, distr_upper) = (distr_upper, distr_lower)
            if lower_bound is not None: assert distr_lower >= lower_bound
            if upper_bound is not None: assert distr_upper <= upper_bound
            self.initial_distribution = (distribution_type, distr_lower, distr_upper)
        elif distribution_type == 'normal':
            mean    = float(a)
            std_dev = float(b)
            assert lower_bound <= mean <= upper_bound
            assert lower_bound <= mean - std_dev
            assert mean + std_dev <= upper_bound
            self.initial_distribution = (distribution_type, mean, std_dev)
        else:
            raise ValueError(f'Unrecognized type of random distribution "{distribution_type}"!')

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
    def get_valid_range(self):              return self.valid_range
    def get_initial_value(self):
        """ Returns either the initial_value or the initial_distribution. """
        if self.initial_distribution is not None:
            return self.initial_distribution
        else:
            return self.initial_value
    def get_class(self) -> DB_Class:        return self.db_class
    def get_database(self) -> Database:     return self.db_class.database

    def get_memory_space(self) -> str:
        """
        Returns the current location of this data component:

        Returns "host" when located in python's memory space.
        Returns "cuda" when located in CUDA's memory space.
        """
        return self.memory_space

    def is_reference(self):
        """
        Always use this method because "bool(self.reference)" can return False
        if the referenced DB_Class has no instances: "len(self.reference) == 0"
        """
        return self.reference is not False

    def free(self):
        """
        Release the memory used by this data component. The next time the data
        is accessed it will be reallocated and set to its initial_value.

        If an initial_distribution is used instead of an initial_value,
        then new values will be generated upon reallocation.
        """
        raise NotImplementedError(type(self))

    def is_free(self):
        return self.data is None

    def check(self):
        """ Check data for values which are: NaN, NULL, or out of bounds. """
        data = self.get_data()
        if isinstance(self, ClassAttribute):  data = np.array([data])
        elif isinstance(self, SparseMatrix): data = data.data
        xp = self.get_memory_space().get_array_module()
        if not self.allow_invalid:
            if self.is_reference():
                assert xp.all(xp.less(data, self.reference.size)), f"{self.qualname} is NULL"
            else:
                if data.dtype.kind in ("f", "c"):
                    assert not xp.any(xp.isnan(data)), f"{self.qualname} is NaN"
        lower_bound, upper_bound = self.valid_range
        if lower_bound is not None:
            assert xp.all(xp.less_equal(lower_bound, data)), f"{self.qualname} less than {lower_bound}"
        if upper_bound is not None:
            assert xp.all(xp.greater_equal(upper_bound, data)), f"{self.qualname} greater than {upper_bound}"

    def _remove_references_to_destroyed(self):
        raise NotImplementedError(type(self))

    def _type_info(self):
        s = ""
        if self.is_reference(): s += "ref:" + self.reference.name
        else: s += self.dtype.name
        if self.shape != (1,): s += repr(list(self.shape))
        return s

    def __repr__(self):
        return "<%s: %s.%s %s>"%(type(self).__name__, self.db_class.name, self.name, self._type_info())

    _initial_distribution_doc = """
        Argument initial_distribution may be used instead of initial_value.
                This argument has several valid configurations:
                  * initial_distribution = ('uniform', lower_bound, upper_bound)
                  * initial_distribution = ('normal', mean, standard_deviation)
        """

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

class Attribute(DataComponent):
    """ This is the database's internal representation of an instance variable. """
    def __init__(self, db_class, name:str, initial_value=None, *, dtype=Real, shape=(1,),
                doc:str="", units:str="", allow_invalid:bool=False, valid_range=(None, None),
                initial_distribution=None,):
        """ Add an instance variable to a class type.

        Argument initial_value is written to new instances of this attribute.
                This is applied before "base_class.__init__" is called.
                Optional, if not given then the data will not be initialized.
        """
        DataComponent.__init__(self, db_class, name,
            doc=doc, units=units, dtype=dtype, shape=shape, initial_value=initial_value,
            allow_invalid=allow_invalid, valid_range=valid_range, initial_distribution=initial_distribution,)
        self.free()

    __init__.__doc__ += "".join((
        DataComponent._initial_distribution_doc,
        DataComponent._dtype_doc,
        DataComponent._shape_doc,
        DataComponent._units_doc,
        DataComponent._allow_invalid_doc,
        DataComponent._valid_range_doc,))

    def free(self):
        self.data = None

    def _alloc_if_free(self):
        if self.data is None:
            self.data = self._alloc(len(self.db_class))
            self._append(0, len(self.db_class))

    def _getter(self, instance):
        if self.data is None:
            if self.initial_distribution is not None:
                self._alloc_if_free()
                value = self.data[instance._idx]
            else:
                value = self.initial_value
        else:
            value = self.data[instance._idx]
        if self.memory_space is memory_spaces.cuda and hasattr(value, 'get'):
            value = value.get()
        if self.is_reference():
            return self.reference.index_to_object(value)
        return self.dtype.type(value)

    def _setter(self, instance, value):
        self._alloc_if_free()
        if self.is_reference():
            if value is None:
                value = NULL
            else:
                if not isinstance(value, self.reference.instance_type):
                    ref_name  = self.reference.get_name()
                    type_name = type(value).__name__
                    raise TypeError(f"Expected '{ref_name}' got '{type_name}'!")
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
        elif self.initial_distribution is not None:
            distr_type, a, b = self.initial_distribution
            num_append = new_size - old_size
            xp = self.memory_space.array_module
            if distr_type == 'uniform':
                self.data[old_size:new_size] = xp.random.uniform(a, b, size=num_append)
            elif distr_type == 'normal':
                self.data[old_size:new_size] = xp.random.normal(a, b, size=num_append).clip(*self.valid_range)
            else:
                raise NotImplementedError(distr_type)

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
        self._transfer(self.db_class.database.memory_space)
        self._alloc_if_free()
        return self.data[:len(self.db_class)]

    def set_data(self, value):
        size = len(self.db_class)
        assert len(value) == size
        if self.shape == (1,):
            shape = (size,) # Don't append empty trailing dimension.
        else:
            shape = (size,) + self.shape
        # TODO: This should accept whatever memory space it is given, and avoid
        # transfering until someone calls "get_data".
        self.data = self.memory_space.array(value, dtype=self.dtype).reshape(shape)

    def _remove_references_to_destroyed(self):
        if not self.is_reference(): return
        if self.data is None: return
        pointer_data = self.data[:len(self.db_class)]
        destroyed_mask = self.reference._destroyed_mask
        if destroyed_mask is None: return
        xp = memory_spaces.get_array_module(destroyed_mask)

        target_is_dead = xp.take(destroyed_mask, pointer_data, axis=0, mode='clip')
        target_is_dead[pointer_data == NULL] = True
        target_is_dead = xp.nonzero(target_is_dead)[0]
        pointer_data[target_is_dead] = NULL
        if not self.allow_invalid:
            for idx in target_is_dead:
                db_obj = self.db_class.index_to_object(idx)
                if db_obj is not None:
                    db_obj.destroy()
                    destroyed_mask[idx] = True

class ClassAttribute(DataComponent):
    """ This is the database's internal representation of a class variable. """
    def __init__(self, db_class, name:str, initial_value=None, *,
                dtype=Real, shape=(1,),
                doc:str="", units:str="",
                allow_invalid:bool=False, valid_range=(None, None),
                initial_distribution=None,):
        """ Add a class variable to a class type.

        All instance of the class will use a single shared value for this attribute.

        Argument initial_value or initial_distribution is required.
        """
        DataComponent.__init__(self, db_class, name,
                dtype=dtype, shape=shape, doc=doc, units=units, initial_value=initial_value,
                allow_invalid=allow_invalid, valid_range=valid_range, initial_distribution=initial_distribution,
                class_attribute=True)
        if (self.initial_value is None) and (self.initial_distribution is None):
            raise TypeError("Missing 1 required positional argument: 'initial_value'")
        self.free()
        self.memory_space = memory_spaces.host
        if self.is_reference(): raise NotImplementedError

    __init__.__doc__ += "".join((
        DataComponent._initial_distribution_doc,
        DataComponent._dtype_doc,
        DataComponent._shape_doc,
        DataComponent._units_doc,
        DataComponent._allow_invalid_doc,
        DataComponent._valid_range_doc,
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
        if self.initial_distribution is not None:
            distr_type, a, b = self.initial_distribution
            if distr_type == 'uniform':
                self.set_data(np.random.uniform(a, b))
            elif distr_type == 'normal':
                self.set_data(np.random.normal(a, b).clip(*self.valid_range))
            else:
                raise NotImplementedError(distr_type)
        else:
            self.data = self.initial_value

    def _remove_references_to_destroyed(self):
        pass

class SparseMatrix(DataComponent):
    """ """ # TODO-DOC

    # TODO: Figure out if/when to call mat.eliminate_zeros() and sort too.

    # I really want to like to use enums, and they are great an all....
    #       But strings are a lot shorter on the page.
    #               "lil" vs SparseMatrix.Format.lil
    #       Pros & Cons:
    #           + faster to check pointer identity than string equality.
    #           + easier to use, harder to mess up (error checking).
    #           + more organized. Can attach methods to enum. start with: _matrix_class
    #           - more verbose.
    # class Format(enum.Enum):
    #     lil = "lil"
    #     coo = "coo"
    #     csr = "csr"

    def __init__(self, db_class, name, column, *, dtype=Real, doc:str="", units:str="",
                allow_invalid:bool=False, valid_range=(None, None),):
        """
        Add a sparse matrix that is indexed by DB_Objects. This is useful for
        implementing any-to-any connections between entities.

        This db_class is the index for the rows of the sparse matrix.

        Argument column refers to the db_class which is the index for the
                columns of the sparse matrix.
        """
        DataComponent.__init__(self, db_class, name,
                dtype=dtype, shape=None, doc=doc, units=units, initial_value=0.,
                allow_invalid=allow_invalid, valid_range=valid_range)
        self.column = self.db_class.database.get_class(column)
        self.column.referenced_by_matrix_columns.append(self)
        self.shape = (len(self.db_class), len(self.column))
        self.fmt = 'csr'
        self.free()
        if self.is_reference(): raise NotImplementedError

    __init__.__doc__ += "".join((
        DataComponent._dtype_doc,
        DataComponent._units_doc,
        DataComponent._allow_invalid_doc,
        DataComponent._valid_range_doc,))

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
        dead_rows = self.db_class._destroyed_mask
        dead_cols = self.column._destroyed_mask
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
            data  = matrix.data[0]
        elif self.fmt == 'csr':
            index = matrix.indices
            data  = list(matrix.data)
        else: raise NotImplementedError(self.fmt)
        index_to_object = self.column.index_to_object
        return ([index_to_object(x) for x in index], data)

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
        new_shape = self.shape = (len(self.db_class), len(self.column))

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
        if self.memory_space is target_space:
            return
        if self.memory_space is memory_spaces.host:
            if target_space is memory_spaces.cuda:
                # Python-CUDA only supports floating-point and complex numbers.
                dtype = self.dtype
                if dtype.kind in 'bui':
                    if dtype.itemsize <= 2:
                        dtype = np.dtype('f4')
                    else:
                        dtype = np.dtype('f8')
                if dtype.kind not in 'fc':
                    # Unsupported data type.
                    # Silently refuse to transfer, leave data on host.
                    return
                self.memory_space = target_space
                if self.fmt == 'lil':
                    self.fmt = 'csr'
                if self.fmt == 'csr' or self.fmt == 'coo':
                    if self.data is not None:
                        self.data = self._matrix_class(self.data, dtype=dtype)
                else:
                    raise NotImplementedError(self.fmt)
                self._host_lil_mem = None
            else:
                raise NotImplementedError(target_space)
        elif self.memory_space is memory_spaces.cuda:
            if target_space is memory_spaces.host:
                if self.data is not None:
                    self.data = self.data.get()
                self.memory_space = target_space
            else:
                raise NotImplementedError(target_space)
        else:
            raise NotImplementedError(self.memory_space)

    def get_data(self):
        self._transfer(self.db_class.database.memory_space)
        self._alloc_if_free()
        return self.data

    def set_data(self, matrix):
        for mem in (memory_spaces.host, memory_spaces.cuda):
            if mem and isinstance(matrix, mem.matrix_module.spmatrix):
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

class ConnectivityMatrix(SparseMatrix):
    """ """ # TODO-DOC
    def __init__(self, db_class, name, column, *, doc:str=""):
        """ """ # TODO-DOC
        super().__init__(db_class, name, column, doc=doc, dtype=bool,)

    def _getter(self, instance):
        return super()._getter(instance)[0]

    def _setter(self, instance, values):
        super()._setter(instance, (values, [True] * len(values)))

DB_Class.add_attribute.__doc__              = Attribute.__init__.__doc__
DB_Class.add_class_attribute.__doc__        = ClassAttribute.__init__.__doc__
DB_Class.add_sparse_matrix.__doc__          = SparseMatrix.__init__.__doc__
DB_Class.add_connectivity_matrix.__doc__    = ConnectivityMatrix.__init__.__doc__

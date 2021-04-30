""" A custom Entity-Component-System for NEUWON. """

from collections.abc import Callable, Iterable, Mapping
import numpy as np
import cupy
import textwrap

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
        self.components = {}

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

        Argument name:
        Argument dtype:
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

    def create_entity(self, archetype: str, number_of_instances: int = 1) -> list:
        """ Create instances of an archetype.
        Returns their internal ID's. """
        ark = self.archetypes[str(archetype)]
        num = int(number_of_instances); assert(num >= 0)
        old_size = ark.size
        new_size = old_size + num
        ark.size = new_size
        for arr in ark.components: arr._append_entities(old_size, new_size)
        return range(old_size, new_size)

    def destroy_entity(self, archetype: str, instances: list):
        archetype = str(archetype)
        assert(archetype in self.archetypes)
        1/0 # TODO Recursively mark all destroyed instances, make a bitmask for aliveness.
        1/0 # TODO Compress the dead entries out of all data arrays.

    def access(self, name: str):
        """ Returns a components value or GPU data array. """
        x = self.components[str(name)]
        if isinstance(x, _Value): return x.value
        elif isinstance(x, _Array):
            if x.shape == "sparse": return x.data
            else: return x.data[:x.archetype.size]

    def check(self):
        for name, component in self.components.items():
            if not component.check: continue
            if isinstance(component, _Value):
                assert np.isfinite(component), name
            elif isinstance(component, _Array):
                if component.reference:
                    assert not cupy.any(component.data == ROOT), name
                else:
                    kind = component.dtype.kind
                    if kind == "f" or kind == "c":
                        assert cupy.all(cupy.isfinite(component.data)), name

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
    def __init__(self, doc=""):
        self.doc = textwrap.dedent(str(doc)).strip()
        self.size = 0
        self.components = []
        self.referenced_by = []

class _Value:
    def __init__(self, value, doc="",
                user_read=True, check=True):
        self.value = float(value)
        self.doc = textwrap.dedent(str(doc)).strip()
        self.user_read = bool(user_read)
        self.check = bool(check)

class _Array:
    def __init__(self, archetype, doc=doc, dtype=Real, shape=(1,), initial_value=np.nan,
                user_read=False, user_write=False, check=True):
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

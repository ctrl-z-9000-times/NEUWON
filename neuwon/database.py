from collections.abc import Callable, Iterable, Mapping
import numpy as np
import cupy
import textwrap

Real = np.dtype('f4')
epsilon = np.finfo(Real).eps
Location = np.dtype('u4')
ROOT = np.iinfo(Location).max

# TODO: How to deal with children and neighbors? I want that data to be
# accessible via the same API as all of the other data in the database, but I
# don't know how to manage that data. Should it even be on the GPU? Can it be
# transformed into a sparse matrix? How will that work with add/remove'ing instances?
#       Children
#       extracellular Neighbor location
#       extracellular Neighbor distance
#       extracellular Neighbor border_surface_area

# TODO: Now that I can add/remove entities at runtime, I need a stable handle on
# an entity which does not get blown away when the data gets reallocated and remapped.

sep = "/"

class Database:
    """ The Database class is a custom Entity-Component-System. """
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
            dtype=Real, reference=False, shape=(1,), initial_value=None,
            user_read=False, user_write=False, check=True):
        """ Add a new data array to an Archetype.

        The archetype must be specified by prefixing the component name wit hthe
        archetype name followed by a slash "/".
        """
        name = str(name)
        assert(name not in self.components)
        archetype, component = name.split(sep, maxsplit=1)
        assert(archetype in self.archetypes)
        self.components[name] = arr = _Array(self.archetypes[archetype], doc=doc,
            dtype=dtype, reference=reference, shape=shape, initial_value=initial_value,
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
        return np.arange(old_size, new_size)

    def destroy_entity(self, archetype: str, instances: list):
        archetype = str(archetype)
        assert(archetype in self.archetypes)
        1/0 # TODO Recursively mark all destroyed instances, make a bitmask for aliveness.
        1/0 # TODO Compress the dead entries out of all data arrays.

    def access(self, name: str):
        """ Returns a components value or GPU data array. """
        x = self.components[str(name)]
        if isinstance(x, _Value): return x.value
        elif isinstance(x, _Array): return x.data[:x.archetype.size]

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
    def __init__(self, archetype, doc=doc, dtype=Real, reference=False, shape=(1,), initial_value=np.nan,
                user_read=False, user_write=False, check=True):
        self.archetype = archetype; archetype.components.append(self)
        self.doc = textwrap.dedent(str(doc)).strip()
        if isinstance(shape, Iterable):
            self.shape = tuple(int(round(x)) for x in shape)
        else: self.shape = (int(round(shape)),)
        if reference:
            self.dtype = Location
            self.initial_value = ROOT
            self.reference = str(reference)
        else:
            self.dtype = dtype
            self.initial_value = initial_value
            self.reference = False
        self.user_read = bool(user_read)
        self.user_write = bool(user_write)
        self.check = bool(check)
        self.data = cupy.empty(self._realloc_shape(archetype.size), dtype=self.dtype)
        self._append_entities(0, archetype.size)

    def _append_entities(self, old_size, new_size):
        """ Append and initialize some new instances to the data array. """

        # TODO: IIRC CuPy can not deal with numpy structured arrays...
        #       Detect this issue and revert to using numba arrays.
        #       numba.cuda.to_device(numpy.array(data, dtype=dtype))

        if len(self.data) < new_size:
            old_data = self.data
            self.data = cupy.empty(self._realloc_shape(new_size), dtype=self.dtype)
            self.data[:len(old_data)] = old_data
        if self.initial_value is not None:
            self.data[old_size: new_size].fill(self.initial_value)

    def _realloc_shape(self, target_size):
        return (int(round(target_size * 1.25)),) + self.shape

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

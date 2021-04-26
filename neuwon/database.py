from collections.abc import Callable, Iterable, Mapping
import numpy as np
import cupy as cp

from neuwon.common import Location, ROOT, Real
# TODO: Move the location definition into the database file.

# TODO: Consider allowing documentation to be attacted to components & entities.
#       Description, Units, Assign vs Accumulate, Notes...

# TODO: How to deal with children and neighbors? I want that data to be
# accessible via the same API as all of the other data in the database, but I
# don't know how to manage that data. Should it even be on the GPU? Can it be
# transformed into a sparse matrix? How will that work with add/remove'ing instances?
#       Children
#       extracellular Neighbor location
#       extracellular Neighbor distance
#       extracellular Neighbor border_surface_area

# TODO: Consider having an intracellular neighbor & border_surface_area?
#       This would replace children?

# TODO: Now that I can add/remove entities at runtime, I need a stable handle on
# an entity which does not get blown away when the data gets reallocated and remapped.


class Database:
    """ The Database is a custom Entity Component System.

    TODO: Explain how entity types define a prototypical entity type...
    Entities are organized by their type, and each entity type has a standard
    prototypical set of components.

    The database does not care what the entities or components are named.
    The database does not do any computations or execute any given code.
    """
    def __init__(self):
        self.entity_types = {}
        self.components = {}

    def add_entity_type(self, entity_name: str):
        entity_name = str(entity_name)
        assert(entity_name not in self.entity_types)
        self.entity_types[entity_name] = _EntityType()

    def add_global_constant(self, name: str, value):
        name = str(name)
        assert(name not in self.components)
        self.components[name] = float(value)

    def add_component(self, entity_type: str, component_name: str,
            dtype=Real, shape=(1,),
            reference=False,
            initial_value=None,
            user_read=False, user_write=False, check=True):
        entity_type = str(entity_type)
        assert(entity_type in self.entity_types)
        component_name = str(component_name)
        assert(component_name not in self.components)
        self.components[component_name] = x = _Array(dtype=dtype, shape=shape, reference=reference,
            initial_value=initial_value, user_read=user_read, user_write=user_write, check=check)
        x.entity_type = self.entity_types[entity_type]
        x.entity_type.components.append(x)
        if x.reference:
            assert(x.reference in self.entity_types)
            self.entity_types[x.reference].referenced_by.append(x)
        x._create_instances(0, x.entity_type.size)

    def create_instances(self, entity_type: str, number_of_instances: int) -> list:
        ent = self.entity_types[str(entity_type)]
        num = int(number_of_instances); assert(num >= 0)
        old_size = ent.size
        new_size = ent.size + num
        for x in ent.components: x._create_instances(old_size, new_size)
        indexes = np.arange(ent.size, new_size)
        ent.size = new_size
        return indexes

    def destroy_instances(self, entity_type: str, instances: list):
        entity_type = str(entity_type)
        assert(entity_type in self.entity_types)
        1/0 # TODO Recursively mark all destroyed instances, make a bitmask for aliveness.
        1/0 # TODO Compress the dead entries out of all data arrays.

    def access(self, component_name: str):
        x = self.components[str(component_name)]
        if isinstance(x, float): return x
        elif isinstance(x, _Array): return x.data[:x.entity_type.size]

    def check(self):
        for name, component in self.components.items():
            if isinstance(component, float):
                assert(np.isfinite(component), name)
            elif isinstance(component, _Array):
                if not component.check: continue
                if component.reference:
                    1/0 # Check for ROOT
                else:
                    kind = component.dtype.kind
                    if kind == "f" or kind == "c":
                        assert(cp.all(cp.isfinite(component.data)), name)

    def __repr__(self) -> str:
        s = ""
        s += "Entity Types:\n"
        for name, entity in sorted(self.entity_types.items()):
            s += name.ljust(80) + "\n"
            # TODO: Print all of the entity type data here
        s += "\n"
        s += "Component".ljust(40) + "\n"
        for name, entity in sorted(self.components.items()):
            s += name.ljust(40) + "\n"
            # TODO: Make a table of all of the flags for arrays.
        return s

class _EntityType:
    def __init__(self):
        self.size = 0
        self.components = []
        self.referenced_by = []

class _Array:
    def __init__(self, dtype=Real, shape=(1,), initial_value=np.nan, reference=False,
                user_read=False, user_write=False, check=True):
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
        self.data = None

    def _create_instances(self, old_size, new_size):
        """ Append and initialize some new instances to the data array. """
        
        # TODO: IIRC CuPy can not deal with numpy structured arrays...
        #       Detect this issue and revert to using numba arrays.
        #       numba.cuda.to_device(numpy.array(data, dtype=dtype))

        if self.data is None:
            self.data = cp.empty((int(new_size * 1.25),)+self.shape, dtype=self.dtype)
        elif len(self.data) < new_size:
            old_data = self.data
            self.data = cp.empty((int(new_size * 1.25),)+self.shape, dtype=self.dtype)
            self.data[:len(old_data)] = old_data
        if self.initial_value is not None:
            self.data[old_size: new_size].fill(self.initial_value)



if __name__ == "__main__":
    db = Database()
    db.add_entity_type("Location")
    db.add_component("Location", "intra/na/g", initial_value=0)
    db.create_instances("Location", 10)
    print(db)

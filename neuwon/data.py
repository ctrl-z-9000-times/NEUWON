"""
IDEA: Consider replacing AccessHandles with a string path identifier.
instead of >>> model.read(DataAccess('na', conductance=True), location)
do >>> model.read("na/conductance", location)
Or name aliases like: "na/i" and "na/o" "na/g"

So pretty much I want to re-create the entire model except as a directory of
string identifiers. Then I can automate a lot of the boiler plate away.

The objects in the data store can be any of:
      Single omnipresent value
      Array of values
      Array of pointers to other locations

voltage
geometry/property
species_name/in/concentration
species_name/in/release_rate
species_name/out/variable_name
reaction_name/
reaction_name/insertions - dtype=locations
reaction_name/coupled_reaction - dtype=pointer to reaction_instance
reaction_name/states/
"""

import numpy as np

# MEMO: make this code totally agnostic to the contents of the data.
# Also, the name is arbitrary, users can specify any name they like.

class Database:
    def __init__(self):
        self.collections = {}
        self.data = {}

    def add_collection(self, name):
        self.collections[name] = Collection()

    def add_constant(self, name, value):
        1/0

    def add_array(self, name, collection,
            dtype=Real, shape=(1,),
            initial_value=np.nan,
            user_read=False, user_write=False,):
        1/0

    def add_reference(self, name, collection, target):
        1/0

    def add_instance(self, name, collection) -> int:
        1/0

    def host_access(self, name):
        1/0

    def device_access(self, name):
        1/0

    def check(self):
        1/0

class _Collection:
    def __init__(self, size=0):
        self.size = int(round(size))
        self.freelist = []

class Array:
    def __init__(self, dtype=Real, shape=(1,), initial_value=np.nan, collection=None,
                user_read=False, user_write=False,):
        self.dtype = dtype
        if isinstance(shape, Iterable):
            self.shape = tuple(int(round(x)) for x in shape)
        else: self.shape = (int(round(shape)),)
        self.initial_value = initial_value
        self.index = str(index)
        self.user_read = bool(user_read)
        self.user_write = bool(user_write)
        self._host_data = None
        self._device_data = None
        self._host_data_valid = False
        self._device_data_valid = False

class Reference:
    pass

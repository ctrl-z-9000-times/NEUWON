"""
IDEA: Consider replacing AccessHandles with a string path identifier.
instead of >>> model.read(DataAccess('na', conductance=True), location)
do >>> model.read("na/conductance", location)
Or name aliases like: "na/i" and "na/o" "na/g"

voltage
geometry/property
species_name/intra_concentration
species_name/extra_concentration
species_name/intra_release_rate
species_name/extra_release_rate
species_name/conductance
reaction_name/variable_name

So pretty much I want to re-create the entire model except as a directory of
string identifiers.
The objects in the data store can be any of:
      Single omnipresent value
      Array of values
      Array of pointers to other locations
      Sparse matrix
      Species object?


/voltage
/geometry/property
/species_name/in/concentration
/species_name/in/release_rate
/species_name/out/variable_name
/reaction_name/
/reaction_name/insertions - dtype=locations
/reaction_name/coupled_reaction - dtype=pointer to reaction_instance
/reaction_name/states/
"""

class Value:
    pass

class Array:
    def __init__(self, dtype=Real, shape=(1,), initial_value=0.0, index="location",
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

    def read_host(self):
        1/0

    def write_host(self):
        1/0

    def read_device(self):
        1/0

    def write_device(self):
        1/0

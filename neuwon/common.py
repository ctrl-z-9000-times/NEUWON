import numpy as np
from collections.abc import Callable, Iterable, Mapping

F = 96485.3321233100184 # Faraday's constant, Coulombs per Mole of electrons
R = 8.31446261815324 # Universal gas constant
celsius = 37 # Human body temperature
T = celsius + 273.15 # Human body temperature in Kelvins

Real = np.dtype('f4')
epsilon = np.finfo(Real).eps
Location = np.dtype('u4')
ROOT = np.iinfo(Location).max

# TODO: Split Neighbor into three separate properties. Then remove my custom Neighbor dtype.

# IDEA: Consider replacing AccessHandles with a string path identifier.
# instead of >>> model.read(DataAccess('na', conductance=True), location)
# do >>> model.read("na/conductance", location)
# Or name aliases like: "na/i" and "na/o" "na/g"
#       voltage
#       geometry/property
#       species_name/intra_concentration
#       species_name/extra_concentration
#       species_name/intra_release_rate
#       species_name/extra_release_rate
#       species_name/conductance
#       reaction_name/variable_name

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

class AllArrays(dict):
    def __init__(self):
        dict.__init__(self)

    def add_array(self, path, **kw_args):
        x = Array(**kw_args)
        self[path] = x
        return x


class AccessHandle:
    """ The AccessHandle class is an enumeration of all publicly accessible data
    in the Model. AccessHandles are the primary way of getting data into or out
    of the Model.

    AccessHandles are often associated with a name. """

    @property
    def species(self):
        """ """
        return self._species
    @property
    def reaction_instance(self):
        """ Allocate an array of the given numpy.dtype for each inserted instance
        of this reaction. It can be one of:
            * numpy.dtype,
            * Pair of (numpy.dtype, shape) to to make an array.
        Examples:
            np.float32
            (np.float32, 7)
            (np.float32, [4, 4])"""
        return self._reaction_instance
    @property
    def reaction_reference(self):
        """ Access a Reactions instance data.
        This is always a pair of: ("reaction-name", "pointer-name")

        Note: Reactions are not run in a deterministic order. Dynamics which
        span between reactions via reaction_references should operate at a
        significantly slower time scale than the time step. """
        return self._reaction_reference
    @property
    def voltage(self):
        """ Units: Volts. """
        return self._voltage
    @property
    def conductance(self):
        """ Units: Siemens """
        return self._conductance
    @property
    def intra_concentration(self):
        """ Units: Molar """
        return self._intra_concentration
    @property
    def extra_concentration(self):
        """ Units: Molar """
        return self._extra_concentration
    @property
    def intra_release_rate(self):
        """ Units: Molar / Second """
        return self._intra_release_rate
    @property
    def extra_release_rate(self):
        """ Units: Molar / Second """
        return self._extra_release_rate
    @property
    def coordinates(self):
        """ """
        return self._coordinates
    @property
    def diameters(self):
        """ """
        return self._diameters
    @property
    def parents(self):
        """ """
        return self._parents
    @property
    def children(self):
        """ """
        return self._children
    @property
    def surface_areas(self):
        """ """
        return self._surface_areas
    @property
    def cross_sectional_areas(self):
        """ """
        return self._cross_sectional_areas
    @property
    def intra_volumes(self):
        """ """
        return self._intra_volumes
    @property
    def extra_volumes(self):
        """ """
        return self._extra_volumes
    @property
    def neighbors(self):
        """Adjacent locations in the partitioning of the extracellular medium.
        geometry.neighbors[location] = array with data type neuwon.geometry.Neighbor """
        return self._neighbors
    @property
    def geometric(self):
        """ """
        return self._geometric

    @property
    def read(self):
        """ """
        return self._read
    @property
    def write(self):
        """ """
        return self._write
    @property
    def omnipresent(self):
        """ """
        return self._omnipresent
    @property
    def parallel(self):
        """ """
        return self._parallel
        # TODO: Consider renaming this to "instance" or "inserted"
        # TODO: Also consider removing this flag since it is always the opposite of omnipresent.

    def __init__(self, species=None,
            reaction_instance=None,
            reaction_reference=None,
            intra_concentration=False,
            extra_concentration=False,
            intra_release_rate=False,
            extra_release_rate=False,
            voltage=False,
            conductance=False,
            coordinates=False,
            diameters=False,
            parents=False,
            children=False,
            surface_areas=False,
            cross_sectional_areas=False,
            intra_volumes=False,
            extra_volumes=False,
            neighbors=False,):
        self._species = str(species) if species else None
        if reaction_instance is not None: self._reaction_instance = _parse_dtype(reaction_instance)
        else: self._reaction_instance = None
        if reaction_reference:
            reaction_name, handle_name = reaction_reference
            self._reaction_reference = (str(reaction_name), str(handle_name))
        else: self._reaction_reference = None
        self._intra_concentration = bool(intra_concentration)
        self._extra_concentration = bool(extra_concentration)
        self._intra_release_rate  = bool(intra_release_rate)
        self._extra_release_rate  = bool(extra_release_rate)
        self._voltage     = bool(voltage)
        self._conductance = bool(conductance)
        self._coordinates = bool(coordinates)
        self._diameters = bool(diameters)
        self._parents = bool(parents)
        self._children = bool(children)
        self._surface_areas = bool(surface_areas)
        self._cross_sectional_areas = bool(cross_sectional_areas)
        self._intra_volumes = bool(intra_volumes)
        self._extra_volumes = bool(extra_volumes)
        self._neighbors = bool(neighbors)
        self._geometric = sum((self.coordinates, self.diameters,
                self.parents, self.children, self.surface_areas,
                self.cross_sectional_areas, self.intra_volumes,
                self.extra_volumes, self.neighbors))
        assert(1 == bool(self.reaction_instance) + bool(self.reaction_reference) +
                self.voltage + self.conductance +
                self.intra_concentration + self.extra_concentration +
                self.intra_release_rate + self.extra_release_rate + self.geometric)
        self._geometric = bool(self._geometric)
        self._read = (bool(self.reaction_instance) or bool(self.reaction_reference) or
                self.voltage or self.intra_concentration or self.extra_concentration
                or self.geometric)
        self._write = (bool(self.reaction_instance) or bool(self.reaction_reference) or
                self.conductance or self.intra_release_rate or self.extra_release_rate)
        self._omnipresent = (self.voltage or self.conductance or
                self.intra_concentration or self.extra_concentration or
                self.intra_release_rate or self.extra_release_rate or
                self.geometric)
        self._parallel = bool(self.reaction_instance) or bool(self.reaction_reference)
        # TODO: Make a flag for assign or accumulate.

    def NEURON_conversion_factor(self):
        """ """ # TODO!
        if   self.reaction_instance: return 1
        elif self.voltage:           return 1000 # From NEUWONs volts to NEURONs millivolts.
        elif self.conductance:       return 1
        else: raise NotImplementedError(self)

    def __repr__(self):
        name = getattr(self.species, "name", self.species)
        flags = []
        if self.reaction_instance: flags.append(str(self.reaction_instance))
        if self.voltage: flags.append("voltage=True")
        if self.conductance: flags.append("conductance=True")
        if self.intra_concentration: flags.append("intra_concentration=True")
        if self.extra_concentration: flags.append("extra_concentration=True")
        if self.intra_release_rate: flags.append("intra_release_rate=True")
        if self.extra_release_rate: flags.append("extra_release_rate=True")
        return "AccessHandle(Species=%s, %s)"%(name, ", ".join(flags))

    def __eq__(self, other):
        return repr(self) == repr(other)
    def __ne__(self, other):
        return not (self == other)
    def __hash__(self):
        return hash(repr(self))

def _parse_dtype(dtype):
    if isinstance(dtype, Iterable):
        dtype, shape = dtype
        if isinstance(shape, Iterable):
            shape = list(shape)
        else:
            shape = [shape]
    else:
        shape = []
    assert(isinstance(dtype, np.dtype))
    return (dtype, shape)

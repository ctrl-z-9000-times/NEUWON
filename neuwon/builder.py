import numpy as np
import math
from neuwon import Model, Species, Mechanism
from graph_algorithms import depth_first_traversal as dft

class ModelBuilder:
    def __init__(self):
        self.neurons = []
        self.reactions = []
        self.species = {}

    def add_neuron(self, coordinates, diameter):
        n = Neurite(coordinates, diameter)
        self.neurons.append(n)
        return n

    def add_reaction(self, reaction):
        # TODO: Reactions should be able to specify required species, in the
        # same way as mechanisms do.
        self.reactions.append(reactions)

    def add_species(self, species):
        if isinstance(species, Species):
            self.species[species.name] = species
        else:
            self.species[str(species)] = None

    def build_model(self, time_step, stagger=True):
        locations = []
        for n in self.neurons:
            locations.extend(dft(n, lambda x: x.children))
        coordinates = [x.coordinates for x in locations]
        for index, x in enumerate(locations):
            x.index = index
        parents = [getattr(x.parent, "index", None) for x in locations]
        diameters = [x.diameter for x in locations]
        insertions = [x.insertions for x in locations]
        for insertions_list in insertions:
            for mechanism_spec in insertions_list:
                for s in mechanism_spec[0].species:
                    if isinstance(s, Species):
                        if s.name not in self.species:
                            self.species[s.name] = s
                    else:
                        s = str(s)
                        if s not in self.species:
                            self.species[s] = None
        for name, species in self.species.items():
            if species is None:
                raise ValueError("Unresolved species: "+str(name))

        return Model(time_step,
            coordinates=coordinates,
            parents=parents,
            diameters=diameters,
            insertions=insertions,
            reactions=self.reactions,
            species=self.species.values(),
            stagger=stagger,)

def _docstring_wrapper(property_name, docstring):
    def get_prop(self):
        return self.__dict__[property_name]
    def set_prop(self, value):
        self.__dict__[property_name] = value
    return property(get_prop, set_prop, None, docstring)

class Neurite:
    parent      = _docstring_wrapper("parent", "Neurite or None")
    children    = _docstring_wrapper("children", "List of Neurite's")
    coordinates = _docstring_wrapper("coordinates", "Tuple of 3 floats")
    diameter    = _docstring_wrapper("diameter", "Float (positive)")
    insertions  = _docstring_wrapper("insertions", "List of pairs of (mechanisms, mechanisms_arguments_tuple)")
    def __init__(self, coordinates, diameter, parent=None):
        self.parent = parent
        self.children = []
        self.coordinates = coordinates
        self.diameter = diameter
        self.insertions = []

    def add_segment(self, coordinates, diameter, maximum_segment_length=float('inf')):
        coordinates = tuple(float(x) for x in coordinates)
        diameter = float(diameter)
        maximum_segment_length = float(maximum_segment_length)
        assert(diameter >= 0)
        assert(maximum_segment_length > 0)
        parent = self
        parent_diameter = self.diameter
        parent_coordinates = self.coordinates
        length = np.linalg.norm(np.subtract(parent_coordinates, coordinates))
        divisions = max(1, math.ceil(length / maximum_segment_length))
        segments = []
        for i in range(divisions):
            x = (i + 1) / divisions
            _x = 1 - x
            coords = (  coordinates[0] * x + parent_coordinates[0] * _x,
                        coordinates[1] * x + parent_coordinates[1] * _x,
                        coordinates[2] * x + parent_coordinates[2] * _x)
            diam = diameter * x + parent_diameter * _x
            child = Neurite(coords, diam, parent)
            parent.children.append(child)
            segments.append(child)
            parent = child
        return segments

    def insert_mechanism(self, mechanism, *args):
        assert(issubclass(mechanism, Mechanism))
        self.insertions.append((mechanism, args))

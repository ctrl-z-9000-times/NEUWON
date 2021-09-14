from neuwon.growth import GrowthRoutine, Soma, Tree
from neuwon.database import *

default_parameters = {
    "soma_diameter": 8,
    "dendrite_morphology": {
        "carrier_point_density": .0003,
        "balancing_factor": .7,
        "extension_distance": 40,
        "bifurcation_distance": 40,
        "maximum_segment_length": 20,
        "diameter": 1.5,
    },
    "dendrite_mechanisms": {
        "hh": {},
    }
}

class NeuronType(GrowthRoutine):
    def __init__(self, database, name, soma_region):
        self.db_cls = database.add_class(name)
        self.db_cls.add_attribute("root", dtype="Segment")

    def get_all(self):
        return self.db_cls.get_all_instances()

    def grow_synapses(self):
        # Must grow synapses in a separate pass, after all dendrites & axons
        # have been build for all cell types.
        1/0

class Excitatory(NeuronType):
    def __init__(self, Segment, region, parameters={}):
        combined_params = dict(default_parameters)
        combined_params.update(parameters)
        self.soma = Soma(Segment, region, 8)
        self.dendrites = Tree(self.soma, region,
                **combined_params["dendrite_morphology"])
        self.axon = Tree(self.soma, region, .001,
                extend_before_bifurcate = False,
                only_bifurcate = True,
                diameter = .9,)

    def grow(self, num_cells=1):
        self.soma.grow(num_cells)
        self.dendrites.grow()
        self.axon.grow()

    def grow_synapses(self):
        1/0

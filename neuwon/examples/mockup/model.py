from neuwon.growth import GrowthRoutine, Soma, Tree
from neuwon.database import *

class NeuronType(GrowthRoutine):
    def grow_synapses(self):
        # Must grow synapses in a separate pass, after all dendrites & axons
        # have been build for all cell types.
        1/0

class Excitatory(NeuronType):
    def __init__(self, Segment, region, **kwargs):
        self.soma = Soma(Segment, region, 8)
        self.dendrites = Tree(self.soma, region, .0003,
                balancing_factor = .7,
                extension_distance = 40,
                bifurcation_distance = 40,
                extend_before_bifurcate = False,
                only_bifurcate = True,
                maximum_segment_length = 20,
                diameter = 1.5,
        )
        self.axon = Tree(self.soma, region, .0002,
                diameter = .9,)

    def grow(self, num_cells=1):
        self.soma.grow(num_cells)
        self.dendrites.grow()
        self.axon.grow()

    def grow_synapses(self):
        1/0

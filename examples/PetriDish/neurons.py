import numpy as np
from neuwon import *
from neuwon.growth import *
from neuwon.mechanisms import HH, Destexhe1994, Mongillo2008

um3 = (1e6) ** 3

class ExcitatoryNeuron:
    def __init__(self, region):
        soma_diameter = 6e-6
        self.soma = GrowSomata.single(region.sample_point(), soma_diameter)
        self.axon = Growth(self.soma[0], region, 0.00015*um3,
            balancing_factor = 0,
            extension_angle = np.pi / 3,
            extension_distance = 60e-6,
            bifurcation_angle = np.pi / 2,
            bifurcation_distance = 40e-6,
            extend_before_bifurcate = True,
            only_bifurcate = True,
            maximum_segment_length = 20e-6,
            diameter = .7e-6,
        )
        self.dendrite = Growth(self.soma, region, 0.00015*um3,
            balancing_factor = .7,
            extension_distance = 40e-6,
            bifurcation_distance = 40e-6,
            extend_before_bifurcate = False,
            only_bifurcate = True,
            maximum_segment_length = 10e-6,
            diameter = None,
        )
        self.segments = self.soma + self.axon.segments + self.dendrite.segments
        # Insert mechansisms.
        for x in self.soma + self.axon.segments:
            x.insert_mechanism(HH.Leak)
            x.insert_mechanism(HH.VoltageGatedSodiumChannel)
            x.insert_mechanism(HH.VoltageGatedPotassiumChannel)

def excitatory_synapses(axons, dendrites, num_synapses):
    synapses = GrowSynapses(axons, dendrites,
            (0, .6e-6, 3e-6), 1e-6, num_synapses)
    presyn_config = Mongillo2008.Presynapses(
        transmitter = "glutamate",
        minimum_utilization = .2,
        utilization_decay = 200e-3,
        resource_recovery =  1e-3)
    for x in synapses.presynaptic_segments:
        x.insert_mechanism(presyn_config, strength=1e-15)
    for x in synapses.postsynaptic_segments:
        x.insert_mechanism(Destexhe1994.AMPA5)
        # x.insert_mechanism(Destexhe1994.NMDA5)
    return synapses

class InhibitoryNeuron:
    def __init__(self, region):
        1/0


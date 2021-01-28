import numpy as np
import scipy.spatial
import random
import itertools
import matplotlib.pyplot as plt
import argparse

from graph_algorithms import depth_first_traversal as dft
from neuwon import *
from neuwon.regions import *
from neuwon.growth import *
from neuwon.mechanisms import HH, Destexhe1994, Mongillo2008

class Experiment:
    def __init__(self):
        self.time_step = .1e-3
        self.make_model()
        self.run_experiment()

    def make_model(self):
        width = 120e-6 # X & Y dimensions.
        depth = 10e-6 # Z dimension.
        rgn = Intersection([
                Rectangle([-width/2,-width/2,-depth/2], [width/2,width/2,depth/2]),
                Sphere([0,0,0], width/2)]
        )
        soma_diameter = 5e-6
        self.presyn = presyn = GrowSomata.single([-width/2,0, 0], soma_diameter)
        self.postsyn = postsyn = GrowSomata.single([width/2,0, 0], soma_diameter)
        axon = Growth(presyn, rgn, 0.002e18,
            balancing_factor = 0,
            extension_angle = np.pi / 4,
            extension_distance = 60e-6,
            bifurcation_angle = np.pi / 3,
            bifurcation_distance = 20e-6,
            extend_before_bifurcate = True,
            only_bifurcate = True,
            maximum_segment_length = 10e-6,
            diameter = .8e-6,
        )
        dendrite = Growth(postsyn, rgn, 0.002e18,
            balancing_factor = .7,
            extension_distance = 50e-6,
            bifurcation_distance = 50e-6,
            extend_before_bifurcate = False,
            only_bifurcate = True,
            maximum_segment_length = 10e-6,
            diameter = None,
        )
        self.synapses = synapses = GrowSynapses(axon.segments, dendrite.segments,
            (0, .6e-6, 3e-6),
            diameter = 1e-6,
            num_synapses = 100)
        # Insert mechansisms.
        for x in dft(presyn[0], lambda x: x.children):
            x.insert_mechanism(HH.Leak)
            x.insert_mechanism(HH.VoltageGatedSodiumChannel)
            x.insert_mechanism(HH.VoltageGatedPotassiumChannel)
        for x in postsyn:
            x.insert_mechanism(HH.VoltageGatedSodiumChannel)
            x.insert_mechanism(HH.VoltageGatedPotassiumChannel)
            x.insert_mechanism(HH.Leak)
        presyn_config = Mongillo2008.Presynapses(
            transmitter = "glutamate",
            minimum_utilization = .2,
            utilization_decay = 200e-3,
            resource_recovery =  1e-3)
        for x in synapses.presynaptic_segments:
            x.insert_mechanism(presyn_config, strength=100e-15)
        for x in synapses.postsynaptic_segments:
            x.insert_mechanism(Destexhe1994.AMPA5)
            # x.insert_mechanism(Destexhe1994.NMDA5)
        # Assemble the model.
        self.model = Model(self.time_step, presyn + postsyn,
                    reactions=(),
                    species=(Destexhe1994.glutamate,))
        self.glu_data = self.model.species["glutamate"]
        # Measure the voltage at these points:
        self.probes = [presyn[0], postsyn[0],
                        synapses.presynaptic_segments[0], synapses.postsynaptic_segments[0]]

    def run_experiment(self):
        self.time_stamps = []
        self.v = [[] for _ in self.probes]
        self.glu = [[] for _ in self.probes]
        input_times = [20, 40]
        input_times.append(np.inf); input_times.sort(reverse=True)
        t = 0
        while t <= 50:
            if t > input_times[-1]:
                input_times.pop()
                self.presyn[0].inject_current(.5e-9, 1e-3)
            self.model.advance()
            t += self.time_step * 1e3
            self.time_stamps.append(t)
            for idx, p in enumerate(self.probes):
                self.v[idx].append(p.get_voltage() * 1e3)
            for idx, p in enumerate(self.probes):
                self.glu[idx].append(self.glu_data.extra.concentrations[p.location] * 1e-3)

x = Experiment()
if True:
    colors = [(0, 0, 0) for _ in range(len(x.model))]
    for l in dft(x.presyn[0], lambda l: l.children):
        colors[l.location] = (1, 0, 0)
    for l in dft(x.postsyn[0], lambda l: l.children):
        colors[l.location] = (0, 0, 1)
    for l in x.synapses.presynaptic_segments:
        colors[l.location] = (1, 1, 0)
    for l in x.synapses.postsynaptic_segments:
        colors[l.location] = (0, 1, 1)
    x.model.draw_image("test.png", (640*4, 480*4), (0, 0, -180e-6), (0,0,0), colors)

plt.figure()
plt.plot(x.time_stamps, x.v[0], 'r',
         x.time_stamps, x.v[3], 'g',
         x.time_stamps, x.v[1], 'b',)
plt.figure()
plt.plot(x.time_stamps, x.glu[2], 'r',
         x.time_stamps, x.glu[3], 'b')
plt.show()

args = argparse.ArgumentParser(description='')
# args.add_argument()
args = args.parse_args()

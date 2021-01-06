import numpy as np
import scipy.spatial
import random
import itertools
import matplotlib.pyplot as plt
import argparse

from graph_algorithms import depth_first_traversal as dft
from neuwon import *
from neuwon.regions import *
from neuwon.mechanisms import HH, Destexhe1994, Mongillo2008

from load_mnist import load_mnist

class GrowSynapses:
    def __init__(self, axons, dendrites, pre_gap_post, diameter, num_synapses):
        pre_len, gap_len, post_len = pre_gap_post
        f_pre = pre_len / sum(pre_gap_post)
        f_post = post_len / sum(pre_gap_post)
        self.presynaptic_segments = []
        self.postsynaptic_segments = []
        # Find all possible synapses.
        pre = scipy.spatial.cKDTree([x.coordinates for x in axons])
        post = scipy.spatial.cKDTree([x.coordinates for x in dendrites])
        results = pre.query_ball_tree(post, sum(pre_gap_post))
        results = list(itertools.chain.from_iterable(
            ((pre, post) for post in inner) for pre, inner in enumerate(results)))
        # Select some synapses and make them.
        for pre, post in random.sample(results, min(num_synapses, len(results))):
            pre = axons[pre]
            post = dendrites[post]
            if pre_len and len(pre.children) > 1: continue
            if post_len and len(post.children) > 1: continue
            if pre_len == 0:
                self.presynaptic_segments.append(pre)
            else:
                x = (1 - f_pre) * np.array(pre.coordinates) + f_pre * np.array(post.coordinates)
                self.presynaptic_segments.append(Segment(x, diameter, pre))
            if post_len == 0:
                self.postsynaptic_segments.append(post)
            else:
                x = (1 - f_post) * np.array(post.coordinates) + f_post * np.array(pre.coordinates)
                self.postsynaptic_segments.append(Segment(x, diameter, post))
        self.presynaptic_segments = list(set(self.presynaptic_segments))


class Experiment:
    def __init__(self):
        self.time_step = .1e-3
        self.make_model()
        # self.train()
        # self.test()

    def make_model(self):
        spacing = 10e-6
        layer_height_min =  50e-6
        layer_height_max = 100e-6
        soma_diameter = 10e-6
        axon_diameter = 2e-6
        rgn = Rectangle(
                [-spacing/2, layer_height_min, -spacing/2],
                [28*spacing + spacing/2, layer_height_max, 28*spacing + spacing/2])
        um3 = 1e6 * 1e6 * 1e6
        # Make the input axons.
        self.input_terminals = np.zeros((28, 28), dtype=object)
        tips = []
        for r in range(28):
            for c in range(28):
                terminal = GrowSomata.single([r*spacing, 0, c*spacing], axon_diameter)
                tip = terminal[-1].add_segment([r*spacing, 0, c*spacing], axon_diameter, 30e-6)
                self.input_terminals[r, c] = terminal[0]
                tips.append(tip[-1])
        self.input_axons = Growth(tips, rgn, 0.002 * um3,
                balancing_factor = 0,
                extension_angle = np.pi / 4,
                extension_distance = 60e-6,
                bifurcation_angle = np.pi / 3,
                bifurcation_distance = 20e-6,
                extend_before_bifurcate = True,
                only_bifurcate = True,
                maximum_segment_length = 10e-6,
                diameter = axon_diameter,)
        for inp in self.input_terminals.flat:
            for x in dft(inp, lambda x: x.children):
                x.insert_mechanism(HH.Leak)
                x.insert_mechanism(HH.VoltageGatedSodiumChannel)
                x.insert_mechanism(HH.VoltageGatedPotassiumChannel)
        # Make the excitatory cells.
        self.pc_soma = GrowSomata(rgn, 0.0001 * um3, soma_diameter)
        self.pc_dendrites = Growth(self.pc_soma.segments, rgn, 0.001 * um3,
                balancing_factor = .7,
                extension_distance = 50e-6,
                bifurcation_distance = 50e-6,
                extend_before_bifurcate = False,
                only_bifurcate = True,
                maximum_segment_length = 10e-6,
                diameter = None,)
        self.pc_axons = Growth(self.pc_soma.segments, rgn, 0.001 * um3,
                balancing_factor = 0,
                extension_angle = np.pi / 4,
                extension_distance = 60e-6,
                bifurcation_angle = np.pi / 3,
                bifurcation_distance = 20e-6,
                extend_before_bifurcate = True,
                only_bifurcate = True,
                maximum_segment_length = 10e-6,
                diameter = axon_diameter,)
        for x in self.pc_soma.segments + self.pc_axons.segments:
            x.insert_mechanism(HH.Leak)
            x.insert_mechanism(HH.VoltageGatedSodiumChannel)
            x.insert_mechanism(HH.VoltageGatedPotassiumChannel)
        # Make excitatory synapses.
        self.syn_glu = GrowSynapses(
                self.input_axons.segments + self.pc_axons.segments,
                self.pc_soma.segments + self.pc_dendrites.segments,
                (0, .6e-6, 3e-6),
                diameter = 1e-6,
                num_synapses = 100)
        presyn_config = Mongillo2008.PresynapseConfiguration(
                transmitter = "glutamate",
                minimum_utilization = .2,
                utilization_decay = 200e-3,
                resource_recovery =  10e-3)
        for x in self.syn_glu.presynaptic_segments:
            x.insert_mechanism(Mongillo2008.Presynapse, presyn_config,
                    strength=100e-21)
        for x in self.syn_glu.postsynaptic_segments:
            x.insert_mechanism(Destexhe1994.AMPA5)
            x.insert_mechanism(Destexhe1994.NMDA5)
        # Assemble the model.
        self.model = Model(self.time_step,
                list(self.input_terminals.flat) + self.pc_soma.segments,
                reactions=(),
                species=())

    def generate_input(self):
        """ Subject the soma to three pulses of current injection. """
        1/0
        self.time_span = 50e-3
        self.input_current = []
        for step in range(int(self.time_span / self.time_step)):
            t = step * self.time_step
            if 20e-3 <= t < 21e-3:
                self.input_current.append(self.stimulus)
            elif 40e-3 <= t < 41e-3:
                self.input_current.append(self.stimulus)
            else:
                self.input_current.append(None)

    def run_experiment(self):
        1/0
        self.time_stamps = []
        self.v = [[] for _ in self.probes]
        self.glu = [[] for _ in self.probes]
        for t, inp in enumerate(self.input_current):
            if inp is not None:
                self.presyn[0].inject_current(inp)
            self.model.advance()
            self.time_stamps.append((t + 2) * self.time_step * 1e3)

x = Experiment()

x.draw_image()

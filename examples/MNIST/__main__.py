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

from load_mnist import load_mnist
from htm.bindings.algorithms import Classifier
from htm.bindings.sdr import SDR, Metrics

# NEXT TASK: The next thing to work on is getting the MNIST framework setup.
# STart by trying to decode the number straight from the ends of the input
# axons, just to check that the "Encoder -> AP-Propagation -> Classifier"
# sequence is working.

time_step = .1e-3
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
input_terminals = np.zeros((28, 28), dtype=object)
tips = []
for r in range(28):
    for c in range(28):
        terminal = GrowSomata.single([r*spacing, 0, c*spacing], axon_diameter)
        input_terminals[r, c] = terminal[0]
        tip = terminal[-1].add_segment([r*spacing, layer_height_min, c*spacing], axon_diameter, 30e-6)
        tips.append(tip[-1])
# input_axons = Growth(tips, rgn, 0.01 * um3,
#         balancing_factor = 0,
#         extension_angle = np.pi / 4,
#         extension_distance = 60e-6,
#         bifurcation_angle = np.pi / 3,
#         bifurcation_distance = 20e-6,
#         extend_before_bifurcate = True,
#         only_bifurcate = True,
#         maximum_segment_length = 10e-6,
#         diameter = axon_diameter,)
for inp in input_terminals.flat:
    for x in dft(inp, lambda x: x.children):
        x.insert_mechanism(HH.Leak)
        x.insert_mechanism(HH.VoltageGatedSodiumChannel)
        x.insert_mechanism(HH.VoltageGatedPotassiumChannel)
if False:
    # Make the excitatory cells.
    pc_soma = GrowSomata(rgn, 0.0001 * um3, soma_diameter)
    pc_dendrites = Growth(pc_soma.segments, rgn, 0.001 * um3,
            balancing_factor = .7,
            extension_distance = 50e-6,
            bifurcation_distance = 50e-6,
            extend_before_bifurcate = False,
            only_bifurcate = True,
            maximum_segment_length = 10e-6,
            diameter = None,)
    pc_axons = Growth(pc_soma.segments, rgn, 0.001 * um3,
            balancing_factor = 0,
            extension_angle = np.pi / 4,
            extension_distance = 60e-6,
            bifurcation_angle = np.pi / 3,
            bifurcation_distance = 20e-6,
            extend_before_bifurcate = True,
            only_bifurcate = True,
            maximum_segment_length = 10e-6,
            diameter = axon_diameter,)
    for x in pc_soma.segments + pc_axons.segments:
        x.insert_mechanism(HH.Leak)
        x.insert_mechanism(HH.VoltageGatedSodiumChannel)
        x.insert_mechanism(HH.VoltageGatedPotassiumChannel)
    # Make excitatory synapses.
    syn_glu = GrowSynapses(
            input_axons.segments + pc_axons.segments,
            pc_soma.segments + pc_dendrites.segments,
            (0, .6e-6, 3e-6),
            diameter = 1e-6,
            num_synapses = 100)
    presyn_config = Mongillo2008.PresynapseConfiguration(
            transmitter = "glutamate",
            minimum_utilization = .2,
            utilization_decay = 200e-3,
            resource_recovery =  10e-3)
    for x in syn_glu.presynaptic_segments:
        x.insert_mechanism(Mongillo2008.Presynapse, presyn_config,
                strength=100e-21)
    for x in syn_glu.postsynaptic_segments:
        x.insert_mechanism(Destexhe1994.AMPA5)
        x.insert_mechanism(Destexhe1994.NMDA5)
# Assemble the model.
model = Model(time_step,
        # list(input_terminals.flat) + pc_soma.segments,
        list(input_terminals.flat),
        reactions=(),
        species=())
print(len(model), "Segments")
for x in tips:
    model.detect_APs(x)
sdrc = Classifier()

def run(image):
    # Encode the image into binary map.
    image = image >= 100
    for x, y in zip(*np.nonzero(np.squeeze(image))):
        input_terminals[x,y].inject_current()
    for t in range(int(10e-3 / model.time_step)):
        model.advance()
    return model.activity_SDR()

colors = [(0,0,0)] * len(model)
model.draw_image("test.png", (640, 480),
    (0,layer_height_min,-100e-6),
    rgn.sample_point(),
    colors)

train_data, test_data = load_mnist()
# Training Loop
for img, lbl in train_data[:1000]:
    activity = run(img)
    sdrc.learn(activity, lbl)
# Testing Loop
score = 0
for img, lbl in test_data[:100]:
    activity = run(img)
    if lbl == np.argmax(sdrc.infer(activity)):
        score += 1
print('Score: %g %', 100 * score / len(test_data))

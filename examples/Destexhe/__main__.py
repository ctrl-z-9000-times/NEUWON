# Make a synapse using the Destexhe mechanisms. Show what it takes to go from a
# neuroscience specification to runnable experiment. And then analyse how well it
# works.

# TODO: Most of the mod files will need to be modified. Changing them is OK as
# long as it's version controlled.

import numpy as np
import scipy.spatial
import random
import itertools
import matplotlib.pyplot as plt
import argparse

from neuwon.api import *
from neuwon.nmodl import NmodlMechanism
from neuwon.analysis import *

# TODO: Leak mechanism.
reactions = (
    NmodlMechanism("examples/Destexhe/nmodl_files/HH2.mod"),
    NmodlMechanism("examples/Destexhe/nmodl_files/ampa5.mod",
            pointers={"C": Pointer("Glu", extra_concentration=True)}),
    NmodlMechanism("examples/Destexhe/nmodl_files/caL3d.mod",
            pointers={"g": Pointer("ca", conductance=True)}),
)

glu = Pointer("glu", extracellular_concentration=True)

class Experiment:
    def __init__(self, time_step):
        axon_diameter = 0.75e-6
        dendrite_diameter = 2e-6
        length = 100e-6
        z = 2e-6
        self.camera_position = (0, 0, length)
        self.camera_lookat = (length/2, length/2, 0)
        self.axon = [Segment([0,length/2,z], axon_diameter)]
        self.axon.extend(self.axon[0].add_segment([length,length/2,z], axon_diameter, 5e-6))
        self.dendrite = [Segment([length/2,0,0], dendrite_diameter)]
        self.dendrite.extend(self.dendrite[0].add_segment([length/2,length,0], dendrite_diameter, 5e-6))
        # TODO: Insert synapse
        pass
        # TODO: Insert mechanisms everywhere
        for x in self.axon: x.insert_mechanism("hh2")
        for x in self.dendrite: pass
        self.model = Model(time_step, self.axon + self.dendrite,
                species=[Species("L", transmembrane = True, reversal_potential = -54.3e-3,)],
                reactions=reactions)
        self.generate_input()
        self.run()

    def input_function(self):
        # TODO: make a nice sequence of inputs to show the basic operation of synapse.
        #       I only want about 40-50 ms worth of runtime.
        #       ..^.....^.^.^..

        # Give presyn a single and then a train of AP's.
        1/0


    def run(self):
        print("Advancing to steady state...")
        for _ in range(int(30e-3 / self.model.time_step)):
            self.model.advance()
        print("Running ...")
        self.time_stamps = []
        self.presyn_v = []
        self.postsyn_v = []
        self.postsyn_extra_glu = []
        for tick in range(1000):
            self.model.advance()
            self.presyn_v.append(self.presyn.get_voltage())
            self.postsyn_v.append(self.postsyn.get_voltage())
            self.postsyn_extra_glu.append(self.model.read_pointer(glu, self.postsyn.location))

    def plot_data(self):
        plt.subplot(1,3,1)
        plt.title("Presynaptic Voltage")
        plt.plot(1/0)

        plt.subplot(1,3,2)
        plt.title("Extracellular Glutamate Concentration")
        plt.plot(1/0)

        plt.subplot(1,3,3)
        plt.title("Postsynaptic Voltage")
        plt.plot(1/0)

    def draw(self, filename="synapse.png"):
        colors = [(.2, .2, .2) for _ in range(len(self.model))]
        for x in self.dendrite: colors[x.location] = (.7, 0, 0) # red
        for x in self.axon:     colors[x.location] = (0, 0, .7) # blue
        r = 2
        draw_image(self.model, colors, filename, (640*r, 480*r), self.camera_position, self.camera_lookat)


args = argparse.ArgumentParser(description='')
# args.add_argument("input_generator", choices=["single"])
# args.add_argument("--cells", type=int, default=1)
# args.add_argument("--EI", type=float, default=8.0)
# args.add_argument("--syn", "--synapses_per_cell", type=int, default=100.0)
# args.add_argument("--run_time", type=float, default=20.0, help="Milliseconds")
# # Output options.
# args.add_argument("--schematic", action="store_true", help="Draw 2D image.")
# args.add_argument("--voltage", action="store_true", help="Make 3D animation.")
# args.add_argument("--glutamate", action="store_true", help="Make 3D animation.")
# args.add_argument("--gaba", action="store_true", help="Make 3D animation.")

# Allow cmd-line flags to control which receptors are inserted, and allow the user
# to control the scaling factor too.

args = args.parse_args()

x = Experiment(25e-6)

x.draw()

plt.figure()
x.plot_data()
plt.show()

# TODO: Make a 3d voltage animation.
# camera = Animation(x.model, skip=skip, resolution=(640*r, 480*r), camera_coordinates=x.camera_position,)
# camera.add_frame(1/0)

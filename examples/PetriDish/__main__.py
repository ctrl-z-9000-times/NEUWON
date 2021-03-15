import numpy as np
import scipy.spatial
import random
import itertools
import matplotlib.pyplot as plt
import argparse

from neuwon import *
from neuwon.analysis import *
from neuwon.growth import *
from neuwon.regions import *

from neurons import ExcitatoryNeuron, InhibitoryNeuron, excitatory_synapses

class PetriDish:
    def __init__(self, number_of_cells, EI_ratio, synapses_per_cell):
        self.diameter = .2e-3
        self.radius = self.diameter / 2
        self.height = self.diameter / 10
        self.region = Cylinder([0,0,0], [0,0,self.height], self.radius)
        self.camera_position = (0, 0, -self.diameter * 1.1)
        num_excit  = int(round(number_of_cells * (EI_ratio / (EI_ratio + 1))))
        self.excit = [ExcitatoryNeuron(self.region) for _ in range(num_excit)]
        self.inhib = [InhibitoryNeuron(self.region) for _ in range(number_of_cells - num_excit)]
        self.glu = excitatory_synapses(
                itertools.chain.from_iterable(n.axon.segments for n in self.excit),
                itertools.chain.from_iterable(n.dendrite.segments for n in self.excit),
                int(round(number_of_cells * synapses_per_cell)))
        # Assemble the model.
        self.time_step = .1e-3
        self.model = Model(self.time_step, (
                list(n.segments[0] for n in self.excit) +
                list(n.segments[0] for n in self.inhib)),
                species=[neuwon.Species("L", transmembrane = True, reversal_potential = -54.3e-3,)],)

    def draw_schematic(self, filename="schematic.png"):
        colors = [(.2, .2, .2) for _ in range(len(self.model))]
        for x in self.excit:
            for segment in x.axon.segments:
                colors[segment.location] = (0, 0, .7)
            for segment in x.dendrite.segments:
                colors[segment.location] = (0, .7, 0)
        for segment in self.glu.postsynaptic_segments:
            colors[segment.location] = (.7, 0, 0)
        r = 4
        draw_image(self.model, colors, filename, (640*r, 480*r), self.camera_position, (0,0,0))

args = argparse.ArgumentParser(description='')
args.add_argument("input_generator", choices=["single"])
args.add_argument("--cells", type=int, default=1)
args.add_argument("--EI", type=float, default=8.0)
args.add_argument("--syn", "--synapses_per_cell", type=int, default=100.0)
args.add_argument("--run_time", type=float, default=20.0, help="Milliseconds")
# Output options.
args.add_argument("--schematic", action="store_true", help="Draw 2D image.")
args.add_argument("--voltage", action="store_true", help="Make 3D animation.")
args.add_argument("--glutamate", action="store_true", help="Make 3D animation.")
args.add_argument("--gaba", action="store_true", help="Make 3D animation.")
args = args.parse_args()
print("Initializing model...")
x = PetriDish(args.cells, args.EI, args.syn)
if args.schematic: x.draw_schematic()
# Setup data collection.
time_stamps = []
excit_soma = [n.segments[0] for n in x.excit]
excit_v = [[] for _ in excit_soma]
inhib_soma = [n.segments[0] for n in x.excit]
inhib_v = [[] for _ in inhib_soma]
r = 2 # Resolution multipler.
skip = 0
voltage_camera = None
if args.voltage:
    voltage_camera = Animation(x.model, skip=skip, resolution=(640*r, 480*r), camera_coordinates=x.camera_position,)
glutamate_camera = None
if args.glutamate:
    glutamate_camera = Animation(x.model, skip=skip, resolution=(640*r, 480*r), camera_coordinates=x.camera_position,)
# Make input sequence, which is a list of pairs of (time_stamp, segment).
input_sequence = []
if args.input_generator == "single":
    input_sequence.append((1e-3, random.choice(excit_soma)))
# elif args.input_generator == "foobar":
input_sequence.sort(reverse=True)
print("Advancing to steady state...")
for _ in range(int(30e-3 / x.model.time_step)):
    x.model.advance()
print("Running for %g milliseconds with %s input generator"%(args.run_time, args.input_generator))
for tick in range(int(round(args.run_time * 1e-3 / x.model.time_step))):
    t = tick * x.model.time_step
    while input_sequence and input_sequence[-1][0] > t:
        _, segment = input_sequence.pop()
        segment.inject_current(2e-9, 1.4e-3)
    x.model.advance()
    t += x.model.time_step
    time_stamps.append(t * 1e3)
    for soma, v_signal in zip(excit_soma, excit_v):
        v_signal.append(soma.get_voltage())
    for soma, v_signal in zip(inhib_soma, inhib_v):
        v_signal.append(soma.get_voltage())
    if voltage_camera is not None:
        v = ((x.model.electrics.voltages - min_v) / (max_v - min_v)).get()
        voltage_camera.add_frame(
            colors = [(x, 0, 1-x) for x in v],
            text = "{:6.2f} milliseconds".format(t * 1e3))
    if glutamate_camera is not None:
        glu = x.model.electrics.species["glutamate"]
        glu = glu.extra.concentrations / 1000
        glutamate_camera.add_frame(
            colors = [(x, 0, 1-x) for x in glu],
            text = "{:6.2f} milliseconds".format(t * 1e3))
print("Plotting outputs...")
if voltage_camera is not None:
    voltage_camera.save("voltages.gif")
if glutamate_camera is not None:
    glutamate_camera.save("glutamate.gif")
plt.figure()
for v in excit_v:
    plt.plot(time_stamps, v, 'r')
plt.show()

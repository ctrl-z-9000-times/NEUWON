import numpy as np
import scipy.spatial
import random
import itertools
import matplotlib.pyplot as plt
import argparse

from neuwon.model import *
from neuwon.growth import *
from neuwon.regions import *

min_v = -90e-3
max_v = +70e-3

um3 = (1e6) ** 3

class ExcitatoryNeuron:
    def __init__(self, model, region):
        soma_diameter = 5e-6
        self.soma = model.create_segment(None, region.sample_point(), soma_diameter).pop()
        self.axon = Growth(model, self.soma, region, 0.00015*um3,
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
        self.dendrite = Growth(model, self.soma, region, 0.00015*um3,
                balancing_factor = .7,
                extension_distance = 40e-6,
                bifurcation_distance = 40e-6,
                extend_before_bifurcate = False,
                only_bifurcate = True,
                maximum_segment_length = 10e-6,
                diameter = 1.5e-6,
        )
        self.segments = [self.soma] + self.axon.segments + self.dendrite.segments
        model.get_reaction("hh").new_instances(model, self.axon.segments,
                scale = 0.5)
        self.voltage = [] # millivolts.

    def collect_data(self):
        self.voltage.append(self.soma.voltage() * 1000)

class InhibitoryNeuron:
    def __init__(self, region):
        1/0

def excitatory_synapses(axons, dendrites, num_synapses):
    synapses = GrowSynapses(axons, dendrites,
            (0, .6e-6, 3e-6), 1e-6, num_synapses)
    for x in synapses.presynaptic_segments:
        # x.insert_mechanism("rel")
        pass
    for x in synapses.postsynaptic_segments:
        x.insert_mechanism("AMPA5")
    return synapses

class PetriDish:
    def __init__(self, num_excit, num_inhib):
        self.tick = 0
        self.time_step = .1e-3
        self.time_stamps = []
        self.input_queue = [] # List of pairs of (time_stamp, segment)
        self.diameter = .2e-3
        self.radius = self.diameter / 2
        self.height = self.diameter / 10
        self.region = Cylinder([0,0,0], [0,0,self.height], self.radius)
        self.camera_position = (0, 0, -self.diameter * 1.1)
        # 
        self.model = m = Model(self.time_step)
        m.add_species("k")
        m.add_species("na")
        m.add_species(Species("L", transmembrane = True, reversal_potential = -54.3e-3,))
        hh = m.add_reaction("hh")
        self.excit = [ExcitatoryNeuron(m, self.region) for _ in range(num_excit)]
        self.inhib = [InhibitoryNeuron(m, self.region) for _ in range(num_inhib)]
        # self.glu = excitatory_synapses(
        #         itertools.chain.from_iterable(n.axon.segments for n in self.excit),
        #         itertools.chain.from_iterable(n.dendrite.segments for n in self.excit),
        #         int(round(number_of_cells * synapses_per_cell)))
        print("Advancing to steady state...")
        for _ in range(int(30e-3 / self.time_step)): m.advance()

    @property
    def t(self):
        """ Milliseconds """
        return self.tick * self.time_step * 1000

    def inject_stimulus(self, num_aps=3, num_active=1, stim_duration=5):
        soma = [x.soma for x in self.excit]
        soma = random.sample(soma, num_active)
        for _ in range(num_aps):
            cell = random.choice(soma)
            t = self.t + random.uniform(0, stim_duration)
            self.input_queue.append((t, cell))
        self.input_queue.sort(reverse=True)

    def all_quiet(self):
        for x in self.excit + self.inhib:
            if x.voltage[-1] > -50:
                return False
        return True

    def advance(self):
        while self.input_queue and self.input_queue[-1][0] > self.t:
            _, segment = self.input_queue.pop()
            segment.inject_current(2e-9, 1.4e-3)
        self.model.advance()
        self.tick += 1
        self.time_stamps.append(self.t)
        for x in self.excit: x.collect_data()
        for x in self.inhib: x.collect_data()

    def draw_schematic(self, filename="schematic.png"):
        colors = np.zeros((len(self.model), 3))
        for x in self.excit:
            colors[x.soma.index] = [0, .7, 0] # Green
            for segment in x.axon.segments:
                colors[segment.index] = [0, 0, .7] # Blue
            for segment in x.dendrite.segments:
                colors[segment.index] = [.7, 0, 0] # Red
        r = 4
        self.model.render_frame(colors, filename, (640*r, 480*r), self.camera_position)

    def animate(self):
        r = 1 # Resolution multiplier.
        skip = 2
        def color_function(db_access):
            v = db_access("membrane/voltages")
            v = ((v - min_v) / (max_v - min_v)).get()
            return [(x, 0, 1-x) for x in v]
        def text_function(db_access):
            text = "{:6.2f} milliseconds".format(x.t)
            return text
        return Animation(x.model, color_function, text_function,
                skip=skip, resolution=(640*r, 480*r), camera_coordinates=x.camera_position,)

    def plot_voltages(self):
        plt.figure()
        for x in self.excit:
            plt.plot(self.time_stamps, x.voltage, 'r')
        plt.show()

args = argparse.ArgumentParser(description='')
args.add_argument("NUM_EXCIT", type=int)
args.add_argument("NUM_INHIB", type=int)
args.add_argument("--run_time", type=float, default=20.0, help="Milliseconds")
# Output options.
args.add_argument("--schematic", action="store_true", help="Draw 2D image.")
args.add_argument("--animation", action="store_true", help="Make 3D animation.")
args = args.parse_args()
print("Initializing model...")
x = PetriDish(args.NUM_EXCIT, args.NUM_INHIB)
if args.schematic: x.draw_schematic()
if args.animation: camera = x.animate()

print("Running for %g milliseconds..."%(args.run_time))
rest_period = 10e-3
rest_remaining = rest_period / 4
while x.t <= args.run_time:
    x.advance()
    if not x.all_quiet():
        rest_remaining = rest_period
    elif rest_remaining > 0:
        rest_remaining -= x.time_step
    else:
        x.inject_stimulus()
        rest_remaining = rest_period

print("Plotting outputs...")
if args.animation: camera.save("voltages.gif")
x.plot_voltages()

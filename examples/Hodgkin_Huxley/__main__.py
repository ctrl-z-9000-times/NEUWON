""" Sanity tests with the Hodgkin-Huxley model

The model is a single long axon with Hodgkin-Huxley channels to experiment with.

Run from the command line as:
$ python ./NEUWON/examples/Hodgkin_Huxley propagation
"""
from neuwon import *
from neuwon.model import *
import numpy as np
import bisect
import matplotlib.pyplot as plt
import argparse

class Experiment:
    def __init__(self,
            axon_length   = 1000e-6,
            axon_diameter = 1e-6,
            soma_diameter = 20e-6,
            time_step     = 1e-6,
            length_step   = 20e-6,
            stagger       = True,
            probes        = None,
            stimulus      = 2e-10,
        ):
        self.time_step = time_step
        self.length_step = length_step
        self.stagger = stagger
        self.length = axon_length
        self.axon_diameter = axon_diameter
        self.soma_diameter = soma_diameter
        self.stimulus = stimulus
        self.probe_locations = list(probes) if probes is not None else [1.0]
        self.make_model()
        self.generate_input()
        self.run_experiment()

    def make_model(self):
        """ Construct a soma with a single long axon. """
        self.model = Model(self.time_step)
        self.model.add_species(Species("L", transmembrane = True, reversal_potential = -54.3e-3,))
        # self.model.add_reaction("hh")
        self.soma = self.model.add_segment()

        self.soma = [Segment([0,0,0], self.soma_diameter)]
        self.soma.extend(self.soma[0].add_segment([0,0,self.soma_diameter], self.axon_diameter))
        if self.length > 0:
            self.axon = self.soma[-1].add_segment(
                    [0,0,self.length + self.soma_diameter],
                    self.axon_diameter, self.length_step)
            self.tip = self.axon[-1]
        else:
            self.axon = []
            self.tip = self.soma[-1]
        self.probes = [self.axon[int(round(p * (len(self.axon)-1)))] for p in self.probe_locations]
        for x in self.soma + self.axon:
            x.insert_mechanism("hh")
            # x.insert_mechanism("na11a", scale=3)
            # x.insert_mechanism("Kv11_13States_temperature2", scale=3)
        print("Number of Locations:", len(self.model))
        # sa  = sum(self.model.geometry.surface_areas[x.index] for x in self.soma)
        # sa += sum(self.model.geometry.surface_areas[x.index] for x in self.axon)
        # print("Surface area:", sa, "m^2")
        # print()

    def generate_input(self):
        """ Subject the soma to three pulses of current injection. """
        self.time_span = 50e-3
        self.input_current = []
        ap_times = [10e-3, 25e-3, 40e-3]
        for step in range(int(self.time_span / self.time_step)):
            t = step * self.time_step
            self.input_current.append(any(abs(x - t) < self.time_step / 2 for x in ap_times))

    def input_pattern(self, time):
        ap_times = [10e-3, 25e-3, 40e-3]
        for ap_start in ap_times:
            if time >= ap_start and time < ap_start + 1e-3:
                return self.stimulus
        return 0

    def run_experiment(self):
        self.time_stamps = []
        self.v = [[] for _ in self.probes]
        self.m = [[] for _ in self.probes]
        m = AccessHandle(reaction_reference=("hh", "m"))
        gna = AccessHandle("na", conductance=True)
        for tick, inp in enumerate(self.input_current):
            if inp:
                self.soma[0].inject_current(self.stimulus, duration=1e-3)
            self.model.advance()
            self.time_stamps.append((tick + 1) * self.time_step * 1e3)
            for idx, p in enumerate(self.probes):
                self.v[idx].append(p.get_voltage() * 1e3)
                self.m[idx].append(self.model.read_pointer(m, p.location))

def analyze_accuracy():
    caption = ""
    # These parameters approximately match Figure 4.9 & 4.10 of the NEURON book.
    args = {
        "axon_length": 4e-6,
        "axon_diameter": 4e-6,
        "soma_diameter": 4e-6,
        "stimulus": 0.025e-9,
        "length_step": .5e-6,
        "probes": [0],
    }
    def make_label(x):
        if x.model.stagger:
            return "staggered, dt = %g ms"%(x.time_step * 1e3)
        else:
            return "unstaggered, dt = %g ms"%(x.time_step * 1e3)

    x_1 = Experiment(time_step = 1e-6, **args)

    def measure_error(experiment):
        error_v = []
        error_m = []
        for idx, t in enumerate(experiment.time_stamps):
            v = experiment.v[0][idx]
            m = experiment.m[0][idx]
            loc = bisect.bisect_left(x_1.time_stamps, t, hi=len(x_1.time_stamps)-1)
            error_v.append(abs(v - x_1.v[0][loc]))
            error_m.append(abs(m - x_1.m[0][loc]))
        return (error_v, error_m)

    def make_figure(stagger):
        x_250 = Experiment(time_step = 300e-6, stagger=stagger, **args)
        x_slow = Experiment(time_step = 150e-6, stagger=stagger, **args)
        x_fast = Experiment(time_step =  75e-6, stagger=stagger, **args)

        x_250_error = measure_error(x_250)
        x_fast_error = measure_error(x_fast)
        x_slow_error = measure_error(x_slow)

        plt.subplot(2,2,1)
        plt.plot(x_250.time_stamps, x_250.v[0], 'r',
                label=make_label(x_250))
        plt.plot(x_slow.time_stamps, x_slow.v[0], 'g',
                label=make_label(x_slow))
        plt.plot(x_fast.time_stamps, x_fast.v[0], 'b',
                label=make_label(x_fast))
        plt.plot(x_1.time_stamps, x_1.v[0], 'k',
                label=make_label(x_1))
        plt.legend()
        plt.xlabel('ms')
        plt.ylabel('mV')

        plt.subplot(2,2,3)
        plt.plot(x_250.time_stamps, x_250.m[0], 'r',
                label=make_label(x_250))
        plt.plot(x_slow.time_stamps, x_slow.m[0], 'g',
                label=make_label(x_slow))
        plt.plot(x_fast.time_stamps, x_fast.m[0], 'b',
                label=make_label(x_fast))
        plt.plot(x_1.time_stamps, x_1.m[0], 'k',
                label=make_label(x_1))
        plt.legend()
        plt.xlabel('ms')
        plt.ylabel('m')

        plt.subplot(2,2,2)
        plt.plot(x_250.time_stamps, x_250_error[0], 'r',
                label=make_label(x_250))
        plt.plot(x_slow.time_stamps, x_slow_error[0], 'g',
                label=make_label(x_slow))
        plt.plot(x_fast.time_stamps, x_fast_error[0], 'b',
                label=make_label(x_fast))
        plt.legend()
        plt.xlabel('ms')
        plt.ylabel('|v error|')

        plt.subplot(2,2,4)
        plt.plot(x_250.time_stamps, x_250_error[1], 'r',
                label=make_label(x_250))
        plt.plot(x_slow.time_stamps, x_slow_error[1], 'g',
                label=make_label(x_slow))
        plt.plot(x_fast.time_stamps, x_fast_error[1], 'b',
                label=make_label(x_fast))
        plt.legend()
        plt.xlabel('ms')
        plt.ylabel('|m error|')

    plt.figure("Unstaggered Time Steps")
    make_figure(False)
    plt.figure("Staggered Time Steps")
    make_figure(True)

def analyze_propagation():
    caption = """
Simulated action potential propagating through a single long axon with Hodgkin-
Huxley type channels. A current injection at the soma of 0.2 nA for 1 ms causes
the action potential. The axon terminates after 1000 μm which slightly alters
the dynamics near that point."""
    x = Experiment(probes=[0, .2, .4, .6, .8, 1.0], time_step=2.5e-6,)
    colors = 'k purple b g y r'.split()
    plt.figure("AP Propagation")
    soma_coords = x.soma[-1].coordinates
    for i, p in enumerate(x.probes):
        dist = np.linalg.norm(np.subtract(soma_coords, p.coordinates))
        plt.plot(x.time_stamps, x.v[i], colors[i],
            label="Distance from soma: %g μm"%(dist*1e6))
    plt.legend()
    plt.title("Action Potential Propagation")
    plt.xlabel('ms')
    plt.figtext(0.5, 0.01, caption, horizontalalignment='center', fontsize=14)
    plt.ylabel('mV')

def analyze_length_step():
    x2 = Experiment(length_step=10e-6)
    x3 = Experiment(length_step=20e-6)
    x4 = Experiment(length_step=100e-6)
    x5 = Experiment(length_step=200e-6)
    plt.figure("Segment Length")
    plt.plot(x2.time_stamps, x2.v[0], 'b', label="Maximum Inter-Nodal Length: %g μm"%(x2.length_step*1e6))
    plt.plot(x3.time_stamps, x3.v[0], 'g', label="Maximum Inter-Nodal Length: %g μm"%(x3.length_step*1e6))
    plt.plot(x4.time_stamps, x4.v[0], 'y', label="Maximum Inter-Nodal Length: %g μm"%(x4.length_step*1e6))
    plt.plot(x5.time_stamps, x5.v[0], 'r', label="Maximum Inter-Nodal Length: %g μm"%(x5.length_step*1e6))
    plt.legend()
    plt.title("Effect of inter-nodal length on simulation accuracy")
    plt.xlabel('ms')
    plt.ylabel('mV')

def analyze_axon_diameter():
    x1 = Experiment(axon_diameter = .5e-6, stimulus=3e-9, time_step=2.5e-6)
    x2 = Experiment(axon_diameter = 1e-6,  stimulus=3e-9, time_step=2.5e-6)
    x3 = Experiment(axon_diameter = 2e-6,  stimulus=3e-9, time_step=2.5e-6)
    x4 = Experiment(axon_diameter = 4e-6,  stimulus=3e-9, time_step=2.5e-6)
    x5 = Experiment(axon_diameter = 8e-6,  stimulus=3e-9, time_step=2.5e-6)
    plt.figure("Axon Diameter")
    plt.plot(x1.time_stamps, x1.v[0], 'purple', label="Axon Diameter: %g μm"%(x1.axon_diameter*1e6))
    plt.plot(x2.time_stamps, x2.v[0], 'b', label="Axon Diameter: %g μm"%(x2.axon_diameter*1e6))
    plt.plot(x3.time_stamps, x3.v[0], 'g', label="Axon Diameter: %g μm"%(x3.axon_diameter*1e6))
    plt.plot(x4.time_stamps, x4.v[0], 'y', label="Axon Diameter: %g μm"%(x4.axon_diameter*1e6))
    plt.plot(x5.time_stamps, x5.v[0], 'r', label="Axon Diameter: %g μm"%(x5.axon_diameter*1e6))
    plt.legend()
    plt.title("Effect of axon diameter on AP propagation speed")
    plt.xlabel('ms')
    plt.ylabel('mV')

def animation():
    import animation

experiments_index = {
    "accuracy":     analyze_accuracy,
    "propagation":  analyze_propagation,
    "length":       analyze_length_step,
    "diameter":     analyze_axon_diameter,
    "animation":    animation,
}

args = argparse.ArgumentParser(description='Sanity tests with the Hodgkin-Huxley model')
args.add_argument('experiment', choices=list(experiments_index.keys()) + ['all'])
args = args.parse_args()
if args.experiment == 'all':
    [x() for name, x in experiments_index.items()]
else:
    experiments_index[args.experiment]()
plt.show()

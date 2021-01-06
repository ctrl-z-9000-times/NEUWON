""" Sanity tests with the Hodgkin-Huxley model

The model is a single long axon with Hodgkin-Huxley channels to experiment with.

Run from the command line as:
$ python ./NEUWON/examples/Hodgkin_Huxley propagation
"""
from neuwon import Model, Segment, Species
import neuwon.mechanisms.HH as HH
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
            species       = (),
        ):
        self.time_step = time_step
        self.length_step = length_step
        self.stagger = stagger
        self.length = axon_length
        self.axon_diameter = axon_diameter
        self.soma_diameter = soma_diameter
        self.stimulus = stimulus
        self.species = list(species)
        self.probe_locations = list(probes) if probes is not None else [1.0]
        self.make_model()
        self.generate_input()
        self.run_experiment()

    def make_model(self):
        """ Construct a soma with a single long axon. """
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
        for x in self.soma:
            x.insert_mechanism(HH.Leak)
            x.insert_mechanism(HH.VoltageGatedSodiumChannel)
            x.insert_mechanism(HH.VoltageGatedPotassiumChannel)
        for x in self.axon:
            x.insert_mechanism(HH.Leak)
            x.insert_mechanism(HH.VoltageGatedSodiumChannel)
            x.insert_mechanism(HH.VoltageGatedPotassiumChannel)
        self.model = Model(self.time_step, [self.soma[0]],
            reactions=(),
            species=self.species,
            stagger=self.stagger)
        print("Number of Locations:", len(self.model))
        # sa  = sum(self.model.geometry.surface_areas[x.index] for x in self.soma)
        # sa += sum(self.model.geometry.surface_areas[x.index] for x in self.axon)
        # print("Surface area:", sa, "m^2")
        # print()

    def generate_input(self):
        """ Subject the soma to three pulses of current injection. """
        self.time_span = 50e-3
        self.input_current = []
        for step in range(int(self.time_span / self.time_step)):
            t = step * self.time_step
            if 10e-3 <= t < 11e-3:
                self.input_current.append(self.stimulus)
            elif 25e-3 <= t < 26e-3:
                self.input_current.append(self.stimulus)
            elif 40e-3 <= t < 41e-3:
                self.input_current.append(self.stimulus)
            else:
                self.input_current.append(None)

    def run_experiment(self):
        self.time_stamps = []
        self.v = [[] for _ in self.probes]
        for t, inp in enumerate(self.input_current):
            if inp is not None:
                self.soma[0].inject_current(inp)
            self.model.advance()
            self.time_stamps.append((t + 2) * self.time_step * 1e3)
            for idx, p in enumerate(self.probes):
                self.v[idx].append(p.get_voltage() * 1e3)

def analyze_time_step():
    caption = """
Reproduction of figures 4.9 and 4.10 from "The NEURON Book, 2003". A patch of membrane with
Hodgkin-Huxley channels is stimulated with 0.025 nA of current for 1 ms. This figure demonstrates
the effects of different integration strategies on simulation accuracy. The standard for accuracy is
staggered integration with ∆t = 0.001 ms.

Top Left: The spike was noticeably delayed compared to the standard for accuracy. The unstaggered
method advances the solution as follows. First the membrane potential is advanced from v(t) to
v(t + ∆t/2) and then to v(t + ∆t) using the membrane conductances g(t). Second the membrane
conductances are advanced from g(t) to g(t + ∆t) using the membrane potential v(t + ∆t/2).

Top Right: Plot of the absolute error of the membrane potential reveal that the error is
proportional to the integration time step ∆t, i.e. the solution has only first order accuracy.

Bottom Left: The spike was almost identical to the standard for accuracy. The staggered method is
the same as unstaggered method, except that it reports the membrane potential at v(t + ∆t/2) in
place of v(t + ∆t). The key insight to the staggered method is that both the membrane potential and
conductances are advanced using input values from halfway through their time step.

Bottom Right: Plot of the absolute error of the membrane potential show that the error is greatly
reduced, and that the remaining error is not proportional to the time step ∆t. """

    # These parameters approximately match Figure 4.10 of the NEURON book.
    args = {
        "axon_length": 4e-6,
        "axon_diameter": 4e-6,
        "soma_diameter": 4e-6,
        "stimulus": 0.025e-9,
        "probes": [0],
    }
    def make_label(x):
        if x.model.stagger:
            return "staggered, dt = %g ms"%(x.time_step * 1e3)
        else:
            return "unstaggered, dt = %g ms"%(x.time_step * 1e3)

    gold = Experiment(time_step = 1e-6, **args)

    def measure_error(experiment):
        error = []
        for t, v in zip(experiment.time_stamps, experiment.v[0]):
            loc = bisect.bisect_left(gold.time_stamps, t, hi=len(gold.time_stamps)-1)
            error.append(abs(v - gold.v[0][loc]))
        return error

    lockstep = Experiment(time_step = 100e-6, stagger=False, **args)
    plt.figure("Staggered Time Steps")
    plt.figtext(0.05, 0.01, caption, horizontalalignment='left', fontsize=14)
    plt.suptitle("Efficient handling of nonlinearity")
    plt.subplot(2,2,1)
    plt.plot(gold.time_stamps, gold.v[0], 'k',
            label=make_label(gold))
    plt.plot(lockstep.time_stamps, lockstep.v[0], 'r',
            label=make_label(lockstep))
    plt.legend()
    plt.title("Membrane Potential")
    plt.xlabel('ms')
    plt.ylabel('mV')

    plt.subplot(2,2,2)
    lockstep_slow = Experiment(time_step = 20e-6, stagger=False, **args)
    lockstep_fast = Experiment(time_step = 10e-6, stagger=False, **args)
    plt.plot(lockstep_slow.time_stamps, measure_error(lockstep_slow), 'r',
            label=make_label(lockstep_slow))
    plt.plot(lockstep_fast.time_stamps, measure_error(lockstep_fast), 'b',
            label=make_label(lockstep_fast))
    plt.legend()
    plt.title("Absolute Error")
    plt.xlabel('ms')
    plt.ylabel('mV')

    plt.subplot(2,2,3)
    staggered = Experiment(time_step = 100e-6, **args)
    plt.plot(gold.time_stamps, gold.v[0], 'k',
            label=make_label(gold))
    plt.plot(staggered.time_stamps, staggered.v[0], 'r',
            label=make_label(staggered))
    plt.legend()
    plt.title("Membrane Potential")
    plt.xlabel('ms')
    plt.ylabel('mV')

    plt.subplot(2,2,4)
    staggered_slow = Experiment(time_step = .1e-3, **args)
    staggered_fast = Experiment(time_step = .05e-3, **args)
    plt.plot(staggered_slow.time_stamps, measure_error(staggered_slow), 'r',
            label=make_label(staggered_slow))
    plt.plot(staggered_fast.time_stamps, measure_error(staggered_fast), 'b',
            label=make_label(staggered_fast))
    plt.legend()
    plt.title("Absolute Error")
    plt.xlabel('ms')
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

experiments_index = {
    "time":         analyze_time_step,
    "length":       analyze_length_step,
    "diameter":     analyze_axon_diameter,
    "propagation":  analyze_propagation,
}

args = argparse.ArgumentParser(description='Sanity tests with the Hodgkin-Huxley model')
args.add_argument('experiment', choices=list(experiments_index.keys()) + ['all'])
args = args.parse_args()
if args.experiment == 'all':
    [x() for name, x in experiments_index.items()]
else:
    experiments_index[args.experiment]()
plt.show()

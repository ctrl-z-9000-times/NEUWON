""" Sanity tests with the Hodgkin-Huxley model

The neuron consists of a soma and a single long axon with Hodgkin Huxley
channels to experiment with.

Run from the command line as:
$ python ./NEUWON/examples/Hodgkin_Huxley propagation
"""
from neuwon.model import *
import numpy as np
import matplotlib.pyplot as plt
import argparse

class Experiment:
    def __init__(self,
            axon_length   = 1000e-6,
            axon_diameter = 1e-6,
            soma_diameter = 11e-6,
            time_step     = 1e-3,
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
        self.model = m = Model(self.time_step)
        m.add_species(Species("l", transmembrane = True, reversal_potential = -54.3,))
        m.add_species("k")
        m.add_species("na")
        m.add_reaction("hh")
        self.soma = m.create_segment(None, [0,0,0], self.soma_diameter)
        if self.length > 0:
            self.axon = m.create_segment(self.soma[-1],
                    [0,0,self.length], self.axon_diameter,
                    maximum_segment_length=self.length_step)
            self.tip = self.axon[-1]
        else:
            self.axon = []
            self.tip = self.soma[-1]
        self.segments = self.soma + self.axon
        self.hh = m.get_reaction("hh").new_instances(m, self.segments, scale=1)
        self.probes = []
        self.v = []
        self.m = []
        for p in self.probe_locations:
            idx = int(round(p * (len(self.segments)-1)))
            segment = self.segments[idx]
            hh_ent  = Entity(self.model.db, "hh", self.hh[idx])
            self.probes.append(segment)
            v_tracker = TimeSeries(segment.entity, "membrane/voltages")
            self.v.append(v_tracker)
            self.model.add_callback(v_tracker)
            m_tracker = TimeSeries(hh_ent, "hh/data/m")
            self.m.append(m_tracker)
            self.model.add_callback(m_tracker)
        if True:
            print("Number of Locations:", len(self.model))
            sa = sum(x.read("membrane/surface_areas") for x in self.soma)
            print("Soma surface area:", sa, "m^2")
            sa += sum(x.read("membrane/surface_areas") for x in self.axon)
            print("Total surface area:", sa, "m^2")
        if False:
            print(str(self.model))
            self.model.db.browse_docs()
            1/0
        if False: print(repr(self.model))
        if False:
            import tkinter
            from neuwon.gui import PlotData
            from threading import Thread
            # Thread.run
            m.data_plots.add(PlotData(self.tip.entity, 'membrane/voltages'))

    def generate_input(self):
        """ Subject the soma to three pulses of current injection. """
        self.time_span = 50
        self.input_current = []
        ap_times = [10, 25, 40]
        for step in range(int(self.time_span / self.time_step)):
            t = step * self.time_step
            self.input_current.append(any(abs(x - t) < self.time_step / 2 for x in ap_times))

    def input_pattern(self, time):
        ap_times = [10, 25, 40]
        for ap_start in ap_times:
            if time >= ap_start and time < ap_start + 1:
                return self.stimulus
        return 0

    def run_experiment(self):
        for inp in self.input_current:
            if inp:
                self.soma[0].inject_current(self.stimulus, duration=1)
            if self.stagger:
                self.model.advance()
            else:
                self.model._advance_lockstep()
            if False: self.model.check()

def analyze_accuracy():
    # These parameters approximately match Figure 4.9 & 4.10 of the NEURON book.
    args = {
        "axon_length": 0,
        "soma_diameter": 5.7e-6,
        "stimulus": 0.025e-9,
        "probes": [0],
    }

    gold = Experiment(time_step = 1e-3, **args)
    gold_timestamps = gold.v[0].timestamps

    def measure_error(experiment):
        v = experiment.v[0].interpolate(gold_timestamps)
        m = experiment.m[0].interpolate(gold_timestamps)
        error_v = np.abs(v - gold.v[0].timeseries)
        error_m = np.abs(m - gold.m[0].timeseries)
        return (error_v, error_m)

    def make_label(x):
        if x.stagger:   return "staggered, dt = %g ms"%(x.time_step)
        else:           return "unstaggered, dt = %g ms"%(x.time_step)

    def make_figure(stagger):
        slow   = Experiment(time_step = 200e-3, stagger=stagger, **args)
        medium = Experiment(time_step = 100e-3, stagger=stagger, **args)
        fast   = Experiment(time_step =  50e-3, stagger=stagger, **args)

        slow_error   = measure_error(slow)
        medium_error = measure_error(medium)
        fast_error   = measure_error(fast)

        plt.subplot(2,2,1)
        plt.plot(slow.v[0].timestamps, slow.v[0].timeseries, 'r',
                label=make_label(slow))
        plt.plot(medium.v[0].timestamps, medium.v[0].timeseries, 'g',
                label=make_label(medium))
        plt.plot(fast.v[0].timestamps, fast.v[0].timeseries, 'b',
                label=make_label(fast))
        plt.plot(gold.v[0].timestamps, gold.v[0].timeseries, 'k',
                label=make_label(gold))
        gold.v[0].label_axes()
        plt.legend()

        plt.subplot(2,2,3)
        plt.plot(slow.m[0].timestamps, slow.m[0].timeseries, 'r',
                label=make_label(slow))
        plt.plot(medium.m[0].timestamps, medium.m[0].timeseries, 'g',
                label=make_label(medium))
        plt.plot(fast.m[0].timestamps, fast.m[0].timeseries, 'b',
                label=make_label(fast))
        plt.plot(gold.m[0].timestamps, gold.m[0].timeseries, 'k',
                label=make_label(gold))
        plt.legend()
        gold.m[0].label_axes()

        plt.subplot(2,2,2)
        plt.plot(gold_timestamps, slow_error[0], 'r',
                label=make_label(slow))
        plt.plot(gold_timestamps, medium_error[0], 'g',
                label=make_label(medium))
        plt.plot(gold_timestamps, fast_error[0], 'b',
                label=make_label(fast))
        plt.legend()
        plt.xlabel('ms')
        plt.ylabel('|mV error|')

        plt.subplot(2,2,4)
        plt.plot(gold_timestamps, slow_error[1], 'r',
                label=make_label(slow))
        plt.plot(gold_timestamps, medium_error[1], 'g',
                label=make_label(medium))
        plt.plot(gold_timestamps, fast_error[1], 'b',
                label=make_label(fast))
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
Huxley channels. Current injections at the soma of 0.2 nA for 1 ms cause the
action potentials. The axon terminates after 1000 μm which slightly alters the
dynamics near that point."""
    x = Experiment(probes=[0, .2, .4, .6, .8, 1.0], time_step=2.5e-3,)
    colors = 'k purple b g y r'.split()
    plt.figure("AP Propagation")
    soma_coords = x.soma[0].coordinates
    for c, p, v_buf in zip(colors, x.probes, x.v):
        dist = np.linalg.norm(np.subtract(soma_coords, p.coordinates))
        plt.plot(v_buf.timestamps, v_buf.timeseries, c,
            label="Distance from soma: %g μm"%(dist*1e6))
    v_buf.label_axes()
    plt.legend()
    plt.title("Action Potential Propagation")
    plt.figtext(0.5, 0.01, caption, horizontalalignment='center', fontsize=14)

def analyze_length_step():
    1/0 # TODO: Rewrite this to use the new TimeSeries class.
    x2 = Experiment(time_step=25e-3, length_step=10e-6)
    x3 = Experiment(time_step=25e-3, length_step=20e-6)
    x4 = Experiment(time_step=25e-3, length_step=100e-6)
    x5 = Experiment(time_step=25e-3, length_step=200e-6)
    plt.figure("Segment Length")
    plt.plot(x2.time_stamps, x2.v[0], 'k', label="Maximum Inter-Nodal Length: %g μm"%(x2.length_step*1e6))
    plt.plot(x3.time_stamps, x3.v[0], 'b', label="Maximum Inter-Nodal Length: %g μm"%(x3.length_step*1e6))
    plt.plot(x4.time_stamps, x4.v[0], 'g', label="Maximum Inter-Nodal Length: %g μm"%(x4.length_step*1e6))
    plt.plot(x5.time_stamps, x5.v[0], 'r', label="Maximum Inter-Nodal Length: %g μm"%(x5.length_step*1e6))
    plt.legend()
    plt.title("Effect of inter-nodal length on simulation accuracy")
    plt.xlabel('ms')
    plt.ylabel('mV at axon tip (1000 μm from soma)')

def animation():
    import animation

experiments_index = {
    "accuracy":     analyze_accuracy,
    "propagation":  analyze_propagation,
    "length":       analyze_length_step,
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

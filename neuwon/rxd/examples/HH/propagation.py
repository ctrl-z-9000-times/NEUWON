""" Plot the propagation of an action potential through an axon.

This module is executable with the command:
    $ python -m neuwon.rxd.examples.HH.propagation
"""
from neuwon.database import TimeSeries
from . import make_model_with_hh
from sys import argv
import matplotlib.pyplot as plt
import numpy as np

class main:
    def __init__(self,
            axon_length   = 2000,
            axon_diameter = .5,
            soma_diameter = 9,
            time_step     = 2.5e-3,
            length_step   = 20,
            num_probes    = 6,
            stimulus      = 2e-9,
        ):
        self.time_step      = time_step
        self.length_step    = length_step
        self.axon_length    = axon_length
        self.axon_diameter  = axon_diameter
        self.soma_diameter  = soma_diameter
        self.stimulus       = stimulus
        self.num_probes     = num_probes
        self.make_model()
        self.init_steady_state()
        self.setup_measurements()
        self.run_experiment()
        if '--noshow' not in argv:
            self.plot()
            plt.show()

    def make_model(self):
        """ Construct a soma with a single long axon. """
        self.model = m = make_model_with_hh(self.time_step)
        hh        = m.mechanisms['hh']
        self.soma = m.Neuron([0,0,0], self.soma_diameter).root
        self.axon = self.soma.add_section(
                [0,0,self.axon_length], self.axon_diameter,
                maximum_segment_length=self.length_step)
        self.segments = [self.soma] + self.axon
        self.hh = [hh(seg, scale=1) for seg in self.segments]
        if True:
            print("Number of Locations:", len(self.model))
            sa_units = self.soma.get_database_class().get("surface_area").get_units()
            sa = self.soma.surface_area
            print("Soma surface area:", sa, sa_units)
            sa += sum(x.surface_area for x in self.axon)
            print("Total surface area:", sa, sa_units)

    def init_steady_state(self):
        while self.model.clock() < 40:
            self.model.advance()
        self.model.clock.reset()

    def setup_measurements(self):
        self.probe_segments = []
        self.voltage_probes = []
        self.m_probes       = []
        for idx in np.linspace(0, len(self.segments)-1, self.num_probes):
            idx     = round(idx)
            segment = self.segments[idx]
            hh      = self.hh[idx]
            self.probe_segments.append(segment)
            self.voltage_probes.append(TimeSeries().record(segment, "voltage"))
            self.m_probes.append(TimeSeries().record(hh, "m"))

    def run_experiment(self):
        ap_times = [10, 25, 40]
        while self.model.clock() < 50:
            if ap_times and self.model.clock() >= ap_times[0]:
                ap_times.pop(0)
                self.soma.inject_current(self.stimulus, duration=1)
            self.model.advance()
        self.model.check()

    def plot(self):
        caption = f"""
    Simulated action potential propagating through a single long axon with Hodgkin-
    Huxley channels. Current injections at the soma of 2 nA for 1 ms cause the
    action potentials. The axon terminates after {self.axon_length} μm."""
        colors = 'k purple b g y r'.split()
        plt.figure("AP Propagation")
        for color, seg, v_data in zip(colors, self.probe_segments, self.voltage_probes):
            dist = np.linalg.norm(np.subtract(self.soma.coordinates, seg.coordinates))
            plt.plot(v_data.get_timestamps(), v_data.get_data(), color,
                label="Distance from soma: %g μm"%(dist))
        v_data.label_axes()
        plt.legend()
        plt.title("Action Potential Propagation")
        plt.figtext(0.5, 0.01, caption, horizontalalignment='center', fontsize=14)

if __name__ == "__main__": main()

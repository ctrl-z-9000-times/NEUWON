from neuwon.database.time import TimeSeries
import matplotlib.pyplot as plt
import neuwon.model
import numpy as np

class Model:
    def __init__(self,
            time_step     = 1e-3,
            stagger       = True,
            # These parameters approximately match Figure 4.9 & 4.10 of the NEURON book.
            soma_diameter = 5.642,
            stimulus      = 0.025e-9,
        ):
        self.time_step      = time_step
        self.stagger        = stagger
        self.soma_diameter  = soma_diameter
        self.stimulus       = stimulus
        self.make_model()
        self.init_steady_state()
        self.run_experiment()

    def make_model(self):
        self.model = m = neuwon.model.Model(self.time_step, celsius=6.3)
        na_cls  = m.add_species("na", reversal_potential = +60)
        k_cls   = m.add_species("k",  reversal_potential = -88)
        l_cls   = m.add_species("l",  reversal_potential = -54.3,)
        hh_cls  = m.add_reaction("./nmodl_library/hh.mod")
        self.soma = m.Segment(None, [0,0,0], self.soma_diameter)
        self.hh = hh_cls(self.soma, scale=1)
        if True:
            sa_units = self.soma.get_database_class().get("surface_area").get_units()
            print("Soma surface area:", self.soma.surface_area, sa_units)

    def init_steady_state(self):
        while self.model.clock() < 40:
            self.model.advance()
        self.model.clock.reset()

    def run_experiment(self):
        self.v_data = TimeSeries().record(self.soma, "voltage")
        self.m_data = TimeSeries().record(self.hh, "m")
        ap_times = [10, 25, 40]
        while self.model.clock() < 50:
            if ap_times and self.model.clock() >= ap_times[0]:
                ap_times.pop(0)
                self.soma.inject_current(self.stimulus, duration=1)
            if self.stagger:
                self.model.advance()
            else:
                self.model._advance_lockstep()
        self.v_data.stop()
        self.m_data.stop()
        self.model.check()

def main():
    gold = Model(time_step = 1e-3)

    def measure_error(model):
        model.v_data.interpolate(gold.v_data)
        model.m_data.interpolate(gold.m_data)
        error_v = np.abs(np.subtract(model.v_data.get_data(), gold.v_data.get_data()))
        error_m = np.abs(np.subtract(model.m_data.get_data(), gold.m_data.get_data()))
        return (model.v_data.get_timestamps(), error_v, error_m)

    def make_label(x):
        if x.stagger:   return "staggered, dt = %g ms"%(x.time_step)
        else:           return "unstaggered, dt = %g ms"%(x.time_step)

    def make_figure(stagger):
        slow   = Model(time_step = 2*200e-3, stagger=stagger)
        medium = Model(time_step = 2*100e-3, stagger=stagger)
        fast   = Model(time_step = 2* 50e-3, stagger=stagger)

        slow_times,   slow_error_v,   slow_error_m   = measure_error(slow)
        medium_times, medium_error_v, medium_error_m = measure_error(medium)
        fast_times,   fast_error_v,   fast_error_m   = measure_error(fast)

        plt.subplot(2,2,1)
        plt.plot(slow.v_data.get_timestamps(), slow.v_data.get_data(), 'r',
                label=make_label(slow))
        plt.plot(medium.v_data.get_timestamps(), medium.v_data.get_data(), 'g',
                label=make_label(medium))
        plt.plot(fast.v_data.get_timestamps(), fast.v_data.get_data(), 'b',
                label=make_label(fast))
        plt.plot(gold.v_data.get_timestamps(), gold.v_data.get_data(), 'k',
                label=make_label(gold))
        gold.v_data.label_axes()
        plt.legend()

        plt.subplot(2,2,3)
        plt.plot(slow.m_data.get_timestamps(), slow.m_data.get_data(), 'r',
                label=make_label(slow))
        plt.plot(medium.m_data.get_timestamps(), medium.m_data.get_data(), 'g',
                label=make_label(medium))
        plt.plot(fast.m_data.get_timestamps(), fast.m_data.get_data(), 'b',
                label=make_label(fast))
        plt.plot(gold.m_data.get_timestamps(), gold.m_data.get_data(), 'k',
                label=make_label(gold))
        plt.legend()
        gold.m_data.label_axes()

        plt.subplot(2,2,2)
        plt.plot(slow_times, slow_error_v, 'r',
                label=make_label(slow))
        plt.plot(medium_times, medium_error_v, 'g',
                label=make_label(medium))
        plt.plot(fast_times, fast_error_v, 'b',
                label=make_label(fast))
        plt.legend()
        plt.xlabel('ms')
        plt.ylabel('|mV error|')

        plt.subplot(2,2,4)
        plt.plot(slow_times, slow_error_m, 'r',
                label=make_label(slow))
        plt.plot(medium_times, medium_error_m, 'g',
                label=make_label(medium))
        plt.plot(fast_times, fast_error_m, 'b',
                label=make_label(fast))
        plt.legend()
        plt.xlabel('ms')
        plt.ylabel('|m error|')

    plt.figure("Unstaggered Time Steps")
    make_figure(False)
    plt.figure("Staggered Time Steps")
    make_figure(True)
    plt.show()

if __name__ == "__main__": main()

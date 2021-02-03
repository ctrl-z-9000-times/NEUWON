import numpy as np
from neuwon import celsius
from neuwon.mechanisms import Mechanism

class Nav1_Model(Mechanism):
    """ A six-state markovian kinetic model of ionic channel.

    Balbi P, Massobrio P, Hellgren Kotaleski J (2017) A single Markov-type
    kinetic model accounting for the macroscopic currents of all human
    voltage-gated sodium channel isoforms. PLoS Comput Biol 13(9): e1005737.
    https://doi. org/10.1371/journal.pcbi.1005737
    """
   def required_species(self):
        return ["Na"]

    def __init__(self,
            C1C2b2, C1C2v2, C1C2k2, C2C1b1, C2C1v1, C2C1k1, C2C1b2, C2C1v2,
            C2C1k2, C2O1b2, C2O1v2, C2O1k2, O1C2b1, O1C2v1, O1C2k1, O1C2b2,
            O1C2v2, O1C2k2, C2O2b2, C2O2v2, C2O2k2, O2C2b1, O2C2v1, O2C2k1,
            O2C2b2, O2C2v2, O2C2k2, O1I1b1, O1I1v1, O1I1k1, O1I1b2, O1I1v2,
            O1I1k2, I1O1b1, I1O1v1, I1O1k1, I1C1b1, I1C1v1, I1C1k1, C1I1b2,
            C1I1v2, C1I1k2, I1I2b2, I1I2v2, I1I2k2, I2I1b1, I2I1v1, I2I1k1,
            gbar = 10):
        self.C1C2b2 = float(C1C2b2)
        self.C1C2v2 = float(C1C2v2)
        self.C1C2k2 = float(C1C2k2)
        self.C2C1b1 = float(C2C1b1)
        self.C2C1v1 = float(C2C1v1)
        self.C2C1k1 = float(C2C1k1)
        self.C2C1b2 = float(C2C1b2)
        self.C2C1v2 = float(C2C1v2)
        self.C2C1k2 = float(C2C1k2)
        self.C2O1b2 = float(C2O1b2)
        self.C2O1v2 = float(C2O1v2)
        self.C2O1k2 = float(C2O1k2)
        self.O1C2b1 = float(O1C2b1)
        self.O1C2v1 = float(O1C2v1)
        self.O1C2k1 = float(O1C2k1)
        self.O1C2b2 = float(O1C2b2)
        self.O1C2v2 = float(O1C2v2)
        self.O1C2k2 = float(O1C2k2)
        self.C2O2b2 = float(C2O2b2)
        self.C2O2v2 = float(C2O2v2)
        self.C2O2k2 = float(C2O2k2)
        self.O2C2b1 = float(O2C2b1)
        self.O2C2v1 = float(O2C2v1)
        self.O2C2k1 = float(O2C2k1)
        self.O2C2b2 = float(O2C2b2)
        self.O2C2v2 = float(O2C2v2)
        self.O2C2k2 = float(O2C2k2)
        self.O1I1b1 = float(O1I1b1)
        self.O1I1v1 = float(O1I1v1)
        self.O1I1k1 = float(O1I1k1)
        self.O1I1b2 = float(O1I1b2)
        self.O1I1v2 = float(O1I1v2)
        self.O1I1k2 = float(O1I1k2)
        self.I1O1b1 = float(I1O1b1)
        self.I1O1v1 = float(I1O1v1)
        self.I1O1k1 = float(I1O1k1)
        self.I1C1b1 = float(I1C1b1)
        self.I1C1v1 = float(I1C1v1)
        self.I1C1k1 = float(I1C1k1)
        self.C1I1b2 = float(C1I1b2)
        self.C1I1v2 = float(C1I1v2)
        self.C1I1k2 = float(C1I1k2)
        self.I1I2b2 = float(I1I2b2)
        self.I1I2v2 = float(I1I2v2)
        self.I1I2k2 = float(I1I2k2)
        self.I2I1b1 = float(I2I1b1)
        self.I2I1v1 = float(I2I1v1)
        self.I2I1k1 = float(I2I1k1)
        self.gbar   = float(gbar)
        assert(self.gbar >= 0)

    def set_time_step(self, time_step):
        Q10 = 3**((celsius - 20) / 10)
        def rates2(v, b, vv, k):
            return Q10 * b / (1 + np.exp((v - vv) / k))
        self.kinetics = neuwon.mechanisms.KineticModel(
                time_step = time_step * 1e3,
                input_ranges = [-100e-3, 100e-3],
                states = "C1 C2 O1 O2 I1 I2".split(),
                initial_state = "C1",
                conserve_sum = 1,
                kinetics = [
                    ("C1", "C2",
                        lambda v:  rates2(v, self.C1C2b2, self.C1C2v2, self.C1C2k2),
                        lambda v: (rates2(v, self.C2C1b1, self.C2C1v1, self.C2C1k1)
                                 + rates2(v, self.C2C1b2, self.C2C1v2, self.C2C1k2))),
                    ("C2", "O1",
                        lambda v:  rates2(v, self.C2O1b2, self.C2O1v2, self.C2O1k2),
                        lambda v: (rates2(v, self.O1C2b1, self.O1C2v1, self.O1C2k1)
                                 + rates2(v, self.O1C2b2, self.O1C2v2, self.O1C2k2))),
                    ("C2", "O2",
                        lambda v:  rates2(v, self.C2O2b2, self.C2O2v2, self.C2O2k2),
                        lambda v: (rates2(v, self.O2C2b1, self.O2C2v1, self.O2C2k1)
                                 + rates2(v, self.O2C2b2, self.O2C2v2, self.O2C2k2))),
                    ("O1", "I1",
                        lambda v: (rates2(v, self.O1I1b1, self.O1I1v1, self.O1I1k1)
                                 + rates2(v, self.O1I1b2, self.O1I1v2, self.O1I1k2)),
                        lambda v:  rates2(v, self.I1O1b1, self.I1O1v1, self.I1O1k1)),
                    ("I1", "C1",
                        lambda v:  rates2(v, self.I1C1b1, self.I1C1v1, self.I1C1k1),
                        lambda v:  rates2(v, self.C1I1b2, self.C1I1v2, self.C1I1k2)),
                    ("I1", "I2",
                        lambda v:  rates2(v, self.I1I2b2, self.I1I2v2, self.I1I2k2),
                        lambda v:  rates2(v, self.I2I1b1, self.I2I1v1, self.I2I1k1)),
                ],)

    def instance_dtype(self):
        return (Real, 7)

    def new_instance(self, time_step, location, geometry, *args):
        gbar = geometry.surface_areas[location] * self.gbar
        return np.append(self.kinetics.initial_state, [gbar])

    def advance(self, locations, instances, time_step, reaction_inputs, reaction_outputs):
        v = reaction_inputs.v[locations] * 1e3
        self.kinetics.advance((v,), instances)
        O1 = self.kinetics.states.index(O1)
        O2 = self.kinetics.states.index(O2)
        gbar = 6
        g = instances[:, gbar] * (instances[:, O1] + instances[:, O2])
        reaction_outputs.conductances.na[locations] += g

Nav11 = Nav1_Model(
        C1C2b2 = 18,
        C1C2v2 = -7,
        C1C2k2 = -10,
        C2C1b1 = 3,
        C2C1v1 = -37,
        C2C1k1 = 10,
        C2C1b2 = 18,
        C2C1v2 = -7,
        C2C1k2 = -10,
        C2O1b2 = 18,
        C2O1v2 = -7,
        C2O1k2 = -10,
        O1C2b1 = 3,
        O1C2v1 = -37,
        O1C2k1 = 10,
        O1C2b2 = 18,
        O1C2v2 = -7,
        O1C2k2 = -10,
        C2O2b2 = 0.08,
        C2O2v2 = -10,
        C2O2k2 = -15,
        O2C2b1 = 2,
        O2C2v1 = -50,
        O2C2k1 = 7,
        O2C2b2 = 0.2,
        O2C2v2 = -20,
        O2C2k2 = -10,
        O1I1b1 = 8,
        O1I1v1 = -37,
        O1I1k1 = 13,
        O1I1b2 = 17,
        O1I1v2 = -7,
        O1I1k2 = -15,
        I1O1b1 = 0.00001,
        I1O1v1 = -37,
        I1O1k1 = 10,
        I1C1b1 = 0.21,
        I1C1v1 = -61,
        I1C1k1 = 7,
        C1I1b2 = 0.3,
        C1I1v2 = -61,
        C1I1k2 = -5.5,
        I1I2b2 = 0.0015,
        I1I2v2 = -90,
        I1I2k2 = -5,
        I2I1b1 = 0.0075,
        I2I1v1 = -90,
        I2I1k1 = 15,)

Nav12 = Nav1_Model(
        C1C2b2 = 16,
        C1C2v2 = -5,
        C1C2k2 = -10,
        C2C1b1 = 3,
        C2C1v1 = -35,
        C2C1k1 = 10,
        C2C1b2 = 16,
        C2C1v2 = -5,
        C2C1k2 = -10,
        C2O1b2 = 16,
        C2O1v2 = -10,
        C2O1k2 = -10,
        O1C2b1 = 3,
        O1C2v1 = -40,
        O1C2k1 = 10,
        O1C2b2 = 16,
        O1C2v2 = -10,
        O1C2k2 = -10,
        C2O2b2 = 0.13,
        C2O2v2 = -20,
        C2O2k2 = -15,
        O2C2b1 = 2,
        O2C2v1 = -60,
        O2C2k1 = 6,
        O2C2b2 = 0.7,
        O2C2v2 = -10,
        O2C2k2 = -15,
        O1I1b1 = 3,
        O1I1v1 = -41,
        O1I1k1 = 12,
        O1I1b2 = 16,
        O1I1v2 = -11,
        O1I1k2 = -12,
        I1O1b1 = 0.00001,
        I1O1v1 = -42,
        I1O1k1 = 10,
        I1C1b1 = 0.55,
        I1C1v1 = -65,
        I1C1k1 = 7,
        C1I1b2 = 0.55,
        C1I1v2 = -65,
        C1I1k2 = -11,
        I1I2b2 = 0.0022,
        I1I2v2 = -90,
        I1I2k2 = -5,
        I2I1b1 = 0.017,
        I2I1v1 = -90,
        I2I1k1 = 15,)

Nav13 = Nav1_Model(
        C1C2b2 = 8,
        C1C2v2 = -7,
        C1C2k2 = -9,
        C2C1b1 = 2,
        C2C1v1 = -37,
        C2C1k1 = 9,
        C2C1b2 = 8,
        C2C1v2 = -7,
        C2C1k2 = -9,
        C2O1b2 = 8,
        C2O1v2 = -17,
        C2O1k2 = -9,
        O1C2b1 = 2,
        O1C2v1 = -47,
        O1C2k1 = 9,
        O1C2b2 = 8,
        O1C2v2 = -17,
        O1C2k2 = -9,
        C2O2b2 = 0.13,
        C2O2v2 = -15,
        C2O2k2 = -5,
        O2C2b1 = 1,
        O2C2v1 = -40,
        O2C2k1 = 3,
        O2C2b2 = 0.2,
        O2C2v2 = -20,
        O2C2k2 = -3,
        O1I1b1 = 2,
        O1I1v1 = -52,
        O1I1k1 = 13,
        O1I1b2 = 8,
        O1I1v2 = -22,
        O1I1k2 = -13,
        I1O1b1 = 0.00001,
        I1O1v1 = -52,
        I1O1k1 = 10,
        I1C1b1 = 0.062,
        I1C1v1 = -70,
        I1C1k1 = 10,
        C1I1b2 = 0.09,
        C1I1v2 = -68,
        C1I1k2 = -8,
        I1I2b2 = 0.0001,
        I1I2v2 = -90,
        I1I2k2 = -5,
        I2I1b1 = 0.0001,
        I2I1v1 = -90,
        I2I1k1 = 15,)

Nav14 = Nav1_Model(
        C1C2b2 = 16,
        C1C2v2 = -3,
        C1C2k2 = -9,
        C2C1b1 = 3,
        C2C1v1 = -33,
        C2C1k1 = 9,
        C2C1b2 = 16,
        C2C1v2 = -3,
        C2C1k2 = -9,
        C2O1b2 = 16,
        C2O1v2 = -8,
        C2O1k2 = -9,
        O1C2b1 = 1,
        O1C2v1 = -38,
        O1C2k1 = 9,
        O1C2b2 = 16,
        O1C2v2 = -8,
        O1C2k2 = -9,
        C2O2b2 = 0.03,
        C2O2v2 = -20,
        C2O2k2 = -8,
        O2C2b1 = 3,
        O2C2v1 = -50,
        O2C2k1 = 8,
        O2C2b2 = 0.1,
        O2C2v2 = -20,
        O2C2k2 = -8,
        O1I1b1 = 0,
        O1I1v1 = -10,
        O1I1k1 = 10,
        O1I1b2 = 16,
        O1I1v2 = -10,
        O1I1k2 = -10,
        I1O1b1 = 0.00001,
        I1O1v1 = -10,
        I1O1k1 = 10,
        I1C1b1 = 0.35,
        I1C1v1 = -70,
        I1C1k1 = 10,
        C1I1b2 = 0.8,
        C1I1v2 = -70,
        C1I1k2 = -7,
        I1I2b2 = 0.0015,
        I1I2v2 = -70,
        I1I2k2 = -12,
        I2I1b1 = 0.007,
        I2I1v1 = -70,
        I2I1k1 = 12,)

Nav15 = Nav1_Model(
        C1C2b2 = 16,
        C1C2v2 = -3,
        C1C2k2 = -9,
        C2C1b1 = 3,
        C2C1v1 = -33,
        C2C1k1 = 9,
        C2C1b2 = 16,
        C2C1v2 = -3,
        C2C1k2 = -9,
        C2O1b2 = 16,
        C2O1v2 = -8,
        C2O1k2 = -9,
        O1C2b1 = 1,
        O1C2v1 = -38,
        O1C2k1 = 9,
        O1C2b2 = 16,
        O1C2v2 = -8,
        O1C2k2 = -9,
        C2O2b2 = 0.03,
        C2O2v2 = -20,
        C2O2k2 = -8,
        O2C2b1 = 3,
        O2C2v1 = -50,
        O2C2k1 = 8,
        O2C2b2 = 0.1,
        O2C2v2 = -20,
        O2C2k2 = -8,
        O1I1b1 = 0,
        O1I1v1 = -10,
        O1I1k1 = 10,
        O1I1b2 = 16,
        O1I1v2 = -10,
        O1I1k2 = -10,
        I1O1b1 = 0.00001,
        I1O1v1 = -10,
        I1O1k1 = 10,
        I1C1b1 = 0.35,
        I1C1v1 = -70,
        I1C1k1 = 10,
        C1I1b2 = 0.8,
        C1I1v2 = -70,
        C1I1k2 = -7,
        I1I2b2 = 0.0015,
        I1I2v2 = -70,
        I1I2k2 = -12,
        I2I1b1 = 0.007,
        I2I1v1 = -70,
        I2I1k1 = 12,)

Nav16 = Nav1_Model(
        C1C2b2 = 14,
        C1C2v2 = -8,
        C1C2k2 = -10,
        C2C1b1 = 2,
        C2C1v1 = -38,
        C2C1k1 = 9,
        C2C1b2 = 14,
        C2C1v2 = -8,
        C2C1k2 = -10,
        C2O1b2 = 14,
        C2O1v2 = -18,
        C2O1k2 = -10,
        O1C2b1 = 4,
        O1C2v1 = -48,
        O1C2k1 = 9,
        O1C2b2 = 14,
        O1C2v2 = -18,
        O1C2k2 = -10,
        C2O2b2 = 0.0001,
        C2O2v2 = -10,
        C2O2k2 = -8,
        O2C2b1 = 0.0001,
        O2C2v1 = -55,
        O2C2k1 = 10,
        O2C2b2 = 0.0001,
        O2C2v2 = -20,
        O2C2k2 = -5,
        O1I1b1 = 6,
        O1I1v1 = -40,
        O1I1k1 = 13,
        O1I1b2 = 10,
        O1I1v2 = 15,
        O1I1k2 = -18,
        I1O1b1 = 0.00001,
        I1O1v1 = -40,
        I1O1k1 = 10,
        I1C1b1 = 0.1,
        I1C1v1 = -86,
        I1C1k1 = 9,
        C1I1b2 = 0.08,
        C1I1v2 = -55,
        C1I1k2 = -12,
        I1I2b2 = 0.00022,
        I1I2v2 = -50,
        I1I2k2 = -5,
        I2I1b1 = 0.0018,
        I2I1v1 = -90,
        I2I1k1 = 30,)

Nav17 = Nav1_Model(
        C1C2b2 = 16,
        C1C2v2 = -18,
        C1C2k2 = -9,
        C2C1b1 = 6,
        C2C1v1 = -48,
        C2C1k1 = 9,
        C2C1b2 = 16,
        C2C1v2 = -18,
        C2C1k2 = -9,
        C2O1b2 = 16,
        C2O1v2 = -23,
        C2O1k2 = -9,
        O1C2b1 = 2,
        O1C2v1 = -53,
        O1C2k1 = 9,
        O1C2b2 = 16,
        O1C2v2 = -23,
        O1C2k2 = -9,
        C2O2b2 = 0.01,
        C2O2v2 = -35,
        C2O2k2 = -5,
        O2C2b1 = 3,
        O2C2v1 = -75,
        O2C2k1 = 5,
        O2C2b2 = 0.01,
        O2C2v2 = -35,
        O2C2k2 = -5,
        O1I1b1 = 4,
        O1I1v1 = -52,
        O1I1k1 = 12,
        O1I1b2 = 8,
        O1I1v2 = -27,
        O1I1k2 = -12,
        I1O1b1 = 0.00001,
        I1O1v1 = -52,
        I1O1k1 = 10,
        I1C1b1 = 0.085,
        I1C1v1 = -110,
        I1C1k1 = 5,
        C1I1b2 = 0.025,
        C1I1v2 = -55,
        C1I1k2 = -20,
        I1I2b2 = 0.00001,
        I1I2v2 = -80,
        I1I2k2 = -20,
        I2I1b1 = 0.00001,
        I2I1v1 = -80,
        I2I1k1 = 20,)

Nav18 = Nav1_Model(
        C1C2b2 = 5,
        C1C2v2 = 17,
        C1C2k2 = -8,
        C2C1b1 = 1,
        C2C1v1 = -23,
        C2C1k1 = 8,
        C2C1b2 = 5,
        C2C1v2 = 17,
        C2C1k2 = -8,
        C2O1b2 = 5,
        C2O1v2 = 13,
        C2O1k2 = -8,
        O1C2b1 = 1,
        O1C2v1 = -27,
        O1C2k1 = 8,
        O1C2b2 = 5,
        O1C2v2 = 13,
        O1C2k2 = -8,
        C2O2b2 = 0.02,
        C2O2v2 = 15,
        C2O2k2 = -8,
        O2C2b1 = 0.8,
        O2C2v1 = -60,
        O2C2k1 = 5,
        O2C2b2 = 0.002,
        O2C2v2 = 10,
        O2C2k2 = -6,

        O1I1b1 = 0.8,
        O1I1v1 = -21,
        O1I1k1 = 10,
        O1I1b2 = 1,
        O1I1v2 = -1,
        O1I1k2 = -7,
        I1O1b1 = 0.00001,
        I1O1v1 = -21,
        I1O1k1 = 10,
        I1C1b1 = 0.28,
        I1C1v1 = -61,
        I1C1k1 = 9.5,
        C1I1b2 = 0.02,
        C1I1v2 = -10,
        C1I1k2 = -20,
        I1I2b2 = 0.001,
        I1I2v2 = -50,
        I1I2k2 = -3,
        I2I1b1 = 0.0003,
        I2I1v1 = -50,
        I2I1k1 = 5,)

Nav19 = Nav1_Model(
        C1C2b2 = 0.8,
        C1C2v2 = -21,
        C1C2k2 = -9,
        C2C1b1 = 0.05,
        C2C1v1 = -56,
        C2C1k1 = 10,
        C2C1b2 = 0.8,
        C2C1v2 = -21,
        C2C1k2 = -9,
        C2O1b2 = 0.8,
        C2O1v2 = -61,
        C2O1k2 = -9,
        O1C2b1 = 0.5,
        O1C2v1 = -96,
        O1C2k1 = 10,
        O1C2b2 = 0.8,
        O1C2v2 = -61,
        O1C2k2 = -9,
        C2O2b2 = 0.0001,
        C2O2v2 = -5,
        C2O2k2 = -8,
        O2C2b1 = 0.0001,
        O2C2v1 = -65,
        O2C2k1 = 7,
        O2C2b2 = 0.0001,
        O2C2v2 = -15,
        O2C2k2 = -12,
        O1I1b1 = 0.04,
        O1I1v1 = -59,
        O1I1k1 = 8,
        O1I1b2 = 0.8,
        O1I1v2 = 1,
        O1I1k2 = -10,
        I1O1b1 = 0.0001,
        I1O1v1 = -60,
        I1O1k1 = 8,
        I1C1b1 = 0.06,
        I1C1v1 = -59,
        I1C1k1 = 8,
        C1I1b2 = 0.04,
        C1I1v2 = -59,
        C1I1k2 = -8,
        I1I2b2 = 0.0016,
        I1I2v2 = -60,
        I1I2k2 = -20,
        I2I1b1 = 0.0115,
        I2I1v1 = -100,
        I2I1k1 = 8,)

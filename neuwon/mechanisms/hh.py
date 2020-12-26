""" Hodgkin-Huxley Model """

import math
from neuwon import Mechanism, Conductance

g_leak = Conductance("g_leak", reversal_potential=-49.42e-3)
g_na = Conductance("g_na", reversal_potential=55.17e-3)
g_k = Conductance("g_k", reversal_potential=-72.14e-3)

class Leak(Mechanism):
    species = []
    conductances = [g_leak]
    G_LEAK = .0003 * 10000
    def __init__(self, time_step, location, geometry, *args):
        self.location = location
        self.g = self.G_LEAK * geometry.surface_areas[location]

    def advance(self, reaction_inputs, reaction_outputs):
        reaction_outputs["g_leak"][self.location] += self.g

class VoltageGatedSodiumChannel(Mechanism):
    species = []
    conductances = [g_na]
    G_NA = .12 * 10000
    def __init__(self, time_step, location, geometry, *args):
        self.time_step = time_step * 1000
        self.location = location
        self.g = self.G_NA * geometry.surface_areas[location]
        self.m = 0
        self.h = 0

    def advance(self, reaction_inputs, reaction_outputs):
        v = reaction_inputs["v"][self.location] * 1e3
        m_alpha = .1 * _vtrap(-(v + 35), 10)
        m_beta = 4 * math.exp(-(v + 60)/18)
        m_sum = m_alpha + m_beta
        m_inf = m_alpha / m_sum
        self.m = m_inf + (self.m - m_inf) * math.exp(-self.time_step * m_sum)
        h_alpha = .07 * math.exp(-(v+60)/20)
        h_beta = 1 / (math.exp(-(v+30)/10) + 1)
        h_sum = h_alpha + h_beta
        h_inf = h_alpha / h_sum
        self.h = h_inf + (self.h - h_inf) * math.exp(-self.time_step * h_sum)
        reaction_outputs["g_na"][self.location] += self.g * self.m**3 * self.h

class VoltageGatedPotassiumChannel(Mechanism):
    species = []
    conductances = [g_k]
    G_K = .036 * 10000
    def __init__(self, time_step, location, geometry, *args):
        self.time_step = time_step * 1000
        self.location = location
        self.g = self.G_K * geometry.surface_areas[location]
        self.n = 0

    def advance(self, reaction_inputs, reaction_outputs):
        v = reaction_inputs["v"][self.location] * 1e3
        n_alpha = .01*_vtrap(-(v+50),10)
        n_beta = .125*math.exp(-(v+60)/80)
        n_sum = n_alpha + n_beta
        n_inf = n_alpha / n_sum
        self.n = n_inf + (self.n - n_inf) * math.exp(-self.time_step * n_sum)
        reaction_outputs["g_k"][self.location] += self.g * self.n**4

def _vtrap(x, y):
    """ Traps for 0 in denominator of rate eqns. """
    if abs(x / y) < 1e-6:
        return y * (1 - x / y / 2)
    else:
        return x / (math.exp(x / y) - 1)

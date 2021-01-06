""" Hodgkin-Huxley Model """

import numpy as np
import math
from neuwon import Mechanism, Species, Real
import numba

leak = Species("leak", transmembrane=True, reversal_potential = -49.42e-3)
na = Species("na", transmembrane=True, reversal_potential = 55.17e-3)
k = Species("k", transmembrane=True, reversal_potential = -72.14e-3)

class Leak(Mechanism):
    G_LEAK = .0003 * 10000
    @classmethod
    def required_species(cls):
        return leak
    @classmethod
    def instance_dtype(cls):
        return [("g", Real)]
    @classmethod
    def new_instance(cls, time_step, location, geometry):
        return cls.G_LEAK * geometry.surface_areas[location]
    @classmethod
    def advance_instance(cls, instance, time_step, location, reaction_inputs, reaction_outputs):
        reaction_outputs.conductances.leak[location] += instance['g']

class VoltageGatedSodiumChannel(Mechanism):
    G_NA = .12 * 10000
    @classmethod
    def required_species(cls):
        return na
    @classmethod
    def instance_dtype(cls):
        return [("g", Real), ("m", Real), ("h", Real)]
    @classmethod
    def new_instance(cls, time_step, location, geometry):
        g = cls.G_NA * geometry.surface_areas[location]
        return (g, 0, 0)
@numba.njit()
def _vgsc_advance_instance(instance, time_step, location, reaction_inputs, reaction_outputs):
    time_step = time_step * 1000
    v = reaction_inputs.v[location] * 1e3
    m_alpha = .1 * _vtrap(-(v + 35), 10)
    m_beta = 4 * math.exp(-(v + 60)/18)
    m_sum = m_alpha + m_beta
    m_inf = m_alpha / m_sum
    instance['m'] = m_inf + (instance['m'] - m_inf) * math.exp(-time_step * m_sum)
    h_alpha = .07 * math.exp(-(v+60)/20)
    h_beta = 1 / (math.exp(-(v+30)/10) + 1)
    h_sum = h_alpha + h_beta
    h_inf = h_alpha / h_sum
    instance['h'] = h_inf + (instance['h'] - h_inf) * math.exp(-time_step * h_sum)
    reaction_outputs.conductances.na[location] += instance['g'] * instance['m']**3 * instance['h']
VoltageGatedSodiumChannel.advance_instance = _vgsc_advance_instance

class VoltageGatedPotassiumChannel(Mechanism):
    G_K = .036 * 10000
    @classmethod
    def required_species(cls):
        return k
    @classmethod
    def instance_dtype(cls):
        return [('g', Real), ('n', Real)]
    @classmethod
    def new_instance(cls, time_step, location, geometry):
        g = cls.G_K * geometry.surface_areas[location]
        return (g, 0)
@numba.njit()
def _vgkc_advance_instance(instance, time_step, location, reaction_inputs, reaction_outputs):
    time_step = time_step * 1000
    v = reaction_inputs.v[location] * 1e3
    n_alpha = .01*_vtrap(-(v+50),10)
    n_beta = .125*math.exp(-(v+60)/80)
    n_sum = n_alpha + n_beta
    n_inf = n_alpha / n_sum
    instance['n'] = n_inf + (instance['n'] - n_inf) * math.exp(-time_step * n_sum)
    reaction_outputs.conductances.k[location] += instance['g'] * instance['n']**4
VoltageGatedPotassiumChannel.advance_instance = _vgkc_advance_instance

@numba.njit()
def _vtrap(x, y):
    """ Traps for 0 in denominator of rate eqns. """
    if abs(x / y) < 1e-6:
        return y * (1 - x / y / 2)
    else:
        return x / (math.exp(x / y) - 1)

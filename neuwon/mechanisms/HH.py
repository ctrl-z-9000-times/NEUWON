""" Hodgkin-Huxley Model """

import numpy as np
import cupy as cp
import math
from neuwon import Mechanism, Species, Real
import numba

leak = Species("leak", transmembrane=True, reversal_potential = -49.42e-3)
na = Species("na", transmembrane=True, reversal_potential = 55.17e-3)
k = Species("k", transmembrane=True, reversal_potential = -72.14e-3)

class Leak(Mechanism):
    G_LEAK = .0003 * 10000
    @classmethod
    def name(self):
        return "HH_Leak"
    @classmethod
    def required_species(cls):
        return "L"
    @classmethod
    def instance_dtype(cls):
        return np.dtype([("g", Real)])
    @classmethod
    def new_instance(cls, time_step, location, geometry):
        return cls.G_LEAK * geometry.surface_areas[location]
    @classmethod
    def advance(cls, locations, instances, time_step, reaction_inputs, reaction_outputs):
        threads = 128
        blocks = (instances.size + (threads - 1)) // threads
        _leak_advance[blocks, threads](locations, instances, time_step,
                reaction_outputs.conductances.L)
@numba.cuda.jit()
def _leak_advance(locations, instances, time_step, g_leak):
    index = numba.cuda.grid(1)
    if index >= instances.size:
        return
    location = locations[index]
    instance = instances[index]
    g_leak[location] += instance['g']

class VoltageGatedSodiumChannel(Mechanism):
    G_NA = .12 * 10000
    @classmethod
    def name(self):
        return "HH_VoltageGatedSodiumChannel"
    @classmethod
    def required_species(cls):
        return "Na"
    @classmethod
    def instance_dtype(cls):
        return np.dtype([("g", Real), ("m", Real), ("h", Real)])
    @classmethod
    def new_instance(cls, time_step, location, geometry):
        g = cls.G_NA * geometry.surface_areas[location]
        return (g, 0, 0)
    @classmethod
    def advance(cls, locations, instances, time_step, reaction_inputs, reaction_outputs):
        threads = 128
        blocks = (instances.size + (threads - 1)) // threads
        _vgsc_advance[blocks, threads](locations, instances, time_step,
                reaction_inputs.v,
                reaction_outputs.conductances.Na)
@numba.cuda.jit()
def _vgsc_advance(locations, instances, time_step, v, g_na):
    index = numba.cuda.grid(1)
    if index >= instances.size:
        return
    location = locations[index]
    instance = instances[index]
    time_step = time_step * 1000
    v = v[location] * 1e3
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
    g_na[location] += instance['g'] * instance['m']**3 * instance['h']

class VoltageGatedPotassiumChannel(Mechanism):
    G_K = .036 * 10000
    @classmethod
    def name(self):
        return "HH_VoltageGatedPotassiumChannel"
    @classmethod
    def required_species(cls):
        return "K"
    @classmethod
    def instance_dtype(cls):
        return np.dtype([('g', Real), ('n', Real)])
    @classmethod
    def new_instance(cls, time_step, location, geometry):
        g = cls.G_K * geometry.surface_areas[location]
        return (g, 0)
    @classmethod
    def advance(cls, locations, instances, time_step, reaction_inputs, reaction_outputs):
        threads = 128
        blocks = (instances.size + (threads - 1)) // threads
        _vgkc_advance[blocks, threads](locations, instances, time_step,
                reaction_inputs.v,
                reaction_outputs.conductances.K)
@numba.cuda.jit()
def _vgkc_advance(locations, instances, time_step, v, g_k):
    index = numba.cuda.grid(1)
    if index >= instances.size:
        return
    location = locations[index]
    instance = instances[index]
    time_step = time_step * 1000
    v = v[location] * 1e3
    n_alpha = .01*_vtrap(-(v+50),10)
    n_beta = .125*math.exp(-(v+60)/80)
    n_sum = n_alpha + n_beta
    n_inf = n_alpha / n_sum
    instance['n'] = n_inf + (instance['n'] - n_inf) * math.exp(-time_step * n_sum)
    g_k[location] += instance['g'] * instance['n']**4

@numba.cuda.jit(device=True)
def _vtrap(x, y):
    """ Traps for 0 in denominator of rate eqns. """
    if abs(x / y) < 1e-6:
        return y * (1 - x / y / 2)
    else:
        return x / (math.exp(x / y) - 1)

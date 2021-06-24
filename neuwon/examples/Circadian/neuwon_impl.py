# CIRCADIAN

import matplotlib.pyplot as plt
from neuwon.model import *
from neuwon.nmodl import NmodlMechanism

reaction = NmodlMechanism("neuwon/examples/Circadian/circadian.mod")

def declare_parameters(**kwargs):
    '''enables clean declaration of parameters in top namespace'''
    for key, value in kwargs.items():
        globals()[key] = rxd.Parameter(r, name=key, initial=value)


def declare_species(**kwargs):
    '''enables clean declaration of species in top namespace'''
    for key, value in kwargs.items():
        globals()[key] = rxd.Species(r, name=key, initial=value, atolscale=1e-3 * nM)



cell = h.Section(name='cell')
cell.diam = cell.L = 5
r = rxd.Region([cell], nrn_region='i')


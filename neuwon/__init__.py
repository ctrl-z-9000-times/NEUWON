""" A tool kit for simulating brains.

NEUWON is a simulation framework for neuroscience and artificial intelligence
specializing in conductance based models. This software is a modern remake of
the NEURON simulator. It is accurate, efficient, and easy to use.
"""

from neuwon.database import TimeSeries, Trace
from neuwon.model import Model

__all__ = ('Model', 'TimeSeries', 'Trace',)

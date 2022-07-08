"""
Simulator for Linear Time-Invariant Kinetic Models.

For more information see:
    Exact digital simulation of time-invariant linear systems with applications
    to neuronal modeling. Rotter S, Diesmann M (1999).
    https://doi.org/10.1007/s004220050570
"""

from .inputs import LinearInput, LogarithmicInput
from .lti_model import LTI_Model

__all__ = ('LTI_Model', 'LinearInput', 'LogarithmicInput')

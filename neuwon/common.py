import numpy as np
from collections.abc import Callable, Iterable, Mapping

Real = np.dtype('f4')
epsilon = np.finfo(Real).eps
Location = np.dtype('u4')
ROOT = np.iinfo(Location).max

# TODO: DELETE THIS FILE!


class AccessHandle:
    @property
    def reaction_instance(self):
        """ Allocate an array of the given numpy.dtype for each inserted instance
        of this reaction. It can be one of:
            * numpy.dtype,
            * Pair of (numpy.dtype, shape) to to make an array.
        Examples:
            np.float32
            (np.float32, 7)
            (np.float32, [4, 4])"""
        return self._reaction_instance
    @property
    def reaction_reference(self):
        """ Access a Reactions instance data.
        This is always a pair of: ("reaction-name", "pointer-name")

        Note: Reactions are not run in a deterministic order. Dynamics which
        span between reactions via reaction_references should operate at a
        significantly slower time scale than the time step. """
        return self._reaction_reference
    @property
    def voltage(self):
        """ Units: Volts. """
        return self._voltage
    @property
    def conductance(self):
        """ Units: Siemens """
        return self._conductance
    @property
    def intra_concentration(self):
        """ Units: Molar """
        return self._intra_concentration
    @property
    def extra_concentration(self):
        """ Units: Molar """
        return self._extra_concentration
    @property
    def intra_release_rate(self):
        """ Units: Molar / Second """
        return self._intra_release_rate
    @property
    def extra_release_rate(self):
        """ Units: Molar / Second """
        return self._extra_release_rate

    def NEURON_conversion_factor(self):
        """ """ # TODO!
        if   self.reaction_instance: return 1
        elif self.voltage:           return 1000 # From NEUWONs volts to NEURONs millivolts.
        elif self.conductance:       return 1
        else: raise NotImplementedError(self)

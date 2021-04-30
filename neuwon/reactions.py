from collections.abc import Callable, Iterable, Mapping
from neuwon.common import Location
from neuwon.nmodl import library, NmodlMechanism
import copy
import cupy
import itertools
import numba.cuda
import numpy

class Reaction:
    """ Abstract class for specifying reactions and mechanisms. """
    @classmethod
    def name(self):
        """ A unique name for this reaction and all of its instances. """
        raise TypeError("Abstract method called by %s."%repr(self))

    @classmethod
    def initialize(self, database):
        """ (Optional) This method is called after the Model has been created.
        This method is called on a deep copy of each Reaction object.

        Argument database is a function(name) -> value

        (Optional) Returns a new Reaction object to use in place of this one. """
        pass

    @classmethod
    def new_instances(self, database_access, locations, *args, **kw_args):
        """ """
        pass

    @classmethod
    def advance(self, database_access):
        """ Advance all instances of this reaction.

        Argument database_access is function: f(component_name) -> value
        """
        raise TypeError("Abstract method called by %s."%repr(self))

def reactions_advance(model):
    for s in model._species.values():
        if s.transmembrane: s.conductances.fill(0)
        if s.extra: s.extra.release_rates.fill(0)
        if s.intra: s.intra.release_rates.fill(0)
    for r in model._reactions.values():
        r.reaction.advance(model.db.access)

from collections.abc import Callable, Iterable, Mapping
from neuwon.common import Location
import copy
import cupy
import itertools
import numba.cuda
import numpy

class Reaction:
    """ Abstract class for specifying reactions and mechanisms.

    Note: all Reactions will be deep copied. """
    @classmethod
    def name(self):
        """ A unique name for this reaction and all of its instances. """
        raise TypeError("Abstract method called by %s."%repr(self))

    @classmethod
    def omnipresent(self):
        """ Omnipresent reactions are always happening everywhere.
        Non-omnipresent reactions are only happen at locations where they've
        been inserted into the model.

        The default value is: False """
        return False

    @classmethod
    def pointers(self):
        """ Returns a mapping of string names to AccessHandle objects.
        All external data access is declared by this method. """
        raise TypeError("Abstract method called by %s."%repr(self))
        # TODO: Rename this API method BC pointers got renamed...

    @classmethod
    def bake(self, time_step, initial_values):
        """ Optional, This method is called after the Model is created.
        This method is called on a deep copy of each Reaction object.

        Argument time_step is in units of seconds.
        Argument initial_values is dictionary of pointer names to values.

        Optionally returns a new Reaction object to use in place of this one. """
        pass

    @classmethod
    def new_instance(self, time_step, location, geometry, *args, **kw_args):
        """ Returns a mapping of pointer names to their initial values at this
        location. Must contain entries for all pointers of the reaction_instance
        type. """
        return {}
        # TODO: Consider zero init'ing the instance data and allowing missing
        # initializations here. It would be convenient, for them and me too...

        # TODO: Consider passing the initial_values to new_instance too.

    @classmethod
    def advance(self, time_step, locations, **pointers):
        """ Advance all instances of this reaction.

        Argument locations ...
        Argument **pointers ...
        """
        raise TypeError("Abstract method called by %s."%repr(self))

class _AllReactions(dict):
    def __init__(self, reactions_argument, insertions):
        # The given arguments take priority, add them first.
        for r in reactions_argument: self._add_reaction(r)
        # Add all reactions which are referenced by the insertions.
        self.insertions = list(insertions)
        for insertions_here in self.insertions:
            for name, args, kwargs in insertions_here:
                self._add_reaction(name)

    def _add_reaction(self, new_reaction):
        """ Adds a new type of reaction if its name is new/unique.
        Reaction names are registered in a first come first serve manner.

        Argument new_reaction must be one of:
          * An instance or subclass of the Reaction class, or
          * The name of a reaction from the standard library. """
        if isinstance(new_reaction, Reaction) or (
                isinstance(new_reaction, type) and issubclass(new_reaction, Reaction)):
            name = str(new_reaction.name())
            if name not in self:
                self[name] = _ReactionData(new_reaction)
        else:
            from  neuwon.nmodl import library, NmodlMechanism
            name = str(new_reaction)
            if name not in self:
                if name in library:
                    nmodl_file_path, kw_args = library[name]
                    import neuwon.nmodl
                    new_reaction = NmodlMechanism(nmodl_file_path, **kw_args)
                    self[name] = _ReactionData(new_reaction)
                    assert(name == new_reaction.name())
                else: raise ValueError("Unrecognized Reaction: %s."%name)

    def pointers(self):
        return set(itertools.chain.from_iterable(
                    r.pointers.values() for r in self.values()))

    def bake(self, time_step, geometry, initial_values):
        for r in self.values():
            if hasattr(r.reaction, "bake"):
                r.reaction = copy.deepcopy(r.reaction)
                retval = r.reaction.bake(time_step, {name: initial_values[ptr]
                            for name, ptr in r.pointers.items() if ptr.read})
                if retval is not None: r.reaction = retval
        for location, insertions_here in enumerate(self.insertions):
            for name, args, kwargs in insertions_here:
                self[name].insert(time_step, location, geometry, *args, **kwargs)
        del self.insertions
        for r in self.values(): r.to_cuda_device()

    @staticmethod
    def advance(model):
        for x in model._species.values():
            if x.transmembrane: x.conductances.fill(0)
            if x.extra: x.extra.release_rates.fill(0)
            if x.intra: x.intra.release_rates.fill(0)
        for container in model._reactions.values():
            args = {}
            for name, ptr in container.pointers.items():
                if ptr.species: species = model._species[ptr.species]
                if ptr.reaction_instance: args[name] = container.state[name]
                elif ptr.reaction_reference:
                    reaction_name, pointer_name = ptr.reaction_reference
                    args[name] = model._reactions[reaction_name].state[pointer_name]
                elif ptr.voltage: args[name] = model._electrics.voltages
                elif ptr.conductance: args[name] = species.conductances
                elif ptr.intra_concentration: args[name] = species.intra.concentrations
                elif ptr.extra_concentration: args[name] = species.extra.concentrations
                elif ptr.intra_release_rate: args[name] = species.intra.release_rates
                elif ptr.extra_release_rate: args[name] = species.extra.release_rates
                else: raise NotImplementedError(ptr)
            container.reaction.advance(model.time_step, container.locations, **args)

    def check_data(self):
        for r in self.values():
            for ptr_name, array in r.state.items():
                kind = array.dtype.kind
                if kind == "f" or kind == "c":
                    assert cupy.all(cupy.isfinite(array)), (r.name(), ptr_name)

class _ReactionData:
    """ Container to hold all instances of one type of reaction. """
    def __init__(self, reaction):
        self.reaction = reaction
        self.pointers = dict(self.reaction.pointers())
        self.state = {name: [] for name, ptr in self.pointers.items() if ptr.reaction_instance}
        self.locations = [] if self.state else None

    def insert(self, time_step, location, geometry, *args, **kwargs):
        if not self.state:
            raise TypeError("Reaction \"%s\" is global and so it can not be inserted."%self.reaction.name())
        self.locations.append(location)
        new_instance = self.reaction.new_instance(time_step, location, geometry, *args, **kwargs)
        for name in self.state: self.state[name].append(new_instance[name])
        assert(len(new_instance) == len(self.state))

    def to_cuda_device(self):
        """ Move locations and state data from python lists to cuda device arrays. """
        self.locations = cupy.array(self.locations, dtype=Location)
        for name, data in self.state.items():
            dtype, shape = self.pointers[name].reaction_instance
            data = numpy.array(data, dtype=dtype).reshape([-1] + shape)
            self.state[name] = numba.cuda.to_device(data)

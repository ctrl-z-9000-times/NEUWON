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
        """ Returns a mapping of string names to Pointer objects.
        All external data access is declared by this method. """
        raise TypeError("Abstract method called by %s."%repr(self))

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

def _init_reactions(reactions_argument, insertions, time_step, geometry):
    reactions = {}
    # The given arguments take priority, add them first.
    for r in reactions_argument: _add_reaction(reactions, r, time_step)
    # Add all inserted reactions.
    for location, insertions_here in enumerate(insertions):
        for name, args, kwargs in insertions_here:
            container = _add_reaction(reactions, name, time_step)
            container.append_new_mechanism(time_step, location, geometry, *args, **kwargs)
    # Copy data from python objects to GPU arrays.
    for container in reactions.values(): container._to_cuda_device()
    return reactions

def _add_reaction(reactions_dict, new_reaction, time_step):
    """ Adds a new reaction to the dictionary if its name is new/unique.

    Argument new_reaction must be one of:
      * An instance or subclass of the Reaction class, or
      * The name of a reaction from the standard library.

    Returns the ReactionContainer for the new_reaction.
    """
    if isinstance(new_reaction, Reaction) or (
            isinstance(new_reaction, type) and issubclass(new_reaction, Reaction)):
        name = str(new_reaction.name())
        if name not in reactions_dict:
            reactions_dict[name] = ReactionContainer(new_reaction, time_step)
    else:
        from  neuwon.nmodl import library, NmodlMechanism
        name = str(new_reaction)
        if name not in reactions_dict:
            if name in library:
                nmodl_file_path, kw_args = library[name]
                import neuwon.nmodl
                new_reaction = NmodlMechanism(nmodl_file_path, **kw_args)
                reactions_dict[name] = ReactionContainer(new_reaction, time_step)
                assert(name == new_reaction.name())
            else:
                raise ValueError("Unrecognized Reaction: %s."%name)
    return reactions_dict[name]

class ReactionContainer:
    """ Container to hold all instances of a type of reaction. """
    def __init__(self, reaction, time_step):
        if hasattr(reaction, "set_time_step"):
            reaction = copy.deepcopy(reaction)
            reaction.set_time_step(time_step)
        self.reaction = reaction
        self.pointers = dict(self.reaction.pointers())
        self.state = {name: [] for name, ptr in self.pointers.items() if ptr.dtype}
        self.locations = [] if self.state else None

    def append_new_mechanism(self, time_step, location, geometry, *args, **kwargs):
        if not self.state:
            raise TypeError("Reaction \"%s\" is global and so it can not be inserted."%self.reaction.name())
        self.locations.append(location)
        new_instance = self.reaction.new_instance(time_step, location, geometry, *args, **kwargs)
        for name in self.state: self.state[name].append(new_instance[name])
        assert(len(new_instance) == len(self.state))

    def _to_cuda_device(self):
        """ Move locations and state data from python lists to cuda device arrays. """
        self.locations = cupy.array(self.locations, dtype=Location)
        for name, data in self.state.items():
            dtype, shape = self.pointers[name].dtype
            data = numpy.array(data, dtype=dtype).reshape([-1] + shape)
            self.state[name] = numba.cuda.to_device(data)


class Reaction:
    """ Abstract class for specifying reactions between omnipresent species. """
    @classmethod
    def required_species(self):
        """ Optional, Returns the Species required by this mechanism.
        Allowed return types: Species, names of species, or a list or either. """
        return []
    @classmethod
    def set_time_step(self, time_step):
        """ Optional, This method is called on a deep copy of each reaction instance."""
    def advance(self, time_step, reaction_inputs, reaction_outputs):
        """ Advance all state of the reaction throughout the while model. """
        raise TypeError("Abstract method called!")

"""
TODO: Make some tools for specifying reactions and implementing them.
Something like:
Reaction(reactants, products, forward, reverse, **integration_kw_args)
    Where reactants and products are: species, strings, list or list of either.
    Where forward and reverse are constant rates.
"""


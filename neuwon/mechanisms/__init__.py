
class Mechanism:
    """ Abstract class for specifying mechanisms which are localized and stateful. """
    def required_species(self):
        """ Optional, Returns the Species required by this mechanism.
        Allowed return types: Species, names of species, and lists either. """
        return []
    def instance_dtype(self):
        """ Returns the numpy data type for a structured array. """
        raise TypeError("Abstract method called!")
    def new_instance(self, time_step, location, geometry, *args):
        """ """
        raise TypeError("Abstract method called!")
    def advance_instance(self, instance, time_step, location, reaction_inputs, reaction_outputs):
        """ """
        raise TypeError("Abstract method called!")

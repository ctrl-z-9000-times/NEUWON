class Mechanism:
    """ Abstract class for specifying chemical reactions and mechanisms. """
    __slots__ = ()
    @classmethod
    def initialize(cls, rxd_model, name):
        """
        Optional method to setup this mechanism.

        This is called after the Model has been created.

        Optionally may return a new Mechanism object to use in place of this one.
        """
        pass

    def set_magnitude(self, magnitude):
        """
        Optional method. Sets the strength of this element.
        """
        raise NotImplementedError

    @classmethod
    def add_nonspecific_conductance(cls, conductance, reversal_potential, segment='segment'):
        mechanism_data = cls.get_database_class()
        1/0
        # TODO: This should make hidden attributes for the mechanism factory to pick up.
        #       The model will give the mech-factory the species input clock as another argument.

    @classmethod
    def advance(cls):
        """ Advance all instances of this mechanism. """
        raise NotImplementedError

class MechanismsFactory(dict):
    def __init__(self, rxd_model, parameters:dict):
        super().__init__()
        self.rxd_model  = rxd_model
        self.add_parameters(parameters)

    def add_parameters(self, parameters:dict) -> [Mechanism]:
        return [self.add_mechanism(name, mechanism)
                    for name, mechanism in parameters.items()]

    def add_mechanism(self, name, mechanism) -> Mechanism:
        assert name not in self
        if isinstance(mechanism, str):
            if mechanism.endswith(".mod"):
                mechanism = neuwon.rxd.nmodl.NMODL(mechanism)
            else:
                raise ValueError("File extension not understood")
        if hasattr(mechanism, "initialize"):
            retval = mechanism.initialize(self.rxd_model, name)
            if retval is not None:
                mechanism = retval
        self[name] = mechanism
        # TODO: something like this:
        if getattr(mechanism, '_nonspecific_conductance', False):
            1/0 # Register the computation with the model here.
        return mechanism

import neuwon.rxd.nmodl

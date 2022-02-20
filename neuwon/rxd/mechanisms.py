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
    def advance(cls):
        """ Advance all instances of this mechanism. """
        raise NotImplementedError

class MechanismsFactory(dict):
    def __init__(self, rxd_model, parameters:dict):
        super().__init__()
        self._rxd_model = rxd_model
        self.add_parameters(parameters)

    def add_parameters(self, parameters:dict) -> [Mechanism]:
        return [self.add_mechanism(name, mechanism)
                    for name, mechanism in parameters.items()]

    def add_mechanism(self, name, mechanism) -> Mechanism:
        name = str(name)
        assert name not in self
        if isinstance(mechanism, str):
            if mechanism.endswith(".mod"):
                mechanism = neuwon.rxd.nmodl.NMODL(mechanism)
            else:
                raise ValueError("File extension not understood")
        if hasattr(mechanism, "initialize"):
            retval = mechanism.initialize(self._rxd_model, name)
            if retval is not None:
                mechanism = retval
        self[name] = mechanism
        return mechanism

import neuwon.rxd.nmodl

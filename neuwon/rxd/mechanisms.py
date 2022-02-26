
class OmnipresentMechanism:
    """
    Abstract class for specifying chemical reactions and mechanisms which exist
    everywhere throughout the model. These mechanisms can only interact with
    other omnipresent entities, such as species and segments, and they can not
    be inserted into any specific place in the model.
    """
    def initialize(self, model, mechanism_name:str):
        """ Optional method to setup this mechanism. """
        pass

    def advance(self):
        """ Advance this mechanism. """
        raise NotImplementedError(type(self))

class LocalMechanism:
    """
    Abstract class for specifying chemical reactions and mechanisms which only
    exist at specific locations in the model. These mechanisms are inserted
    onto segments and can interact with other LocalMechanism's which are
    inserted onto the same segment.
    """

    __slots__ = ()

    def initialize(self, model, mechanism_name:str):
        """
        Optional method to setup this mechanism.

        Optionally may return a new LocalMechanism class to use in place of this
        one. This method is only called once at startup and so the replacement
        class does not need to implement this method.
        """
        pass

    @classmethod
    def other_mechanisms(cls) -> [str]:
        """
        Returns a list of LocalMechanism names which will be created and given
        to this mechanism's constructor.
        """
        return []

    def __init__(self, segment, magnitude, *other_mechanisms):
        raise NotImplementedError(type(self))

    def set_magnitude(self, magnitude=1.0):
        """
        Sets the strength of this element.
        A magnitude of one should always be a sensible value.
        """
        raise NotImplementedError(type(self))

    @classmethod
    def advance(cls):
        """ Advance all instances of this mechanism. """
        raise NotImplementedError(cls)

Mechanism = LocalMechanism # TODO: remove this alias.

class MechanismsFactory(dict):
    def __init__(self, model, parameters:dict):
        super().__init__()
        self._model = model
        self.add_parameters(parameters)

    def add_parameters(self, parameters:dict) -> [Mechanism]:
        return [self.add_mechanism(name, mechanism)
                    for name, mechanism in parameters.items()]

    def add_mechanism(self, name, mechanism) -> Mechanism:
        """ """
        # Unpack the specification & parameters.
        name = str(name)
        assert name not in self
        if isinstance(mechanism, str):
            parameters = {}
        elif isinstance(mechanism, LocalMechanism) or isinstance(mechanism, OmnipresentMechanism):
            parameters = {}
        else:
            mechanism, parameters = mechanism
        # Create the Mechanism object.
        if isinstance(mechanism, str):
            if mechanism.endswith(".mod"):
                mechanism = neuwon.rxd.nmodl.NMODL(mechanism, parameters)
            elif mechanism.endswith(".py"):
                1/0 # TODO?
            else:
                raise ValueError("File extension not understood")
        # 
        if hasattr(mechanism, "initialize"):
            retval = mechanism.initialize(self._model, name)
            if retval is not None:
                mechanism = retval
        self[name] = mechanism
        return mechanism

import neuwon.rxd.nmodl


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

class LocalMechanismInstance:
    """
    Base class for instances of LocalMechanisms.

    All subclasses must also inherit from `DB_Object`, meaning they must be
    created by the method `database.add_class()` and the retrieved by the
    method `database.get_instance_type()`.
    """

    __slots__ = ()

    def __init__(self, segment, magnitude, *other_mechanisms):
        raise NotImplementedError(type(self))

    def set_magnitude(self, magnitude):
        """
        Sets the strength of this element.
        A magnitude of one should always be a sensible value.
        """
        raise NotImplementedError(type(self))

    @classmethod
    def advance(cls):
        """ Advance all instances of this mechanism. """
        raise NotImplementedError(cls)

class LocalMechanismSpecification:
    """
    Abstract class for specifying chemical reactions and mechanisms which only
    exist at specific locations in the model. These mechanisms are inserted
    onto segments and can interact with other LocalMechanismInstance's which
    are inserted onto the same segment.
    """
    def initialize(self, model, mechanism_name:str) -> LocalMechanismInstance:
        """
        Setup this mechanism.

        Returns a LocalMechanismInstance which implements this mechanism.
        """
        raise NotImplementedError(type(self))

    def other_mechanisms(self) -> [str]:
        """
        Returns a list of LocalMechanism names which will be created and given
        to this mechanism's constructor.
        """
        return []

class MechanismsFactory(dict):
    def __init__(self, model, parameters:dict):
        super().__init__()
        self._model = model
        self._local_dependencies = {}
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
        elif isinstance(mechanism, LocalMechanismSpecification) or isinstance(mechanism, OmnipresentMechanism):
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
        omnipresent = isinstance(mechanism, OmnipresentMechanism)
        local       = isinstance(mechanism, LocalMechanismSpecification)
        assert (local != omnipresent)
        # 
        if local:
            dependencies = mechanism.other_mechanisms()
            assert not isinstance(dependencies, str)
            self._local_dependencies[name] = tuple(str(x) for x in dependencies)
        # 
        retval = mechanism.initialize(self._model, name)
        if local or (omnipresent and retval is not None):
            mechanism = retval
        if local:
            assert issubclass(mechanism, LocalMechanismInstance) and issubclass(mechanism, DB_Object)
        self[name] = mechanism
        return mechanism

import neuwon.rxd.nmodl

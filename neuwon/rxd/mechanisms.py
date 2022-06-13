from collections.abc import Iterable
from neuwon.database import Real, DB_Object
from neuwon.database.data_components import Attribute

class Mechanism:
    """
    Abstract class for specifying chemical reactions and mechanisms.

    Mechanisms can be inserted onto segments and can interact with other
    Mechanisms which are inserted onto the same segment.

    Use the "omnipresent" class attribute to specify mechanisms which exist
    everywhere throughout the model. These mechanisms can only interact with
    other omnipresent entities, and they will not be inserted into any specific
    place in the model.
    """
    __slots__ = ()

    omnipresent = False

    @classmethod
    def get_parameters(cls) -> {}:
        """ Define the parameters for the graphical user interface. """
        return {}

    @classmethod
    def initialize(cls, model, mechanism_name:str, **parameters) -> 'Mechanism':
        """ Method to setup this mechanism.

        Returns a "Mechanism" subclass which implements this mechanism.
        """
        pass

    @classmethod
    def other_mechanisms(self) -> [str]:
        """
        Returns a list of Mechanism names which will be created and given
        to this mechanism's constructor.

        This is called after initialize().
        """
        return []

    def __init__(self, segment, magnitude, *other_mechanisms, outside):
        """
        Inserts a new instance of this Mechanism.

        Argument magnitude is a positive number which controls the strength of
        this mechanism. The default value is 1.
        """
        raise NotImplementedError(type(self))

    @classmethod
    def advance(cls):
        """ Advance this mechanism. """
        raise NotImplementedError(cls)

class MechanismsFactory(dict):
    def __init__(self, model, parameters:dict):
        super().__init__()
        self._model = model
        self._local_dependencies = {}
        self.add_parameters(parameters)

    def add_parameters(self, parameters:dict) -> '[Mechanism]':
        new_mechanisms = []
        for name, mechanism in parameters.items():
            if isinstance(mechanism, Iterable):
                mechanism, parameters = mechanism
            else:
                parameters = {}
            self.add_mechanism(name, mechanism, parameters)

    def add_mechanism(self, name, mechanism, parameters={}) -> 'Mechanism':
        """ """
        # Unpack the specification & parameters.
        name = str(name)
        assert name not in self
        # Load the Mechanism class.
        if isinstance(mechanism, str):
            if mechanism.endswith(".mod"):
                mechanism = neuwon.rxd.nmodl.NMODL(mechanism)
            elif mechanism.endswith(".py"):
                mechanism = import_python_mechanism(mechanism)
            else:
                raise ValueError("File extension not understood")
        assert (isinstance(mechanism, Mechanism) or
                (isinstance(mechanism, type) and issubclass(mechanism, Mechanism)))
        # Initialize the mechanism.
        mechanism = mechanism.initialize(self._model, name, **parameters)
        assert (isinstance(mechanism, Mechanism) or
                (isinstance(mechanism, type) and issubclass(mechanism, Mechanism)))
        # 
        if not mechanism.omnipresent:
            dependencies = mechanism.other_mechanisms()
            if isinstance(dependencies, str):
                self._local_dependencies[name] = (dependencies,)
            else:
                self._local_dependencies[name] = tuple(str(x) for x in dependencies)
        # 
        self[name] = mechanism
        return mechanism

def import_python_mechanism(filename):
    assert filename.endswith(".py")
    with open(filename, 'rt') as f:
        src = f.read()
    globals_ = {}
    exec(src, globals_)
    for x in globals_.values():
        if isinstance(x, type) and issubclass(x, Mechanism) and (x is not Mechanism):
            return x

import neuwon.rxd.nmodl # Import namespace after defining the Mechanisms API to prevent circular dependency.

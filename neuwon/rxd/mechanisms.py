from neuwon.database import Real, DB_Object
from neuwon.database.data_components import Attribute

class Mechanism:
    """
    Abstract class for specifying chemical reactions and mechanisms.

    Mechanisms can be inserted onto segments and can interact with other
    Mechanisms which are inserted onto the same segment.

    Use the "omnipresent" class attribute to specify that the mechanism exists
    everywhere throughout the model. These mechanisms can only interact with
    other omnipresent entities, and they can not be inserted into any specific
    place in the model.
    """
    __slots__ = ()

    omnipresent = False

    @classmethod
    def initialize(cls, model) -> 'Mechanism':
        """ Method to setup this mechanism.

        Optionally returns a new "Mechanism" subclass to implement this mechanism.
        """
        pass

    @classmethod
    def get_name(cls) -> str:
        raise NotImplementedError(cls)

    @classmethod
    def other_mechanisms(self) -> [str]:
        """
        Returns a list of Mechanism names which will be created and given
        to this mechanism's constructor.

        This is called after initialize().
        """
        return []

    def __init__(self, segment, outside, magnitude, *other_mechanisms):
        """
        Inserts a new instance of this Mechanism.

        Argument magnitude is a positive number which controls the strength of
        this mechanism. The default value is 1.
        """
        raise NotImplementedError(type(self))

    @classmethod
    def advance(cls):
        """ Advance the state of all instances of this mechanism. """
        raise NotImplementedError(cls)

class _MechanismsFactory(dict):
    def __init__(self, model, parameters:list):
        super().__init__()
        self._model = model
        self._local_dependencies = {}
        for mechanism in parameters:
            self._load_mechanism(mechanism)

    def _load_mechanism(self, mechanism_file):
        if isinstance(mechanism_file, str):
            if mechanism_file.endswith(".mod"):
                x = neuwon.rxd.nmodl.NMODL(mechanism_file)
                self._add_mechanism(x)
            elif mechanism_file.endswith(".py"):
                for x in _import_python_mechanisms(mechanism_file):
                    self._add_mechanism(x)
            else:
                raise ValueError("File extension not understood")
        else:
            self._add_mechanism(mechanism_file) # Developers API.

    def _add_mechanism(self, mechanism):
        """ """
        assert (isinstance(mechanism, Mechanism) or
                (isinstance(mechanism, type) and issubclass(mechanism, Mechanism)))
        # Initialize the mechanism.
        retval = mechanism.initialize(self._model)
        if retval is not None:
            mechanism = retval
        # 
        name = str(mechanism.get_name())
        if mechanism.omnipresent:
            assert (isinstance(mechanism, Mechanism) or
                    (isinstance(mechanism, type) and issubclass(mechanism, Mechanism)))
        else:
            assert isinstance(mechanism, type) and issubclass(mechanism, Mechanism)
            dependencies = mechanism.other_mechanisms()
            if isinstance(dependencies, str):
                self._local_dependencies[name] = (dependencies,)
            else:
                self._local_dependencies[name] = tuple(str(x) for x in dependencies)
        # 
        assert name not in self
        self[name] = mechanism

def _import_python_mechanisms(filename):
    assert filename.endswith(".py")
    with open(filename, 'rt') as f:
        src = f.read()
    globals_ = {}
    exec(src, globals_)
    mechanisms = []
    for x in globals_.values():
        if isinstance(x, type) and issubclass(x, Mechanism) and (x is not Mechanism):
            mechanisms.append(x)
    return mechanisms

import neuwon.rxd.nmodl # Import namespace after defining the Mechanisms API to prevent circular dependency.

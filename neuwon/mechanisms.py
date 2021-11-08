import neuwon.nmodl

class Mechanism:
    """ Abstract class for specifying chemical reactions and mechanisms. """
    __slots__ = ()
    @classmethod
    def get_name(self):
        """ A unique name for this mechanism and all of its instances. """
        name = getattr(self, "name", False)
        if name:
            return name
        # Fallback: return class name.
        if isinstance(self, type):
            return self.__name__
        else:
            return type(self).__name__

    @classmethod
    def initialize(self, database, time_step, celsius, input_clock):
        """
        Optional method; this is called after the Model has been created.

        Optionally may return a new Mechanism object to use in place of this one. """
        pass

    @classmethod
    def advance(self):
        """ Advance all instances of this mechanism. """
        raise TypeError(f"Abstract method called by {self.get_name()}.")

class MechanismsFactory(dict):
    def __init__(self, parameters:dict, database, time_step, celsius):
        super().__init__()
        self.database   = database
        self.time_step  = time_step
        self.celsius    = celsius
        self.add_parameters(parameters)

    def add_parameters(self, parameters:dict):
        for name, mechanism in parameters.items():
            self.add_mechanism(name, mechanism)

    def add_mechanism(self, name, mechanism) -> Mechanism:
        if name in self:
            return self[name]
        if isinstance(mechanism, str):
            if mechanism.endswith(".mod"):
                mechanism = neuwon.nmodl.NmodlMechanism(mechanism)
            else:
                raise ValueError("File extension not understood")
        if hasattr(mechanism, "initialize"):
            retval = mechanism.initialize(self.database,
                    time_step=self.time_step,
                    celsius=self.celsius,)
            if retval is not None: mechanism = retval
        mech_name = mechanism.get_name()
        if name != mech_name:
            raise AssertionError(
                f"Mechanism referred to by multiple different names: '{name}' and '{mech_name}'.")
        self[name] = mechanism
        return mechanism

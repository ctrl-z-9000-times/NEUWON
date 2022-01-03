class Mechanism:
    """ Abstract class for specifying chemical reactions and mechanisms. """
    __slots__ = ()
    @classmethod
    def initialize(cls, database, name, time_step, celsius):
        """
        Optional method. This is called after the Model has been created.

        Optionally may return a new Mechanism object to use in place of this one. """
        pass

    @classmethod
    def add_nonspecific_conductance(cls, conductance, reversal_potential, segment='segment'):
        mechanism_data = cls.get_database_class()
        raise NotImplementedError
        # TODO: This should make hidden attributes for the mechanism factory to pick up.
        #       The model will give the mech-factory the species input clock as another argument.

    @classmethod
    def advance(self):
        """ Advance all instances of this mechanism. """
        raise TypeError(f"Abstract method called by {type(self)}.")

class MechanismsFactory(dict):
    def __init__(self, parameters:dict, database, time_step, celsius, accumulate_conductances_hook):
        super().__init__()
        self.database   = database
        self.time_step  = time_step
        self.celsius    = celsius
        self.add_parameters(parameters)

    def add_parameters(self, parameters:dict):
        for name, mechanism in parameters.items():
            self.add_mechanism(name, mechanism)

    def add_mechanism(self, name, mechanism) -> Mechanism:
        assert name not in self
        if isinstance(mechanism, str):
            if mechanism.endswith(".mod"):
                mechanism = neuwon.rxd.nmodl.NMODL(mechanism)
            else:
                raise ValueError("File extension not understood")
        if hasattr(mechanism, "initialize"):
            retval = mechanism.initialize(self.database, name,
                    time_step=self.time_step,
                    celsius=self.celsius,)
            if retval is not None:
                mechanism = retval
        self[name] = mechanism
        # TODO: something like this:
        if getattr(mechanism, '_nonspecific_conductance', False):
            1/0 # Register the computation with the model here.
        return mechanism

import neuwon.rxd.nmodl

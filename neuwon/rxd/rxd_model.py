from collections.abc import Callable, Iterable, Mapping
from neuwon.database import Database, epsilon
from neuwon.database.time import Clock
from neuwon.parameters import Parameters
from neuwon.rxd.neuron.neuron import Neuron
from neuwon.rxd.outside.outside import Outside
from neuwon.rxd.mechanisms import MechanismsFactory
from neuwon.rxd.species import SpeciesFactory

default_parameters = Parameters({
    'time_step': 0.1,
    'celsius': 37,
    'fh_space': 300e-10, # Frankenhaeuser Hodgkin Space, in Angstroms
    'max_outside_radius':20e-6,
    'outside_volume_fraction':.20,
    'outside_tortuosity':1.55,
    'cytoplasmic_resistance': 100,
    'membrane_capacitance': 1, # uf/cm^2
    'initial_voltage': -70,
})

class RxD_Model:
    def __init__(self, parameters={}, species={}, mechanisms={},):
        self.parameters = Parameters(parameters)
        self.parameters.update_with_defaults(default_parameters)
        self.database   = db = Database()
        self.clock      = db.add_clock(self.parameters['time_step'], units='ms')
        self.time_step  = self.clock.get_tick_period()
        self.celsius    = float(self.parameters['celsius'])
        self.Neuron, self.Segment = Neuron._initialize(db,
                initial_voltage         = self.parameters['initial_voltage'],
                cytoplasmic_resistance  = self.parameters['cytoplasmic_resistance'],
                membrane_capacitance    = self.parameters['membrane_capacitance'],)
        self.Segment._model = self # todo: replace with the species input clock.
        # self.Outside = Outside._initialize(db)
        self.species = SpeciesFactory(species, db,
                                        0.5 * self.time_step, self.celsius)
        self.mechanisms = MechanismsFactory(mechanisms, db,
                self.time_step, self.celsius, self.species.input_hook)

    def __len__(self):
        return len(self.Segment.get_database_class())

    def __repr__(self):
        return repr(self.database)

    def get_database(self):
        return self.database

    def get_Neuron(self):
        return self.Neuron

    def get_Segment(self):
        return self.Segment

    def get_Outside(self):
        return self.Outside

    def check(self):
        self.database.check()

    def advance(self):
        """ Advance the state of the model by one time_step. """
        """
        Both systems (mechanisms & electrics) are integrated using input values
        from halfway through their time step. Tracing through the exact
        sequence of operations is difficult because both systems see the other
        system as staggered halfway through their time step.

        For more information see: The NEURON Book, 2003.
        Chapter 4, Section: Efficient handling of nonlinearity.
        """
        self.database.sort()
        with self.database.using_memory_space('host'):
            self._advance_species()
            self._advance_mechanisms()
            self._advance_species()
            self.Neuron._advance_AP_detector()
        self.clock.tick()

    def _advance_lockstep(self):
        """ Naive integration strategy, for reference only. """
        self.database.sort()
        self._advance_species()
        self._advance_species()
        self._advance_mechanisms()
        self.Neuron._advance_AP_detector()
        self.clock.tick()

    def _advance_species(self):
        """ Note: Each call to this method integrates over half a time step. """
        self._accumulate_conductances()
        self.Segment._advance_electric(self.species.time_step)
        self.species._advance()

    def _accumulate_conductances(self):
        sum_conductance = self.database.get_data("Segment.sum_conductance")
        driving_voltage = self.database.get_data("Segment.driving_voltage")
        # Zero the accumulators.
        sum_conductance.fill(0.0)
        driving_voltage.fill(0.0)
        # Sum the species conductances & driving-voltages into the accumulators.
        self.species.input_hook()
        # 
        driving_voltage /= sum_conductance
        # If conductance is zero then the driving_voltage is also zero.
        xp = self.database.get_array_module()
        driving_voltage[:] = xp.nan_to_num(driving_voltage)

    def _advance_mechanisms(self):
        self.species._zero_accumulators()
        for name, m in self.mechanisms.items():
            try: m.advance()
            except Exception: raise RuntimeError("in mechanism " + name)

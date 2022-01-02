from collections.abc import Callable, Iterable, Mapping
from neuwon.database import Database, epsilon
from neuwon.database.time import Clock
from neuwon.rxd.neurons import Neuron
from neuwon.rxd.extracellular import Extracellular
from neuwon.rxd.mechanisms import MechanismsFactory
from neuwon.rxd.species import SpeciesFactory

class RxD_Model:
    def __init__(self, time_step = 0.1, *,
                celsius = 37,
                initial_voltage = -70,
                cytoplasmic_resistance = 100,
                membrane_capacitance = 1, # uf/cm^2
                extracellular_tortuosity = 1.55,
                extracellular_max_distance = 20e-6,
                species={},
                mechanisms={},):
        self.database   = db = Database()
        self.clock      = db.add_clock(time_step, units='ms')
        self.time_step  = self.clock.get_tick_period()
        self.celsius    = float(celsius)
        self.Neuron = Neuron._initialize(db,
                initial_voltage         = initial_voltage,
                cytoplasmic_resistance  = cytoplasmic_resistance,
                membrane_capacitance    = membrane_capacitance,)
        self.Segment = db.get_class('Segment').get_instance_type()
        self.Segment._model = self # todo: replace with the species input clock.
        self.Extracellular = Extracellular._initialize(db,
                tortuosity       = extracellular_tortuosity,
                maximum_distance = extracellular_max_distance,)
        self.species = SpeciesFactory(species, db,
                                        0.5 * self.time_step, self.celsius)
        self.mechanisms = MechanismsFactory(mechanisms, db,
                self.time_step, self.celsius, self.species.input_hook)

    def __len__(self):
        return len(self.Segment.get_database_class())

    def __repr__(self):
        return repr(self.database)

    def get_celsius(self) -> float:     return self.celsius
    def get_clock(self):                return self.clock
    def get_database(self):             return self.database
    def get_Extracellular(self):        return self.Extracellular
    def get_mechanisms(self) -> dict:   return dict(self.mechanisms)
    def get_Neuron(self):               return self.Neuron
    def get_species(self) -> dict:      return dict(self.species)
    def get_time_step(self) -> float:   return self.time_step

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
        self.species.input_hook.tick()
        # 
        driving_voltage /= sum_conductance
        # If conductance is zero then the driving_voltage is also zero.
        xp = self.database.get_array_module()
        driving_voltage[:] = xp.nan_to_num(driving_voltage)

    def _advance_mechanisms(self):
        self.species._zero_input_accumulators()
        for name, m in self.mechanisms.items():
            try: m.advance()
            except Exception: raise RuntimeError("in mechanism " + name)

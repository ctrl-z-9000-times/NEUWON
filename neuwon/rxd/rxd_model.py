from neuwon.database import Database, Clock
from neuwon.rxd.neuron import Neuron
from neuwon.rxd.extracellular import Extracellular
from neuwon.rxd.mechanisms import MechanismsFactory
from neuwon.rxd.species import SpeciesFactory
import numpy as np

class RxD_Model:
    def __init__(self, time_step: 'ms' = 0.1, *,
                celsius = 37,
                initial_voltage: 'mv' = -70,
                cytoplasmic_resistance = 100,
                membrane_capacitance: 'uf/cm^2' = 1,
                extracellular_tortuosity = 1.55,
                extracellular_max_distance: 'um' = 20e-6,
                species={},
                mechanisms={},):
        """ """
        self.time_step      = float(time_step)
        self.celsius        = float(celsius)
        self.database       = db = Database()
        self.input_hook                  = Clock(0.5 * self.time_step, units='ms')
        self.advance_hook   = self.clock = Clock(      self.time_step, units='ms')
        db.add_clock(self.advance_hook)
        self.Neuron = Neuron._initialize(db,
                initial_voltage         = initial_voltage,
                cytoplasmic_resistance  = cytoplasmic_resistance,
                membrane_capacitance    = membrane_capacitance,)
        self.Segment = db.get_class('Segment').get_instance_type()
        self.Segment._model = self
        self.Extracellular = Extracellular._initialize(db,
                tortuosity       = extracellular_tortuosity,
                maximum_distance = extracellular_max_distance,)
        self.species = SpeciesFactory(species, db, self.input_hook, self.celsius)
        self.mechanisms = MechanismsFactory(self, mechanisms)

    def get_celsius(self) -> float:     return self.celsius
    def get_clock(self):                return self.clock
    def get_database(self):             return self.database
    def get_Extracellular(self):        return self.Extracellular
    def get_mechanisms(self) -> dict:   return dict(self.mechanisms)
    def get_Neuron(self):               return self.Neuron
    def get_Segment(self):              return self.Segment
    def get_species(self) -> dict:      return dict(self.species)
    def get_time_step(self) -> float:   return self.time_step

    def register_input_callback(self, function: 'f() -> bool'):
        """ """
        self.input_hook.register_callback(function)
    def register_advance_callback(self, function: 'f() -> bool'):
        """ """
        self.advance_hook.register_callback(function)

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
        self.advance_hook.tick()

    def _advance_lockstep(self):
        """ Naive integration strategy, for reference only. """
        self.database.sort()
        self._advance_species()
        self._advance_species()
        self._advance_mechanisms()
        self.Neuron._advance_AP_detector()
        self.advance_hook.tick()

    def _advance_species(self):
        """ Note: Each call to this method integrates over half a time step. """
        sum_conductance = self.database.get_data("Segment.sum_conductance")
        driving_voltage = self.database.get_data("Segment.driving_voltage")
        # Zero the accumulators.
        sum_conductance.fill(0.0)
        driving_voltage.fill(0.0)
        # Call the input_hook, which does the following:
        #       Accumulate the species conductances & driving-voltages.
        #       Currect injection.
        self.input_hook.tick()
        # 
        driving_voltage /= sum_conductance
        # If conductance is zero then the driving_voltage is also zero.
        xp = self.database.get_array_module()
        driving_voltage[:] = xp.nan_to_num(driving_voltage)
        self.Segment._advance_electric(self.species.time_step)
        self.species._advance()

    def _advance_mechanisms(self):
        self.species._zero_input_accumulators()
        for name, m in self.mechanisms.items():
            try: m.advance()
            except Exception: raise RuntimeError("in mechanism " + name)

    def filter_segments_by_type(self, neuron_types, segment_types):
        assert self.database.is_sorted()
        if not neuron_types and not segment_types:
            segment_db_class = self.Segment.get_database_class()
            return segment_db_class.get_all_instances()
        neuron_types_list = self.Neuron.neuron_types_list
        if neuron_types:
            neuron_types = [neuron_types_list.index(x) for x in neuron_types]
            neuron_mask  = np.zeros(len(neuron_types_list), dtype=bool)
            neuron_mask[neuron_types] = True
        else:
            neuron_mask = np.ones(len(neuron_types_list), dtype=bool)
        segment_types_list = self.Segment.segment_types_list
        if segment_types:
            segment_types = [segment_types_list.index(x) for x in segment_types]
            segment_mask  = np.zeros(len(segment_types_list), dtype=bool)
            segment_mask[segment_types] = True
        else:
            segment_mask = np.ones(len(segment_types_list), dtype=bool)
        filter_values = self.Segment._filter_by_type(None, neuron_mask, segment_mask)
        index = np.nonzero(filter_values)[0]
        if True:
            index_to_object = self.Segment.get_database_class().index_to_object
            return [index_to_object(x) for x in index]
        else:
            return index

    def filter_neurons_by_type(self, neuron_types):
        assert self.database.is_sorted()
        if not neuron_types:
            neuron_db_class = self.Neuron.get_database_class()
            return neuron_db_class.get_all_instances()
        neuron_types_list = self.Neuron.neuron_types_list
        neuron_types = [neuron_types_list.index(x) for x in neuron_types]
        neuron_mask  = np.zeros(len(neuron_types_list), dtype=bool)
        neuron_mask[neuron_types] = True
        filter_values = self.Neuron._filter_by_type(None, neuron_mask)
        index = np.nonzero(filter_values)[0]
        if True:
            index_to_object = self.Neuron.get_database_class().index_to_object
            return [index_to_object(x) for x in index]
        else:
            return index

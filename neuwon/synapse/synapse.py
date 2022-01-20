from neuwon.database import Pointer, NULL, Compute
from neuwon.rxd import Mechanism
from .constraints import Constraints
import math


class Synapse(Mechanism):
    def __init__(self, presynapse: 'Neuron', postsynapse: 'Segment'):
        postsynapse._num_presyn += 1

    @Compute
    def postsynapse_event(self) -> bool:
        over = self.postsynapse.voltage >= self.postsyn_threshold
        event = over and not self.postsynapse_event_state
        self.postsynapse_event_state = event
        return event


class SynapseGrowthProgram:
    def __init__(self, factory, synapse_type, *,
                transmitter,
                constraints,
                number = None,
                number_per_neuron = None,
                stp = None,
                stdp = None,
                ):
        self.name           = self.synapse_type = str(synapse_type)
        self.factory        = factory
        self.transmitter    = self.factory.rxd.species[str(transmitter)]
        self.database       = self.factory.rxd.get_database()
        self.constraints    = Constraints(self.database, **constraints)
        self.Neuron   = self.database.get_class('Neuron')
        self.Segment  = self.database.get_class('Segment')
        self.Segment.get_database_class().add_attribute('_num_presyn', dtype=np.uint8)

        self.Synapse = self._make_synapse_class()

        self._initial_growth(number, number_per_neuron)

    def _make_synapse_class(self):
        bases = [Synapse, STP]
        synapse_data = self.database.add_class(self.name, bases)
        synapse_data.add_attribute('presynapse',  'Segment')
        synapse_data.add_attribute('postsynapse', 'Segment')
        STP.initialize(synapse_data, **self.stp)
        return synapse_data.get_instance_type()

    def _initial_growth(self, number, number_per_neuron):
        # Clean the inputs.
        assert (number is None) or (number_per_neuron is None)
        if number_per_neuron is not None:
            postsyn_segments    = self.constraints.get_postsynapse_candidates()
            segment_dot_neuron  = self.database.get_data('Segment.neuron')
            postsyn_neurons     = segment_dot_neuron[postsyn_segments]
            num_postsyn_neurons = len(set(postsyn_neurons))
            number              = round(number_per_neuron * num_postsyn_neurons)
        # 
        pairs = self.constraints.find_all_candidates()
        # Iterate over and filter out candidates which are already taken.
        _num_presyn = self.database.get_data('Synapse._num_presyn')
        index_to_object = self.Segment.get_database_class().index_to_object
        while len(self.Synapse.database_class) < number and pairs:
            presyn, postsyn = pairs.pop()
            if not self.constraints.share_postsynapses and _num_presyn[postsyn]:
                continue
            # Make the synapse.
            presyn  = index_to_object(presyn)
            postsyn = index_to_object(postsyn)
            self.Synapse(presyn, postsyn)

def SynapsesFactory(rxd_model, parameters: dict):
    self = {}
    for name, synapse_parameters in parameters.items():
        p = SynapseGrowthProgram.initialize(rxd_model, name, **synapse_parameters)
        self[p.name] = p.Synapse
    return self

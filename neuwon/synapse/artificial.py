from neuwon.database import Pointer, NULL, Compute
from neuwon.rxd import Mechanism
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
                presynapse_neuron_types=[],
                presynapse_segment_types=[],
                postsynapse_neuron_types=[],
                postsynapse_segment_types=[],
                share_postsynapses=False,
                maximum_distance=math.inf,
                number = None,
                stp = None,
                stdp = None,
                ):
        self.name     = self.synapse_type = str(synapse_type)
        self.factory  = factory
        self.transmitter = self.factory.rxd.species[str(transmitter)]
        self.database = self.factory.rxd.get_database()
        self.Neuron   = self.database.get_class('Neuron')
        self.Segment  = self.database.get_class('Segment')
        self.Segment.get_database_class().add_attribute('_num_presyn', dtype=np.uint8)

        self.Synapse = self.make_synapse_class()


    def make_synapse_class(self):
        synapse_data = self.database.add_class(self.name)
        synapse_data.add_attribute('presynapse',  'Segment')
        synapse_data.add_attribute('postsynapse', 'Segment')
        STP.initialize(synapse_data, **self.stp)
        return synapse_data.get_database_class()


class SynapsesFactory(dict):
    def __init__(self, rxd, parameters: dict):
        super().__init__()
        self.rxd = rxd
        self.add_parameters(parameters)

    def add_parameters(self, parameters: dict):
        for name, syn in self.parameters.items():
            self.add_synapse_type(name, syn)

    def add_synapse_type(self, name: str, synapse_parameters: dict) -> Synapse:
        assert name not in self
        self[name] = syn_cls = Synapse.initialize(self.rxd, name, **synapse_parameters)
        return syn_cls

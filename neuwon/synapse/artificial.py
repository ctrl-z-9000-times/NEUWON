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
                presynapse_neuron_types=[],
                presynapse_segment_types=[],
                postsynapse_neuron_types=[],
                postsynapse_segment_types=[],
                share_postsynapses=False,
                maximum_distance=math.inf,
                number = None,
                stp = None,
                stdp = None,
                transmitter,
                ):
        self.name     = self.synapse_type = str(synapse_type)
        self.factory  = factory
        self.transmitter = self.factory.rxd.species[str(transmitter)]
        self.database = self.factory.rxd.get_database()
        self.Neuron   = self.database.get_class('Neuron')
        self.Segment  = self.database.get_class('Segment')
        self.Segment.get_database_class().add_attribute('_num_presyn', dtype=np.uint8)
        self.presynapse_neuron_types = [self.Neuron.neuron_types_list.index(neuron_type)
                                    for neuron_type in presynapse_neuron_types]
        self.presynapse_segment_types = [self.Segment.segment_types_list.index(segment_type)
                                    for segment_type in presynapse_segment_types]
        self.postsynapse_neuron_types = [self.Neuron.neuron_types_list.index(neuron_type)
                                    for neuron_type in postsynapse_neuron_types]
        self.postsynapse_segment_types = [self.Segment.segment_types_list.index(segment_type)
                                    for segment_type in postsynapse_segment_types]
        self.maximum_distance = float(maximum_distance)

        self._type_mask = self.Segment.get_database_class().add_method(self._type_mask)

        self.Synapse = self.make_synapse_class()


    def make_synapse_class(self):
        synapse_data = self.database.add_class(self.name)
        synapse_data.add_attribute('presynapse',  'Segment')
        synapse_data.add_attribute('postsynapse', 'Segment')
        STP.initialize(synapse_data, **self.stp)
        return synapse_data.get_database_class()


    def find_all_candidates(self) -> '[(presyn, postsyn), ...]':
        presyn_segs  = self.filter_segments(self.presynapse_neuron_types, self.presynapse_segment_types)
        postsyn_segs = self.filter_segments(self.postsynapse_neuron_types, self.postsynapse_segment_types)
        coordinates  = self.Segment.get_database_class().get_data('coordinates')
        presyn_tree  = scipy.spatial.cKDTree([coordinates[presyn_segs]])
        postsyn_tree = scipy.spatial.cKDTree([coordinates[postsyn_segs]])
        results = presyn_tree.query_ball_tree(postsyn_tree, self.maximum_distance)
        results = list(itertools.chain.from_iterable(
                ((presyn_segs[pre_idx], postsyn_segs[post_idx])
                    for post_idx in inner) for pre_idx, inner in enumerate(results)))
        random.shuffle(results)
        return results
 
    def filter_segments(self, neuron_types, segment_types):
        if neuron_types:
            neuron_mask = np.zeros(len(self.Neuron.neuron_types_list), dtype=bool)
            neuron_mask[neuron_types] = True
        else:
            neuron_mask = np.ones(len(self.Neuron.neuron_types_list), dtype=bool)
        if segment_types:
            segment_mask = np.zeros(len(self.Segment.segment_types_list), dtype=bool)
            segment_mask[segment_types] = True
        else:
            segment_mask = np.ones(len(self.Segment.segment_types_list), dtype=bool)
        return np.nonzero(self._type_mask(None, neuron_mask, segment_mask))[0]

    def make_postsynapse_coinhabits_mask(self):
        mask = np.zeros(len(self.Segment.get_database_class()), dtype=bool)
        for synapse_type, synapse_growth_program in self.factory.items():
            if synapse_type in self.postsynapse_coinhabits:
                continue
            synapse_growth_program.
            mask[] = True


    @Compute
    def _type_mask(seg: 'Segment', neuron_mask, segment_mask) -> bool:
        return segment_mask[seg.segment_type_id] and neuron_mask[seg.neuron.neuron_type_id]


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

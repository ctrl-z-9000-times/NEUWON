from neuwon.database import Pointer, NULL
from neuwon.parameters import Parameters

default_synapse_parameters = Parameters({
    'filters': {},
    'ap_detector': {},
    'presynapse': {},
    'extracellular': {},
    'postsynapse': {},
})




class ____Synapses:
    def __init__(self, model, axons, dendrites, pre_gap_post, diameter, num_synapses):
        self.model = model
        self.axons = list(axons)
        self.dendrites = list(dendrites)
        num_synapses = int(num_synapses)
        pre_len, gap_len, post_len = pre_gap_post
        f_pre = pre_len / sum(pre_gap_post)
        f_post = post_len / sum(pre_gap_post)
        self.presynaptic_segments = []
        self.postsynaptic_segments = []
        # Find all possible synapses.
        pre = scipy.spatial.cKDTree([x.coordinates for x in self.axons])
        post = scipy.spatial.cKDTree([x.coordinates for x in self.dendrites])
        results = pre.query_ball_tree(post, sum(pre_gap_post))
        results = list(itertools.chain.from_iterable(
            ((pre, post) for post in inner) for pre, inner in enumerate(results)))
        # Select some synapses and make them.
        random.shuffle(results)
        for pre, post in results:
            if num_synapses <= 0:
                break
            pre = self.axons[pre]
            post = self.dendrites[post]
            if pre_len and len(pre.children) > 1: continue
            if post_len and len(post.children) > 1: continue
            if pre_len == 0:
                self.presynaptic_segments.append(pre)
            else:
                x = (1 - f_pre) * np.array(pre.coordinates) + f_pre * np.array(post.coordinates)
                self.presynaptic_segments.append(model.create_segment(pre, x, diameter)[0])
            if post_len == 0:
                self.postsynaptic_segments.append(post)
            else:
                x = (1 - f_post) * np.array(post.coordinates) + f_post * np.array(pre.coordinates)
                self.postsynaptic_segments.append(model.create_segment(post, x, diameter)[0])
            num_synapses -= 1
        self.presynaptic_segments = list(set(self.presynaptic_segments))
        self.segments = self.presynaptic_segments + self.postsynaptic_segments




class Synapse(Mechanism):
    def __init__(self, presynapse: 'Neuron', postsynapse: 'Segment'):
        1/0

    @staticmethod
    def initialize(database, name,
                filters = {},
                presynapse = {},
                extracellular = {},
                postsynapse = {},):

        synapse_data = database.add_class(name, (Synapse,))
        synapse_data.add_attribute('presynapse', 'Neuron')
        synapse_data.add_attribute('postsynapse', 'Segment')
        event_queue = AP_Event.initialize()
        return synapse_data.get_instance_type()

    def initialize_presynapse(model, args, kwargs):
        inherit = []
        model = str(model).lower()
        if model == 'mongillo2008':
            from .Mongillo2008 import Mongillo2008
            inherit.append(Mongillo2008)
        else:
            raise NotImplementedError(model)

    def initialize_postsynapse(self, parameters):
        # TODO: Make a big switch case stmt over all of the different things
        # that the user can insert at the postsynapse.

        # THOUGHT: Is everything at the postsynapse a regular Mechanism?
        #          If so then this should be pretty easy to implement.

        1/0

    def filter(self, presynapse, postsynapse) -> bool:
        for filter_type, filter_parameters in self.parameters['filters'].items():
            if filter_type == 'presynaptic_neuron':
                neuron_type = presynapse.neuron.neuron_type
                if filter_parameters != neuron_type:
                    return False
            elif filter_type == 'postsynaptic_neuron':
                neuron_type = postsynapse.neuron.neuron_type
                if filter_parameters != neuron_type:
                    return False
            elif filter_type == 'presynaptic_dendrite':
                segment_type = presynapse.segment_type
                if filter_parameters != segment_type:
                    return False
            elif filter_type == 'postsynaptic_dendrite':
                segment_type = postsynapse.segment_type
                if filter_parameters != segment_type:
                    return False
            else:
                raise ValueError(f'Unrecognized filter type "{filter_type}"')


    def scan_for_APs():
        1/0

    def dispatch_to_synapses():
        1/0


    @Compute
    def postsynapse_event(self) -> bool:
        over = self.postsynapse.voltage >= self.postsyn_threshold
        event = over and not self.postsynapse_event_state
        self.postsynapse_event_state = event
        return event





class SynapsesFactory(dict):
    def __init__(self, database, parameters: dict):
        super().__init__()
        self.add_parameters(database, parameters)

    def add_parameters(self, database, parameters: dict):
        for name, syn in self.parameters.items():
            self.add_synapse(database, name, syn)

    def add_synapse(self, database, name: str, synapse_parameters: dict) -> Synapse:
        assert name not in self
        self[name] = syn_cls = Synapse.initialize(database, name, **synapse_parameters)
        return syn_cls

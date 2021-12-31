from neuwon.parameters import Parameters

default_synapse_parameters = Parameters({
    'filters': {},
    'ap_detector': {},
    'presynapse': {},
    'extracellular': {},
    'postsynapse': {},
})



class Presynapse:
    # TODO: Should the presynapse also receive the timestep & temperature?
    def initialize(database, name, **parameters):
        raise TypeError(f"Abstract method called!")

    def compute(self, timestamp) -> float:
        raise TypeError(f"Abstract method called!")

class Extracellular:
    def __init__(database, name, **parameters):
        raise TypeError(f"Abstract method called!")

    def compute(cls, presynapse_APs, release):
        raise TypeError(f"Abstract method called!")


class ArtificialSynapseType(Mechanism):
    def __init__(self, name, parameters):
        self.name = str(name)
        self.parameters = Parameters(parameters)
        self.parameters.update_with_defaults(default_synapse_parameters)

    def initialize(self, database):
        self.synapse_data = database.add_class(self.name)
        self.synapse_data.add_attribute('presynapse', 'Segment')
        self.synapse_data.add_attribute('postsynapse', 'Segment')

    def initialize_presynapse(self, parameters):
        1/0

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


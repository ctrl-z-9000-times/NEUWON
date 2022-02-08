from .morphology import NeuronTypeFactory
from .regions import RegionFactory
from .rxd import RxD_Model
from .synapse import SynapsesFactory

class Model(RxD_Model):
    def __init__(self, rxd_parameters={},
                species={}, mechanisms={},
                regions={}, neurons={},
                synapses={},):
        self.parameters = {
                'rxd_parameters': rxd_parameters,
                'species': species,
                'mechanisms': mechanisms,
                'regions': regions,
                'neurons': neurons,
                'synapses': synapses,
        }
        super().__init__(species=species, mechanisms=mechanisms, **rxd_parameters)
        self.regions  = RegionFactory(regions)
        self.neurons  = NeuronTypeFactory(self, neurons)
        self.synapses = SynapsesFactory(self, synapses)

    def get_parameters(self):
        return self.parameters

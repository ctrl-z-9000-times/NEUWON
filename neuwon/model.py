from .morphology import NeuronTypeFactory
from .regions import RegionFactory
from .rxd import RxD_Model
from .synapses import SynapsesFactory

class Model(RxD_Model):
    def __init__(self, simulation={},
                species={}, mechanisms={},
                regions={}, neurons={},
                synapses={},):
        self.parameters = {
                'simulation': simulation,
                'species': species,
                'mechanisms': mechanisms,
                'regions': regions,
                'neurons': neurons,
                'synapses': synapses,
        }
        super().__init__(species=species, mechanisms=mechanisms, **simulation)
        self.regions  = RegionFactory(regions)
        self.neurons  = NeuronTypeFactory(self, neurons)
        self.synapses = SynapsesFactory(self, synapses)

    def get_parameters(self):
        return self.parameters

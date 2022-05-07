from .morphology import NeuronFactory
from .regions import RegionFactory
from .rxd import RxD_Model
from .synapses import SynapsesFactory

class Model(RxD_Model):
    def __init__(self, simulation={}, *, species={}, mechanisms={}, regions={},
                segments={}, neurons={}, synapses={},):
        self.parameters = {
                'simulation':   simulation,
                'species':      species,
                'mechanisms':   mechanisms,
                'regions':      regions,
                'segments':     segments,
                'neurons':      neurons,
                'synapses':     synapses,
        }
        super().__init__(species=species, mechanisms=mechanisms, **simulation)
        self.regions  = RegionFactory(regions)
        self.neurons  = NeuronFactory(self, neurons, segments)
        self.synapses = SynapsesFactory(self, synapses)

    def get_parameters(self):
        return self.parameters

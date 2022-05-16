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

    # TODO: Implement the pickle API.
    #       Pickle the parameters, and then pickle all of the data in the DB.
    #               But do not pickle the DB itself, @Compute will never pickle.
    # 
    #       Reconstruct the model from the parameters, except make zero neurons/synapses.
    #       Then Overwrite the new model with the saved data.

import neuwon.morphology
import neuwon.regions
import neuwon.rxd
import neuwon.synapse

class Model(neuwon.rxd.RxD_Model):
    def __init__(self, rxd_parameters={},
                species={}, mechanisms={},
                regions={}, neurons={},
                synapses={},):
        self.parameters = {
                'rxd_parameters':rxd_parameters,
                'species':species,
                'mechanisms':mechanisms,
                'regions':regions,
                'neurons':neurons,
                'synapses':synapses,
        }
        super().__init__(species=species, mechanisms=mechanisms, **rxd_parameters)
        self.regions  = neuwon.regions.RegionFactory(regions)
        self.neurons  = neuwon.morphology.NeuronTypeFactory(self, neurons)
        self.synapses = neuwon.synapse.SynapsesFactory(self, synapses)

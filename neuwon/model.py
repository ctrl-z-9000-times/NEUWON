import neuwon.rxd
import neuwon.synapse
import neuwon.morphology.neurons
import neuwon.morphology.regions

__all__ = ('Model', 'TimeSeries', 'Trace',)

class Model(neuwon.rxd.RxD_Model):
    def __init__(self, rxd_parameters={},
                species={}, mechanisms={},
                regions={}, neurons={},
                synapses={},):
        _rxd.Rx_Model.__init__(self, species=species, mechanisms=mechanisms, **rxd_parameters)
        self.regions = _rgn.RegionFactory(regions)
        self.neurons = _nrn.NeuronTypeFactory(self, neurons)

        # TODO Synapses!

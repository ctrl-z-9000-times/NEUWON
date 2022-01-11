from .neurons import NeuronTypeFactory
from .regions import RegionFactory
from neuwon.rxd import RxD_Model

class Brain:
    def __init__(self,
                rxd_parameters={},
                species={},
                mechanisms={},
                regions={},
                neurons={},
                ):
        self.rxd_model  = RxD_Model(species=species, mechanisms=mechanisms, **rxd_parameters)
        self.database   = self.rxd_model.get_database()
        self.regions    = RegionFactory(regions)
        self.neurons    = NeuronTypeFactory(self, neurons)

    def advance(self):
        self.rxd_model.advance()

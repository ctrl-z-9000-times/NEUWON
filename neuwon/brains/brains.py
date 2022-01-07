from .neurons import NeuronTypesFactory
from .regions import RegionsFactory
from neuwon.rxd.rxd_model import RxD_Model

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
        self.regions    = RegionsFactory(regions)
        self.neurons    = NeuronTypesFactory(self, neurons)

import numpy as np

class NeuronMethods:
    @classmethod
    def intialize(cls, database):
        Neuron_data = database.add_class(cls)
        Neuron_data.add_attribute('root', dtype='Segment')

        Segment_data = database.get_class('Segment')
        Segment_data.add_attribute('neuron', dtype='Neuron')

        Neuron_cls          = Neuron_data.get_instance_type()
        Neuron_cls._Segment = Segment_data.get_instance_type()
        return Neuron_cls

    def __init__(self, coordinates, diameter):
        self.root = self._Segment(parent=None, coordinates=coordinates, diameter=diameter)


# TODO: Move the NeuronFactory and NeuronType systems into a different module,
#       BC the basic neuron model should be able to function w/o the type sys.

class NeuronFactory(dict):
    def __init__(self, parameters: dict):
        super().__init__()
        self.add_parameters(parameters)

    def add_parameters(self, parameters: dict):
        for name, nt_params in parameters.items():
            self.add_neuron_type(name, nt_params)

    def add_neuron_type(self, name, parameters) -> 'NeuronType':
        if name in self:
            return self[name]
        1/0


class NeuronType:
    pass



class Neuron:
    __slots__ = ()
    @staticmethod
    def _intialize(database):
        Neuron_data = database.add_class(cls)
        Neuron_data.add_attribute('root', dtype='Segment')

        Segment_data = database.get_class('Segment')
        Segment_data.add_attribute('neuron', dtype='Neuron')

        Neuron_cls          = Neuron_data.get_instance_type()
        Neuron_cls._Segment = Segment_data.get_instance_type()
        return Neuron_cls

    def __init__(self, coordinates, diameter):
        self.root = self._Segment(parent=None, coordinates=coordinates, diameter=diameter)

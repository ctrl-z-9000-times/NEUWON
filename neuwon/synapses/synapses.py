# from .constraints import Constraints
import math
import numpy as np
import random
import scipy.spatial

class Synapse:
    """ """
    __slots__ = ()
    @classmethod
    def _initialize(cls, model, synapse_type, *,
                number=0,
                maximum_distance=math.inf,
                cleft=None,
                attachment_points,
                ):
        database = model.get_database()
        syn_data = database.add_class(str(synapse_type), cls)
        syn_cls  = syn_data.get_instance_type()
        syn_cls._maximum_distance = float(maximum_distance)
        syn_cls._attachment_points = []
        for index, parameters in enumerate(attachment_points):
            syn_cls._initialize_attachment_point(syn_data, index, **parameters)
        if cleft is not None:
            syn_cls._initialize_cleft(syn_data, **cleft)
        if number: syn_cls.grow(number)
        return syn_cls

    @classmethod
    def _initialize_attachment_point(syn_cls, syn_data, index, *, constraints={}, mechanisms={}):
        name = 'x' + str(index)
        syn_data.add_attribute(name, dtype='Segment')
        syn_cls._attachment_points.append(mechanisms)
        for mechanism_name in mechanisms:
            mechanism_name = str(mechanism_name)
            syn_data.add_attribute(name + '_' + mechanism_name, dtype=mechanism_name)

    @classmethod
    def _initialize_cleft(syn_cls, syn_data, *, volume, spillover_area=0.0):
        syn_cls._cleft_volume           = float(volume)
        syn_cls._cleft_spillover_area   = float(spillover_area)
        syn_data.add_attribute('cleft', dtype='Extracellular')

    def __init__(self, *attachment_points):
        syn_cls  = type(self)
        syn_data = syn_cls.get_database_class()

        cleft = None
        cleft_volume  = getattr(syn_cls, '_cleft_volume', None)
        if cleft_volume is not None:
            spillover_area = syn_cls._cleft_spillover_area
            database = syn_data.get_database()
            ECS = database.get_instance_type('Extracellular')
            coordinates = np.zeros(3)
            for s in attachment_points:
                coordinates += s.coordinates
            coordinates /= len(attachment_points)
            self.cleft = cleft = ECS(coordinates, cleft_volume)
            assert spillover_area == 0, 'Unimplemented!'

        for index, segment in enumerate(attachment_points):
            name = 'x' + str(index)
            setattr(self, name, segment)
            mechanisms = syn_cls._attachment_points[index]
            mechanisms = segment.insert(mechanisms, outside=cleft)
            for mechanism_name, mechanism_instance in mechanisms.items():
                setattr(self, name + '_' + mechanism_name, mechanism_instance)

    @classmethod
    def grow(cls, number=1) -> ['Synapse']:
        number = int(number)
        candidates = []
        # for at in :
        #     cls.model.filter_segments_by_type(neuron_types, segment_types)
        #     candidates.append()


        pairs = self.constraints.find_all_candidates()
        random.shuffle(pairs)
        # Iterate over and filter out candidates which are already taken.
        _num_presyn = self.database.get_data('Synapse._num_presyn')
        index_to_object = self.Segment.get_database_class().index_to_object
        while len(self.Synapse.database_class) < number and pairs:
            presyn, postsyn = pairs.pop()
            if not self.constraints.share_postsynapses and _num_presyn[postsyn]:
                continue
            # Make the synapse.
            presyn  = index_to_object(presyn)
            postsyn = index_to_object(postsyn)
            self.Synapse(presyn, postsyn)


class SynapsesFactory(dict):
    def __init__(self, rxd_model, parameters: dict):
        super().__init__()
        self.rxd_model = rxd_model
        self.add_parameters(parameters)

    def add_parameters(self, parameters: dict):
        for synapse_type, synapse_parameters in parameters.items():
            self.add_synapse_type(synapse_type, synapse_parameters)

    def add_synapse_type(self, synapse_type: str, synapse_parameters: dict):
        synapse_type = str(synapse_type)
        assert synapse_type not in self
        self[synapse_type] = Synapse._initialize(self.rxd_model, synapse_type, **synapse_parameters)



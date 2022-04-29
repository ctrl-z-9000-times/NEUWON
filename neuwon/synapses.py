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
        syn_cls._model = model
        syn_cls._maximum_distance = float(maximum_distance)
        syn_cls._constraints = []
        syn_cls._mechanisms  = []
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
        syn_cls._constraints.append(constraints)
        syn_cls._mechanisms.append(mechanisms)
        for mechanism_name in mechanisms:
            mechanism_name = str(mechanism_name)
            syn_data.add_attribute(name + '_' + mechanism_name, dtype=mechanism_name)

    @classmethod
    def _initialize_cleft(syn_cls, syn_data, *, volume, spillover_area=0.0):
        syn_cls._cleft_volume           = float(volume)
        syn_cls._cleft_spillover_area   = float(spillover_area)
        syn_data.add_attribute('cleft', dtype='Extracellular')

    @classmethod
    def grow(syn_cls, number=1) -> ['Synapse']:
        number = int(number)
        assert len(syn_cls._constraints) == 2
        candidates = syn_cls._find_growth_candidates(*syn_cls._constraints)
        random.shuffle(candidates)
        index_to_object = syn_cls._model.get_database().get_class('Segment').index_to_object
        while number > 0 and candidates:
            seg_1, seg_2 = candidates.pop()
            # TODO: Reject candidates which are already taken.
            pass
            # Make the synapse.
            seg_1 = index_to_object(seg_1)
            seg_2 = index_to_object(seg_2)
            syn_cls(seg_1, seg_2)
            number -= 1

    @classmethod
    def _find_growth_candidates(syn_cls, constaints_1, constaints_2):
        Segment     = syn_cls._model.Segment
        segs_1      = syn_cls._model.filter_segments_by_type(**constaints_1, _return_objects=False)
        segs_2      = syn_cls._model.filter_segments_by_type(**constaints_2, _return_objects=False)
        coordinates = syn_cls._model.get_database().get_data('Segment.coordinates')
        tree_1      = scipy.spatial.cKDTree(coordinates[segs_1])
        tree_2      = scipy.spatial.cKDTree(coordinates[segs_2])
        results     = tree_1.query_ball_tree(tree_2, syn_cls._maximum_distance)
        pairs = []
        for idx_1, inner in enumerate(results):
            for idx_2 in inner:
                pairs.append((segs_1[idx_1], segs_2[idx_2]))
        return pairs

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
            mechanisms = syn_cls._mechanisms[index]
            mechanisms = segment.insert(mechanisms, outside=cleft)
            for mechanism_name, mechanism_instance in mechanisms.items():
                setattr(self, name + '_' + mechanism_name, mechanism_instance)

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



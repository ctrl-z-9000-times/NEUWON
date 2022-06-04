import math
import numpy as np
import random
import scipy.spatial
from collections import namedtuple

class _AttachmentPoint:
    def __init__(self, db_class, index, *,
                constraints = {},
                mechanisms = {}):
        self.constraints = constraints
        self.mechanisms = mechanisms
        self.max_share = self.constraints.pop("maximum_share", math.inf)
        db_class.add_attribute(f"x{index+1}", dtype='Segment',
                doc=f"Attachment point #{index+1}")
        for mechanism_name in mechanisms:
            mechanism_name = str(mechanism_name)
            db_class.add_attribute(f"x{index+1}_{mechanism_name}", dtype=mechanism_name)

class Synapse:
    """ """
    __slots__ = ()
    @classmethod
    def _initialize(cls, model, synapse_type, *,
                attachment_points,
                cleft={},
                region=None,
                maximum_distance=math.inf,
                number=0,):
        database = model.get_database()
        db_class = database.add_class(str(synapse_type), cls)
        cls = db_class.get_instance_type()
        cls._model = model
        cls._maximum_distance = float(maximum_distance)
        cls._attachment_points = []
        for index, syn_parameters in enumerate(attachment_points):
            cls._attachment_points.append(
                    _AttachmentPoint(db_class, index, **syn_parameters))
        cls._initialize_cleft(db_class, **cleft)
        if number: cls.grow(number)
        return cls

    @classmethod
    def _initialize_cleft(cls, db_class, *, volume=0.0, spillover_area=0.0):
        cls._cleft_volume           = float(volume)
        cls._cleft_spillover_area   = float(spillover_area)
        assert cls._cleft_volume >= 0.0
        assert cls._cleft_spillover_area >= 0.0
        if cls._cleft_volume != 0.0:
            db_class.add_attribute('cleft', dtype='Extracellular')

    @classmethod
    def grow(cls, number=1) -> ['Synapse']:
        assert cls._model.get_database().is_sorted()
        assert len(cls._attachment_points) == 2 # unimplemented.
        number = int(number)
        candidates = cls._find_growth_candidates()
        random.shuffle(candidates)
        index_to_object = cls._model.get_database().get_class('Segment').index_to_object
        while number > 0 and candidates:
            segments = []
            # Reject candidates which are already taken.
            taken = False
            for x, seg_idx in zip(self._attachment_points, candidates.pop()):
                seg = index_to_object(seg_idx)
                if seg._num_synapses >= x.max_share:
                    taken = True
                    break
                segments.append(seg)
            if not taken:
                # Make the synapse.
                cls(*segments)
                number -= 1

    @classmethod
    def _find_growth_candidates(cls):
        coordinates = cls._model.get_database().get_data('Segment.coordinates')
        segments = []
        trees    = []
        for x in cls._attachment_points:
            segments.append(cls._model.filter_segments_by_type(**x.constraints, _return_objects=False))
            trees.append(scipy.spatial.cKDTree(coordinates[segments[-1]]))

        if len(cls._attachment_points) == 2:
            segs_1, segs_2 = segments
            tree_1, tree_2 = trees
            results = tree_1.query_ball_tree(tree_2, cls._maximum_distance)
            pairs = []
            for idx_1, inner in enumerate(results):
                for idx_2 in inner:
                    pairs.append((segs_1[idx_1], segs_2[idx_2]))
            return pairs

    def __init__(self, *attachment_points):
        cls = type(self)
        db_class = cls.get_database_class()
        # Initialize the synaptic cleft's extracellular volume.
        cleft = None
        if hasattr(db_class, 'cleft'):
            volume          = cls._cleft_volume
            spillover_area  = cls._cleft_spillover_area
            ECS = db_class.get_database().get_instance_type('Extracellular')
            coordinates = np.mean([x.coordinates for x in attachment_points])
            self.cleft = cleft = ECS(coordinates, volume)
            assert spillover_area == 0, 'Unimplemented!'
        # 
        for index, segment in enumerate(attachment_points):
            segment._num_synapses += 1
            setattr(self, f"x{index+1}", segment)
            mechanisms = cls._attachment_points[index].mechanisms
            mechanisms = segment.insert(mechanisms, outside=cleft)
            for mechanism_name, mechanism_instance in mechanisms.items():
                setattr(self, f"x{index+1}_{mechanism_name}", mechanism_instance)

class SynapsesFactory(dict):
    def __init__(self, rxd_model, parameters: dict):
        super().__init__()
        self.rxd_model = rxd_model
        segment_db_class = self.rxd_model.get_database().get_class("Segment")
        segment_db_class.add_attribute("_num_synapses", 0, dtype=np.uint32)
        self.add_parameters(parameters)

    def add_parameters(self, parameters: dict):
        for synapse_type, synapse_parameters in parameters.items():
            self.add_synapse_type(synapse_type, synapse_parameters)

    def add_synapse_type(self, synapse_type: str, synapse_parameters: dict):
        synapse_type = str(synapse_type)
        assert synapse_type not in self
        self[synapse_type] = Synapse._initialize(self.rxd_model, synapse_type, **synapse_parameters)

"""
Models the physical & structural constraints on synapses,
and finds potential sites for growing new synapses.
"""

from neuwon.database import Compute
import math
import numpy as np
import scipy.spatial

class Constraints:
    def __init__(self, rxd, *,
                presynapse_neuron_types=[],
                presynapse_segment_types=[],
                postsynapse_neuron_types=[],
                postsynapse_segment_types=[],
                maximum_distance=math.inf,
                share_postsynapses=False,):
        self.Neuron   = rxd.Neuron
        self.Segment  = rxd.Segment
        self.presynapse_neuron_types = [self.Neuron.neuron_types_list.index(neuron_type)
                                    for neuron_type in presynapse_neuron_types]
        self.presynapse_segment_types = [self.Segment.segment_types_list.index(segment_type)
                                    for segment_type in presynapse_segment_types]
        self.postsynapse_neuron_types = [self.Neuron.neuron_types_list.index(neuron_type)
                                    for neuron_type in postsynapse_neuron_types]
        self.postsynapse_segment_types = [self.Segment.segment_types_list.index(segment_type)
                                    for segment_type in postsynapse_segment_types]
        self.maximum_distance = float(maximum_distance)
        self.share_postsynapses = bool(share_postsynapses)

        self._filter_method = self.Segment.get_database_class().add_method(self._filter_method)

    def find_all_candidates(self) -> '[(presyn, postsyn), ...]':
        presyn_segs  = self.get_presynapse_candidates()
        postsyn_segs = self.get_postsynapse_candidates()
        coordinates  = self.Segment.get_database_class().get_data('coordinates')
        presyn_tree  = scipy.spatial.cKDTree(coordinates[presyn_segs])
        postsyn_tree = scipy.spatial.cKDTree(coordinates[postsyn_segs])
        results = presyn_tree.query_ball_tree(postsyn_tree, self.maximum_distance)
        pairs = []
        for pre_idx, inner in enumerate(results):
            for post_idx in inner:
                pairs.append((presyn_segs[pre_idx], postsyn_segs[post_idx]))
        return pairs

    def get_presynapse_candidates(self):
        return self._filter_segments(self.presynapse_neuron_types, self.presynapse_segment_types, True)
    def get_postsynapse_candidates(self):
        return self._filter_segments(self.postsynapse_neuron_types, self.postsynapse_segment_types, self.share_postsynapses)

    def _filter_segments(self, neuron_types, segment_types, share):
        if neuron_types:
            neuron_mask = np.zeros(len(self.Neuron.neuron_types_list), dtype=bool)
            neuron_mask[neuron_types] = True
        else:
            neuron_mask = np.ones(len(self.Neuron.neuron_types_list), dtype=bool)
        if segment_types:
            segment_mask = np.zeros(len(self.Segment.segment_types_list), dtype=bool)
            segment_mask[segment_types] = True
        else:
            segment_mask = np.ones(len(self.Segment.segment_types_list), dtype=bool)
        filter_values = self._filter_method(None, neuron_mask, segment_mask, share)
        return np.nonzero(filter_values)[0]

    @Compute
    def _filter_method(segment, neuron_mask, segment_mask, share) -> bool:
        if not share and segment._num_presyn > 0:
            return False
        return segment_mask[segment.segment_type_id] and neuron_mask[segment.neuron.neuron_type_id]

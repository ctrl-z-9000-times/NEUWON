import numpy as np
from database import *
import htm

# Maybe instead of an htm I could make a simpler recurrent network?

class SpatialPooler:
    def __init__(self):
        self.db      = db      = Database()
        self.Cell    = Cell    = db.add_class("Cell")
        self.Segment = Segment = db.add_class("Segment")
        self.Synapse = Synapse = db.add_class("Synapse")

        Synapse.add_attribute("permanence")
        Synapse.add_attribute("presyn",  dtype=Cell)
        Synapse.add_attribute("postsyn", dtype=Segment)

        Segment.add_attribute("v", dtype=np.uint32)
        Segment.add_attribute("cell")
        Segment.add_attribute("num_syn", dtype=np.uint32)
        Segment.add_attribute("distal", dtype=bool)

        Cell.add_sparse_matrix("connections", Segment, dtype=bool)
        Cell.add_attribute("proximal_input", dtype=np.uint32)
        Cell.add_attribute("distal_input", dtype=np.uint32)
        Cell.add_attribute("state", dtype=np.uint8)

    def compute(self, input_sdr, learn=True):
        self._feed_forward(input_sdr)
        self._activate()
        if learn: self._learn()

    def _feed_forward(self, input_sdr):
        input_sdr.sparse
        connections = self.Cell.get_component("connections").to_fmt('csr').get()
        seg_accum = self.Segment.get_component("num_active_syn").get()
        seg_accum.fill(0)
        for row in input_sdr.sparse:
            for col in connections[row].rows:
                seg_accum[col] += 1

    def _activate(self):
        pass

    def _learn(self):
        1/0


def test_htm():
    c = Connections()

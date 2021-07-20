from htm import SDR
from htm.encoders.coordinate import CoordinateEncoder
from neuwon.database import *
import math
import numpy as np
import random

default_parameters = {
    'num_place_cells': 201,
    'place_cell_radius': 4,
    'num_grid_cells': 100,
    'stability_rate': 0.1,
    'fatigue_rate':   0.033,
    'a0': 3.,
    's0': 0.3,
    'theta_rate': 0.001, # Not specified.
    'gain_rate':  0.001, # Not specified.
    'psi_sat': 30.,
    'learning_rate': 0.001,
    'learning_desensitization_rate': 0.1,  # Not specified.
}

class Model:
    def __init__(self, parameters):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.db = Database()
        self.GridCell = self.db.add_class("GridCell")
        self.PlaceCell = self.db.add_class("PlaceCell")

        self.pc_encoder = CoordinateEncoder(9, parameters['num_place_cells'])
        self.pc_sdr = SDR(self.pc_encoder.n)

        self.PlaceCell.add_attribute("r", doc="Firing rate")
        self.PlaceCell.add_sparse_matrix("J", self.GridCell, doc="Synapse weights")
        self.PlaceCell.add_attribute("avg_r", initial_value=0)

        self.GridCell.add_attribute("psi", doc="Firing rate")
        self.GridCell.add_attribute("avg_psi", initial_value=0)
        self.GridCell.add_attribute("r_act", initial_value=0)
        self.GridCell.add_attribute("r_inact", initial_value=0)

        self.place_cells = [self.PlaceCell() for _ in range(self.num_place_cells)]
        self.grid_cells = [self.GridCell() for _ in range(self.num_grid_cells)]
        J = self.PlaceCell.get_component("J")
        J.set(np.random.uniform(0.0, 0.1, size=J.shape))

        self.reset()

    def reset(self):
        self.theta = 0
        self.gain = 1
        self.GridCell.get_component("r_act").get().fill(0)
        self.GridCell.get_component("r_inact").get().fill(0)
        self.GridCell.get_component("avg_psi").get().fill(0)
        self.PlaceCell.get_component("avg_r").get().fill(0)

    def advance(self, coordinates, learn=True):
        coordinates = np.array((int(round(x)) for x in coordinates))
        self.pc_encoder.encode((coordinates, self.place_cell_radius), self.pc_sdr)
        r = self.PlaceCell.get_component("r").get()
        r[:] = self.pc_sdr.dense
        # Compute the feed forward synaptic activity.
        J = self.PlaceCell.get_component("J").get()
        r_ = r.reshape((1, len(r)))
        h  = (r_ * J).T.squeeze()
        h *= (1 / len(self.PlaceCell))
        # Apply stability and fatigue dynamics.
        r_act   = self.GridCell.get_component("r_act").get()
        r_inact = self.GridCell.get_component("r_inact").get()
        h -= r_inact
        r_act   += self.stability_rate * (h - r_act)
        r_inact += self.fatigue_rate * h
        # Apply activation function.
        psi = self.GridCell.get_component("psi").get()
        psi[:] = (self.psi_sat * (2 / math.pi)
                * np.arctan(self.gain * np.maximum(0, r_act - self.theta)))
        # Update theta & gain to control the mean activity and sparsity.
        psi_sum = np.sum(psi)
        psi2_sum = np.sum(psi * psi)
        a = psi_sum / len(self.grid_cells)
        if psi2_sum != 0:
            s = psi_sum ** 2 / psi2_sum
        else: s = 0
        self.theta += self.theta_rate * (a - self.a0)
        self.gain  += self.gain_rate * (s - self.s0)
        # Hebbian learning.
        if not learn: return
        avg_r   = self.PlaceCell.get_component("avg_r").get()
        avg_psi = self.GridCell.get_component("avg_psi").get()
        J = self.PlaceCell.get_component("J").to_coo().get()
        J.data += self.learning_rate * (r[J.row] * psi[J.col] - avg_r[J.row] * avg_psi[J.col])
        # Update moving averages of cell activity.
        avg_r   *= 1 - self.learning_desensitization_rate
        avg_r   += r * self.learning_desensitization_rate
        avg_psi *= 1 - self.learning_desensitization_rate
        avg_psi += psi * self.learning_desensitization_rate

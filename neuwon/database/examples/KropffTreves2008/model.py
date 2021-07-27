from htm import SDR
from htm.encoders.coordinate import CoordinateEncoder
from neuwon.database import *
import math
import numpy as np
import random

default_parameters = {
    'num_place_cells': 201,
    'place_cell_radius': 4, # Not specified by authors.
    'place_cell_num_active': 9, # Not specified by authors.
    'num_grid_cells': 100,
    'stability_rate': 0.1,
    'fatigue_rate':   0.033,
    'a0': 3.,
    's0': 0.3,
    'theta_rate': 0.001, # Not specified by authors.
    'gain_rate':  0.001, # Not specified by authors.
    'psi_sat': 30.,
    'learning_rate': 0.001,
    'learning_desensitization_rate': 0.1,  # Not specified by authors.
}

class Model:
    def __init__(self, parameters=default_parameters):
        # Assign the parameters to 'self' as attributes.
        # For example: >>> parameters = {'x': 3}
        # Becomes:     >>> self.x = 3
        for key, value in parameters.items(): setattr(self, key, value)

        self.db = Database()
        self.GridCell = self.db.add_class("GridCell")
        self.PlaceCell = self.db.add_class("PlaceCell")

        self.pc_encoder = CoordinateEncoder(self.place_cell_num_active, self.num_place_cells)
        self.pc_sdr = SDR(self.pc_encoder.n)

        self.PlaceCell.add_attribute("r", doc="Firing rate")
        self.PlaceCell.add_sparse_matrix("J", self.GridCell, doc="Synapse weights")
        self.PlaceCell.add_attribute("avg_r", 0)

        self.GridCell.add_attribute("psi", doc="Firing rate")
        self.GridCell.add_attribute("avg_psi", 0)
        self.GridCell.add_attribute("r_act", 0)
        self.GridCell.add_attribute("r_inact", 0)

        PC = self.PlaceCell.get_instance_type()
        GC = self.GridCell.get_instance_type()
        self.place_cells = [PC() for _ in range(self.num_place_cells)]
        self.grid_cells = [GC() for _ in range(self.num_grid_cells)]
        J = self.db.get("PlaceCell.J")
        initial_weights = np.random.uniform(0.0, 0.1, size=J.shape)
        # initial_weights *= SDR(J.shape).randomize(.5).dense
        J.set_data(initial_weights)

        self.reset()

    def reset(self):
        self.theta = 0
        self.gain = 1
        self.db.get_data("GridCell.r_act").fill(0)
        self.db.get_data("GridCell.r_inact").fill(0)
        self.db.get_data("GridCell.avg_psi").fill(0)
        self.db.get_data("PlaceCell.avg_r").fill(0)

    def advance(self, coordinates, learn=True):
        # Set the place cell activity based on the positional coordinates.
        coordinates = np.array((int(round(x)) for x in coordinates))
        self.pc_encoder.encode((coordinates, self.place_cell_radius), self.pc_sdr)
        self.PlaceCell.get("r").set_data(self.pc_sdr.dense)
        # Compute the feed forward synaptic activity.
        r = self.PlaceCell.get_data("r")
        J = self.PlaceCell.get_data("J")
        r_ = r.reshape((1, len(r)))
        h  = (r_ * J).T.squeeze()
        h *= (1 / len(self.PlaceCell))
        # Apply stability and fatigue dynamics.
        r_act   = self.GridCell.get_data("r_act")
        r_inact = self.GridCell.get_data("r_inact")
        h -= r_inact
        r_act   += self.stability_rate * (h - r_act)
        r_inact += self.fatigue_rate * h
        # Apply activation function.
        psi = self.GridCell.get_data("psi")
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
        avg_r   = self.PlaceCell.get_data("avg_r")
        avg_psi = self.GridCell.get_data("avg_psi")
        J = self.PlaceCell.get("J").to_coo().get_data()
        J.data += self.learning_rate * (r[J.row] * psi[J.col] - avg_r[J.row] * avg_psi[J.col])
        # Update moving averages of cell activity.
        avg_r   *= 1 - self.learning_desensitization_rate
        avg_r   += r * self.learning_desensitization_rate
        avg_psi *= 1 - self.learning_desensitization_rate
        avg_psi += psi * self.learning_desensitization_rate

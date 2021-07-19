#!/usr/bin/python
from htm import SDR, Metrics
from htm.encoders.coordinate import CoordinateEncoder
from neuwon.database import *
from scipy.ndimage.filters import maximum_filter
import argparse
import cv2
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.ndimage
import scipy.signal
import scipy.stats

default_parameters = {
    'num_place_cells': 201,
    'place_cell_radius': 4,
    'num_grid_cells': 100,
    'stability_rate': 0.1,
    'fatigue_rate':   0.033,
    'a0': 3.,
    's0': 0.3,
    'theta_rate': 0.2, # Not specified.
    'gain_rate': 0.2, # Not specified.
    'psi_sat': 30.,
    # 'learning_rate': 0.001,
    'learning_rate': 0.01,
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
        self.theta = 1
        self.gain = 100
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

class Environment:
    """ The environment is a 2D square, in first quadrant with corner at origin. """
    def __init__(self, size):
        self.size     = size
        self.position = (size/2, size/2)
        self.speed    = 2.0 ** .5
        self.angle    = 0
        self.course   = []

    def in_bounds(self, position):
        x, y = position
        if True:
            # Circle arena
            radius = self.size / 2.
            hypot  = (x - radius) ** 2 + (y - radius) ** 2
            return hypot < radius ** 2
        else:
            # Square arena
            x_in = x >= 0 and x < self.size
            y_in = y >= 0 and y < self.size
            return x_in and y_in

    def move(self):
        max_rotation = 2 * math.pi / 20
        self.angle += random.uniform(-max_rotation, max_rotation)
        self.angle = (self.angle + 2 * math.pi) % (2 * math.pi)
        vx = self.speed * math.cos(self.angle)
        vy = self.speed * math.sin(self.angle)
        x, y = self.position
        new_position = (x + vx, y + vy)
        if self.in_bounds(new_position):
            self.position = new_position
            self.course.append(self.position)
        else:
            # On failure, recurse and try again.
            assert(self.in_bounds(self.position))
            self.angle = random.uniform(0, 2 * math.pi)
            self.move()

    def plot_course(self, show=True):
        plt.figure("Path")
        plt.ylim([0, self.size])
        plt.xlim([0, self.size])
        self.course = self.course[:100*1000]
        x, y = zip(*self.course)
        plt.plot(x, y, 'k-')
        if show: plt.show()

class Experiment:
    def __init__(self):
        # self.env = Environment(size = 200)
        self.env = Environment(size = 100)
        self.model = Model(default_parameters)

    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--steps', type=int, default = 1000 * 1000,)
        args = parser.parse_args()
        x = Experiment()
        x.measure_receptive_fields(args.steps)
        x.analyze_grid_properties()
        x.find_alignment_points()
        x.select_exemplar_cells(20)
        x.plot()

    def run_for(self, steps):
        print("Running for %d steps ..."%steps)
        self.model.reset()
        for step in range(steps):
            if step and step % 10000 == 0: print("Step %d"%step)
            self.env.move()
            self.model.advance(self.env.position)
            # self.model.db.check()

    def measure_receptive_fields(self, steps, enc_samples=12):
        env_sz = self.env.size
        num_pc = len(self.model.place_cells)
        num_gc = len(self.model.grid_cells)
        enc_samples  = random.sample(range(num_pc), enc_samples)
        self.enc_rfs = [np.zeros((env_sz, env_sz)) for idx in enc_samples]
        self.gc_rfs  = [np.zeros((env_sz, env_sz)) for idx in range(num_gc)]
        for _ in range(steps):
            self.env.move()
            position = tuple(int(q) for q in self.env.position)
            self.model.advance(self.env.position)
            for rf_idx, enc_idx in enumerate(enc_samples):
                self.enc_rfs[rf_idx][position] = self.model.pc_sdr.dense[enc_idx]
            gc_activity = self.model.GridCell.get_component("psi").get()
            for gc_idx in range(num_gc):
                self.gc_rfs[gc_idx][position] = gc_activity[gc_idx] / self.model.psi_sat

    def analyze_grid_properties(self):
        self.xcor      = []
        self.gridness  = []
        self.zoom = .5
        dim = int(self.env.size - 1) # Dimension of the X-Correlation.
        circleMask = cv2.circle( np.zeros([dim,dim]), (dim//2,dim//2), dim//2, [1], -1)
        for rf in self.gc_rfs:
            # Shrink images before the expensive auto correlation.
            rf = scipy.ndimage.zoom(rf, self.zoom, order=1)
            X  = scipy.signal.correlate2d(rf, rf)
            # Crop to a circle.
            X *= circleMask
            self.xcor.append( X )
            # Correlate the xcor with a rotation of itself.
            rot_cor = {}
            for angle in (30, 60, 90, 120, 150):
                rot = scipy.ndimage.rotate( X, angle, order=1, reshape=False )
                r, p = scipy.stats.pearsonr( X.reshape(-1), rot.reshape(-1) )
                rot_cor[ angle ] = r
            score = (+ (rot_cor[60] + rot_cor[120]) / 2.
                     - (rot_cor[30] + rot_cor[90] + rot_cor[150]) / 3.)
            if not math.isfinite(score): score = 0
            self.gridness.append(score)

    def find_alignment_points(self):
        self.alignment = []
        dim = int(self.env.size - 1)
        for X in self.xcor:
            # Find alignment points, the local maxima in the x-correlation.
            half   = X[ : , : dim//2 + 1 ]
            maxima = maximum_filter( half, size=10 ) == half
            maxima = np.logical_and( maxima, half > 0 ) # Filter out areas which are all zeros.
            maxima = np.nonzero( maxima )
            align_pts = []
            for idx in np.argsort(-half[ maxima ])[ 1 : 4 ]:
                x_coord, y_coord = maxima[0][idx], maxima[1][idx]
                # Fix coordinate scale & offset.
                x_coord = (x_coord - dim/2 + .5) / self.zoom
                y_coord = (dim/2 - y_coord - .5) / self.zoom
                align_pts.append( ( x_coord, y_coord ) )
            self.alignment.append(align_pts)

    def select_exemplar_cells(self, num=20):
        # Analyze the best examples of grid cells, by gridness scores.
        if num < 1: num = int(round(len(self.gridness) * num))
        gc_samples = np.argsort(self.gridness)[ -num : ]
        self.gridness_all = self.gridness[:] # Save all gridness scores for histogram.
        gc_samples   = sorted(gc_samples, key=lambda x: self.gridness[x])
        # Get the selected data.
        self.gc_rfs       = [ self.gc_rfs[idx]    for idx in gc_samples ]
        self.xcor         = [ self.xcor[idx]      for idx in gc_samples ]
        self.gridness     = [ self.gridness[idx]  for idx in gc_samples ]
        self.alignment    = [ self.alignment[idx] for idx in gc_samples ]
        self.score = np.mean(self.gridness)
        print("Score:", self.score)
        return self.score

    def plot(self):
        if False:
            # Show how the agent traversed the environment.
            self.env.plot_course(show=False)

        if True:
            # Show the Input/Encoder Receptive Fields.
            plt.figure("Input Receptive Fields")
            nrows = int(len(self.enc_rfs) ** .5)
            ncols = math.ceil((len(self.enc_rfs)+.0) / nrows)
            for subplot_idx, rf in enumerate(self.enc_rfs):
              plt.subplot(nrows, ncols, subplot_idx + 1)
              plt.imshow(rf, interpolation='nearest')

        if False:
            # Show Histogram of gridness scores.
            plt.figure("Histogram of Gridness Scores")
            plt.hist( self.gridness_all, bins=28, range=[-.3, 1.1] )
            plt.ylabel("Number of cells")
            plt.xlabel("Gridness score")

        if True:
            # Show the Grid Cells Receptive Fields.
            plt.figure("Grid Cell Receptive Fields")
            nrows = int(len(self.gc_rfs) ** .5)
            ncols = math.ceil((len(self.gc_rfs)+.0) / nrows)
            for subplot_idx, rf in enumerate(self.gc_rfs):
              plt.subplot(nrows, ncols, subplot_idx + 1)
              plt.title("Gridness score %g"%self.gridness[subplot_idx])
              plt.imshow(rf, interpolation='nearest')

        if True:
            # Show the autocorrelations of the grid cell receptive fields.
            plt.figure("Grid Cell RF Autocorrelations")
            for subplot_idx, X in enumerate(self.xcor):
              plt.subplot(nrows, ncols, subplot_idx + 1)
              plt.title("Gridness score %g"%self.gridness[subplot_idx])
              plt.imshow(X, interpolation='nearest')

        if False:
            # Show locations of the first 3 maxima of each x-correlation.
            alignment_flat = [];
            for pts in self.alignment: alignment_flat.extend(pts)
            # Replace duplicate points with larger points in the image.
            coord_set = list(set(alignment_flat))
            defaultDotSz = mpl.rcParams['lines.markersize'] ** 2
            scales = [alignment_flat.count(c) * defaultDotSz for c in coord_set]
            if coord_set:
                x_coords, y_coords = zip(*coord_set)
                if x_coords and y_coords:
                    plt.figure("Spacing & Orientation")
                    plt.scatter( x_coords, y_coords, scales )

        plt.show()


if __name__ == "__main__":
    Experiment().main()

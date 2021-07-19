#!/usr/bin/python
from htm import SDR, Metrics
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
    'num_place_cells': 200,
    'num_grid_cells': 100,
    'b1': 0,
    'b2': 0,
    'a0': 0,
    's0': 0,
    'b3': 0,
    'b4': 0,
    'alpha': 0,
}

class Model:
    def __init__(self, parameters):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.db = Database()
        self.GridCell = self.db.add_class("GridCell")
        self.PlaceCell = self.db.add_class("PlaceCell")

        self.pc_encoder = CoordinateEncoder(11, 201)
        self.pc_sdr = self.pc_encoder.encode(self.env.position)

        self.PlaceCell.add_attribute("r", doc="Firing rate")
        self.PlaceCell.add_sparse_matrix("J", self.GridCell, doc="Synapse weights")
        self.PlaceCell.add_attribute("avg_r", initial_value=0)

        self.GridCell.add_attribute("psi", doc="Firing rate")
        self.GridCell.add_attribute("avg_psi", initial_value=0)
        self.GridCell.add_attribute("r_act", initial_value=0)
        self.GridCell.add_attribute("r_inact", initial_value=0)

        self.place_cells = [self.GridCell() for _ in range(self.num_place_cells)]
        self.grid_cells = [self.PlaceCell() for _ in range(self.num_grid_cells)]
        J = self.PlaceCell.get_component("J").to_lil().get()
        # J is all random, all to all...

    def reset(self):
        self.GridCell.get_component("r_act").get().fill(0)
        self.GridCell.get_component("r_inact").get().fill(0)

    def advance(self, coordinates, learn=True):
        self.pc_encoder.encode(coordinates, self.pc_sdr)
        r = self.PlaceCell.get_component("r").get()
        r[:] = self.pc_sdr.dense
        # Compute the feed forward synaptic activity.
        J = self.PlaceCell.get_component("J").to_csr().get()
        h = J.dot(r) / len(self.PlaceCell)
        # Apply stability and fatigue dynamics.
        r_act   = self.GridCell.get_component("r_act").get()
        r_inact = self.GridCell.get_component("r_inact").get()
        h -= r_inact
        r_act   += self.b1 * (h - r_act)
        r_inact += self.b2 * h
        # Apply activation function.
        psi[:] = (self.psi_sat * (2 / math.pi)
                * np.atan(self.gain * np.maximum(0, r_act - self.theta)))
        # Update theta & gain to control the mean activity and sparsity.
        psi_sum = np.sum(psi)
        a = psi_sum / len(self.grid_cells)
        s = psi_sum ** 2 / np.sum(psi * psi)
        self.theta += self.b3 * (a - self.a0)
        self.gain  += self.b4 * (s - self.s0)
        # Hebbian learning.
        if not learn: return
        avg_r   = self.PlaceCell.get_component("avg_r").get()
        avg_psi = self.GridCell.get_component("avg_psi").get()
        J.data += self.learning_rate * 1/0
        # Update moving averages of cell activity.
        avg_r   *= 1 - self.alpha
        avg_r   += r * self.alpha
        avg_psi *= 1 - self.alpha
        avg_psi += psi * self.alpha

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
        x, y = zip(*self.course)
        plt.plot(x, y, 'k-')
        if show: plt.show()

class Experiment:
    def __init__(self):
        self.env = Environment(size = 200)
        self.model = Model(default_parameters)

    def run_for(self, steps):
        print("Running for %d steps ..."%steps)
        self.model.reset()
        for step in range(steps):
            if step and step % 10000 == 0: print("Step %d"%step)
            self.env.move()
            self.model.advance(self.env.position)

    def measure_receptive_fields(self, enc_samples=12):
        print("Measuring Receptive Fields ...")
        enc_samples  = random.sample(range(self.enc.parameters.size), enc_samples)
        self.enc_rfs = [np.zeros((env.size, env.size)) for idx in enc_samples]
        num_gc = len(self.model.grid_cells)
        self.gc_rfs  = [np.zeros((env.size, env.size)) for idx in range(num_gc)]
        for x in range(env.size):
            for y in range(env.size):
                position = (x, y)
                if not env.in_bounds(position): continue
                self.model.reset()
                self.model.advance(position, learn=False)
                for rf_idx, enc_idx in enumerate(enc_samples):
                    self.enc_rfs[rf_idx][position] = self.model.pc_sdr.dense[enc_idx]
                gc_activity = self.model.GridCell.get_component("psi")
                gc_active_threshold = .2
                gc_sdr = SDR(num_gc)
                gc_sdr.sparse = np.nonzero(gc_activity >= gc_active_threshold)[0]
                for gc_idx in range(num_gc):
                    self.gc_rfs[gc_idx][position] = gc_sdr.dense[gc_idx]

    def analyze_grid_properties(self):
        xcor      = []
        gridness  = []
        alignment = []
        zoom = .25 if args.debug else .5
        dim  = int(env.size * 2 * zoom - 1) # Dimension of the X-Correlation.
        circleMask = cv2.circle( np.zeros([dim,dim]), (dim//2,dim//2), dim//2, [1], -1)
        for rf in gc_rfs:
            # Shrink images before the expensive auto correlation.
            rf = scipy.ndimage.zoom(rf, zoom, order=1)
            X  = scipy.signal.correlate2d(rf, rf)
            # Crop to a circle.
            X *= circleMask
            xcor.append( X )
            # Correlate the xcor with a rotation of itself.
            rot_cor = {}
            for angle in (30, 60, 90, 120, 150):
                rot = scipy.ndimage.rotate( X, angle, order=1, reshape=False )
                r, p = scipy.stats.pearsonr( X.reshape(-1), rot.reshape(-1) )
                rot_cor[ angle ] = r
            gridness.append( + (rot_cor[60] + rot_cor[120]) / 2.
                             - (rot_cor[30] + rot_cor[90] + rot_cor[150]) / 3.)
            # Find alignment points, the local maxima in the x-correlation.
            half   = X[ : , : dim//2 + 1 ]
            maxima = maximum_filter( half, size=10 ) == half
            maxima = np.logical_and( maxima, half > 0 ) # Filter out areas which are all zeros.
            maxima = np.nonzero( maxima )
            align_pts = []
            for idx in np.argsort(-half[ maxima ])[ 1 : 4 ]:
                x_coord, y_coord = maxima[0][idx], maxima[1][idx]
                # Fix coordinate scale & offset.
                x_coord = (x_coord - dim/2 + .5) / zoom
                y_coord = (dim/2 - y_coord - .5) / zoom
                align_pts.append( ( x_coord, y_coord ) )
            alignment.append(align_pts)

    def select_exemplar_cells(self):
        # Analyze the best examples of grid cells, by gridness scores.
        if verbose:
            # Select exactly 20 cells to display.
            gc_num_samples = 20
        else:
            # Top 20% of grid cells.
            gc_num_samples = int(round(len(gridness) * .20))
        gc_samples = np.argsort(gridness)[ -gc_num_samples : ]

        # Get the selected data.
        gridness_all = gridness[:] # Save all gridness scores for histogram.
        gc_samples   = sorted(gc_samples, key=lambda x: gridness[x])
        gc_rfs       = [ gc_rfs[idx]    for idx in gc_samples ]
        xcor         = [ xcor[idx]      for idx in gc_samples ]
        gridness     = [ gridness[idx]  for idx in gc_samples ]
        alignment    = [ alignment[idx] for idx in gc_samples ]
        score = np.mean(gridness)
        print("Score:", score)

    def plot(self):
        # Show how the agent traversed the environment.
        if args.steps <= 100000:
            env.plot_course(show=False)
        else:
            print("Not going to plot course, too long.")

        # Show the Input/Encoder Receptive Fields.
        plt.figure("Input Receptive Fields")
        nrows = int(len(enc_rfs) ** .5)
        ncols = math.ceil((len(enc_rfs)+.0) / nrows)
        for subplot_idx, rf in enumerate(enc_rfs):
          plt.subplot(nrows, ncols, subplot_idx + 1)
          plt.imshow(rf, interpolation='nearest')

        # Show Histogram of gridness scores.
        plt.figure("Histogram of Gridness Scores")
        plt.hist( gridness_all, bins=28, range=[-.3, 1.1] )
        plt.ylabel("Number of cells")
        plt.xlabel("Gridness score")

        # Show the Grid Cells Receptive Fields.
        plt.figure("Grid Cell Receptive Fields")
        nrows = int(len(gc_rfs) ** .5)
        ncols = math.ceil((len(gc_rfs)+.0) / nrows)
        for subplot_idx, rf in enumerate(gc_rfs):
          plt.subplot(nrows, ncols, subplot_idx + 1)
          plt.title("Gridness score %g"%gridness[subplot_idx])
          plt.imshow(rf, interpolation='nearest')

        # Show the autocorrelations of the grid cell receptive fields.
        plt.figure("Grid Cell RF Autocorrelations")
        for subplot_idx, X in enumerate(xcor):
          plt.subplot(nrows, ncols, subplot_idx + 1)
          plt.title("Gridness score %g"%gridness[subplot_idx])
          plt.imshow(X, interpolation='nearest')

        # Show locations of the first 3 maxima of each x-correlation.
        plt.figure("Spacing & Orientation")
        alignment_flat = [];
        [alignment_flat.extend(pts) for pts in alignment]
        # Replace duplicate points with larger points in the image.
        coord_set = list(set(alignment_flat))
        defaultDotSz = mpl.rcParams['lines.markersize'] ** 2
        scales = [alignment_flat.count(c) * defaultDotSz for c in coord_set]
        x_coords, y_coords = zip(*coord_set)
        if x_coords and y_coords:
          plt.scatter( x_coords, y_coords, scales )
        else:
          plt.scatter( [], [] )
          print("No alignment points found!")

        plt.show()


def test_model():
    1/0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default = 1000 * 1000,)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    x = Experiment()
    x.run_for(args.steps)
    x.measure_receptive_fields()
    x.analyze_grid_properties()
    x.select_exemplar_cells()
    x.plot()
    return x.score

if __name__ == "__main__": main()

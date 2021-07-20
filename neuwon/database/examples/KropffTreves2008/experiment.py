#!/usr/bin/python
from .environment import Environment
from .model import Model
from scipy.ndimage.filters import maximum_filter
import cv2
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.ndimage
import scipy.signal
import scipy.stats

class Experiment:
    def __init__(self, size=200):
        self.env = Environment(size=size)
        self.model = Model()
        self.zoom = .5 # Shrink images by this factor before doing the auto correlation.
        self.num_pc_samples = 12
        self.num_pc = len(self.model.place_cells)
        self.num_gc = len(self.model.grid_cells)
        env_sz = self.env.size
        self.pc_samples = random.sample(range(self.num_pc), self.num_pc_samples)
        self.pc_rfs = [np.zeros((env_sz, env_sz)) for idx in range(self.num_pc_samples)]
        self.gc_rfs = [np.zeros((env_sz, env_sz)) for idx in range(self.num_gc)]

    def run(self, steps):
        measure_period = 5 * self.env.size ** 2
        for step in range(steps):
            if step and step % 10000 == 0: print("Step %d / %d"%(step, steps))
            self.env.move()
            self.model.advance(self.env.position)
            if step > steps - measure_period:
                # Measure each cells receptive field at this location.
                position = tuple(int(q) for q in self.env.position)
                for rf_idx, pc_idx in enumerate(self.pc_samples):
                    self.pc_rfs[rf_idx][position] = self.model.pc_sdr.dense[pc_idx]
                gc_activity = self.model.GridCell.get_component("psi").get()
                for gc_idx in range(self.num_gc):
                    self.gc_rfs[gc_idx][position] = gc_activity[gc_idx] / self.model.psi_sat

    def analyze_grid_properties(self):
        self.xcor      = []
        self.gridness  = []
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
        gc_samples = sorted(gc_samples, key=lambda x: self.gridness[x])
        # Get the selected data.
        self.gc_rfs       = [self.gc_rfs[idx]    for idx in gc_samples]
        self.xcor         = [self.xcor[idx]      for idx in gc_samples]
        self.gridness     = [self.gridness[idx]  for idx in gc_samples]
        self.alignment    = [self.alignment[idx] for idx in gc_samples]
        self.score = np.mean(self.gridness)
        print("Score:", self.score)
        return self.score

    def plot(self):
        # Show how the agent traversed the environment.
        self.env.plot_course(show=False)

        # Show the Input/Encoder Receptive Fields.
        plt.figure("Input Receptive Fields")
        nrows = int(len(self.pc_rfs) ** .5)
        ncols = math.ceil((len(self.pc_rfs)+.0) / nrows)
        for subplot_idx, rf in enumerate(self.pc_rfs):
          plt.subplot(nrows, ncols, subplot_idx + 1)
          plt.imshow(rf, interpolation='nearest')

        # Show Histogram of gridness scores.
        plt.figure("Histogram of Gridness Scores")
        plt.hist( self.gridness_all, bins=28, range=[-.3, 1.1] )
        plt.ylabel("Number of cells")
        plt.xlabel("Gridness score")

        # Show the Grid Cells Receptive Fields.
        plt.figure("Grid Cell Receptive Fields")
        nrows = int(len(self.gc_rfs) ** .5)
        ncols = math.ceil((len(self.gc_rfs)+.0) / nrows)
        for subplot_idx, rf in enumerate(self.gc_rfs):
          plt.subplot(nrows, ncols, subplot_idx + 1)
          plt.title("Gridness score %g"%self.gridness[subplot_idx])
          plt.imshow(rf, interpolation='nearest')

        # Show the autocorrelations of the grid cell receptive fields.
        plt.figure("Grid Cell RF Autocorrelations")
        for subplot_idx, X in enumerate(self.xcor):
          plt.subplot(nrows, ncols, subplot_idx + 1)
          plt.title("Gridness score %g"%self.gridness[subplot_idx])
          plt.imshow(X, interpolation='nearest')

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

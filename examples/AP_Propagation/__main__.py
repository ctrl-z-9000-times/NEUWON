""" Model of an action potential propagating through a large axonal arbor. """

import numpy as np
from neuwon import *
import neuwon.mechanisms.hh as hh
min_v = hh.g_k.reversal_potential
max_v = hh.g_na.reversal_potential
from graph_algorithms import depth_first_traversal as dft
import tempfile
import os
from PIL import Image, ImageFont, ImageDraw

# Setup a neuron and its axonal arbor.
r = 30e-6
w = 100e-6
soma = GrowSomata.single([-w, -w-r/2, 0], 10e-6)
rgn = Union([
    Sphere(  [-w, -w, 0], r),
    Cylinder([-w, -w, 0], [-w, w, 0], r),
    Sphere(  [-w, w, 0], r),
    Cylinder([-w, w, 0], [w, w, 0], r),
    Sphere(  [w, w, 0], r),
    Cylinder([w, w, 0], [w, -w, 0], r),
    Sphere(  [w, -w, 0], r),
    Cylinder([w, -w, 0], [0, -w, w*2], r),
    Sphere(  [0, -w, w*2], r),
    Cylinder([0, -w, w*2], [0, 0, w*6], r),
    Sphere(  [0, 0, w*6], r),
])
axon = Growth(soma, rgn, 0.00025e18,
    balancing_factor = .0,
    extension_angle = np.pi / 6,
    extension_distance = 60e-6,
    bifurcation_angle = np.pi / 3,
    bifurcation_distance = 40e-6,
    extend_before_bifurcate = True,
    maximum_segment_length = 30e-6,)
for x in soma:
    x.insert_mechanism(hh.Leak)
    x.insert_mechanism(hh.VoltageGatedSodiumChannel)
    x.insert_mechanism(hh.VoltageGatedPotassiumChannel)
for x in axon.segments:
    x.diameter = .5e-6
    x.insert_mechanism(hh.Leak)
    x.insert_mechanism(hh.VoltageGatedSodiumChannel)
    x.insert_mechanism(hh.VoltageGatedPotassiumChannel)
model = Model(.1e-3, soma, reactions=(), species=(), conductances=())
print("Number of segments:", len(model))
# Wait for the system to settle into its steady state.
for _ in range(int(20e-3 / model.time_step)):
    model.advance()
# Run the simulation.
stimulus = (1e-3, 3e-3) # Beginning and end times of stimulation.
stimulus_current = 2e-9 # Stimulation magnitude.
frames_dir = tempfile.TemporaryDirectory()
frames = []
skip = 0 # Skip rendering this many frames to make this program run faster.
res = 4 # Integer factor to control image resolution: lower to run faster.
for t in range(int(100e-3 / model.time_step) + 1):
    if stimulus[0] <= t * model.time_step < stimulus[1]:
        soma[0].inject_current(stimulus_current)
    model.advance()
    v = (model.electrics.voltages - min_v) / (max_v - min_v)
    if t % (skip+1) == 0:
        # Render a frame.
        colors = [(f, 0, 1-f) for f in v]
        frames.append(os.path.join(frames_dir.name, str(t)+".png"))
        model.draw_image(frames[-1], (int(640*res), int(480*res)),
                (0, 0, -270e-6), (0,0,0), colors)
        # Shrink the image to reduce the filesize for github's limit.
        img = Image.open(frames[-1])
        img = img.resize((int(640*1.5), int(480*1.5)), resample=Image.LANCZOS)
        # Draw the elapsed time since onset of stimulation onto the frame.
        draw = ImageDraw.Draw(img)
        text = "{:6.2f} milliseconds since onset of stimulus.".format(
                (t * model.time_step - stimulus[0]) * 1e3)
        draw.text((5, 5), text, (0, 0, 0))
        img.save(frames[-1])
        print("Rendered frame at t =" + text)
    # Terminate the model after it reaches a post stimulation steady state.
    if t * model.time_step > stimulus[1] + 10e-3:
        if all(v < 0.1):
            break
# Save into a GIF file that loops forever.
frames = [Image.open(i) for i in frames] # Load all of the frames.
gif_file = os.path.abspath('AP_Propagation.gif')
frames[0].save(gif_file, format='GIF', append_images=frames[1:], save_all=True,
        duration=int((skip+1) * model.time_step * 1e6), # Milliseconds per frame.
        optimize=True, quality=0,
        loop=0,) # Loop forever.
print("Saved output to: \"%s\""%gif_file)

""" Model of an action potential propagating through a large axonal arbor. """

import numpy as np
from neuwon import *
from neuwon.regions import *
from neuwon.growth import *
from neuwon.analysis import Animation
min_v = -90e-3
max_v = +70e-3
from graph_algorithms import depth_first_traversal as dft

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
    x.insert_mechanism("hh")
for x in axon.segments:
    x.diameter = .5e-6
    x.insert_mechanism("hh")
model = Model(.1e-3, soma, reactions=(), species=())
print("Number of segments:", len(model))
r = 4 # Integer factor to control image resolution: lower to run faster.
skip = 0 # Skip rendering this many frames to make this program run faster.
video_camera = Animation(model,
        resolution = (int(640*r), int(480*r)),
        skip = skip,
        camera_coordinates = (0, 0, -270e-6))
# Wait for the system to settle into its steady state.
for _ in range(int(20e-3 / model.time_step)):
    model.advance()
# Run the simulation.
stimulus_tick = int(round(1e-3 / model.time_step))
stimulus_current = 2e-9
for tick in range(int(100e-3 / model.time_step) + 1):
    if tick == stimulus_tick:
        soma[0].inject_current(stimulus_current, 2e-3)
    model.advance()
    v = ((model.electrics.voltages - min_v) / (max_v - min_v)).clip(0, 1).get()
    t = (tick - stimulus_tick) * model.time_step * 1e3
    video_camera.add_frame(
            colors = [(x, 0, 1-x) for x in v],
            shrink = 0.3, # Shrink the image for githubs filesize limit.
            text = "{:6.2f} milliseconds since onset of stimulus.".format(t))
    print("t = %g"%t)
    # Terminate the model after it reaches a post stimulation steady state.
    if t > 3 and all(model.electrics.voltages.get() < -50e-3):
        break
video_camera.save('AP_Propagation.gif')

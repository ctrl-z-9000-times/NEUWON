""" Model of an action potential propagating through a large axonal arbor. """

import numpy as np
from neuwon.model import *
from neuwon.regions import *
from neuwon.growth import *
min_v = -90e-3
max_v = +70e-3

time_step= 100e-6
model = Model(time_step)
model.add_species("na")
model.add_species("k")
model.add_species(Species("L", transmembrane = True, reversal_potential = -54.3e-3,))
model.add_reaction("hh")
hh = model.get_reaction("hh")

# Setup a neuron and its axonal arbor.
r = 30e-6
w = 100e-6
soma = model.create_segment(None, [-w, -w-r/2, 0], 10e-6)
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
axon = Growth(model, soma, rgn, 0.00025e18,
    balancing_factor = .0,
    extension_angle = np.pi / 6,
    extension_distance = 60e-6,
    bifurcation_angle = np.pi / 3,
    bifurcation_distance = 40e-6,
    extend_before_bifurcate = True,
    maximum_segment_length = 30e-6,
    diameter = .5e-6)
locations = [x.index for x in soma + axon.segments]
hh.new_instances(model.db, locations)
print("Number of segments:", len(model))

# Run with no inputs until it reaches a steady state.
for _ in range(int(20e-3 / time_step)):
    model.advance()

# Run the simulation.
stimulus_tick = int(round(1e-3 / time_step))
stimulus_current = 2e-9
tick = 0
t = None
def text_function(database_access):
    return "{:6.2f} milliseconds since onset of stimulus.".format(t)
def color_function(database_access):
    v = ((database_access("membrane/voltages") - min_v) / (max_v - min_v)).clip(0, 1).get()
    return [(x, 0, 1-x) for x in v]
r = 4 # Integer factor to control image resolution: lower to run faster.
video_camera = Animation(model, color_function, text_function,
        resolution = (int(640*r), int(480*r)),
        skip = 0, # Skip rendering this many frames to make this program run faster.
        scale = 0.3, # Shrink the image for githubs filesize limit.
        camera_coordinates = (0, 0, -270e-6))
for tick in range(int(100e-3 / time_step) + 1):
    if tick == stimulus_tick:
        soma[0].inject_current(stimulus_current, 2e-3)
    t = (tick - stimulus_tick) * time_step * 1e3
    model.advance()
    print("t = %g"%t)
    # Terminate the model after it reaches a post stimulation steady state.
    if t > 3 and all(model.access("membrane/voltages").get() < -50e-3):
        break
video_camera.save('AP_Propagation.gif')

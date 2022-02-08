from math import pi
import argparse
import numpy as np
import random

from .load_mnist import load_mnist
from neuwon import *
from neuwon.gui.viewport import Viewport

# from htm.bindings.algorithms import Classifier
# from htm.bindings.sdr import SDR, Metrics

spacing = 10 # microns


default_parameters = {
    'rxd_parameters': {
        'time_step': .1,
    },
    'species': {
        'na': {'reversal_potential': +60,},
        'k': {'reversal_potential': -88,},
        'l': {'reversal_potential': -54.3,},
    },
    'mechanisms': {
        'hh': './nmodl_library/hh.mod',
    },
    'regions': {
        'input_layer': ("Rectangle", [0,0,0], [28*spacing, .1,        28*spacing]),
        'main_layer':  ("Rectangle", [0,0,0], [28*spacing, 2*spacing, 28*spacing]),
    },
    'neurons': {
        'input_neuron': (
            {
                'segment_type': 'input_soma',
                'region': 'input_layer',
                'number': 28 * 28 * 1,
                'diameter': 3,
                'mechanisms': {'hh'}
            },
            {
                'segment_type': 'input_axon',
                'region': 'main_layer',
                'diameter': .5,
                'morphology': {
                    'carrier_point_density': 0.000025,
                    'balancing_factor': .0,
                    'extension_angle': pi / 6,
                    'extension_distance': 60,
                    'bifurcation_angle': pi / 3,
                    'bifurcation_distance': 40,
                    'extend_before_bifurcate': True,
                    'maximum_segment_length': 30,},
                'mechanisms': {'hh'}
            },
        )
    },
    'synapses': {},
}

def main(parameters=default_parameters, verbose=True):
    model = Model(**parameters)
    train_data, test_data = load_mnist()

    # Organize all of the sensory input neurons into a grid.
    input_terminals = [[[] for row in range(28)] for col in range(28)]
    for n in model.neurons['input_neuron']:
        x, y, z = n.root.coordinates
        x = min(int(x / spacing), 28-1)
        z = min(int(z / spacing), 28-1)
        input_terminals[x][z].append(n)
    def apply_sensory_input(image):
        image = (image >= 100) # Encode the image into binary map.
        for x, y in zip(*np.nonzero(np.squeeze(image))):
            for n in input_terminals[x][y]:
                n.root.inject_current(1)
    # Setup the GUI.
    if verbose:
        view = Viewport(camera_position=[0,0,400])
        view.set_scene(model)
    def update_viewport():
        if not verbose: return
        max_v   = +60
        min_v   = -88
        voltage = model.get_database().get_data('Segment.voltage')
        voltage = ((voltage - min_v) / (max_v - min_v)).clip(0, 1)
        colors  = [(x, 0, 1-x) for x in voltage]
        view.tick(colors)
    
    # Training Loop
    for img, lbl in train_data[:1000]:
        print("Label:", lbl)
        model.check()
        apply_sensory_input(img)
        model.check()
        for _ in range(round(10 / model.get_time_step())):
            model.advance()
            model.check()
            update_viewport()
            print(model.clock())

        # sdrc.learn(activity, lbl)


    # Testing Loop
    score = 0
    for img, lbl in test_data[:0]:
        activity = run(img)
        if lbl == np.argmax(sdrc.infer(activity)):
            score += 1
    print('Score: %g %', 100 * score / len(test_data))
    return 

if __name__ == "__main__":
    main()

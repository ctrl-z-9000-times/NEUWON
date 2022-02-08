from math import pi
import argparse
import numpy as np
import random

from .load_mnist import load_mnist
from neuwon import Model, TimeSeries
from neuwon.gui import Viewport

from htm import SDR, Metrics
from htm.algorithms import Classifier

spacing = 10 # Microns
stim = 1e-9 # Amps
period = 10 # Milliseconds

default_parameters = {
    'rxd_parameters': {
        'time_step': .1,
        'celsius': 6.3,
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
        'input_layer': ("Rectangle", [0,0,0], [28*spacing, .1,         28*spacing]),
        'main_layer':  ("Rectangle", [0,0,0], [28*spacing, 10*spacing, 28*spacing]),
    },
    'neurons': {
        'input_neuron': [
            {
                'segment_type': 'input_soma',
                'region': 'input_layer',
                'number': 28 * 28 * 1,
                'diameter': 3,
                'mechanisms': {'hh': 1.0}
            },
            {
                'segment_type': 'input_axon',
                'region': 'main_layer',
                'diameter': .5,
                'morphology': {
                    'carrier_point_density': 0.0001,
                    'balancing_factor': .0,
                    'extension_angle': pi / 6,
                    'extension_distance': 60,
                    'bifurcation_angle': pi / 3,
                    'bifurcation_distance': 40,
                    'extend_before_bifurcate': True,
                    'maximum_segment_length': 30,},
                'mechanisms': {'hh': 1.0}
            },
        ],
        # 'excit_neuron': [],
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
        input_terminals[x][27-z].append(n)
    def apply_sensory_input(image):
        image = (image >= 100) # Encode the image into binary.
        for x, y in zip(*np.nonzero(np.squeeze(image))):
            for n in input_terminals[x][y]:
                n.root.inject_current(stim)
    # Setup the output timeseries buffers and classifier.
    outputs = []
    for n in model.neurons['input_neuron']: # DEBUGGING, record from the input neurons instead of the exictatory neurons.
        outputs.append(TimeSeries().record(n.root, 'voltage', discard_after=period))
    def get_activity():
        q = SDR(len(outputs))
        q.dense = [any(v > 0 for v in ts.get_data()) for ts in outputs]
        return q
    sdrc = Classifier()
    sdrc.learn(SDR(len(outputs)), 9)
    # Setup the GUI.
    if verbose:
        view = Viewport(camera_position=[14*spacing,28*spacing,14*spacing], fps=60)
        view.set_scene(model)
    def update_viewport():
        if not verbose: return
        max_v   = +60
        min_v   = -88
        voltage = model.get_database().get_data('Segment.voltage')
        voltage = ((voltage - min_v) / (max_v - min_v)).clip(0, 1)
        colors  = [(x, 0, 1-x) for x in voltage]
        view.tick(colors)
    # Training Loop.
    for img, lbl in train_data:
        apply_sensory_input(img)
        for _ in range(round(period / model.get_time_step())):
            model.advance()
            update_viewport()
        activity = get_activity()
        if verbose:
            print("Label:", lbl, "Output:", np.argmax(sdrc.infer(activity)))
        sdrc.learn(activity, lbl)
    # Testing Loop.
    score = 0
    for img, lbl in test_data:
        apply_sensory_input(img)
        for _ in range(round(period / model.get_time_step())):
            model.advance()
        activity = get_activity()
        if lbl == np.argmax(sdrc.infer(activity)):
            score += 1
    score = score / len(test_data)
    print('Score: %g%%'%(100 * score))
    return score

if __name__ == "__main__":
    main()

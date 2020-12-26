import numpy as np
import scipy.spatial
import random
from abc import ABC, abstractmethod
from neuwon import *
import itertools
import heapq as q

class GrowSynapses:
    def __init__(self, axons, dendrites, pre_gap_post, diameter, num_synapses):
        pre_len, gap_len, post_len = pre_gap_post
        f_pre = pre_len / sum(pre_gap_post)
        f_post = post_len / sum(pre_gap_post)
        self.presynaptic_segments = []
        self.postsynaptic_segments = []
        # Find all possible synapses.
        pre = scipy.spatial.cKDTree([x.coordinates for x in axons])
        post = scipy.spatial.cKDTree([x.coordinates for x in dendrites])
        results = pre.query_ball_tree(post, sum(pre_gap_post))
        results = itertools.chain.from_iterable(
            ((pre, post) for post in inner) for pre, inner in enumerate(results))
        # Select some synapses and make them.
        for pre, post in random.sample(flat_results, num_synapses):
            pre = axons[pre]
            post = dendrites[post]
            if pre_len == 0:
                self.presynaptic_segments.append(pre)
            else:
                x = f_pre * np.array(pre.coordinates) + (1 - f_pre) * np.array(post.coordinates)
                self.presynaptic_segments.append(Segment(x, diameter, pre))
            if post_len == 0:
                self.postsynaptic_segments.append(post)
            else:
                x = f_post * np.array(post.coordinates) + (1 - f_post) * np.array(pre.coordinates)
                self.postsynaptic_segments.append(Segment(x, diameter, post))

if __name__ == "__main__":
    from neuwon import *
    from graph_algorithms import depth_first_traversal as dft
    w = 120e-6
    h = 120e-6
    d = 20e-6
    rgn1 = Rectangle([-w/2,-h/2,-d/2], [w/2,h/2,d/2])
    rgn2 = Sphere([0,0,0], w/2)
    rgn = Intersection([rgn1, rgn2])
    soma = [Segment([0,0,0],    20e-6, None),
            Segment([-w/8,0,0], 20e-6, None),
            Segment([w/8,0,0],  20e-6, None),
            Segment([-w*2/8,0,0], 20e-6, None),
            Segment([w*2/8,0,0],  20e-6, None),
            Segment([-w*3/8,0,0], 20e-6, None),
            Segment([w*3/8,0,0],  20e-6, None),
            ]
    dendrites = TREESwithROOTS(soma, rgn, 0.0025e18,
        balancing_factor = .7,
        extension_angle = np.pi / 3,
        extension_distance = 30e-6,
        bifurcation_angle = np.pi / 2,
        bifurcation_distance = 15e-6,
        extend_before_bifurcate = False)
    for x in soma:
        x.set_diameter()
    model = Model(1e-6, soma,
                reactions=(),
                species=(),
                conductances=())

    colors = [(1, 1, 1) for _ in range(len(model))]
    for x in dft(soma[0], lambda x:x.children):
        colors[x.location] = (1, 0, 0)
    for x in dft(soma[1], lambda x:x.children):
        colors[x.location] = (0, 1, 0)
    for x in dft(soma[2], lambda x:x.children):
        colors[x.location] = (0, 0, 1)
    model.to_povray("test.png", (640*4, 480*4), (0, 0, -180e-6), (0,0,0), colors)

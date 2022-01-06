from neuwon.brains.growth import growth_algorithm
from neuwon.brains.regions import Sphere
from neuwon.rxd.rxd_model import RxD_Model

def test_growth_algorithm():
    # Make a very simple tree, and verify that it creates at least one segment.
    m = RxD_Model(.1)
    root = m.Neuron([0,0,0], 10).root
    region = Sphere([0,10,0], 10)

    segments = growth_algorithm([root], region, 0.2, segment_parameters={'diameter':1})
    assert len(segments) > 3

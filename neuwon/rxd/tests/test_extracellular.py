from neuwon.database import Database
from neuwon.rxd.extracellular import Extracellular
from neuwon.rxd.neuron import Neuron
import pytest


def test_simple():
    db = Database()
    Neuron._initialize(db)
    ECS = Extracellular._initialize(db, maximum_distance=10)
    x1 = ECS([0,0,0], 1)
    x2 = ECS([0,0,2], 2)
    x3 = ECS([0,0,5], .5)

    ECS._clean()

    n = x2.neighbors
    assert x1 in n
    assert x3 in n

    assert x2.voronoi_volume == pytest.approx( ((2 * 10) ** 2) * (5 / 2) )

    # Check neighbor distances.
    distances = list(zip(*x2.neighbor_distances))
    for n, d in distances:
        if   n == x1: assert d == pytest.approx(2)
        elif n == x3: assert d == pytest.approx(3)
        else: assert False

    # Check neighbor border areas.
    border_areas = list(zip(*x2.neighbor_border_areas))
    for n, a in border_areas:
        assert a == pytest.approx( (2 * 10) ** 2 )


@pytest.mark.skip
def test_diffusion():
    """
    Model a very simple diffusion problem at different resolutions
    of time & space and verify the accuracy of the results.
    """
    1/0 # TODO!


@pytest.mark.skip
def test_conservation():
    """
    Verify that mass is conserved.

    Model a species diffusing through a random artifical system. Modify the
    system while the test is running, by randomly adding and removing points.
    """
    1/0 # TODO!


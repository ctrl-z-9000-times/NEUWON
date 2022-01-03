from neuwon.database import Database
from neuwon.rxd.extracellular import Extracellular
from neuwon.rxd.neurons import Neuron
import pytest

def test_simple():
    db = Database()
    Neuron._initialize(db)
    ECS = Extracellular._initialize(db)
    x1 = ECS([0,0,0], 1)
    x2 = ECS([0,0,2], 2)
    x3 = ECS([0,0,5], .5)

    ECS._clean()

    n = x2.neighbors
    assert x1 in n
    assert x3 in n

@pytest.mark.skip
def test_diffusion():
    """
    Model a very simple diffusion problem at different resolutions
    of time & space and verify accurate results.
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

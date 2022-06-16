from neuwon.gui.viewport import Viewport
import neuwon
import pytest
import time

def test_viewport():
    m = neuwon.Model()
    v = Viewport((400,400))
    v.set_model(m)

    m.Neuron([0, 0, -3], 1)
    m.database.sort()
    v.set_model(m)

    time.sleep(2)

    assert v._process.is_alive()

    del v

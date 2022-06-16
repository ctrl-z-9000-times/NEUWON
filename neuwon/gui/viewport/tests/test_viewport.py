from neuwon.gui.viewport import Viewport
import neuwon
import pytest
import time

def test_viewport():
    m = neuwon.Model()
    v = Viewport((400,400))
    v.set_model(m)

    m.Neuron([1,2,3], 4)
    m.database.sort()
    v.set_model(m)

    time.sleep(2)

    del v

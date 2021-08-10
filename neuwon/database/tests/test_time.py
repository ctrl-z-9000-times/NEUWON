import random
from neuwon.database.time import *

def test_clock():
    c = Clock(1)
    c = Clock(.1)
    c = Clock(1/40, 'ms')
    assert c.time() == 0

    for x in range(40): c.tick()
    assert c() == 1
    c.reset()
    assert c() == 0

    c.set_time(99)
    for x in range(40): c.tick()
    assert c.clock() == 100

# def test_time_series_buffers():
#     c = Clock(.1)
#     b = TimeSeriesBuffer(c)

#     b.record()

    # TODO: Test record & play back




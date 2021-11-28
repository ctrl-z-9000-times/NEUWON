import numpy as np
from neuwon.database import Database
from neuwon.synapses.Mongillo2008 import Mongillo2008

def test_basic():
    db = Database()
    syn_data = db.add_class('MySynapse', Mongillo2008)
    MySynapse = syn_data.get_instance_type()
    Mongillo2008.initialize(syn_data,
            minimum_utilization = 0.2,
            utilization_decay = 1000.0,
            resource_recovery = 5.0,)

    x = MySynapse()
    db.check()

    assert x.compute(1.0) == x.minimum_utilization

    print("\nRESET\n")
    x.reset()
    for t in np.linspace(0, 100, 4):
        print('TIME', t)
        print('release', x.compute(t))
        print('utilization', x.utilization)
        print('resources', x.resources)
        print()

    db.check()
    x.reset()
    assert x.compute(1.0) == 0.2
    assert x.compute(2.0) > 0.2 # facilitation

    for t in range(3,20):
        x.compute(float(t))
    assert x.compute(2.0) < 0.2 # depression

import numpy as np
from neuwon.database import Database
from neuwon.synapses.stp import STP

def test_basic():
    db = Database()
    syn_data = db.add_class('MySynapse', STP)
    MySynapse = syn_data.get_instance_type()
    STP.initialize(syn_data,
            minimum_utilization = 0.2,
            utilization_decay = 1000.0,
            resource_recovery = 5.0,)

    x = MySynapse()
    db.check()

    assert x.activate_presynapses(1.0) == x.minimum_utilization

    print("\nRESET\n")
    x.reset_presynapses()
    for t in np.linspace(0, 100, 4):
        print('TIME', t)
        print('release', x.activate_presynapses(t))
        print('utilization', x.utilization)
        print('resources', x.resources)
        print()

    db.check()
    x.reset_presynapses()
    assert x.activate_presynapses(1.0) == 0.2
    assert x.activate_presynapses(2.0) > 0.2 # facilitation

    for t in range(3,20):
        x.activate_presynapses(float(t))
    assert x.activate_presynapses(2.0) < 0.2 # depression

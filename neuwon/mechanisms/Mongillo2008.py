""" Presynapses

Synaptic Theory of Working Memory
Gianluigi Mongillo, et al.
Science 319, 1543 (2008);
DOI: 10.1126/science.1150769
"""

import numpy as np
import cupy as cp
import numba
import math
from neuwon import Mechanism, Real

class Presynapses(Mechanism):
    def __init__(self, transmitter, minimum_utilization, utilization_decay, resource_recovery):
        self.minimum_utilization = float(minimum_utilization)
        self.utilization_decay = float(utilization_decay)
        self.resource_recovery = float(resource_recovery)
        self.transmitter = str(transmitter)

    def instance_dtype(self):
        # TODO: Make a structure of arrays by returning a dictionary. Only half
        # of the data is used every cycle and the other half is very rarely used
        # so this should be factored into two arrays.
        return np.dtype([
                ("last_update", np.int64),
                ("utilization", Real),
                ("resources", Real),
                ("strength", Real),
                ("above_threshold", np.bool),
        ])

    def new_instance(self, time_step, location, geometry, strength):
        assert(strength >= 0)
        instance = np.empty(1, dtype=self.instance_dtype())
        instance["strength"]        = float(strength) / geometry.extra_volumes[location]
        instance["last_update"]     = 0
        instance["utilization"]     = self.minimum_utilization
        instance["resources"]       = 1
        instance["above_threshold"] = True
        return instance

    def advance(self, locations, instances, time_step, reaction_inputs, reaction_outputs):
        release_rate = getattr(reaction_outputs.extra, self.transmitter)
        release_rate -= getattr(reaction_inputs.extra, self.transmitter) / self.resource_recovery
        triggered = cp.empty(len(instances), dtype=np.bool)
        threads = 128
        blocks = (instances.shape[0] + (threads - 1)) // threads
        _detect_presyn_ap[blocks,threads](locations, instances, reaction_inputs.v, triggered)
        triggered = cp.nonzero(triggered)[0]
        if len(triggered) == 0: return
        blocks = (triggered.size + (threads - 1)) // threads
        _release[blocks,threads](locations, instances, time_step, release_rate,
                self.minimum_utilization, self.utilization_decay, self.resource_recovery)

@numba.cuda.jit()
def _detect_presyn_ap(locations, instances, voltages, triggered):
    index = numba.cuda.grid(1)
    if index >= instances.shape[0]:
        return
    location = locations[index]
    instance = instances[index]
    instance["last_update"] += 1
    # Detect rising edge of voltage spike.
    above_threshold = voltages[location] > 0e-3
    triggered[index] = above_threshold and not instance["above_threshold"]
    instance["above_threshold"] = above_threshold

@numba.cuda.jit()
def _release(locations, instances, time_step, release_rate,
        minimum_utilization, utilization_decay, resource_recovery):
    index = numba.cuda.grid(1)
    if index >= instances.shape[0]:
        return
    location = locations[index]
    instance = instances[index]
    # Update the state of the presynapse over the course of the elapsed time.
    elapsed_time = instance["last_update"] * time_step
    utilization_alpha = math.exp(-elapsed_time / utilization_decay)
    resources_alpha   = math.exp(-elapsed_time / resource_recovery)
    instance["utilization"] += utilization_alpha * (minimum_utilization - instance["utilization"])
    instance["resources"]   += resources_alpha * (1.0 - instance["resources"])
    # Compute the immediate AP induced release.
    release_percent = instance["utilization"] * instance["resources"]
    instance["utilization"] += minimum_utilization * (1.0 - instance["utilization"])
    instance["resources"]   -= release_percent
    delta_concentration = instance["strength"] * release_percent
    release_rate[location] += delta_concentration / time_step

""" Presynapses

Synaptic Theory of Working Memory
Gianluigi Mongillo, et al.
Science 319, 1543 (2008);
DOI: 10.1126/science.1150769
"""

import math
from neuwon import Mechanism, Real

class PresynapseConfiguration:
    def __init__(self, transmitter, minimum_utilization, utilization_decay, resource_recovery):
        self.minimum_utilization = float(minimum_utilization)
        self.utilization_decay = float(utilization_decay)
        self.resource_recovery = float(resource_recovery)
        self.transmitter = str(transmitter)

class Presynapse(Mechanism):
    species = []
    conductances = []
    def __init__(self, time_step, location, geometry, config, strength=None):
        self.time_step = time_step
        self.location = location
        self.config = config
        self.strength = float(strength) / geometry.extra_volumes[location]
        assert(isinstance(self.config, PresynapseConfiguration))
        assert(self.strength >= 0)
        self.reset()

    def reset(self):
        self.last_update = 0
        self.utilization = self.config.minimum_utilization
        self.resources   = 1
        self.triggered   = True

    def advance(self, reaction_inputs, reaction_outputs):
        concentration = getattr(reaction_inputs.extra, self.config.transmitter)[self.location]
        release_rate = getattr(reaction_outputs.extra, self.config.transmitter)
        release_rate[self.location] -= self.strength / self.config.resource_recovery
        self.last_update += 1
        v = reaction_inputs.v[self.location]
        # Detect rising edge of voltage spike.
        triggered = v > 0
        if not triggered or self.triggered:
            self.triggered = triggered
            return
        self.triggered = triggered
        # Update the state of the presynapse over the course of the elapsed time.
        elapsed_time = self.last_update * self.time_step
        utilization_alpha = math.exp(-elapsed_time / self.config.utilization_decay)
        resources_alpha   = math.exp(-elapsed_time / self.config.resource_recovery)
        self.utilization += utilization_alpha * (self.config.minimum_utilization - self.utilization)
        self.resources   += resources_alpha * (1.0 - self.resources)
        # Compute the immediate AP induced release.
        release_percent    = self.utilization * self.resources
        self.utilization  += self.config.minimum_utilization * (1.0 - self.utilization)
        self.resources    -= release_percent
        delta_concentration = self.strength * release_percent
        release_rate[self.location] += delta_concentration / self.time_step

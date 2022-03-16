import math
from neuwon.database import Compute, Real, epsilon

class STP:
    """ Model of Short-Term-Plasticity in presynapses.

    Reference:
        Synaptic theory of working memory.
        Mongillo, Barak, Tsodyks. 2008
        doi: 10.1126/science.1150769.
    """
    __slots__ = ()

    def __init__(self, *
                minimum_utilization,
                utilization_decay,
                resource_recovery):
        self.minimum_utilization = float(minimum_utilization)
        self.utilization_decay   = float(utilization_decay)
        self.resource_recovery   = float(resource_recovery)

    @classmethod
    def initialize(db_class):
        db_class.add_class_attribute('minimum_utilization',
                initial_value = self.minimum_utilization,
                valid_range = (0.0, 1.0),
                doc = "")
        db_class.add_class_attribute('utilization_decay',
                initial_value = self.utilization_decay,
                valid_range = (epsilon, math.inf),
                doc = "")
        db_class.add_class_attribute('resource_recovery',
                initial_value = self.resource_recovery,
                valid_range = (epsilon, math.inf),
                doc = "")
        db_class.add_attribute('last_update',
                initial_value = 0.0,
                valid_range = (0.0, math.inf),
                doc = "")
        db_class.add_attribute('resources',
                initial_value = 1.0,
                valid_range = (0.0, 1.0),
                doc = "")
        db_class.add_attribute('utilization',
                initial_value = self.minimum_utilization,
                valid_range = (0.0, 1.0),
                doc = "")

    def reset_presynapses(cls):
        db_class = cls.get_database_class()
        db_class.get_data('last_update').fill(0.0)
        db_class.get_data('resources'  ).fill(1.0)
        db_class.get_data('utilization').fill(cls.minimum_utilization)

    @Compute
    def activate_presynapses(self, timestamp) -> Real:
        # Unpack the synapse parameters & data.
        utilization         = self.utilization
        resources           = self.resources
        minimum_utilization = self.minimum_utilization
        # Compute temporal effects, since last activation of this synapse.
        elapsed_time = timestamp - self.last_update
        utilization_alpha = 1.0 - math.exp( -elapsed_time / self.utilization_decay)
        resources_alpha   = 1.0 - math.exp( -elapsed_time / self.resource_recovery)
        utilization += utilization_alpha * (minimum_utilization - utilization)
        resources   += resources_alpha * (1.0 - resources)
        # Compute fractional neurotransmitter release.
        syn_release   = utilization * resources
        utilization  += minimum_utilization * (1.0 - utilization)
        resources    -= syn_release
        # Update synapse data and return.
        self.last_update = timestamp
        self.utilization = utilization
        self.resources   = resources
        return syn_release

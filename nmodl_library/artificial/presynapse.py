import numpy as np
from neuwon.rxd import Mechanism
from neuwon.database import Compute, Real

class Presynapse(Mechanism):
    __slots__ = ()

    @classmethod
    def get_parameters(cls):
        return {'neurotransmitter': "<Species>"
                'initial_strength': 1e-3,
                'minimum_utilization': 0.2,
                'utilization_decay': 5,
                'resource_recovery': 20,}

    @classmethod
    def initialize(cls, model, mechanism_name, **kwargs):
        db = model.get_database()
        db_class = db.add_class(mechanism_name, cls, sort_key='segment')

        db_class.add_attribute('magnitude', kwargs['initial_strength'])

        db_class.add_attribute('segment', dtype='Segment')
        db_class.add_attribute('outside', dtype='Outside')
        db_class.add_attribute('_AP', dtype=np.bool)

        db_class.add_class_attribute('minimum_utilization', kwargs['minimum_utilization'],
                valid_range=(0, 1))
        db_class.add_class_attribute('utilization_decay', kwargs['utilization_decay'],
                units='ms',
                doc='Time period of exponential decay',
                valid_range=(0, None))
        db_class.add_class_attribute('resource_recovery', kwargs['resource_recovery'],
                units='ms',
                doc='Time period of exponential recovery',
                valid_range=(0, None))

        db_class.add_attribute('utilization', kwargs['minimum_utilization'])
        db_class.add_attribute('resources', 1)
        db_class.add_attribute('last_update', 0)

        cls = db_class.get_instance_type()
        cls.neurotransmitter = db.get_data('Outside.' + kwargs['neurotransmitter'])
        return cls

    def __init__(self, segment, magnitude, outside):
        self.segment = segment
        self.magnitude *= magnitude
        self.outside = outside

    @classmethod
    def reset(cls, timestamp):
        1/0

    @classmethod
    def advance(cls):
        db_class = cls.get_database_class()
        active  = np.nonzero(cls._AP_detected())[0]
        release = cls._activate(active, timestamp)
        outside = db_class.get_data('outside')
        transmitter = cls.neurotransmitter.get_data()
        transmitter[outside] += release

    @Compute
    def _AP_detected(self) -> np.bool:
        AP   = (self.segment.voltage > 20)
        edge = AP and not self._AP
        self._AP = AP
        return edge

    @Compute
    def _activate(self, timestamp) -> Real:
        # Unpack the synapse parameters & state.
        utilization  = self.utilization
        resources    = self.resources
        minimum_utilization = self.minimum_utilization
        # Compute temporal effects, since last activation of this synapse.
        elapsed_time = timestamp - self.last_update
        utilization_alpha = math.exp( -elapsed_time / self.utilization_decay)
        resources_alpha   = math.exp( -elapsed_time / self.resource_recovery)
        utilization += utilization_alpha * (minimum_utilization - utilization)
        resources   += resources_alpha * (1.0 - resources)
        # Compute fractional neurotransmitter release.
        release       = utilization * resources;
        utilization  += minimum_utilization * (1.0 - utilization);
        resources    -= release;
        # Update synapse data and return.
        self.last_update = timestamp
        self.utilization = utilization
        self.resources   = resources
        return release * self.magnitude / self.outside.volume

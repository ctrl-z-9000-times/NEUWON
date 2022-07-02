"""
These classes transform the inputs between real world values and indexes which
are suitable for use with a look up table.
"""

import numpy as np

class Input:
    """ Abstract base class. """
    def __init__(self, name, minimum, maximum, initial=None):
        self.name       = str(name)
        self.minimum    = float(minimum)
        self.maximum    = float(maximum)
        self.range      = self.maximum - self.minimum
        assert self.minimum < self.maximum
        if initial is None:
            self.initial = self.minimum
        else:
            self.initial = float(initial)
            assert self.minimum <= self.initial < self.maximum

    def __repr__(self):
        return f"lti_sim.{type(self).__name__}({self.name}, {self.minimum}, {self.maximum}, initial={self.initial})"

    def set_num_buckets(self, num_buckets):
        self.num_buckets = int(num_buckets)
        assert self.num_buckets >= 1

    def get_bucket_value(self, input_value):
        raise NotImplementedError(type(self))

    def get_input_value(self, bucket_value):
        raise NotImplementedError(type(self))

    def get_bucket_location(self, input_value):
        """ Returns pair of (bucket_index, location_within_bucket) """
        location = self.get_bucket_value(input_value)
        bucket_index = int(location)
        if bucket_index < 0:
            bucket_index = 0
            location     = 0.0
        elif bucket_index >= self.num_buckets:
            bucket_index = self.num_buckets - 1
            location     = 1.0
        else:
            location -= bucket_index
        return (bucket_index, location)

    def sample_space(self, number):
        """ Note: this includes the endpoint. """
        number = int(number)
        samples = np.linspace(0, self.num_buckets, number, endpoint=True)
        for sample_index, bucket_location in enumerate(samples):
            samples[sample_index] = self.get_input_value(bucket_location)
        # Fix any numeric instability.
        samples[0]  = self.minimum
        samples[-1] = self.maximum
        return samples

    def bisect_inputs(self, input_value1, input_value2):
        b1  = self.get_bucket_value(input_value1)
        b2  = self.get_bucket_value(input_value2)
        mid = 0.5 * (b1 + b2)
        return self.get_input_value(mid)

    def random(self, size=1, dtype=np.float64, array_module=np):
        bucket_values = array_module.random.uniform(0, self.num_buckets, size)
        input_values = self.get_input_value(bucket_values)
        return array_module.array(input_values, dtype=dtype)

class LinearInput(Input):
    """ """
    def set_num_buckets(self, num_buckets):
        super().set_num_buckets(num_buckets)
        self.bucket_frq     = self.num_buckets / self.range
        self.bucket_width   = self.range / self.num_buckets

    def get_bucket_value(self, input_value):
        return (input_value - self.minimum) * self.bucket_frq

    def get_input_value(self, bucket_value):
        return self.minimum + bucket_value * self.bucket_width

class LogarithmicInput(Input):
    """ """
    def __init__(self, name, minimum, maximum, initial=None):
        super().__init__(name, minimum, maximum, initial)
        assert self.minimum == 0.0, 'Logarithmic inputs must have minimum value of 0.'

    def set_num_buckets(self, num_buckets, scale=None):
        super().set_num_buckets(num_buckets)
        if scale is None:
            assert hasattr(self, "scale")
        else:
            self.scale      = float(scale)
        self.log2_minimum   = np.log2(self.minimum + self.scale)
        self.log2_maximum   = np.log2(self.maximum + self.scale)
        self.log2_range     = self.log2_maximum - self.log2_minimum
        self.bucket_frq     = self.num_buckets / self.log2_range
        self.bucket_width   = self.log2_range / self.num_buckets

    def get_bucket_value(self, input_value):
        log2_value = np.log2(input_value + self.scale)
        return (log2_value - self.log2_minimum) * self.bucket_frq

    def get_input_value(self, bucket_value):
        log2_value = self.log2_minimum + bucket_value * self.bucket_width
        return 2.0 ** log2_value - self.scale

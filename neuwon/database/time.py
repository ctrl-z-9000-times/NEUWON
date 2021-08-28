from collections.abc import Callable, Iterable, Mapping
import neuwon.database
import collections
import matplotlib.pyplot
import numpy as np
import scipy.interpolate
import weakref

def _weakref_wrapper(method):
    # Note: Don't use weakrefs if the callback writes to the database or any
    # other globally visible state.
    method_ref = weakref.WeakMethod(method)
    def call_if_able():
        method = method_ref()
        if method is not None:
            return method()
    return call_if_able

class Clock:
    """ Clock and notification system. """
    def __init__(self, database:neuwon.database.Database, tick_period:float, units:str=""):
        """
        Argument tick_period is a duration of time.

        Argument units is the physical units for 'tick_period'. Optional.
        """
        assert isinstance(database, neuwon.database.Database)
        if database.clock is not None:
            raise RuntimeError("Database already has a clock.")
        database.clock = self
        self.database = database
        self.dt = float(tick_period)
        self.ticks = 0
        self.units = str(units)
        self.callbacks = []

    def get_time(self) -> float:
        """ Returns the current time. """
        return self.ticks * self.dt

    def __call__(self) -> float:
        """ Returns the current time. """
        return self.get_time()

    def get_tick_period(self) -> float:
        """ Returns the duration of each tick. """
        return self.dt

    def get_units(self) -> str:
        """ Returns the physical units of time used by this clock. """
        return self.units

    def register_callback(self, function: Callable):
        """
        Argument function will be called immediately after every clock tick.

        The function must return a True value to keep the itself registered.
        """
        assert isinstance(function, Callable)
        self.callbacks.append(function)

    def reset(self):
        """ Set the clock to zero and then call all callbacks. """
        self.ticks = 0
        self._call_callbacks()

    def set_time(self, new_time: float):
        self.ticks = round(float(new_time) / self.dt)
        assert self.ticks >= 0

    def tick(self):
        """ Advance the clock by `tick_period` and then call all callbacks. """
        self.ticks += 1
        self._call_callbacks()

    def _call_callbacks(self):
        for i in reversed(range(len(self.callbacks))):
            keep_alive = self.callbacks[i]()
            if not keep_alive:
                self.callbacks[i] = self.callbacks[-1]
                self.callbacks.pop()

class TimeSeriesBuffer:
    """ Buffer for timeseries data, and associated helper methods. """
    def __init__(self, max_length:float=np.inf):
        """ Create a new empty buffer for managing time series data.

        Argument max_length is the maximum duration of time that the buffer may
                contain before it discards the oldest data samples.
        """
        self.max_length = float(np.inf if max_length is None else max_length)
        self.timeseries = collections.deque()
        self.timestamps = collections.deque()
        self.stop()

    def stop(self) -> 'self':
        """ Immediately stop recording / playing. """
        self.record_duration = 0
        self.play_index = None
        return self

    def is_stopped(self) -> bool:
        return not self.is_recording() and not self.is_playing()

    def is_recording(self) -> bool:
        return self.record_duration > 0

    def is_playing(self) -> bool:
        return self.play_index is not None

    def clear(self) -> 'self':
        """ Reset the buffer. Removes all data samples from the buffer. """
        self.timeseries.clear()
        self.timestamps.clear()
        return self

    def _setup_pointers(self, db_object, component, clock):
        assert isinstance(db_object, neuwon.database._DB_Object)
        self.db_object  = db_object
        db_class        = self.db_object.get_database_class()
        if clock is not None:
            assert isinstance(clock, Clock)
            self.clock  = clock
        else:
            self.clock  = db_class.get_database().get_clock()
        self.component  = db_class.get(component)
        self.component_name = self.component.get_name()
        # TODO: I feel like this should guard against users changing the component.
        # 
        # IDEA: If I dis-allow changing components then I can store the
        # component after first usage and make the argument optional therafter.

    def record(self, db_object: neuwon.database._DB_Object, component: str,
            duration:float=np.inf,
            clock:Clock=None,
            ) -> 'self':
        """ Record data samples immediately after each clock tick.

        Recording can be interrupted at any time by the "stop" method.

        Argument db_object is a database managed object.

        Argument component is the name of an attribute of db_object.

        Argument duration is the period of time that it records for.

        Argument clock is optional, if not given then this uses the database's
                default clock. See method: "Database.get_clock()".
        """
        assert self.is_stopped()
        self._setup_pointers(db_object, component, clock)
        self.clock.register_callback(_weakref_wrapper(self._record_implementation))
        self.record_duration = float(duration)
        self._record_implementation() # Collect the current value as the first data sample.
        return self

    def _record_implementation(self):
        if not self.is_recording(): return False
        self.timeseries.append(getattr(self.db_object, self.component_name))
        self.timestamps.append(self.clock())
        while self.timestamps[-1] - self.timestamps[0] > self.max_length:
            self.timeseries.popleft()
            self.timestamps.popleft()
        self.record_duration -= self.clock.dt
        return True

    def play(self, db_object: neuwon.database._DB_Object, component: str,
            mode:str="+=",
            loop:bool=False,
            clock:Clock=None,
            ) -> 'self':
        """
        Play back the time series data, writing one value after each clock tick.

        Play back can be interrupted at any time by the "stop" method.

        Argument db_object is a database managed object.

        Argument component is the name of an attribute of db_object.

        Argument loop causes the playback to restart at the beginning when it
                reaches the end of the buffer.

        Argument clock is optional, if not given then this uses the database's
                default clock. See method: "Database.get_clock()".
        """
        assert self.is_stopped()
        self._setup_pointers(db_object, component, clock)
        self.clock.register_callback(self._play_implementation)
        self.play_index = 0
        self.play_loop = bool(loop)
        return self

    def _play_implementation(self):
        value = self.timeseries[self.play_index]
        setattr(self.db_object, self.component_name, value)
        self.play_index += 1
        if self.play_index >= len(self):
            if self.play_loop:
                self.play_index = 0
            else:
                self.play_index = None
                return False
        return True

    def set_data(self, data_samples, timestamps):
        """ Overwrite the data in this buffer. """
        assert self.is_stopped()
        raise NotImplementedError("todo: low priority.")
        # This should interpolate the given data onto this object's grid.

    @property
    def y(self):
        """ Data samples """
        return self.timeseries
    @property
    def x(self):
        """ Timestamps """
        return self.timestamps

    def interpolate(self, timestamps):
        """
        Interpolate the value of this timeseries at the given timestamps.
        This uses linear interpolation.
        """
        f = scipy.interpolate.interp1d(self.x, self.y)
        min_t = self.timestamps[0]
        max_t = self.timestamps[-1]
        min_v = self.timeseries[0]
        max_v = self.timeseries[-1]
        results = np.empty(len(timestamps))
        for i, t in enumerate(timestamps):
            if   t < min_t: results[i] = min_v
            elif t > max_t: results[i] = max_v
            else: results[i] = f(t)
        return results

    def plot(self, show=True):
        """ Plot a line graph of the time series using matplotlib. """
        plt = matplotlib.pyplot
        plt.figure(self.component_name)
        plt.title("Time Series of: " + self.component_name)
        self.label_axes()
        plt.plot(self.x, self.y)
        if show: plt.show()

    def label_axes(self, axes=None):
        """ Label the figure Axes with physical units.

        Argument Axes is a set of matplotlib Axes, or the default current ones
                    if this argument is not given.
        """
        if axes is None: axes = matplotlib.pyplot.gca()
        axes.set_ylabel(self.component.get_units())
        axes.set_xlabel(self.clock.get_units())
        return axes

    def __len__(self):
        """ Returns the number of data samples in this buffer. """
        return len(self.timeseries)

class Trace:
    """ Exponentially weighted mean and standard deviation.

    After each clock tick this class will automatically update its measurements
    of a time-series variable's mean (average) and standard deviation (spread).
    """
    # References:
    #       "Incremental calculation of weighted mean and variance"
    #       Tony Finch, February 2009
    #       http://web.archive.org/web/http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
    #       See Chapter 9.
    #
    #       "The correct way to start an Exponential Moving Average (EMA)"
    #       David Owen, 2017-01-31
    #       https://blog.fugue88.ws/archives/2017-01/The-correct-way-to-start-an-Exponential-Moving-Average-EMA
    #       Accessed: Aug 22, 2021.
    #
    def __init__(self, db_value, period:float, mean:str=True, variance:str=True, start:str=True):
        """
        Argument db_value is a pair of (db_object, attribute)
                where db_object is a database managed object,
                where attribute is the name of the attribute to measure.

        Argument period controls the weight of each data sample.
                Each sample's weight is: exp(-Δt / period)
                                         Where Δt = current_time - sample_time

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        Alternatively,

        Argument db_value may be a database component, in which case this class
                will compute the mean and standard-deviation for every instance
                of that attribute.

        Arguments mean, variance, & start are names for the database attributes.
              * Optional, if not given then unique names are generated based on
                db_value's name.
              * If variance is a False value, then variance and
                standard-deviation will not be calculated.
              * If start is a False value, then this class will assume a past
                history: that before the trace was created the data was always
                its initial_value. This saves computer time and memory, at the
                expense of accuracy in the time immediately after creation.
        """
        # Determine which mode of operation to use.
        self.trace_attr = isinstance(db_value, neuwon.database._DataComponent)
        self.trace_obj  = not self.trace_attr
        # Get the data component.
        if self.trace_attr:
            self.component = db_value
        elif self.trace_obj:
            self.db_object, attribute = db_value
            assert isinstance(self.db_object, neuwon.database._DB_Object)
            self.component = self.db_object.get_database_class().get(attribute)
        # Save and check remaining arguments.
        self.period = float(period)
        assert self.period > 0.0
        assert mean
        if self.trace_obj:
            assert mean     is True
            assert variance is True
            assert start    is True
        # Access all of the meta-data from the data component.
        self.component_name = self.component.get_name()
        db_class            = self.component.get_class()
        dtype               = self.component.get_dtype()
        assert dtype.kind == "f"
        shape               = self.component.get_shape()
        units               = self.component.get_units()
        initial_value       = self.component.get_initial_value()
        if initial_value is None or start:
            initial_value = self._zero()
        # Initialize mean and variance.
        if self.trace_attr:
            if mean     is True: mean     = f"{self.component_name}_mean"
            if variance is True: variance = f"{self.component_name}_variance"
            if start    is True: start    = f"_{self.component_name}_start"
            if isinstance(self.component, neuwon.database.Attribute):
                add_attr = db_class.add_attribute
            elif isinstance(self.component, neuwon.database.ClassAttribute):
                add_attr = db_class.add_class_attribute
            else:
                raise TypeError(self.component)
            self.mean = add_attr(mean, initial_value,
                    dtype=dtype, shape=shape, units=units,
                    doc=f"Exponential moving average of '{db_class.get_name()}.{self.component_name}'",
                    allow_invalid=self.component.allow_invalid,
                    valid_range=self.component.valid_range,)
            if not start: self.mean.set_data(self.component.get_data())
            self.var = add_attr(variance, self._zero(),
                    dtype=dtype, shape=shape, units=units,
                    doc=f"Exponential moving variance of '{db_class.get_name()}.{self.component_name}'",
                    allow_invalid=self.component.allow_invalid,) if variance else None
            self.start = add_attr(start, 1.0,
                    dtype=dtype,
                    doc="Weight of moving average samples which occurred before initialization.",
                    allow_invalid=False,
                    valid_range=(0.0, 1.0),) if start else None
        elif self.trace_obj:
            self.mean  = initial_value
            self.var   = self._zero()
            self.start = 1.0
        # Calculate the exponential rates: alpha & beta.
        self.clock = self.component.get_database().get_clock()
        dt         = self.clock.get_tick_period()
        self.alpha = np.exp(-dt / self.period)
        self.beta  = 1.0 - self.alpha
        # Register an on-tick callback with the clock.
        if self.trace_attr:
            if self.start is not None:
                callback = self._attr_callback_start
            else:
                callback = self._attr_callback_nostart
        elif self.trace_obj:
            callback = _weakref_wrapper(self._obj_callback)
        self.clock.register_callback(callback)
        callback() # Collect the current value as the first data sample.

    def reset(self):
        if self.trace_attr:
            for comp in (self.mean, self.var, self.start):
                if comp is not None:
                    comp.get_data().fill(comp.get_initial_value())
        elif self.trace_obj:
            self.mean  = self._zero()
            self.var   = self._zero()
            self.start = 1.0

    def _zero(self):
        dtype = self.component.get_dtype()
        shape = self.component.get_shape()
        if shape == 1 or shape == (1,):
            return 0.0
        else:
            return np.zeros(shape, dtype=dtype)

    def _consolidate_memory_spaces(self):
        if self.component.get_memory_space() == "host":
            for comp in (self.mean, self.var, self.start):
                if comp is not None:
                    comp.to_host()
        elif self.component.get_memory_space() == "cuda":
            for comp in (self.mean, self.var, self.start):
                if comp is not None:
                    comp.to_device()

    def _attr_callback_nostart(self):
        self._consolidate_memory_spaces()
        value = self.component.get_data()
        mean  = self.mean.get_data()
        diff  = value - mean
        incr  = self.beta * diff
        mean += incr
        self.mean.set_data(mean)
        if self.var is not None:
            var = self.var.get_data()
            var = self.alpha * (var + diff * incr)
            self.var.set_data(var)
        return True

    def _attr_callback_start(self):
        self._consolidate_memory_spaces()
        value  = self.component.get_data()
        mean   = self.mean.get_data()
        mean  += self.beta * (value - mean)
        self.mean.set_data(mean)
        start  = self.start.get_data()
        start *= self.alpha
        self.start.set_data(start)
        if self.var is not None:
            true_mean = mean / (1.0 - start)
            var  = self.var.get_data()
            var += self.beta * (value - true_mean) ** 2
            var *= self.alpha
            self.var.set_data(var)
        return True

    def _obj_callback(self):
        value       = getattr(self.db_object, self.component_name)
        self.mean  += self.beta * (value - self.mean)
        self.start *= self.alpha
        true_mean   = self.mean / (1.0 - self.start)
        self.var   += self.beta * (value - true_mean) ** 2
        self.var   *= self.alpha
        return True

    def get_mean(self):
        if self.trace_attr:
            if self.start is not None:
                return self.mean.get_data() / (1.0 - self.start.get_data())
            else:
                return self.mean.get_data()
        elif self.trace_obj:
            return self.mean / (1.0 - self.start)

    def get_variance(self):
        if self.trace_attr:
            assert self.var is not None
            if self.start is not None:
                return self.var.get_data() / (1.0 - self.start.get_data())
            else:
                return self.var.get_data()
        elif self.trace_obj:
            return self.var / (1.0 - self.start)

    def get_standard_deviation(self):
        return self.get_variance() ** 0.5

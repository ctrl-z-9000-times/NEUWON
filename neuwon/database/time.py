from collections.abc import Callable, Iterable, Mapping
from .data_components import DataComponent, Attribute, ClassAttribute
from .database import DB_Object
import collections
import functools
import math
import matplotlib.pyplot
import numpy as np
import scipy.interpolate
from weakref import WeakMethod

class CallbackHook:
    """ This class aggregates and manages callbacks. """
    def __init__(self):
        self._callbacks = []

    def register(self, function: 'f() -> bool', weakref=False):
        """ Append a callback to this hook.

        Callbacks are removed if they return True. They are guaranteed to always
        be called in the same relative order that they were registered in.

        Note: Don't use weakrefs if the callback writes to the database or any
        other globally visible state.
        """
        assert isinstance(function, Callable)
        if weakref:
            function = CallbackHook._weakref_wrapper(function)
        self._callbacks.append(function)

    def __call__(self):
        """ Call all registered callbacks and remove any that return true. """
        any_removed = False
        for idx, callback in enumerate(self._callbacks):
            remove = callback()
            if remove:
                self._callbacks[idx] = None
                any_removed = True
        if any_removed:
            self._callbacks = [x for x in self._callbacks if x is not None]

    @staticmethod
    def _weakref_wrapper(method):
        method_ref = WeakMethod(method)
        def call_if_able():
            method = method_ref()
            if method is not None:
                return method()
            else:
                return True
        return call_if_able

class Clock:
    """ Clock and notification system. """
    def __init__(self, tick_period:float, units:str=""):
        """
        Argument tick_period is a duration of time.

        Argument units is the physical units for 'tick_period'. Optional.
        """
        self.dt = self.time_step = self.tick_period = float(tick_period)
        self.ticks = 0
        self.units = str(units)
        self._callbacks = CallbackHook()

    def get_time(self) -> float:
        """ Returns the current time. """
        return self.ticks * self.dt

    def get_ticks(self) -> int:
        """ Returns the current number of ticks. """
        return self.ticks

    def __call__(self) -> float:
        """ Returns the current time. """
        return self.get_time()

    def get_time_step(self) -> float:
        """ Returns the duration of each tick. """
        return self.dt

    def get_tick_period(self) -> float:
        """ Returns the duration of each tick. """
        return self.dt

    def get_units(self) -> str:
        """ Returns the physical units of time used by this clock. """
        return self.units

    def register_callback(self, function: 'f() -> bool', period=1, weakref=False):
        """
        Argument function will be called immediately after every clock tick.

        Callbacks are removed if they return True. They are guaranteed to always
        be called in the same relative order that they were registered in.
        """
        if period == 1:
            self._callbacks.register(function, weakref)
        else:
            1/0 # TODO!
            # This would be useful for scheduling periodic maintenance tasks.
            # For example: synapse death & growth.

    def reset(self):
        """ Set the clock to zero and then call all callbacks. """
        self.ticks = 0
        self._callbacks()

    def set_time(self, new_time: float):
        self.ticks = round(float(new_time) / self.dt)
        assert self.ticks >= 0

    def tick(self):
        """ Advance the clock by `tick_period` and then call all callbacks. """
        self.ticks += 1
        self._callbacks()

class TimeSeries:
    """ Buffer for time-series data, and associated helper methods. """
    def __init__(self, initial_data=None, timestamps=None):
        """ Create a new empty buffer. """
        self.stop()
        self.clear()
        if initial_data: self.set_data(initial_data, timestamps)

    def stop(self) -> 'self':
        """ Immediately stop recording / playing. """
        self.record_duration = 0
        self.play_data = None
        return self

    def is_stopped(self) -> bool:
        return not self.is_recording() and not self.is_playing()

    def is_recording(self) -> bool:
        return self.record_duration > 0

    def is_playing(self) -> bool:
        return self.play_data is not None

    def clear(self) -> 'self':
        """ Reset the buffer. Removes all data samples from the buffer. """
        assert self.is_stopped()
        self.data = collections.deque()
        self.timestamps = collections.deque()
        return self

    def get_data(self) -> collections.deque:
        """ Returns a list containing all of the data samples. """
        return self.data

    def get_timestamps(self) -> collections.deque:
        """ Returns a list containing all of the timestamps. """
        return self.timestamps

    def set_data(self, data_samples, timestamps=None) -> 'self':
        """ Overwrite the data in this buffer. """
        assert self.is_stopped()
        if isinstance(data_samples, TimeSeries):
            assert timestamps is None
            timestamps   = data_samples.get_timestamps()
            data_samples = data_samples.get_data()
        elif timestamps is None:
            raise TypeError("TimeSeries.set_data() missing required positional argument: 'timestamps'")
        self.data = collections.deque(data_samples)
        self.timestamps = collections.deque(timestamps)
        assert len(self.data) == len(self.timestamps)
        for i in range(len(self.timestamps) - 1):
            assert self.timestamps[i] <= self.timestamps[i + 1]
        return self

    def _setup_pointers(self, db_object, component, clock):
        assert isinstance(db_object, DB_Object)
        self.db_object  = db_object
        db_class        = self.db_object.get_database_class()
        if clock is not None:
            assert isinstance(clock, Clock)
            self.clock  = clock
        else:
            self.clock  = db_class.get_database().get_clock()
        self.component  = db_class.get(component)
        self.component_name = self.component.get_name()

    def record(self, db_object: DB_Object, component: str,
            record_duration:float=np.inf,
            discard_after:float=np.inf,
            clock:Clock=None,
            ) -> 'self':
        """ Record data samples immediately after each clock tick.

        Recording can be interrupted at any time by the "stop" method.

        Argument db_object is a database managed object.

        Argument component is the name of an attribute of db_object.

        Argument record_duration is the period of time to record for.

        Argument discard_after is the maximum length of time that the buffer
                can contain before it discards the oldest data samples.

        Argument clock is optional, if not given then this uses the database's
                default clock. See method: "Database.get_clock()".
        """
        assert self.is_stopped()
        self._setup_pointers(db_object, component, clock)
        assert self.clock, "Argument 'clock' not given and database has no default clock set."
        self.clock.register_callback(self._record_implementation, weakref=True)
        self.record_duration = float(record_duration)
        self.discard_after = float(discard_after)
        self._record_implementation() # Collect the current value as the first data sample.
        return self

    def _record_implementation(self):
        if not self.is_recording(): return True
        self.data.append(getattr(self.db_object, self.component_name))
        self.timestamps.append(self.clock())
        while self.timestamps[-1] - self.timestamps[0] > self.discard_after:
            self.data.popleft()
            self.timestamps.popleft()
        self.record_duration -= self.clock.dt

    @classmethod
    def record_many(cls, db_objects: [DB_Object], component: str,
            record_duration:float=np.inf,
            discard_after:float=np.inf,
            clock:Clock=None,) -> ['TimeSeries']:
        """ Convenience method to record from multiple objects. """
        return [cls().record(obj, component,
                             record_duration=record_duration,
                             discard_after=discard_after,
                             clock=clock,)
                for obj in db_objects]

    def play(self, db_object: DB_Object, component: str,
            mode:str="+=",
            loop:bool=False,
            clock:Clock=None,
            ) -> 'self':
        """
        Play back the time series data, writing one value after each clock tick.

        Play back can be interrupted at any time by the "stop" method.

        Argument db_object is a database managed object.

        Argument component is the name of an attribute of db_object.

        Argument mode controls how the signal is superimposed on the object.
                If mode is "=" then the timeseries overwrites the existing values.
                If mode is "+=" then the signals are added.

        Argument loop causes the playback to restart at the beginning when it
                reaches the end of the buffer.

        Argument clock is optional, if not given then this uses the database's
                default clock. See method: "Database.get_clock()".
        """
        assert self.is_stopped()
        self._setup_pointers(db_object, component, clock)
        self.clock.register_callback(self._play_implementation)
        start = math.floor(self.timestamps[ 0] / self.clock.dt)
        end   = math.ceil( self.timestamps[-1] / self.clock.dt)
        timestamps = self.clock.get_tick_period() * np.arange(start, end)
        self.play_data  = self.interpolate_function()(timestamps)
        self.play_index = 0
        self.play_loop  = bool(loop)
        self.mode       = str(mode)
        self._play_implementation() # Immediately write the first value.
        return self

    def _play_implementation(self):
        if self.play_data is None: return True
        value = self.play_data[self.play_index]
        if self.mode == "=":
            setattr(self.db_object, self.component_name, value)
        elif self.mode == "+=":
            setattr(self.db_object, self.component_name,
                    value + getattr(self.db_object, self.component_name))
        else: raise NotImplementedError(self.mode)
        self.play_index += 1
        if self.play_index >= len(self.play_data):
            if self.play_loop:
                self.play_index = 0
            else:
                self.play_data = None
                return True

    def interpolate(self, *timeseries):
        """ Interpolate any number of TimeSeries to a common set of timestamps.

        This method accepts any number of arguments which are either:
          * Instances of TimeSeries to be interpolated.
                This instance is automatically included.
          * Lists of timestamps to interpolate at.

        This method finds the union of all timestamps, including all timestamps
        contained inside of TimeSeries as well as any lists of timestamps.

        All TimeSeries are interpolated at the union of timestamps.
        This modifies the TimeSeries in-place!

        After calling this, all given TimeSeries have identical timestamps and
        their data arrays can be directly compared.
        """
        if isinstance(self, TimeSeries):
            timeseries = [self] + list(timeseries)
        else:
            # In this case we were called as a classmethod.
            timeseries = list(timeseries)
        timestamps = []
        for idx, ts in enumerate(timeseries):
            if isinstance(ts, TimeSeries):
                timestamps.append(ts.get_timestamps())
            else:
                timestamps.append(np.array(ts, dtype=float))
                timeseries[idx] = None
        timestamps = functools.reduce(np.union1d, timestamps)
        for ts in timeseries:
            if ts is not None:
                ts.set_data(ts.interpolate_function()(timestamps), timestamps)

    def interpolate_function(self) -> Callable:
        """ Returns the function: interpolate(timestamp) -> value

        This uses linear interpolation.
        """
        assert self.is_stopped()
        f = scipy.interpolate.interp1d(self.get_timestamps(), self.get_data(),
                            fill_value = (self.data[0], self.data[-1]),
                            bounds_error = False,)
        return np.vectorize(f)

    def plot(self, *args, show:bool=True, **kwargs):
        """ Plot a line graph of the time series using matplotlib.

        Argument show causes this to immediately display the plot, which will
                also block this thread of execution until the user closes the
                plot's window. If false then the plot is not displayed until
                the caller calls: `matplotlib.pyplot.show()`.

        Extra positional and keyword arguments are passed through to the method
                `matplotlib.pyplot.plot`.
        """
        plt = matplotlib.pyplot
        name = getattr(self, "component_name", None)
        if name is not None:
            plt.figure(name)
            plt.title("Time Series of " + name)
            self.label_axes()
        else:
            plt.figure()
        plt.plot(self.get_timestamps(), self.get_data(), *args, **kwargs)
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

    @classmethod
    def plot_many(cls, timeseries: ['TimeSeries'], spacing, *args, show:bool=True, **kwargs):
        """ Plot multiple TimeSeries on a single matplotlib figure.

        Argument spacing controls the distance between the line plots.
        """
        plt = matplotlib.pyplot
        assert isinstance(timeseries, Iterable)
        assert len(timeseries) > 0
        self = timeseries[0]
        self.plot(*args, show=False, **kwargs)
        # Hide the Y-axis ticks and numbers. The Y-axis numbers are not valid
        # because there are multiple lines plotted at various offsets.
        plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        for idx, ts in enumerate(timeseries):
            if idx == 0: continue
            data = np.array(ts.get_data()) - idx * spacing
            plt.plot(ts.get_timestamps(), data, *args, **kwargs)
        if show: plt.show()

    def __len__(self):
        """ Returns the number of data samples in this buffer. """
        return len(self.data)

    def constant_wave(self, value, duration) -> 'self':
        """ Overwrite this buffer with the given function. """
        return self.set_data([value, value], [0, duration])

    def square_wave(self, minimum, maximum, period, duty_cycle=0.5) -> 'self':
        """ Overwrite this buffer with one cycle of the given periodic function. """
        min_        = float(minimum)
        max_        = float(maximum)
        period      = float(period)
        duty_cycle  = float(duty_cycle)
        assert 0.0 <= duty_cycle <= 1.0
        start = 0
        mid   = period * duty_cycle
        end   = period
        return self.set_data([max_, max_, min_, min_], [start, mid, mid, end])

    def sine_wave(self, minimum, maximum, period) -> 'self':
        """ Overwrite this buffer with one cycle of the given periodic function. """
        min_        = float(minimum)
        max_        = float(maximum)
        period      = float(period)
        mult        = 0.5 * (max_ - min_)
        add         = min_ + mult
        n           = 1000
        return self.set_data(
                add + mult * np.sin(np.linspace(0.0, 2.0 * np.pi, n)),
                np.linspace(0, period, n))

    def triangle_wave(self, minimum, maximum, period) -> 'self':
        """ Overwrite this buffer with one cycle of the given periodic function. """
        min_        = float(minimum)
        max_        = float(maximum)
        period      = float(period)
        return self.set_data([min_, max_, min_], [0.0, 0.5 * period, period])

    def sawtooth_wave(self, minimum, maximum, period) -> 'self':
        """ Overwrite this buffer with one cycle of the given periodic function. """
        min_        = float(minimum)
        max_        = float(maximum)
        period      = float(period)
        return self.set_data([min_, max_, min_], [0.0, period, period])

# Thought: The Trace class does too much. Because it covers two use cases
# (single instances and whole components) its API is unintuitive, excessively
# documented, and allows nonsensical combinations of features.
# I should split it into two classes: "Trace" and "TraceAll".

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
        self.trace_attr = isinstance(db_value, DataComponent)
        self.trace_obj  = not self.trace_attr
        # Get the data component.
        if self.trace_attr:
            self.component = db_value
        elif self.trace_obj:
            self.db_object, attribute = db_value
            assert isinstance(self.db_object, DB_Object)
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
            if isinstance(self.component, Attribute):
                add_attr = db_class.add_attribute
            elif isinstance(self.component, ClassAttribute):
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
            self.clock.register_callback(callback)
        elif self.trace_obj:
            callback = self._obj_callback
            self.clock.register_callback(callback, weakref=True)
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

    def _attr_callback_nostart(self):
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

    def _attr_callback_start(self):
        # db = self.component.get_database()
        # with db.using_memory_space(self.component.get_memory_space()):
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

    def _obj_callback(self):
        value       = getattr(self.db_object, self.component_name)
        self.mean  += self.beta * (value - self.mean)
        self.start *= self.alpha
        true_mean   = self.mean / (1.0 - self.start)
        self.var   += self.beta * (value - true_mean) ** 2
        self.var   *= self.alpha

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

from collections.abc import Callable, Iterable, Mapping
from neuwon.database import _DB_Object, Attribute, ClassAttribute, _Component
import collections
import matplotlib.pyplot
import numpy as np
import scipy.interpolate
import weakref

# IDEA: What if I anchored the clock to the database, as a faux global?
#       I could literally just write it into the db object.
#   Pros:
#       Simpler, cleaner API. user litteral can't fuck it up.
#   Cons:
#       Clock would be effectively global (one per database)
#       Clock would be tied to database (but all the other tools already are as well...)
#       Clock would be harder to test? no.
# 
#   How would I creat, access & control it? I need to sketch out an API before implementing.
# The clock object will remain globally visible, for documentation purposes.
# 



class Clock:
    """ Clock and notification system. """
    def __init__(self, tick_period:float, units:str=""):
        """
        Argument tick_period is a duration of time.

        Argument units is the physical units for 'tick_period'. Optional.
        """
        self.dt = float(tick_period)
        self.ticks = 0
        self.units = str(units)
        self.callbacks = []

    def clock(self) -> float:
        """ Returns the current time. """
        return self.ticks * self.dt

    def time(self) -> float:
        """ Returns the current time. """
        return self.ticks * self.dt

    def __call__(self) -> float:
        """ Returns the current time. """
        return self.clock()

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
            try: keep_alive = self.callbacks[i]()
            except Exception:
                raise RuntimeError("in callback "+repr(self.callbacks[i]))
            if not keep_alive:
                self.callbacks[i] = self.callbacks[-1]
                self.callbacks.pop()

class TimeSeriesBuffer:
    """ Buffer for timeseries data, and associated tools. """
    def __init__(self, clock: Clock, max_length:float=np.inf):
        """ Create a new empty buffer for managing time series data.

        Argument max_length is the maximum duration of time that the buffer may
                contain before it discards the oldest data samples.
        """
        self.clock = clock
        assert isinstance(self.clock, Clock)
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

    def _setup_pointer(self, db_object, component):
        self.db_object = db_object
        assert isinstance(self.db_object, _DB_Object)
        self.component = self.db_object.get_database_class().get(component)
        self.component_name = self.component.get_name()
        # TODO: I feel like this should guard against users changing the
        # component or the database.
        # 
        # IDEA: If I dis-allow changing components then I can store the
        # component after first usage and make the argument optional therafter.

    def record(self, db_object: _DB_Object, component: str, duration:float=np.inf) -> 'self':
        """ Record data samples immediately after each clock tick.

        Recording can be interrupted at any time by the "stop" method.

        Argument db_object is a database managed object.

        Argument component is the name of an attribute of db_object.

        Argument duration is the period of time that it records for.
        """
        assert self.is_stopped()
        self._setup_pointer(db_object, component)
        self.clock.register_callback(weakref.WeakMethod(self._record_implementation))
        self.record_duration = float(duration)
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

    def play(self, db_object: _DB_Object, component: str, loop:bool=False) -> 'self':
        """ Play back the time series data.

        Play back can be interrupted at any time by the "stop" method.

        Argument db_object is a database managed object.

        Argument component is the name of an attribute of db_object.

        Argument loop causes the playback to restart at the beginning when it
                reaches the end of the buffer.
        """
        assert self.is_stopped()
        self._setup_pointer(db_object, component)
        self.clock.register_callback(weakref.WeakMethod(self._play_implementation))
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
    """
    http://web.archive.org/web/http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
        Skip to Chapter 9.
    """
    # TODO: For now this is going to be constantly ON, with no resets or
    # anything. you make it and it just runs. I would like to add a reset
    # function to it, and make it alter the alpha/period durring the warm up
    # after init/reset. It's not expensive and its a nice thing to do.
    def __init__(self, clock, db_object, period, mean=True, var=True):
        """
        Argument db_object is one of:
            -> Attribute, ClassAttribute
            -> pair of (object, component)
        """
        self.clock = clock
        assert isinstance(self.clock, Clock)
        self.period   = float(period)
        self.alpha    = np.exp(-1.0 / self.period)
        self.beta     = 1.0 - self.alpha
        if var: assert mean
        self.mean = None
        self.var = None

        if isinstance(db_object, _Component):
            # Create database components for the mean and variance.
            if mean is True:
                mean = db_object.get_name() + "_mean"
            if var is True:
                var = db_object.get_name() + "_var"
            db_class = db_object.get_class()
            if isinstance(db_object, Attribute):
                add_attr = db_class.add_attribute
            elif isinstance(db_object, ClassAttribute):
                add_attr = db_class.add_class_attribute
            else:
                raise TypeError(db_object)
            if mean: self.mean = add_attr(mean,)
            if var: self.var = add_attr(var,)
            # Don't use a weakref here because this modifies the global state.
            self.clock.register_callback(self._component_callback)
        else:
            self.db_object, component = db_object
            assert isinstance(self.db_object, _DB_Object)
            component = self.db_object.get_database_class().get(component)
            self.component_name = component.get_name()
            # TODO: check component.dtype is sane.
            #       Also component.shape could be something other than 1.
            if mean:
                initial_value = component.get_initial_value()
                if initial_value is None:
                    self.mean = 0.0
                else:
                    self.mean = initial_value
            if var: self.var = 0.0
            self.clock.register_callback(weakref.WeakMethod(self._object_callback))

    def _component_callback(self):
        mean = self.mean.get()
        var = self.var.get()

    def _object_callback(self):
        value     = getattr(self.db_object, self.component_name)
        diff      = value - self.mean
        incr      = self.beta * diff
        self.mean = self.mean + incr
        self.var  = self.alpha * (self.var + diff * incr)

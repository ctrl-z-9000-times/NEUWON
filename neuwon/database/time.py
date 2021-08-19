from collections.abc import Callable, Iterable, Mapping
import neuwon.database
import collections
import matplotlib.pyplot
import numpy as np
import scipy.interpolate
import weakref

def _weakref_wrapper(method):
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
    """ Buffer for timeseries data, and associated tools. """
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

    def _setup_pointers(self, db_object, component):
        assert isinstance(db_object, neuwon.database._DB_Object)
        self.db_object  = db_object
        db_class        = self.db_object.get_database_class()
        self.clock      = db_class.get_database().get_clock()
        self.component  = db_class.get(component)
        self.component_name = self.component.get_name()
        # TODO: I feel like this should guard against users changing the component.
        # 
        # IDEA: If I dis-allow changing components then I can store the
        # component after first usage and make the argument optional therafter.

    def record(self, db_object: neuwon.database._DB_Object, component: str, duration:float=np.inf) -> 'self':
        """ Record data samples immediately after each clock tick.

        Recording can be interrupted at any time by the "stop" method.

        Argument db_object is a database managed object.

        Argument component is the name of an attribute of db_object.

        Argument duration is the period of time that it records for.
        """
        assert self.is_stopped()
        self._setup_pointers(db_object, component)
        self.clock.register_callback(_weakref_wrapper(self._record_implementation))
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

    def play(self, db_object: neuwon.database._DB_Object, component: str, loop:bool=False) -> 'self':
        """ Play back the time series data.

        Play back can be interrupted at any time by the "stop" method.

        Argument db_object is a database managed object.

        Argument component is the name of an attribute of db_object.

        Argument loop causes the playback to restart at the beginning when it
                reaches the end of the buffer.
        """
        assert self.is_stopped()
        self._setup_pointers(db_object, component)
        self.clock.register_callback(_weakref_wrapper(self._play_implementation))
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
    def __init__(self, db_object, period, mean=True, var=True):
        """
        Argument db_object is one of:
            -> Attribute, ClassAttribute
            -> pair of (object, component)
        """
        self.period = float(period)
        assert self.period > 0
        assert mean
        self.mean = None
        self.var = None

        if isinstance(db_object, neuwon.database._DataComponent):
            # Create new database components for the mean and variance.
            self.component      = db_object
            self.component_name = self.component.get_name()
            self.clock          = self.component.get_database().get_clock()
            if mean is True: mean = self.component_name + "_mean"
            if var  is True: var  = self.component_name + "_var"
            db_class = self.component.get_class()
            if isinstance(self.component, neuwon.database.Attribute):
                add_attr = db_class.add_attribute
            elif isinstance(self.component, neuwon.database.ClassAttribute):
                add_attr = db_class.add_class_attribute
            else:
                raise TypeError(self.component)
            initial_value = self.component.get_initial_value()
            if initial_value is None:
                initial_value = 0.0
            self.mean = add_attr(mean, initial_value,
                    dtype=self.component.get_dtype(),
                    shape=self.component.get_shape(),
                    units=self.component.get_units(),
                    # doc=,
                    # allow_invalid=False,
                    # valid_range=,
            )
            if var:
                self.var = add_attr(var,
                        initial_value=0,
                        dtype=self.component.get_dtype(),
                        shape=self.component.get_shape(),
                        units=self.component.get_units(),
                    )
            # Don't use a weakref here because this modifies the global state.
            self.clock.register_callback(self._component_callback)
        else:
            self.db_object, component = db_object
            assert isinstance(self.db_object, neuwon.database._DB_Object)
            component = self.db_object.get_database_class().get(component)
            self.component_name = component.get_name()
            self.clock = component.get_database().get_clock()
            # TODO: check component.dtype is sane.
            #       Also component.shape could be something other than 1.
            initial_value = component.get_initial_value()
            if initial_value is None:
                self.mean = 0.0
            else:
                self.mean = initial_value
            if var: self.var = 0.0
            self.clock.register_callback(_weakref_wrapper(self._object_callback))

        dt = self.clock.get_tick_period()
        self.alpha  = np.exp(-dt / self.period)
        self.beta   = 1.0 - self.alpha

        # TODO: Set the mean to the current value.
        pass

    def _component_callback(self):
        mean  = self.mean.get_data()
        var   = self.var.get_data()
        value = self.component.get_data()

        diff = value - mean
        incr = self.beta * diff
        mean = mean + incr
        var  = self.alpha * (var + diff * incr)

        self.mean.set_data(mean)
        self.var.set_data(var)
        return True

    def _object_callback(self):
        value     = getattr(self.db_object, self.component_name)
        diff      = value - self.mean
        incr      = self.beta * diff
        self.mean = self.mean + incr
        self.var  = self.alpha * (self.var + diff * incr)
        return True

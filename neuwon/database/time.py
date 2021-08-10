import collections
import matplotlib.pyplot
import neuwon.database
import scipy.interpolate

class Clock:
    """ Clock and notification system. """
    def __init__(self, tick_period: float, units: str=None):
        self.dt = float(tick_period)
        self.ticks = 0
        self.units = None if units is None else str(units)
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
        """ Returns the physical units for 'tick_period'. """
        return self.units

    def register_callback(self, function:collections.abc.Callable):
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
            try: keep_alive = self.callbacks[i](self.db.access)
            except Exception:
                raise RuntimeError("in callback "+repr(self.callbacks[i]))
            if not keep_alive:
                self.callbacks[i] = self.callbacks[-1]
                self.callbacks.pop()

class TimeSeriesBuffer:
    def __init__(self, clock: Clock, max_length:float=None):
        """ """
        self.clock = clock
        assert isinstance(self.clock, Clock)
        self.max_length = float(np.inf if max_length is None else max_length)
        self.timeseries = collections.deque()
        self.timestamps = collections.deque()
        self.stop()

    def stop(self) -> 'self':
        """ Stop recording / playing. """
        self.record_duration = 0
        self.play_state = np.inf
        return self

    def is_stopped(self) -> bool:
        return not self.is_recording() and not self.is_playing()

    def is_recording(self) -> bool:
        return self.record_duration > 0

    def is_playing(self) -> bool:
        return self.play_state < np.inf

    def clear(self) -> 'self':
        """ Reset the buffer. Removes all data samples from the buffer. """
        self.timeseries.clear()
        self.timestamps.clear()
        return self

    def _setup_pointer(self, db_object, component):
        self.db_object = db_object
        assert isinstance(self.db_object, neuwon.database._DB_Object)
        self.component = self.db_object.get_database_class().get(component).get_name()

    def record(self, db_object, component, duration=None) -> 'self':
        """ """ # TODO-DOC
        assert self.is_stopped()
        self._setup_pointer(db_object, component)
        self.clock.register_callback(self._record_implementation)
        self.record_duration = float(np.inf if duration is None else duration)
        return self

    def _record_implementation(self):
        if not self.is_recording(): return False
        self.timeseries.append(getattr(self.db_object, self.component))
        self.timestamps.append(self.clock())
        while self.timestamps[-1] - self.timestamps[0] > self.max_length:
            self.timeseries.popleft()
            self.timestamps.popleft()
        self.record_duration -= self.clock.dt
        return True

    def play(self, db_object, component, loop=False) -> 'self':
        """ """ # TODO-DOC
        assert self.is_stopped()
        self._setup_pointer(db_object, component)
        self.clock.register_callback(self._play_implementation)
        self.play_state = self.timestamps[0]
        return self

    def _play_implementation(self):
        1/0 # TODO
        setattr(self.db_object, self.component, 1/0)

    def set_data(self, data_samples, timestamps):
        """ Overwrite the data buffers. """
        1/0 # TODO

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
        Interpolate the value of the timeseries at the given timestamps.
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
        """ Plot a simple line graph of the data using matplotlib. """
        plt = matplotlib.pyplot
        plt.figure(self.component)
        plt.title("Time Series of: " + self.component)
        self.label_axes()
        plt.plot(self.x, self.y)
        if show: plt.show()

    def label_axes(self, axes=None):
        """ Label the figure Axes with physical units.

        Argument Axes is a set of matplotlib Axes, or the default current ones
                    if this argument is not given.
        """
        if axes is None: axes = matplotlib.pyplot.gca()
        axes.set_ylabel(self.db_object.get_database_class().get_database().get_units(self.component))
        axes.set_xlabel(self.clock.get_units())
        return axes

    def __repr__(self):
        s = "%d samples"
        if self.is_recording():
            s += ", rec. src" + repr(self.db_object)
        elif self.is_playing():
            s += ", play dst " + repr(self.db_object)
        return "<TimeSeriesBuffer: %s>"%s

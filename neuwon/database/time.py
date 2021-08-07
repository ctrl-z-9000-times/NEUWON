import collections
import matplotlib.pyplot
import neuwon.database
import scipy.interpolate
import weakref

class Clock:
    """ Clock and notification system. """
    def __init__(self, dt, units=None):
        self.dt = float(dt)
        self.ticks = 0
        self.units = None if units is None else str(units)
        self.callbacks = []

    def clock(self):
        """ Returns the current time. """
        return self.ticks * self.dt

    def time(self):
        """ Returns the current time. """
        return self.ticks * self.dt

    def __call__(self):
        """ Returns the current time. """
        return self.clock()

    def get_units(self): return self.units

    def register_callback(self, function):
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

    def tick(self):
        """ Advance the clock by `dt` and then call all callbacks. """
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
    def __init__(self, db_object, component, clock, max_length=None):
        """ """
        self.db_object = db_object
        assert isinstance(self.db_object, neuwon.database._DB_Object)
        self.component = str(component)
        getattr(self.db_object, self.component)
        assert isinstance(clock, Clock)
        self.clock = clock
        self.clock.register_callback(weakref.WeakMethod(self._sample_data))
        self.max_length = float(np.inf if max_length is None else max_length)
        self.timeseries = collections.deque()
        self.timestamps = collections.deque()

    def clear(self):
        self.timeseries.clear()
        self.timestamps.clear()

    def _sample_data(self):
        self.timeseries.append(getattr(self.db_object, self.component))
        self.timestamps.append(self.clock())
        while self.timestamps[-1] - self.timestamps[0] > self.max_length:
            self.timeseries.popleft()
            self.timestamps.popleft()
        return True

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
        plt = matplotlib.pyplot
        plt.figure(self.component)
        plt.title("Time Series of: " + self.component)
        self.label_axes()
        plt.plot(self.x, self.y)
        if show: plt.show()

    def label_axes(self, axes=None):
        if axes is None: axes = matplotlib.pyplot.gca()
        axes.set_ylabel(self.db_object._cls.get_database().get_units(self.component))
        axes.set_xlabel(self.clock.get_units(self.clock))
        return axes

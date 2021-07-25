
def eprint(*args, **kwargs):
    """ Prints to standard error (sys.stderr). """
    print(*args, file=sys.stderr, **kwargs)

# I might move this out of the database, as it seems to be very ... specific for
# my current project.
class TimeSeriesBuffer:
    def __init__(self, entity, component, clock_function, max_length=None):
        if hasattr(entity, "entity"): entity = entity.entity
        self.entity = entity
        assert isinstance(self.entity, Entity), self.entity
        self.component = str(component)
        self.clock_function = clock_function
        assert isinstance(self.clock_function, Callable)
        self.max_length = float(np.inf if max_length is None else max_length)
        self.timeseries = collections.deque()
        self.timestamps = collections.deque()

    def clear(self):
        self.timeseries.clear()
        self.timestamps.clear()

    def __call__(self, database_access):
        self.timeseries.append(self.entity.read(self.component))
        self.timestamps.append(database_access(self.clock)())
        while self.timestamps[-1] - self.timestamps[0] > self.max_length:
            self.timeseries.popleft()
            self.timestamps.popleft()
        return True

    def label_axes(self, axes=None):
        if axes is None: axes = matplotlib.pyplot.gca()
        axes.set_ylabel(self.db.get_units(self.component))
        axes.set_xlabel(self.db.get_units(self.clock))
        return axes

    @property
    def y(self):
        """ Data samples """
        return self.timeseries
    @property
    def x(self):
        """ Timestamps """
        return self.timestamps

    def interpolate(self, timestamps):
        """ Interpolate the value of the timeseries at the given timestamps.
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


class KD_Tree(_Component):
    def __init__(self, entity, name, coordinates_attribute, doc=""):
        _Component.__init__(self, entity, name, doc)
        self.component = database.components[str(coordinates_attribute)]
        assert(isinstance(self.component, _Attribute))
        assert(not self.component.reference)
        archetype = database._get_archetype(self.name)
        archetype.kd_trees.append(self)
        assert archetype == self.component.archetype, "KD-Tree and its coordinates must have the same archetype."
        self.tree = None
        self.invalidate()

    def invalidate(self):
        self.up_to_date = False

    def access(self, database):
        if not self.up_to_date:
            data = self.component.access(database).get()
            self.tree = scipy.spatial.cKDTree(data)
            self.up_to_date = True
        return self.tree

    def __repr__(self):
        return "KD Tree   " + self.name

class Linear_System(_Component):
    def __init__(self, class_type, name, function, epsilon, doc="", allow_invalid=False,):
        """ Add a system of linear & time-invariant differential equations.

        Argument function(database_access) -> coefficients

        For equations of the form: dX/dt = C * X
        Where X is a component, of the same archetype as this linear system.
        Where C is a matrix of coefficients, returned by the argument "function".

        The database computes the propagator matrix but does not apply it.
        The matrix is updated after any of the entity are created or destroyed.
        """
        _Component.__init__(self, class_type, name, doc, allow_invalid=allow_invalid)
        self.function   = function
        self.epsilon    = float(epsilon)
        self.data       = None

    def invalidate(self):
        self.data = None

    def get(self):
        if self.data is None: self._compute()
        return self.data

    def _compute(self):
        coef = self.function(self.cls.database)
        coef = scipy.sparse.csc_matrix(coef, shape=(self.archetype.size, self.archetype.size))
        # Note: always use double precision floating point for building the impulse response matrix.
        # TODO: Detect if the user returns f32 and auto-convert it to f64.
        matrix = scipy.sparse.linalg.expm(coef)
        # Prune the impulse response matrix.
        matrix.data[np.abs(matrix.data) < self.epsilon] = 0
        matrix.eliminate_zeros()
        self.data = cupyx.scipy.sparse.csr_matrix(matrix, dtype=Real)


class _Clock:
    """ Clock and notification system mix-in for the model class. """
    def __init__(self):
        self._ticks = 0
        self.db.add_function("clock", self._clock, units='ms')
        self.callbacks = []

    def _clock(self):
        """ Returns the model's internal clock time, in milliseconds. """
        return self._ticks * self.db.access("time_step")

    def add_callback(self, function):
        """
        Argument function is called immediately after the clock ticks.

        The function must return a True value to keep the itself registered.
        """
        assert isinstance(function, Callable)
        self.callbacks.append(function)

    def reset_clock(self):
        self._ticks = 0
        self._call_callbacks()

    def _advance_clock(self):
        self._ticks += 1
        self._call_callbacks()

    def _call_callbacks(self):
        for i in reversed(range(len(self.callbacks))):
            try: keep_alive = self.callbacks[i](self.db.access)
            except Exception:
                eprint("Exception raised by "+repr(self.callbacks[i]))
                raise
            if not keep_alive:
                self.callbacks[i] = self.callbacks[-1]
                self.callbacks.pop()

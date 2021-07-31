
def eprint(*args, **kwargs):
    """ Prints to standard error (sys.stderr). """
    print(*args, file=sys.stderr, **kwargs)

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


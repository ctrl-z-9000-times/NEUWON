""" Private module. """
__all__ = []

from neuwon.database import epsilon
import numpy as np

class ElectricProperties:
    __slots__ = ()
    @staticmethod
    def _initialize(db_cls, *,
                initial_voltage,
                cytoplasmic_resistance,
                membrane_capacitance,):
        db_cls.add_attribute("voltage", initial_value=float(initial_voltage), units="mV")
        db_cls.add_attribute("axial_resistance", units="")
        db_cls.add_attribute("capacitance", units="Farads", valid_range=(0, np.inf))
        db_cls.add_class_attribute("cytoplasmic_resistance", cytoplasmic_resistance,
                units="?",
                valid_range=(epsilon, np.inf))
        db_cls.add_class_attribute("membrane_capacitance", membrane_capacitance,
                units="?",
                valid_range=(epsilon, np.inf))
        db_cls.add_attribute("_sum_conductances", units="Siemens", valid_range=(0, np.inf))
        db_cls.add_attribute("_driving_voltage", units="mV")
        # db.add_linear_system("membrane/diffusion", function=_electric_coefficients, epsilon=epsilon)

    def __init__(self):
        self._compute_passive_electric_properties()

    def _compute_passive_electric_properties(self):
        Ra = self.cytoplasmic_resistance
        Cm = self.membrane_capacitance

        # Compute axial membrane resistance.
        # TODO: This formula only works for cylinders.
        self.axial_resistance = Ra * self.length / self.cross_sectional_area
        # Compute membrane capacitance.
        self.capacitance = Cm * self.surface_area

    @classmethod
    def _electric_advance(cls, time_step):
        db_cls              = cls.get_database_class()
        conductance         = db_cls.get_data("conductance")
        driving_voltage     = db_cls.get_data("driving_voltage")
        capacitance         = db_cls.get_data("capacitance")
        voltage             = db_cls.get_data("voltage")
        integral_v          = db_cls.get_data("integral_v")
        # Update voltages.
        exponent        = -dt * conductances / capacitances
        alpha           = cp.exp(exponent)
        diff_v          = driving_voltages - voltages
        irm             = access("membrane/diffusion")
        voltages[:]     = irm.dot(driving_voltages - diff_v * alpha)
        integral_v[:]   = dt * driving_voltages - exponent * diff_v * alpha

    @classmethod
    def _electric_coefficients(cls):
        """
        Model the electric currents over the membrane surface (in the axial directions).
        Compute the coefficients of the derivative function:
        dV/dt = C * V, where C is Coefficients matrix and V is voltage vector.
        """
        db_cls = cls.get_database_class()
        dt           = access("time_step") / 1000 / _ITERATIONS_PER_TIMESTEP
        parents      = access("membrane/parents").get()
        resistances  = access("membrane/axial_resistances").get()
        capacitances = access("membrane/capacitances").get()
        src = []; dst = []; coef = []
        for child, parent in enumerate(parents):
            if parent == NULL: continue
            r        = resistances[child]
            c_parent = capacitances[parent]
            c_child  = capacitances[child]
            src.append(child)
            dst.append(parent)
            coef.append(+dt / (r * c_parent))
            src.append(child)
            dst.append(child)
            coef.append(-dt / (r * c_child))
            src.append(parent)
            dst.append(child)
            coef.append(+dt / (r * c_child))
            src.append(parent)
            dst.append(parent)
            coef.append(-dt / (r * c_parent))
        return (coef, (dst, src))

        coef = scipy.sparse.csc_matrix(coef, shape=(self.archetype.size, self.archetype.size))
        # Note: always use double precision floating point for building the impulse response matrix.
        # TODO: Detect if the user returns f32 and auto-convert it to f64.
        matrix = scipy.sparse.linalg.expm(coef)
        # Prune the impulse response matrix.
        matrix.data[np.abs(matrix.data) < epsilon] = 0
        matrix.eliminate_zeros()
        self.data = cupyx.scipy.sparse.csr_matrix(matrix, dtype=Real)



    def inject_current(self, current, duration = 1.4):
        duration = float(duration)
        assert(duration >= 0)
        current = float(current)

        # Inject_Current is applied two times every tick, at the start of advance_species.
        # Need to make a second clock for the pre-species-advance stuff, put it in the model.
        # Put a link to the model (and via the model the second clock) in the
        # Segment class (where the _cls slot is).

        clock = 1/0
        dv = current * min(clocl.get_tick_period(), t) / self.capacitances
        TimeSeriesBuffer().set_data([dv, dv], [0,duration]).play(self, "voltage", clock=clock)

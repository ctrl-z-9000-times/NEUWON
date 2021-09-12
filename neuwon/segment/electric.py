""" Private module. """
__all__ = []

from neuwon.database import epsilon, NULL
from neuwon.database.time import TimeSeriesBuffer
import numpy as np
import cupy as cp
import scipy.sparse
import scipy.sparse.linalg

class ElectricProperties:
    __slots__ = ()
    @classmethod
    def _initialize(cls, db_cls, *,
                initial_voltage,
                cytoplasmic_resistance,
                membrane_capacitance,):
        db_cls.add_attribute("voltage", float(initial_voltage), units="mV")
        db_cls.add_attribute("integral_voltage")
        db_cls.add_attribute("axial_resistance", units="")
        db_cls.add_attribute("capacitance", units="Farads", valid_range=(0, np.inf))
        db_cls.add_class_attribute("cytoplasmic_resistance", cytoplasmic_resistance,
                units="?",
                valid_range=(epsilon, np.inf))
        db_cls.add_class_attribute("membrane_capacitance", membrane_capacitance,
                units="?",
                valid_range=(epsilon, np.inf))
        db_cls.add_attribute("sum_conductance", units="Siemens", valid_range=(0, np.inf))
        db_cls.add_attribute("driving_voltage", units="mV")
        cls._clean = False
        db_cls.add_sparse_matrix("electric_propagator_matrix", db_cls)

    def __init__(self):
        self._compute_passive_electric_properties()
        type(self)._clean = False

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
        if not cls._clean:
            cls._compute_propagator_matrix(time_step)
        dt = time_step
        db_cls              = cls.get_database_class()
        sum_conductance     = db_cls.get_data("sum_conductance")
        driving_voltage     = db_cls.get_data("driving_voltage")
        capacitance         = db_cls.get_data("capacitance")
        voltage             = db_cls.get_data("voltage")
        integral_v          = db_cls.get_data("integral_voltage")
        irm                 = db_cls.get("electric_propagator_matrix").to_csr().to_host().get_data()
        # Update voltages.
        exponent        = -dt * sum_conductance / capacitance
        alpha           = np.exp(exponent)
        diff_v          = driving_voltage - voltage
        voltage[:]      = irm.dot(driving_voltage - diff_v * alpha)
        integral_v[:]   = dt * driving_voltage - exponent * diff_v * alpha

    @classmethod
    def _compute_propagator_matrix(cls, time_step):
        """
        Model the electric currents over the membrane surface (in the axial directions).
        Compute the coefficients of the derivative function:
        dV/dt = C * V, where C is Coefficients matrix and V is voltage vector.
        """
        db_cls = cls.get_database_class()
        dt           = time_step / 1000
        parents      = db_cls.get_data("parent")
        resistances  = db_cls.get_data("axial_resistance")
        capacitances = db_cls.get_data("capacitance")
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
        coef = (coef, (dst, src))
        # Note: always use double precision floating point for building the impulse response matrix.
        coef = scipy.sparse.csc_matrix(coef, shape=(len(db_cls), len(db_cls)), dtype=np.float64)
        matrix = scipy.sparse.linalg.expm(coef)
        # Prune the impulse response matrix.
        matrix.data[np.abs(matrix.data) < epsilon] = 0
        matrix.eliminate_zeros()
        db_cls.get("electric_propagator_matrix").to_csr().set_data(matrix)
        cls._clean = True

    def inject_current(self, current, duration = 1.4):
        duration = float(duration)
        assert duration >= 0
        current = float(current)
        clock = type(self)._model.input_clock
        dv = current * clock.get_tick_period() / self.capacitance
        TimeSeriesBuffer().set_data([dv, dv], [0, duration]).play(self, "voltage", clock=clock)

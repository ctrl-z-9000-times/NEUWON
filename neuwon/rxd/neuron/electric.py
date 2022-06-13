from neuwon.database import epsilon, NULL, Compute
from neuwon.database.time import TimeSeries
import cupy
import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

class Electric:
    __slots__ = ()
    @staticmethod
    def _initialize(database, *,
                initial_voltage: 'mV' = -70,
                cytoplasmic_resistance: 'ohm-cm' = 100,
                membrane_capacitance: 'μF/cm²' = 1,):
        seg_data = database.get_class('Segment')
        seg_data.add_attribute("voltage", initial_voltage,
                units="mV")
        seg_data.add_attribute("integral_voltage", 0.0,
                units="mV-ms")
        seg_data.add_attribute("axial_resistance",
                units="ohm",
                valid_range=(0, np.inf))
        seg_data.add_attribute("capacitance",
                units="Farads",
                valid_range=(0, np.inf))
        seg_data.add_class_attribute("cytoplasmic_resistance", cytoplasmic_resistance,
                units="ohm-cm",
                valid_range=(epsilon, np.inf))
        seg_data.add_class_attribute("membrane_capacitance", membrane_capacitance,
                units="μF/cm²",
                valid_range=(epsilon, np.inf))
        seg_data.add_attribute("nonspecific_current", 0.0,
                units="Amperes", # TODO: Consider converting this to use "nA".
                valid_range=(0, np.inf))
        seg_data.add_attribute("sum_current", 0.0,
                units="Amperes")
        seg_data.add_attribute("sum_conductance", 0.0,
                units="Siemens",
                valid_range=(0, np.inf))
        seg_data.add_attribute("driving_voltage", 0.0,
                units="mV")
        seg_data.add_sparse_matrix("electric_propagator_matrix", 'Segment')
        seg_cls = seg_data.get_instance_type()
        seg_cls._matrix_valid = False

    def __init__(self):
        self._compute_passive_electric_properties()
        type(self)._matrix_valid = False

    def _compute_passive_electric_properties(self):
        Ra = self.cytoplasmic_resistance * 1e4 # Convert from ohm-cm to ohm-um.
        Cm = self.membrane_capacitance * 1e-14 # Convert from uf/cm^2 to f/um^2.

        self.axial_resistance = Ra * self.length / self.cross_sectional_area
        self.capacitance = Cm * self.surface_area

    @classmethod
    def _advance_electric(cls, time_step):
        if not cls._matrix_valid:
            cls._compute_propagator_matrix(time_step)
        dt = time_step * 1e-3 # Convert to seconds
        db_cls              = cls.get_database_class()
        xp                  = db_cls.get_database().get_memory_space().array_module
        sum_conductance     = db_cls.get_data("sum_conductance") # Siemens
        driving_voltage     = db_cls.get_data("driving_voltage") # mV
        capacitance         = db_cls.get_data("capacitance") # F
        voltage             = db_cls.get_data("voltage") # mV
        integral_v          = db_cls.get_data("integral_voltage") # mV * seconds
        irm                 = db_cls.get("electric_propagator_matrix").to_csr().get_data()
        sum_current         = db_cls.get_data("sum_current")
        # Update voltages.
        dv_currents     = time_step * sum_current / capacitance
        exponent        = -dt * sum_conductance / capacitance
        alpha           = xp.exp(exponent)
        diff_v          = driving_voltage - voltage
        voltage[:]      = irm.dot(driving_voltage - diff_v * alpha + dv_currents)
        integral_v[:]   = dt * driving_voltage - exponent * diff_v * alpha + (.5 * dt * dv_currents)

    @classmethod
    def _compute_propagator_matrix(cls, time_step):
        """
        Model the electric currents over the membrane surface (in the axial directions).
        Compute the coefficients of the derivative function:
        dV/dt = C * V, where C is Coefficients matrix and V is voltage vector.
        """
        db_cls = cls.get_database_class()
        dt     = time_step * 1e-3
        with db_cls.get_database().using_memory_space('host'):
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
        matrix.data[np.abs(matrix.data) < epsilon] = 0.0
        matrix.eliminate_zeros()
        db_cls.get("electric_propagator_matrix").to_csr().set_data(matrix)
        cls._matrix_valid = True

    def inject_current(self, current: 'nA' = 1.0, duration: 'ms' = 1.0):
        current  = float(current) * 1e-9
        duration = float(duration)
        assert duration >= 0
        input_signal = TimeSeries().constant_wave(current, duration)
        input_signal.play(self, "sum_current",
                          clock=type(self)._model.input_hook, immediate=False)

    def get_time_constant(self):
        return self.capacitance / self.sum_conductance

    def get_length_constant(self):
        rm = 1.0 / (self.sum_conductance / self.length)
        ri = self.axial_resistance / self.length
        return (rm / ri) ** 0.5

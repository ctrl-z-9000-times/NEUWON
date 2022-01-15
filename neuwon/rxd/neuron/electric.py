from neuwon.database import epsilon, NULL
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
                initial_voltage = -70.0,
                cytoplasmic_resistance = 100.0,
                membrane_capacitance = 1.0,):
        seg_data = database.get_class('Segment')
        seg_data.add_attribute("voltage", float(initial_voltage), units="mV")
        seg_data.add_attribute("integral_voltage")
        seg_data.add_attribute("axial_resistance", units="")
        seg_data.add_attribute("capacitance", units="Farads", valid_range=(0, np.inf))
        seg_data.add_class_attribute("cytoplasmic_resistance", cytoplasmic_resistance,
                units="ohm-cm",
                valid_range=(epsilon, np.inf))
        seg_data.add_class_attribute("membrane_capacitance", membrane_capacitance,
                units="?",
                valid_range=(epsilon, np.inf))
        seg_data.add_attribute("sum_conductance", units="Siemens", valid_range=(0, np.inf))
        seg_data.add_attribute("driving_voltage", units="mV")
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
        dt = time_step * 1e-3
        db_cls              = cls.get_database_class()
        xp                  = db_cls.get_database().get_memory_space().array_module
        sum_conductance     = db_cls.get_data("sum_conductance")
        driving_voltage     = db_cls.get_data("driving_voltage")
        capacitance         = db_cls.get_data("capacitance")
        voltage             = db_cls.get_data("voltage")
        integral_v          = db_cls.get_data("integral_voltage")
        irm                 = db_cls.get("electric_propagator_matrix").to_csr().get_data()
        # Update voltages.
        exponent        = -dt * sum_conductance / capacitance
        alpha           = xp.exp(exponent)
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

    def inject_current(self, current, duration = 1.4):
        # TODO: Consider changing the default duration to just "1" so that it's a
        # nice round number that's easy to remember.
        # 
        # TODO:  Conisder switching inject current to units of nano-Amps.
        #        Does anyone else use nA?
        #        What does NEURON use?
        #        Is nA actually a good unit? or is my using it a fluke?
        duration = float(duration)
        assert duration >= 0
        current = float(current)
        clock = type(self)._model.input_hook
        dt = clock.get_time_step()
        dv = current * dt / self.capacitance
        input_signal = TimeSeries().constant_wave(dv, duration)
        input_signal.play(self, "voltage", clock=clock)

    def get_time_constant(self):
        return self.capacitance / self.sum_conductance

    def get_length_constant(self):
        rm = 1.0 / (self.sum_conductance / self.length)
        ri = self.axial_resistance / self.length
        return (rm / ri) ** 0.5

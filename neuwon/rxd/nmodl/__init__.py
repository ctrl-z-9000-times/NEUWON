from collections.abc import Callable, Iterable, Mapping
from neuwon.database import Compute
from neuwon.rxd.mechanisms import OmnipresentMechanism, LocalMechanismSpecification, LocalMechanismInstance
from . import code_gen, cache, solver
from .parser import (NmodlParser, ANT,
        SolveStatement,
        AssignStatement,
        ConserveStatement)
import os.path
import re
import sympy

__all__ = ["NMODL"]


# TODO: support for arrays? - arrays should really be unrolled in an AST pass...

# TODO: Move assignments to conductances to the end of the breakpoint block,
# where they belong. This is needed because the nmodl library inserts
# conductance hints and associated statements at the beginning of the block.
# self.breakpoint_block.statements.sort(key=lambda stmt: bool(
#         isinstance(stmt, AssignStatement)
#         and stmt.pointer and "conductance" in stmt.pointer.name))


class NMODL:
    def __init__(self, filename, parameters={}, use_cache=True):
        """
        Argument filename is an NMODL file to load.
                The standard NMODL file name extension is ".mod"

        Argument parameters is a mapping of parameter names to custom
                floating-point values.
        """
        self.filename = os.path.abspath(str(filename))
        if use_cache and cache.try_loading(self.filename, self): pass
        else:
            try:
                parser = NmodlParser(self.filename)
                self._check_for_unsupported(parser)
                self.nmodl_name, self.point_process,  self.title, self.description = parser.gather_documentation()
                self.parameters = ParameterTable(parser.gather_parameters(), self.nmodl_name)
                self.states = parser.gather_states()
                blocks = parser.gather_code_blocks()
                self.all_blocks = list(blocks.values()) # Unused?
                self.initial_block = blocks['INITIAL']
                self.breakpoint_block = blocks['BREAKPOINT']
                self.derivative_blocks = {k:v for k,v in blocks.items() if v.derivative} # Unused?
                self._gather_IO(parser)
                if self.omnipresent:    self.__class__ = OmnipresentNmodlMechanism
                else:                   self.__class__ = LocalNmodlMechanismSpecification
                self._solve()
                self._fixup_breakpoint_IO()
            except Exception:
                print("ERROR while loading file", self.filename, flush=True)
                raise
            cache.save(self.filename, self)
        self.parameters.update(parameters, strict=True)
        self.instance_parameters = self.parameters.split_instance_parameters(self.point_process)

    def _check_for_unsupported(self, parser):
        # TODO: support for NONLINEAR?
        # TODO: support for INCLUDE?
        # TODO: support for COMPARTMENT?
        disallow = (
            "FUNCTION_TABLE_BLOCK",
            "LON_DIFUSE",
            "NONSPECIFIC_CUR_VAR",
            "TABLE_STATEMENT",
            "VERBATIM",
        )
        for x in disallow:
            if parser.lookup(getattr(ANT, x)):
                raise ValueError("\"%s\"s are not allowed."%x)

    def _gather_IO(self, parser):
        """
        Determine what external data the mechanism accesses.

        Sets attributes: "pointers", "accumulators", "other_mechanisms_", and
        the following flags: "omnipresent", "outside".
        """
        self.pointers = {}
        self.accumulators = set()
        self.other_mechanisms_ = []
        self.omnipresent = False # TODO!
        self.outside = False
        for state in self.states:
            self.pointers[state] = f'self.{state}'
        self.breakpoint_block.gather_arguments()
        self.initial_block.gather_arguments()
        all_args = self.breakpoint_block.arguments + self.initial_block.arguments
        if "v" in all_args:
            self.pointers["v"] = "self.segment.voltage"
        if "area" in all_args:
            self.pointers["area"] = "self.segment.surface_area"
        if "volume" in all_args:
            self.pointers["volume"] = "self.segment.inside_volume"
        for x in parser.lookup(ANT.USEION):
            ion = x.name.value.eval()
            # Automatically generate the variable names for this ion.
            equilibrium = ('e' + ion, ion + '_equilibrium',)
            current     = ('i' + ion, ion + '_current',)
            conductance = ('g' + ion, ion + '_conductance',)
            inside      = (ion + 'i', 'intra_' + ion,)
            outside     = (ion + 'o', 'extra_' + ion,)
            for y in x.readlist:
                var_name = y.name.value.eval()
                if var_name in equilibrium:
                    pass # Ignored, mechanisms output conductances instead of currents.
                elif var_name in inside:
                    self.pointers[var_name] = "self.segment.inside_concentrations/%s"
                elif var_name in outside:
                    self.pointers[var_name] = f"self.outside.{ion}"
                    self.outside = True
                else:
                    raise ValueError(f"Unrecognized USEION READ: \"{var_name}\".")
            for y in x.writelist:
                var_name = y.name.value.eval()
                if var_name in current:
                    pass # Ignored, mechanisms output conductances instead of currents.
                elif var_name in conductance:
                    self.pointers[var_name] = f"self.segment.{ion}_conductance"
                    self.accumulators.add(var_name)
                elif var_name in inside:
                    self.pointers[var_name] = "self.segment.inside_delta_concentrations/%s"%ion
                    self.accumulators.add(var_name)
                elif var_name in outside:
                    self.pointers[var_name] = f"self.outside.{ion}_delta"
                    self.accumulators.add(var_name)
                    self.outside = True
                else:
                    raise ValueError(f"Unrecognized USEION WRITE: \"{var_name}\".")
        for x in parser.lookup(ANT.CONDUCTANCE_HINT):
            var_name = x.conductance.get_node_name()
            if var_name not in self.pointers:
                ion = x.ion.get_node_name()
                self.pointers[var_name] = f"Segment.{ion}_conductance"
                self.accumulators.add(var_name)
        for x in parser.lookup(ANT.POINTER_VAR):
            name = x.get_node_name()
            self.other_mechanisms_.append(name)
            self.pointers[name] = f"self.{name}.magnitude"
        self.other_mechanisms_ = tuple(sorted(self.other_mechanisms_))

    def _solve(self):
        """
        Replace SolveStatements with the solved equations to advance the systems
        of differential equations.
        """
        ode_methods = {
            'euler':            solver.forward_euler,
            'derivimplicit':    solver.backward_euler,
            'cnexp':            solver.crank_nicholson,
            'exact':            solver.sympy_solve_ode,
        }
        idx = 0
        while idx < len(self.breakpoint_block.statements):
            solve_stmt = self.breakpoint_block.statements[idx]
            if not isinstance(solve_stmt, SolveStatement):
                idx += 1
                continue
            solve_block  = solve_stmt.block
            solve_method = solve_stmt.method
            if solve_method in ode_methods:
                method = ode_methods[solve_method]
                for stmt in solve_block:
                    if isinstance(stmt, AssignStatement) and stmt.derivative:
                        method(stmt)
                # Replace the solve statment with the solved block.
                self.breakpoint_block.statements.pop(idx)
                for stmt in solve_block:
                    self.breakpoint_block.statements.insert(idx, stmt)
                    idx += 1
            elif solve_method == "sparse":
                assert solve_block.derivative
                deriv_block = self._compile_derivative_block(solve_block)
                1/0
            else:
                raise NotImplementedError(solve_method)

    def _compile_derivative_block(self, block):
        assert block.derivative
        self.parameters.substitute(block)
        block.gather_arguments()
        for stmt in block:
            if isinstance(stmt, AssignStatement) and stmt.derivative:
                stmt.operation = "+="
                stmt.lhsn = '_d_' + stmt.lhsn
                stmt.derivative = False
        deriv_pycode = f"def {block.name}({', '.join(block.arguments)}):\n"
        for state in self.states:
            deriv_pycode += f"    _d_{state} = 0.0\n"
        deriv_pycode += code_gen.to_python(block, "    ", {})
        deriv_pycode += "    return {"
        deriv_pycode += ', '.join(f"{state}: _d_{state}" for state in self.states)
        deriv_pycode += "}\n\n"
        print(deriv_pycode)

    def _fixup_breakpoint_IO(self):
        for stmt in self.breakpoint_block:
            if isinstance(stmt, AssignStatement):
                if stmt.lhsn in self.accumulators:
                    stmt.operation = '+='
        self.breakpoint_block.map(lambda stmt: stmt.simple_solution()
                                if isinstance(stmt, ConserveStatement) else [stmt])
        self.breakpoint_block.substitute(self.pointers)

    def initialize(self, rxd_model, name):
        database = rxd_model.get_database()
        builtin_parameters = {
                "dt":       rxd_model.get_time_step(),
                "celsius":  rxd_model.get_temperature(),
        }
        self.name = str(name)
        try:
            self.parameters.update(builtin_parameters, strict=True, override=False)
            self._run_initial_block(database)
            self._compile_breakpoint_block()
            if self.omnipresent:
                return self._initialize_omnipresent_mechanism_class(database)
            else:
                return self._initialize_local_mechanism_class(database)
        except Exception:
            print("ERROR while loading file", self.filename, flush=True)
            raise

    def _run_initial_block(self, database):
        """ Use pythons built-in "exec" function to run the INITIAL_BLOCK.
        Sets: initial_state and initial_scope. """
        self.parameters.substitute(self.initial_block)
        initial_pointer_values = {}
        for arg in self.initial_block.arguments:
            if arg in self.pointers:
                db_access = self.pointers[arg]
                db_access = re.sub(r'^self\.segment\.', 'Segment.', db_access)
                db_access = re.sub(r'^self\.inside\.',  'Inside.',  db_access)
                db_access = re.sub(r'^self\.outside\.', 'Outside.', db_access)
                value = database.get(db_access).get_initial_value()
                if value is not None:
                    initial_pointer_values[arg] = value
        self.initial_block.substitute(initial_pointer_values)
        self.initial_block.gather_arguments()
        self.initial_scope = {x: 0.0 for x in self.states}
        missing_arguments = set(self.initial_block.arguments) - set(self.initial_scope)
        if missing_arguments:
            raise ValueError(f"Missing initial values for {', '.join(missing_arguments)}.")
        initial_python = code_gen.to_python(self.initial_block)
        code_gen.exec_string(initial_python, {}, self.initial_scope)
        self.initial_state = {x: self.initial_scope.pop(x) for x in self.states}

    def _compile_breakpoint_block(self):
        # 
        self.parameters.substitute(self.breakpoint_block)
        # 
        magnitude = sympy.Symbol('self.magnitude')
        for name, (value, units) in self.instance_parameters.items():
            self.breakpoint_block.statements.insert(0, AssignStatement(name, value * magnitude))
        # 
        for arg in self.breakpoint_block.arguments:
            if arg in self.pointers: continue
            elif arg in self.initial_scope:
                value = float(self.initial_scope[arg])
                self.breakpoint_block.statements.insert(0, AssignStatement(arg, value))
        # 
        self.advance_pycode = (
                "@Compute\n"
                "def advance(self):\n"
                + code_gen.to_python(self.breakpoint_block, "    ", self.pointers))
        globals_ = {
                'Compute': Compute,
        }
        code_gen.exec_string(self.advance_pycode, globals_)
        self.advance_bytecode = globals_['advance']

    def _initialize_omnipresent_mechanism_class(self, database):
        1/0 # TODO

        # NOTES:
        # 
        # Omnipresent mechanisms are used to implement chemical reactions.
        # 
        # Omnipresent mechanisms must have no state variables, instead they use
        # species concentrations and (possibly?) the standard segment attributes.
        # 
        # All omnipresent mechanisms must be associated to a single entity type:
        #       -> Intracellular
        #       -> Extracellular
        #       -> Segment
        # And their advance method will be an anonymous method over all
        # instances of the associated entity.

    def _initialize_local_mechanism_class(self, database):
        mechanism_superclass = type(self.name, (LocalMechanismInstance,), {
            '__doc__':              self.title + "\n\n" + self.description,
            '__slots__':            (),
            '__init__':             NMODL._instance__init__,
            '_point_process':       self.point_process,
            '_outside':             self.outside,
            '_other_mechanisms':    self.other_mechanisms_,
            'advance':              self.advance_bytecode,
            '_advance_pycode':      self.advance_pycode,
        })
        mech_data = database.add_class(mechanism_superclass)
        mech_data.add_attribute("segment", dtype="Segment")
        if self.outside:
            mech_data.add_attribute("outside", dtype="Extracellular")
        if self.point_process:
            mech_data.add_attribute("magnitude", 1.0,
                    doc="") # TODO?
        else:
            mech_data.add_attribute("magnitude",
                    units = database.get('Segment.surface_area').get_units(),
                    doc="") # TODO?
        for name in self.states:
            mech_data.add_attribute(name, initial_value=self.initial_state[name], units=name)
        for name in self.other_mechanisms_:
            mech_data.add_attribute(name, dtype=name)
        return mech_data.get_instance_type()

    @staticmethod
    def _instance__init__(self, segment, magnitude=1.0, *other_mechanisms, outside=None):
        """ Insert this mechanism onto the given segment. """
        self.segment = segment
        if self._point_process:
            self.magnitude = magnitude
        else:
            self.magnitude = magnitude * segment.surface_area
        if self._outside:
            if outside is not None:
                self.outside = outside
            else:
                self.outside = segment.outside
        assert len(self._other_mechanisms) == len(other_mechanisms)
        for name, ref in zip(self._other_mechanisms, other_mechanisms):
            setattr(self, name, ref)

class ParameterTable(dict):
    """ Dictionary mapping from nmodl parameter name to pairs of (value, units). """

    builtin_parameters = {
        "celsius": (None, "degC"),
        "dt":      (None, "ms"),
    }

    def __init__(self, parameters, mechanism_name):
        dict.__init__(self)
        self.mechanism_name = mechanism_name
        self.update(self.builtin_parameters)
        self.update(parameters)

    def update(self, parameters, strict=False, override=True):
        for name, value in parameters.items():
            name = str(name)
            if not isinstance(value, Iterable): value = (value, None)
            value, units = value
            value = float(value) if value is not None else None
            units = str(units)   if units is not None else None
            if name in self:
                old_value, old_units = self[name]
                if units is None: units = old_units
                elif old_units is not None and (strict or not override):
                    assert units == old_units, "Parameter \"%s\" units changed."%name
                if old_value is not None and not override:
                    value = old_value
            elif strict: raise ValueError("Invalid parameter override \"%s\"."%name)
            self[name] = (value, units)
        return self

    def get_instance_parameters(self, point_process):
        """ Returns the instance_parameters.

        The surface area parameters are special because each segment of neuron
        has its own surface area and so their actual values are different for
        each instance of the mechanism. They are not in-lined directly into the
        source code, instead they are stored alongside the state variables and
        accessed at run time. 
        """
        parameters = {}
        for name, (value, units) in self.items():
            if units:
                if point_process:
                    if "/" + self.mechanism_name in units:
                        parameters[name] = (value, units)
                else:
                    if "/cm2" in units:
                        # Convert from NEUWONs um^2 to NEURONs cm^2.
                        value *= (1e-6 * 1e-6) / (1e-2 * 1e-2)
                        units  = units.replace("/cm2", '')
                        parameters[name] = (value, units)
        return parameters

    def split_instance_parameters(self, point_process):
        """ Removes and returns the instance_parameters. """
        parameters = self.get_instance_parameters(point_process)
        for name in parameters:
            self.pop(name)
        return parameters

    def substitute(self, block):
        substitutions = {}
        for name, (value, units) in self.items():
            if value is None:
                continue
            substitutions[name] = value
            if name == "dt":
                substitutions[solver.dt] = value
        block.substitute(substitutions)

class OmnipresentNmodlMechanism(NMODL, OmnipresentMechanism):
    pass

class LocalNmodlMechanismSpecification(NMODL, LocalMechanismSpecification):
    def other_mechanisms(self):
        return self.other_mechanisms_

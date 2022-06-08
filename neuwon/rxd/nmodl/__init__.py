from collections.abc import Callable, Iterable, Mapping
from neuwon.database import Compute
from neuwon.rxd.mechanisms import OmnipresentMechanism, LocalMechanismSpecification, LocalMechanismInstance
from . import code_gen, cache, solver
from .parser import (NmodlParser, ANT,
        SolveStatement,
        AssignStatement,
        ConserveStatement)
import copy
import os.path
import re
import sympy
import textwrap

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
                self.all_blocks = list(blocks.values())
                self.initial_block = blocks['INITIAL']
                self.breakpoint_block = blocks['BREAKPOINT']
                self.conserve_statements = self._gather_conserve_statements()
                self._gather_IO(parser)
                self.__class__ = OmnipresentNmodlMechanism if self.omnipresent else LocalNmodlMechanismSpecification
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
            "TABLE_STATEMENT",
            "VERBATIM",
        )
        for x in disallow:
            if parser.lookup(getattr(ANT, x)):
                raise ValueError("\"%s\"s are not allowed."%x)

    def _gather_conserve_statements(self):
        conserve_statements = []
        for block in self.all_blocks:
            for stmt in block:
                if isinstance(stmt, ConserveStatement):
                    conserve_statements.append(stmt)
        return conserve_statements

    def _gather_IO(self, parser):
        """
        Determine what persistent data the mechanism accesses.

        Sets attributes: "pointers", "accumulators", "other_mechanisms_", and
        the following flags: "omnipresent", "outside".
        """
        self.pointers = {}
        self.accumulators = set()
        self.omnipresent = False # TODO!
        self.outside = False
        for state in self.states:
            self.pointers[state] = f'self.{state}'
        self._gather_segment_IO()
        for x in parser.lookup(ANT.USEION):
            self._process_useion_statement(x)
        self._gather_conductance_hints(parser)
        self._gather_nonspecific_currents(parser)
        self._gather_other_mechanisms_IO(parser)

    def _gather_segment_IO(self):
        self.breakpoint_block.gather_arguments()
        self.initial_block.gather_arguments()
        all_args = self.breakpoint_block.arguments + self.initial_block.arguments
        if "v" in all_args:
            self.pointers["v"] = "self.segment.voltage"
        if "area" in all_args:
            self.pointers["area"] = "self.segment.surface_area"
        if "volume" in all_args:
            self.pointers["volume"] = "self.segment.inside_volume"

    def _process_useion_statement(self, stmt):
        ion = stmt.name.value.eval()
        # Automatically generate the variable names for this ion.
        equilibrium = ('e' + ion, ion + '_equilibrium',)
        current     = ('i' + ion, ion + '_current',)
        conductance = ('g' + ion, ion + '_conductance',)
        inside      = (ion + 'i', 'intra_' + ion,)
        outside     = (ion + 'o', 'extra_' + ion,)
        for y in stmt.readlist:
            var_name = y.name.value.eval()
            if var_name in equilibrium:
                pass # Ignored, mechanisms output conductances instead of currents.
            elif var_name in inside:
                self.pointers[var_name] = f"self.segment.{ion}"
            elif var_name in outside:
                self.pointers[var_name] = f"self.outside.{ion}"
                self.outside = True
            else:
                raise ValueError(f"Unrecognized USEION READ: \"{var_name}\".")
        for y in stmt.writelist:
            var_name = y.name.value.eval()
            if var_name in current:
                self.pointers[var_name] = f"self.segment.{ion}_current"
                self.accumulators.add(var_name)
            elif var_name in conductance:
                self.pointers[var_name] = f"self.segment.{ion}_conductance"
                self.accumulators.add(var_name)
            elif var_name in inside:
                self.pointers[var_name] = f"self.segment.{ion}_derivative"
                self.accumulators.add(var_name)
            elif var_name in outside:
                self.pointers[var_name] = f"self.outside.{ion}_derivative"
                self.accumulators.add(var_name)
                self.outside = True
            else:
                raise ValueError(f"Unrecognized USEION WRITE: \"{var_name}\".")

    def _gather_conductance_hints(self, parser):
        for x in parser.lookup(ANT.CONDUCTANCE_HINT):
            var_name = x.conductance.get_node_name()
            if var_name not in self.pointers:
                ion = x.ion.get_node_name()
                self.pointers[var_name] = f"self.segment.{ion}_conductance"
                self.accumulators.add(var_name)

    def _gather_nonspecific_currents(self, parser):
        for x in parser.lookup(ANT.NONSPECIFIC_CUR_VAR):
            var_name = x.get_node_name()
            self.pointers[var_name] = f"self.segment.nonspecific_current"
            self.accumulators.add(var_name)

    def _gather_other_mechanisms_IO(self, parser):
        self.other_mechanisms_ = []
        for x in parser.lookup(ANT.POINTER_VAR):
            var_name = x.get_node_name()
            self.other_mechanisms_.append(var_name)
            self.pointers[var_name] = f"self.{var_name}.magnitude"
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
            '':                 solver.sympy_solve_ode,
        }
        def solve(solve_stmt):
            # Solve in-place but return a copy.
            if not isinstance(solve_stmt, SolveStatement):
                return [solve_stmt]
            solve_block  = solve_stmt.block
            solve_method = solve_stmt.method
            if solve_method in ode_methods:
                # Move the CONSERVE statements to the end of the block.
                solve_block.statements.sort(key=lambda stmt: isinstance(stmt, ConserveStatement))
                # Replace CONSERVE statements with a simple multiplicative solution.
                solve_block.map(solver.conserve_statement_solution)
                # 
                method = ode_methods[solve_method]
                assert solve_block.derivative
                solve_block.derivative = False
                for stmt in solve_block:
                    if isinstance(stmt, AssignStatement) and stmt.derivative:
                        method(stmt)
                return [copy.copy(x) for x in solve_block.statements]
            else:
                return [solve_stmt]
        self.breakpoint_block.map(solve)

    def _fixup_breakpoint_IO(self):
        for stmt in self.breakpoint_block:
            if isinstance(stmt, AssignStatement):
                if stmt.lhsn in self.accumulators:
                    stmt.operation = '+='
        self.breakpoint_block.rename_variables(self.pointers)

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

    def _estimate_initial_state(self):
        # Find a reasonable initial state which respects any CONSERVE statements.
        init_state = {state: 0.0 for state in self.states}
        if not self.conserve_statements:
            pass # Zero-init.
        elif len(self.conserve_statements) == 1:
            stmt = self.conserve_statements[0]
            init_value = stmt.conserve_sum / len(stmt.states)
            for state in stmt.states:
                init_state[str(state)] += init_value
        else:
            raise ValueError("Multiple CONSERVE statements are not supported!")
        return init_state

    def _run_initial_block(self, database):
        """
        Use pythons built-in "exec" function to run the INITIAL_BLOCK.
        Sets: initial_state and initial_scope.
        """
        # Initialize the state variables.
        for state, value in self._estimate_initial_state().items():
            self.initial_block.statements.insert(0, AssignStatement(state, value))
            try: self.initial_block.arguments.remove(state)
            except ValueError: pass
        # 
        self.parameters.substitute(self.initial_block)
        # Get the initial values from the database for any pointers that have them.
        for arg in list(self.initial_block.arguments):
            if arg not in self.pointers:
                continue
            db_access = self.pointers[arg]
            db_access = re.sub(r'^self\.segment\.', 'Segment.', db_access)
            db_access = re.sub(r'^self\.inside\.',  'Inside.',  db_access)
            db_access = re.sub(r'^self\.outside\.', 'Outside.', db_access)
            value = database.get(db_access).get_initial_value()
            if value is not None:
                self.initial_block.statements.insert(0, AssignStatement(arg, value))
                self.initial_block.arguments.remove(arg)
        # 
        if self.initial_block.arguments:
            raise ValueError(f"Missing initial values for {', '.join(self.initial_block.arguments)}.")
        # 
        self.initial_block.map(self._solve_steadystate)
        # 
        self.initial_python = code_gen.to_python(self.initial_block)
        self.initial_scope = {}
        code_gen.exec_string(self.initial_python, self.initial_scope)
        self.initial_state = {x: self.initial_scope.pop(x) for x in self.states}

    def _solve_steadystate(self, solve_stmt):
        """ Replace SOLVE STEADYSTATE statements with python code to solve them. """
        if not isinstance(solve_stmt, SolveStatement):
            return [solve_stmt]
        solve_block  = solve_stmt.block
        solve_name   = solve_block.name
        solve_method = solve_stmt.method
        state_list = ', '.join(sorted(self.states))
        max_time   = 60*60*1000 # 1 hour in milliseconds.
        max_delta  = 1e-3
        nstates    = len(self.states)
        py  = f'def __{solve_name}_steadystate():\n'
        py += f'    global {state_list}\n'
        if solve_method == 'sparse':
            py += textwrap.indent(self._derivative_block_to_python(solve_block), "    ")
            py +=  '    import numpy as np\n'
            py +=  '    from scipy.linalg import expm\n'
            py += f'    irm = np.empty(({nstates}, {nstates}))\n'
            py += f'    for idx in range({nstates}):\n'
            py += f'        impulse = [int(x == idx) for x in range({nstates})]\n'
            py += f'        irm[:,idx] = {solve_name}(*impulse)\n'
            py += f'    irm = expm(irm * {max_time})\n'
            py += f'    return irm.dot([{state_list}])\n'
        else:
            assert not solve_block.derivative # Should already be solved!
            py += f"    prev_state = [{state_list}]\n"
            py += f"    for _ in range(int(round({max_time} / dt))):\n"
            py += textwrap.indent(code_gen.to_python(solve_block), "        ")
            py += f"        state = [{state_list}]\n"
            py += f"        if max(abs(a-b) for a,b in zip(prev_state, state)) < {max_delta}: break\n"
            py +=  "        prev_state = state\n"
            py += f"    return state\n"
        py += f'{state_list} = __{solve_name}_steadystate()\n'
        return [py]

    def _derivative_block_to_python(self, block):
        assert block.derivative
        for stmt in block:
            if isinstance(stmt, AssignStatement) and stmt.derivative:
                assert stmt.lhsn in self.states
                stmt.lhsn = f'_d_{stmt.lhsn}'
                stmt.derivative = False
        deriv_pycode = f"def {block.name}({', '.join(self.states)}):\n"
        for state in self.states:
            deriv_pycode += f"    _d_{state} = 0.0\n"
        deriv_pycode += code_gen.to_python(block, "    ")
        deriv_pycode += f"    return [{', '.join(f'_d_{state}' for state in self.states)}]\n\n"
        return deriv_pycode

    def _substitute_initial_scope(self, block):
        block.gather_arguments()
        for arg, value in self.initial_scope.items():
            if arg in self.pointers:
                continue
            if arg not in block.arguments:
                continue
            block.statements.insert(0, AssignStatement(arg, value))
            block.arguments.remove(arg)

    def _compile_breakpoint_block(self):
        globals_ = {
                'Compute': Compute,
        }
        # Insert parameters & initial-values into the solve blocks.
        for solve_stmt in self.breakpoint_block:
            if not isinstance(solve_stmt, SolveStatement):
                continue
            solve_block  = solve_stmt.block
            solve_name   = solve_block.name
            solve_method = solve_stmt.method
            self._substitute_initial_scope(solve_block)
            self.parameters.substitute(solve_block)
            if solve_method == 'sparse':
                globals_[solve_name] = self._initialize_kinetic_model(solve_block)
            else:
                raise ValueError(f'Unsupported SOLVE method {solve_method}.')
        # 
        magnitude = sympy.Symbol('self.magnitude')
        for name, (value, units) in self.instance_parameters.items():
            self.breakpoint_block.statements.insert(0, AssignStatement(name, value * magnitude))
        # 
        self._substitute_initial_scope(self.breakpoint_block)
        self.parameters.substitute(self.breakpoint_block)
        # 
        self.advance_pycode = (
                "@Compute\n"
                "def advance(self):\n"
                + code_gen.to_python(self.breakpoint_block, "    "))
        code_gen.exec_string(self.advance_pycode, globals_)
        self.advance_bytecode = globals_['advance']


    def _initialize_kinetic_model(self, block):
        assert block.derivative


        # Get the derivative function
        pass
        # Build the IRM table
        pass
        # Make Compute'd method to advance the state.
        pass

        return advance_function



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
        block.gather_arguments()
        for name, (value, units) in self.items():
            if value is None:
                continue
            if name not in block.arguments:
                continue
            block.statements.insert(0, AssignStatement(name, value))
            block.arguments.remove(name)

class OmnipresentNmodlMechanism(NMODL, OmnipresentMechanism):
    pass

class LocalNmodlMechanismSpecification(NMODL, LocalMechanismSpecification):
    def other_mechanisms(self):
        return self.other_mechanisms_

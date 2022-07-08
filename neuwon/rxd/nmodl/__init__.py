from collections.abc import Callable, Iterable, Mapping
from neuwon.database import Compute
from neuwon.rxd.mechanisms import Mechanism
from . import code_gen, cache, solver
from .. import lti_sim
from .parser import (NmodlParser, ANT,
        SolveStatement,
        AssignStatement,
        ConserveStatement)
import copy
import os.path
import re
import sympy
import textwrap

__all__ = ['NMODL']


class NMODL(Mechanism):
    def __init__(self, filename, use_cache=True):
        """
        Argument filename is an NMODL file to load.
                The standard NMODL file name extension is ".mod"
        """
        self.filename = os.path.abspath(str(filename))
        self._use_cache = bool(use_cache)

    def initialize(self, rxd_model):
        external_parameters = {
                'dt':       rxd_model.get_time_step(),
                'celsius':  rxd_model.get_temperature(),
        }
        try:
            if self._use_cache and cache.try_loading(self.filename, external_parameters, self): pass
            else:
                parser = NmodlParser(self.filename)
                self._check_for_unsupported(parser)
                self.name, self.point_process,  self.title, self.description = parser.gather_documentation()
                self._gather_parameters(parser.gather_parameters(), **external_parameters)
                self.states = parser.gather_states()
                blocks = parser.gather_code_blocks()
                self.all_blocks         = list(blocks.values())
                self.initial_block      = blocks['INITIAL']
                self.breakpoint_block   = blocks['BREAKPOINT']
                for block in self.all_blocks:
                    block.substitute_parameters(self.parameters)
                self.conserve_statements = self._gather_conserve_statements()
                self._gather_IO(parser)
                self._solve()
                self._fixup_breakpoint_IO()
            cache.save(self.filename, external_parameters, self)

            self._run_initial_block(rxd_model.database)
            self._compile_breakpoint_block()
            if self.omnipresent:
                cls = self._initialize_omnipresent_mechanism_class(rxd_model.database)
            else:
                cls = self._initialize_local_mechanism_class(rxd_model.database)
            self._register_nonspecific_conductances(rxd_model, cls)
            return cls
        except Exception:
            print('ERROR while loading file', self.filename, flush=True)
            raise

    def _check_for_unsupported(self, parser):
        # TODO: support for NONLINEAR?
        # TODO: support for INCLUDE?
        # TODO: support for COMPARTMENT?
        # TODO: support for ARRAY?
        disallow = (
            'FUNCTION_TABLE_BLOCK',
            'LON_DIFUSE',
            'TABLE_STATEMENT',
            'VERBATIM',
        )
        for x in disallow:
            if parser.lookup(getattr(ANT, x)):
                raise ValueError('"%s"s are not allowed.'%x)

    def _gather_parameters(self, parameters, *, dt, celsius):
        """
        Attribute parameters is dictionary of {parameter -> value}.

        Attribute units is dictionary of {parameter -> units}.

        Attribute instance_parameters is dictionary of {parameter -> value}.
            The surface area parameters are special because each segment of neuron
            has its own surface area and so their actual values are different for
            each instance of the mechanism. Point processes also have a special
            unit for instance parameters, which defaults to 1. These parameters are
            not in-lined directly into the source code, instead they are stored
            alongside the state variables and accessed at run time.
        """
        # Split parameters and units into separate dictionaries.
        self.units      = units_dict = {k: u for k, (v,u) in parameters.items()}
        self.parameters = parameters = {k: v for k, (v,u) in parameters.items()}
        # Split off the instance_parameters.
        self.instance_parameters = {}
        for name, units in units_dict.items():
            if units is not None:
                if self.point_process and ('/' + self.mechanism_name) in units:
                    self.instance_parameters[name] = parameters.pop(name)
                elif '/cm2' in units:
                    # Convert from NEURONs cm^2 to NEUWONs um^2.
                    units_dict[name] = units.replace('/cm2', '/um2')
                    x = (1e-6 * 1e-6) / (1e-2 * 1e-2)
                    self.instance_parameters[name] = parameters.pop(name) * x
        # Fill in external parameters.
        assert parameters.get('dt', None) is None, 'Parameter "dt" is reserved.'
        parameters['dt'] = dt
        if units_dict.get('dt', None) is None: units_dict['dt'] = 'ms'
        # Allow NMODL file to override temperature.
        if parameters.get('celsius', None) is None: parameters['celsius'] = celsius
        if units_dict.get('celsius', None) is None: units_dict['celsius'] = 'degC'
        assert units_dict['celsius'] == 'degC'

    def _gather_conserve_statements(self):
        conserve_statements = []
        for block in self.all_blocks:
            for stmt in block:
                if isinstance(stmt, ConserveStatement):
                    conserve_statements.append(stmt)

        if len(conserve_statements) > 1:
            raise ValueError("Multiple CONSERVE statements are not supported.")
        # states  = nmodl.dsl.to_nmodl(stmt.react).split('+')
        # if set(states) != set(self.state_names) or not stmt.expr.is_number():
        #     raise ValueError('CONSERVE statement must be in the form: sum-of-all-states = number.')
        return conserve_statements

    def _gather_IO(self, parser):
        """
        Determine what persistent data the mechanism accesses.

        Sets attributes: "pointers", "accumulators", "other_mechanisms_", and
        the following flags: "omnipresent", "outside".
        """
        self.pointers = {} # Dict of {'nmodl_variable': 'database_access'}
        self.accumulators = set() # NMODL variable names.
        self.nonspecific_conductances = {} # Dict of {'ion_name': 'equilibrium_variable'}
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
        if 'v' in all_args:
            self.pointers['v'] = 'self.segment.voltage'
        if 'diam' in all_args:
            self.pointers['diam'] = 'self.segment.diameter'
        if 'area' in all_args:
            self.pointers['area'] = 'self.segment.surface_area'
        if 'volume' in all_args:
            self.pointers['volume'] = 'self.segment.inside_volume'

    def _process_useion_statement(self, stmt):
        ion = stmt.name.value.eval()
        # Automatically generate the variable names for this ion.
        equilibrium = ('e' + ion, ion + '_equilibrium',)
        current     = ('i' + ion, ion + '_current',)
        conductance = ('g' + ion, ion + '_conductance',)
        inside      = (ion + 'i', ion + '_inside',)
        outside     = (ion + 'o', ion + '_outside',)
        # 
        read_vars   = [x.name.value.eval() for x in stmt.readlist]
        write_vars  = [x.name.value.eval() for x in stmt.writelist]
        nonspecific = any(x in equilibrium for x in write_vars)
        # 
        for var_name in read_vars:
            assert not nonspecific
            if var_name in equilibrium:
                pass # Ignored, mechanisms output conductances instead of currents.
                # self.pointers[var_name] = f'self.segment.{ion}_reversal_potential'
            elif var_name in inside:
                self.pointers[var_name] = f'self.segment.{ion}'
            elif var_name in outside:
                self.pointers[var_name] = f'self.outside.{ion}'
                self.outside = True
            else:
                raise ValueError(f'Unrecognized USEION READ: "{var_name}".')
        for var_name in write_vars:
            if var_name in equilibrium:
                self.pointers[var_name] = f'self.segment.{ion}_reversal_potential'
                self.nonspecific_conductances[ion] = var_name
            elif var_name in current:
                if nonspecific:
                    self.pointers[var_name] = f'self.segment.nonspecific_current'
                else:
                    self.pointers[var_name] = f'self.segment.{ion}_current'
                self.accumulators.add(var_name)
            elif var_name in conductance:
                if nonspecific:
                    self.pointers[var_name] = f'self.{ion}_conductance'
                else:
                    self.pointers[var_name] = f'self.segment.{ion}_conductance'
                self.accumulators.add(var_name)
            elif var_name in inside:
                assert not nonspecific
                self.pointers[var_name] = f'self.segment.{ion}_derivative'
                self.accumulators.add(var_name)
            elif var_name in outside:
                assert not nonspecific
                self.pointers[var_name] = f'self.outside.{ion}_derivative'
                self.accumulators.add(var_name)
                self.outside = True
            else:
                raise ValueError(f'Unrecognized USEION WRITE: "{var_name}".')

    def _gather_conductance_hints(self, parser):
        for x in parser.lookup(ANT.CONDUCTANCE_HINT):
            var_name = x.conductance.get_node_name()
            if var_name not in self.pointers:
                ion = x.ion.get_node_name()
                self.pointers[var_name] = f'self.segment.{ion}_conductance'
                self.accumulators.add(var_name)

    def _gather_nonspecific_currents(self, parser):
        for x in parser.lookup(ANT.NONSPECIFIC_CUR_VAR):
            var_name = x.get_node_name()
            self.pointers[var_name] = f'self.segment.nonspecific_current'
            self.accumulators.add(var_name)

    def _gather_other_mechanisms_IO(self, parser):
        self.other_mechanisms_ = []
        for x in parser.lookup(ANT.POINTER_VAR):
            var_name = x.get_node_name()
            self.other_mechanisms_.append(var_name)
            self.pointers[var_name] = f'self.{var_name}.magnitude'
        self.other_mechanisms_ = tuple(sorted(self.other_mechanisms_))

    def _solve(self):
        """
        Replace SolveStatements with the solved equations to advance the equations.
        """
        # These solver methods assume that every statement is independent,
        # ie: not part of a system of equations.
        independent_methods = {
            'euler':            solver.forward_euler,
            'derivimplicit':    solver.backward_euler,
            '':                 solver.backward_euler,
            'cnexp':            solver.crank_nicholson,
        }
        # First split all of the calls to "SOLVE my_block METHOD my_method" out
        # of the breakpoint block.
        solve_stmts = []
        def find_solve_stmts(stmt):
            if not isinstance(stmt, SolveStatement):
                return [stmt]
            solve_stmts.append(stmt)
            return []
        self.breakpoint_block.map(find_solve_stmts)
        # Then solve the target blocks.
        self.solved_blocks = []
        for stmt in solve_stmts:
            solve_block  = stmt.block
            solve_method = stmt.method
            assert solve_block.derivative

            if method := independent_methods.get(solve_method, False):
                # Move the CONSERVE statements to the end of the block.
                solve_block.statements.sort(key=lambda stmt: isinstance(stmt, ConserveStatement))
                # Replace CONSERVE statements with a simple multiplicative solution.
                solve_block.map(solver.conserve_statement_solution)
                # Solve each equation in-place.
                solve_block.derivative = False
                next_states = [] # Don't modify the state until all equations are computed.
                for stmt in solve_block:
                    if isinstance(stmt, AssignStatement) and stmt.derivative:
                        method(stmt)
                        next_states.append(stmt.lhsn)
                        stmt.lhsn = f'_next_{stmt.lhsn}'
                # Update all states instantaneously.
                for x in next_states:
                    solve_block.statements.append(AssignStatement(x, f'_next_{x}'))
                # Prepend the solved block directly into the breakpoint block.
                self.breakpoint_block.statements = solve_block.statements + self.breakpoint_block.statements

            elif solve_method == 'sparse':
                # Call the LTI_SIM program.
                inputs = self._gather_inputs(solve_block)
                lti_sim.LTI_Model(self.name,
                        inputs,
                        self.states,
                        self._compile_derivative_block(solve_block, inputs),
                        1.0, # TODO: conserve_sum?
                        self.parameters['dt'])

                1/0 # TODO

            else:
                raise ValueError(f'Unsupported SOLVE method {solve_method}.')

    def _gather_inputs(self, block) -> '[lti_sim.Input]':
        """
        Determine the external variable inputs to the given block.
        These inputs effect the derivative on a moment to moment basis.
        """
        inputs = []
        block.gather_arguments()
        for variable in block.arguments:
            if variable not in self.pointers: continue
            if variable in self.states: continue
            db_access = self.pointers[variable].split('.')
            db_access.remove('self')
            db_class_ptr, db_attr = db_access
            if variable == 'v':
                inp = lti_sim.LinearInput(variable, -100, 100)
            else:
                inp = lti_sim.LogarithmicInput(variable, 0, 1000)
            inputs.append(inp)
        return inputs

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
            py += textwrap.indent(self._derivative_block_to_python(solve_block), '    ')
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
            py += f'    prev_state = [{state_list}]\n'
            py += f'    for _ in range(int(round({max_time} / dt))):\n'
            py += textwrap.indent(code_gen.to_python(solve_block), '        ')
            py += f'        state = [{state_list}]\n'
            py += f'        if max(abs(a-b) for a,b in zip(prev_state, state)) < {max_delta}: break\n'
            py +=  '        prev_state = state\n'
            py += f'    return state\n'
        py += f'{state_list} = __{solve_name}_steadystate()\n'
        return [py]

    def _compile_derivative_block(self, block, inputs):
        assert block.derivative
        for stmt in block:
            if isinstance(stmt, AssignStatement) and stmt.derivative:
                assert stmt.lhsn in self.states
                stmt.lhsn = f'__d_{stmt.lhsn}'
                stmt.derivative = False
        1/0 # INITIAL SCOPE!
        arguments = [inp.name for inp in inputs] + self.states
        pycode = f"def {block.name}({', '.join(arguments)}):\n"
        for state in self.states:
            pycode += f"    __d_{state} = 0.0\n"
        pycode += code_gen.to_python(block, "    ")
        pycode += f"    return [{', '.join(f'__d_{state}' for state in self.states)}]\n\n"
        scope = {}
        code_gen.exec_string(pycode, scope)
        return scope[block.name]

    def _substitute_initial_scope(self, block):
        block.gather_arguments()
        for arg, value in self.initial_scope.items():
            if arg in self.pointers:
                continue
            if arg not in block.arguments:
                continue
            block.statements.insert(0, AssignStatement(arg, value))
            block.arguments.remove(arg)

    def _fixup_breakpoint_IO(self):
        for stmt in self.breakpoint_block:
            if isinstance(stmt, AssignStatement):
                if stmt.lhsn in self.accumulators:
                    stmt.operation = '+='
        self.breakpoint_block.rename_variables(self.pointers)
        self.breakpoint_block.substitute_parameters(self.parameters) # Substitute for 'dt'.

    def _compile_breakpoint_block(self):
        globals_ = {
                'Compute': Compute,
        }
        # 
        magnitude = sympy.Symbol('self.magnitude')
        for name, value in self.instance_parameters.items():
            self.breakpoint_block.statements.insert(0, AssignStatement(name, value * magnitude))
        # 
        self._substitute_initial_scope(self.breakpoint_block)
        # 
        self.advance_pycode = (
                '@Compute\n'
                'def advance(self):\n'
                + code_gen.to_python(self.breakpoint_block, '    '))
        code_gen.exec_string(self.advance_pycode, globals_)
        self.advance_bytecode = globals_['advance']

        # if self.lti_advance:
        #     solve_block = self.lti_advance
        #     breakpoint_block = self.advance_bytecode
        #     def advance(self):
        #         solve_block(self)
        #         breakpoint_block(self)
        #     self.advance_bytecode = advance

    def _register_nonspecific_conductances(self, rxd_model, db_class):
        for (ion, e_variable) in self.nonspecific_conductances.items():
            e_value = self.parameters[e_variable]
            e_units = self.units[e_variable]
            if e_units is not None: assert e_units.lower() == 'mv'
            rxd_model.register_nonspecific_conductance(db_class, ion, e_value)

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
        mechanism_superclass = type(self.name, (NmodlMechanism,), {
            '__doc__':              self.title + '\n\n' + self.description,
            '__slots__':            (),
            'name':                 self.name,
            '__init__':             NMODL._instance__init__,
            '_point_process':       self.point_process,
            '_outside':             self.outside,
            '_other_mechanisms':    self.other_mechanisms_,
            'advance':              self.advance_bytecode,
            '_advance_pycode':      self.advance_pycode,
        })
        mech_data = database.add_class(mechanism_superclass)
        mech_data.add_attribute('segment', dtype='Segment')
        if self.outside:
            mech_data.add_attribute('outside', dtype='Extracellular')
        if self.point_process:
            mech_data.add_attribute('magnitude', 1.0,
                    doc="") # TODO?
        else:
            mech_data.add_attribute('magnitude',
                    units = database.get('Segment.surface_area').get_units(),
                    doc="") # TODO?
        for name in self.states:
            mech_data.add_attribute(name, initial_value=self.initial_state[name], units=name)
        for name in self.other_mechanisms_:
            mech_data.add_attribute(name, dtype=name)
        return mech_data.get_instance_type()

    @staticmethod
    def _instance__init__(self, segment, outside=None, magnitude=1.0, *other_mechanisms):
        """ Insert this mechanism onto the given segment. """
        cls = type(self)
        self.segment = segment
        if cls._point_process:
            self.magnitude = magnitude
        else:
            self.magnitude = magnitude * segment.surface_area
        if cls._outside:
            if outside is not None:
                self.outside = outside
            else:
                self.outside = segment.outside
        assert len(cls._other_mechanisms) == len(other_mechanisms)
        for name, ref in zip(cls._other_mechanisms, other_mechanisms):
            setattr(self, name, ref)

class NmodlMechanism(Mechanism):
    __slots__ = ()
    @classmethod
    def get_name(cls):
        return cls.name
    @classmethod
    def other_mechanisms(cls):
        return cls._other_mechanisms

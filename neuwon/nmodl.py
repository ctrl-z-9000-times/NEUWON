import nmodl
import nmodl.ast
import nmodl.dsl
import nmodl.symtab
import sympy
import sympy.printing.pycode
import numba.cuda
import numba
import numpy as np
import math
import os.path
import itertools
import copy
from neuwon.common import celsius, Real, Pointer
import neuwon.units
from neuwon.reactions import Reaction
from scipy.sparse.linalg import expm

ANT = nmodl.ast.AstNodeType

# TODO: Use impulse response integration method in place of sparse solver...
#       How to dispatch to it?
#       TODO: Update the kinetic model given all of the stuff that I've done:
#               Derivative function instead of sparse deriv equations.

# TODO: Write function to check that derivative_functions are Linear &
# time-invariant. Put this in the KineticModel class.

# TODO: Initial state. Mostly works...  Need code to run a simulation until it
# reaches a steady state, given only the derivative function.

# TODO: Initial state...

default_parameters = {
    "celsius": (celsius, "degC")
}

library = {
    "hh": ("neuwon/nmodl_library/hh.mod",
        dict(pointers={"gl": Pointer("L", conductance=True)},
             method_override = "exact",
             parameter_overrides = {"celsius": 6.3})),

    "na11a": ("neuwon/nmodl_library/Balbi2017/Nav11_a.mod",
        dict(method_override="cnexp")),

    "Kv11_13States_temperature2": ("neuwon/nmodl_library/Kv-kinetic-models/hbp-00009_Kv1.1/hbp-00009_Kv1.1__13States_temperature2/hbp-00009_Kv1.1__13States_temperature2_Kv11.mod",
        dict(method_override="cnexp")),

    "AMPA5": ("neuwon/nmodl_library/Destexhe1994/ampa5.mod",
        dict(method_override="cnexp",
             pointers={"C": Pointer("Glu", extra_concentration=True)})),

    "caL": ("neuwon/nmodl_library/Destexhe1994/caL3d.mod",
        dict(method_override="cnexp",
             pointers={"g": Pointer("ca", conductance=True)})),

    # I;m not sure I want to use this one without editing the file first.
    "rel": ("neuwon/nmodl_library/Destexhe1994/release.mod",
        dict(method_override="cnexp",)),

}

class NmodlMechanism(Reaction):
    def __init__(self, filename, pointers={}, method_override=False, parameter_overrides={}):
        self.filename = os.path.normpath(str(filename))
        with open(self.filename, 'rt') as f: nmodl_text = f.read()
        # Parse the NMDOL file into an abstract syntax tree (AST).
        AST = nmodl.dsl.NmodlDriver().parse_string(nmodl_text)
        nmodl.dsl.visitor.ConstantFolderVisitor().visit_program(AST)
        nmodl.symtab.SymtabVisitor().visit_program(AST)
        nmodl.dsl.visitor.InlineVisitor().visit_program(AST)
        nmodl.dsl.visitor.SympyConductanceVisitor().visit_program(AST)
        nmodl.dsl.visitor.KineticBlockVisitor().visit_program(AST)
        # Helpful debugging printouts:
        if False: print(nmodl.dsl.to_nmodl(AST))
        if False: print(AST.get_symbol_table())
        if False: nmodl.ast.view(AST)
        self._set_ast(AST)
        self._check_for_unsupported()
        self._gather_documentation()
        self._gather_units()
        self._gather_parameters(default_parameters, parameter_overrides)
        self.states = set(v.get_name() for v in
                self.symbols.get_variables_with_properties(nmodl.symtab.NmodlType.state_var))
        self._gather_IO(pointers)
        self._gather_functions(method_override)
        self._discard_ast()
        self._run_initial_block()

    def set_time_step(self, time_step):
        """ Second initialize, called during neuwon.Model.__init__() """
        self._compile_breakpoint_block(time_step)

    def _set_ast(self, AST):
        """ Sets visitor, lookup, and symbols. """
        self.visitor = nmodl.dsl.visitor.AstLookupVisitor()
        self.lookup = lambda n: self.visitor.lookup(AST, n)
        nmodl.symtab.SymtabVisitor().visit_program(AST)
        self.symbols = AST.get_symbol_table()
        # Helpful debugging printouts:
        if False: print(nmodl.dsl.to_nmodl(AST))
        if False: print(AST.get_symbol_table())
        if False: nmodl.ast.view(AST)

    def _discard_ast(self):
        """ Cleanup. Delete all references to the nmodl library.

        The nmodl library is implemented in C++ and as such does not
        automatically support some python features. In specific: objects
        returned from the nmodl library do not support copying or pickling. """
        del self.visitor
        del self.lookup
        del self.symbols

    def _check_for_unsupported(self):
        if self.lookup(ANT.FUNCTION_TABLE_BLOCK):
            raise ValueError("\"FUNCTION_TABLE\" is not allowed.")
        if self.lookup(ANT.VERBATIM):
            raise ValueError("\"VERBATIM\" is not allowed.")
        if self.lookup(ANT.LON_DIFUSE):
            raise ValueError("\"LONGITUDINAL_DIFFUSION\" is not allowed.")
        # TODO: No support for NONLINEAR!
        # TODO: No support for INCLUDE!
        # TODO: No support for COMPARTMENT!
        # TODO: Deal with arrays?

    def _gather_documentation(self):
        """ Sets name, title, and description.
        This assumes that the first block comment is the primary documentation. """
        self._name = os.path.split(self.filename)[1]
        for x in self.lookup(ANT.SUFFIX):
            self._name = x.name.get_node_name()
        title = self.lookup(ANT.MODEL)
        if not title:
            self.title = ""
        else:
            self.title = title[0].title.eval().strip()
            if self.title.startswith(self._name + ".mod"):
                self.title = self.title[len(self._name + ".mod"):].strip()
            if self.title:
                self.title = self.title[0].title() + self.title[1:] # Capitalize the first letter.
        block_comments = self.lookup(ANT.BLOCK_COMMENT)
        if block_comments:
            self.description = block_comments[0].statement.eval()
        else:
            self.description = ""

    def name(self):
        return self._name

    def _gather_units(self):
        """
        Sets flag "use_units" which determines how to deal with differences
        between NEURONs and NEUWONs unit systems:
          * True: modify the mechanism to use NEUWONs unit system.
          * False: convert the I/O into NEURON unit system.
        """
        self.units = neuwon.units.Units()
        for AST in self.lookup(ANT.UNIT_DEF):
            self.units.add_unit(AST.unit1.name.eval(), AST.unit2.name.eval())
        self.use_units = not any(x.eval() == "UNITSOFF" for x in self.lookup(ANT.UNIT_STATE))
        self.use_units = False
        if not self.use_units: print("Warning: UNITSOFF detected.")

    def _gather_parameters(self, default_parameters, parameter_overrides):
        """ Sets parameters & surface_area_parameters.

        The surface area parameters are special because each segment of neuron
        has its own surface area and so their actual values are different for
        each instance of the mechanism. They are not in-lined directly into the
        source code, instead they are stored alongside the state variables and
        accessed at run time. """
        # Parse the parameters from the NMODL file.
        self.original_parameters = {}
        self.parameters = dict(default_parameters)
        for assign in self.lookup(ANT.PARAM_ASSIGN):
            name = str(self.visitor.lookup(assign, ANT.NAME)[0].get_node_name())
            value = self.visitor.lookup(assign, [ANT.INTEGER, ANT.DOUBLE])
            units = self.visitor.lookup(assign, ANT.UNIT)
            if value: value = float(value[0].eval())
            else: continue
            assert(not bool(units) or len(units) == 1)
            units = units[0].get_node_name() if units else None
            self.original_parameters[name] = (value, units)
            self.parameters[name]          = (value, units)
        # Deal with all parameter_overrides.
        parameter_overrides = dict(parameter_overrides)
        for name in list(parameter_overrides):
            if name in self.parameters:
                old_value, units = self.parameters.pop(name)
                self.parameters[name] = (parameter_overrides.pop(name), units)
        if parameter_overrides:
            extra_parameters = "%s"%", ".join(str(x) for x in parameter_overrides.keys())
            raise ValueError("Invalid parameter overrides: %s."%extra_parameters)
        # Convert all unit on the parameters into NEUWONs unit system.
        if self.use_units:
            for name, (value, units) in self.parameters.items():
                factor, dimensions = self.units.standardize(units)
                self.parameters[name] = (factor * value, dimensions)
        # Split out surface_area_parameters.
        self.surface_area_parameters = {}
        for name, (value, units) in self.original_parameters.items():
            if units and "/cm2" in units:
                self.surface_area_parameters[name] = self.parameters.pop(name)
        if False:
            print("Parameters:", self.parameters)
            print("Surface Area Parameters:", self.surface_area_parameters)

    def _gather_IO(self, pointers):
        """ Determine what external data the mechanism accesses. """
        self._pointers = dict(pointers)
        assert(all(isinstance(ptr, Pointer) for ptr in self._pointers.values()))
        # Assignments to the variables in output_currents are ignored because
        # NEUWON converts all mechanisms to use conductances instead of currents.
        self.output_currents = []
        self.output_nonspecific_currents = []
        for x in self.lookup(ANT.USEION):
            ion = x.name.value.eval()
            # Automatically generate the variable names for this ion.
            equilibrium = 'e' + ion
            current = 'i' + ion
            intra = ion + 'i'
            extra = ion + 'o'
            for y in x.readlist:
                variable = y.name.value.eval()
                if variable == equilibrium:
                    pass # Ignored, mechanisms output conductances instead of currents.
                elif variable == intra:
                    self._add_pointer(variable, ion, intra_concentration=True)
                elif variable == extra:
                    self._add_pointer(variable, ion, extra_concentration=True)
                else: raise ValueError("Unrecognized ion READ: \"%s\"."%variable)
            for y in x.writelist:
                variable = y.name.value.eval()
                if variable == current:
                    self.output_currents.append(variable)
                elif variable == intra:
                    self._add_pointer(variable, ion, intra_release_rate=True)
                elif variable == extra:
                    self._add_pointer(variable, ion, extra_release_rate=True)
                else: raise ValueError("Unrecognized ion WRITE: \"%s\"."%variable)
        for x in self.lookup(ANT.NONSPECIFIC_CUR_VAR):
            self.output_nonspecific_currents.append(x.name.get_node_name())
            print("Warning: NONSPECIFIC_CURRENT detected.")
        self.output_currents.extend(self.output_nonspecific_currents)
        for x in self.lookup(ANT.CONDUCTANCE_HINT):
            variable = x.conductance.get_node_name()
            ion = x.ion.get_node_name() if x.ion else None
            if variable in self._pointers:
                assert(ion is None or self._pointers[variable].species == ion)
            else:
                self._add_pointer(variable, ion, conductance=True)
        num_conductances = sum(ptr.conductance for ptr in self._pointers.values())
        assert(len(self.output_currents) == num_conductances) # Check for missing CONDUCTANCE_HINTs.
        for name in self.states:
            self._add_pointer(name, reaction_instance=Real)
        for name in self.surface_area_parameters:
            self._add_pointer(name, reaction_instance=Real)
            # # Ensure that all output pointers are written to.
            # surface_area_parameters = sorted(self.surface_area_parameters)
            # for variable, pointer in self._pointers.items():
            #     if pointer.conductance and variable not in self.breakpoint_block.assigned:
            #         if variable in surface_area_parameters:
            #             idx = surface_area_parameters.index(variable)
            #             self.breakpoint_block.statements.append(
            #                     AssignStatement(variable, variable, pointer=pointer))                

    def _add_pointer(self, name, *args, **kw_args):
        pointer = Pointer(*args, **kw_args)
        if name in self._pointers:
            raise ValueError("Name conflict: \"%s\" used for %s and %s"%(
                    name, self._pointers[name], pointer))
        self._pointers[name] = pointer

    def pointers(self):
        return self._pointers

    def _gather_functions(self, method_override):
        """ Sets initial_block, breakpoint_block, and derivative_blocks. """
        self.method_override = str(method_override) if method_override else False
        self.derivative_blocks = {}
        for x in self.lookup(ANT.DERIVATIVE_BLOCK):
            name = x.name.get_node_name()
            self.derivative_blocks[name] = CodeBlock(self, x)
        assert(len(self.derivative_blocks) <= 1) # Otherwise unimplemented.
        self.initial_block = CodeBlock(self, self.lookup(ANT.INITIAL_BLOCK).pop())
        self.breakpoint_block = CodeBlock(self, self.lookup(ANT.BREAKPOINT_BLOCK).pop())
        if "v" in self.breakpoint_block.arguments: self._add_pointer("v", voltage=True)

    def _parse_statement(self, AST):
        """ Returns a list of Statement objects. """
        original = AST
        if AST.is_unit_state():             return []
        if AST.is_local_list_statement():   return []
        if AST.is_conductance_hint():       return []
        if AST.is_if_statement(): return [IfStatement(self, AST)]
        if AST.is_conserve():     return [ConserveStatement(self, AST)]
        if AST.is_expression_statement():
            AST = AST.expression
        else:
            raise ValueError("Unrecognized syntax at %s."%nmodl.dsl.to_nmodl(original))
        if AST.is_statement_block():
            return list(itertools.chain.from_iterable(
                    self._parse_statement(stmt) for stmt in AST.statements))
        if AST.is_solve_block() and not AST.steadystate:
            block = copy.deepcopy(self.derivative_blocks[AST.block_name.get_node_name()])
            AST.ifsolerr # TODO: What does this do?
            if self.method_override:
                block._solve(self.method_override)
            else:
                block._solve(AST.method.get_node_name())
            return block.statements
        if AST.is_solve_block() and AST.steadystate:
            # TODO: Determine initial state here?
            return []
        is_derivative = AST.is_diff_eq_expression()
        if is_derivative: AST = AST.expression
        if AST.is_binary_expression():
            assert(AST.op.eval() == "=")
            lhsn = AST.lhs.name.get_node_name()
            rhs = self._parse_expression(AST.rhs)
            if lhsn in self.output_nonspecific_currents: return []
            if lhsn in self.output_currents: return []
            return [AssignStatement(lhsn, rhs,
                    derivative = is_derivative,
                    pointer = self._pointers.get(lhsn, None),)]
        raise ValueError("Unrecognized syntax at %s."%nmodl.dsl.to_nmodl(original))

    def _parse_expression(self, AST):
        """ Returns a SymPy expression. """
        if AST.is_wrapped_expression() or AST.is_paren_expression():
            return self._parse_expression(AST.expression)
        elif AST.is_name():
            return sympy.symbols(AST.get_node_name())
        elif AST.is_var_name():
            name = AST.name.get_node_name()
            if name in self.parameters:
                return sympy.Float(self.parameters[name][0], 18)
            else:
                return sympy.symbols(AST.name.get_node_name())
        elif AST.is_binary_expression():
            op = AST.op.eval()
            lhs = self._parse_expression(AST.lhs)
            rhs = self._parse_expression(AST.rhs)
            if   op == "+": return lhs + rhs
            elif op == "-": return lhs - rhs
            elif op == "*": return lhs * rhs
            elif op == "/": return lhs / rhs
            elif op == "^": return lhs ** rhs
            elif op == "<": return lhs < rhs
            elif op == ">": return lhs > rhs
            else: raise ValueError("Unrecognized syntax at %s."%nmodl.dsl.to_nmodl(AST))
        elif AST.is_unary_expression():
            op = AST.op.eval()
            if op == "-": return - self._parse_expression(AST.expression)
            else: raise ValueError("Unrecognized syntax at %s."%nmodl.dsl.to_nmodl(AST))
        elif AST.is_function_call():
            name = AST.name.get_node_name()
            if name == "fabs": name = "abs"
            args = [self._parse_expression(x) for x in AST.arguments]
            return sympy.Function(name)(*args)
        elif AST.is_double_unit():
            if self.use_units:
                factor, dimensions = self.units.standardize(AST.unit.name.eval())
            else: factor = 1
            return self._parse_expression(AST.value) * factor
        elif AST.is_integer(): return sympy.Integer(AST.eval())
        elif AST.is_double():  return sympy.Float(AST.eval(), 18)
        else:
            raise ValueError("Unrecognized syntax at %s."%nmodl.dsl.to_nmodl(AST))

    def _run_initial_block(self, initial_assumptions={"v": -70e-3}):
        """ Use pythons built-in "exec" function to run the INITIAL_BLOCK.
        Sets: initial_state and initial_scope. """
        initial_globals = {'math': math}
        self.initial_scope = initial_locals = {}
        for arg in self.initial_block.arguments:
            if arg not in initial_assumptions: raise ValueError("Missing initial value for \"%s\"."%arg)
            initial_globals[arg] = initial_assumptions[arg]
        initial_state = 1 / len(self.states) # TODO: Determine this from the conserve statement!
        # TODO: If there are not conserve statements to constrain the initial state then assume zero init.
        initial_globals["_index_"] = 0
        for x in self.states: initial_locals["_" + x] = [initial_state]
        initial_python = self.initial_block.to_python()
        try:
            exec(initial_python, initial_globals, initial_locals)
        except:
            print("Error while exec'ing the following python code:")
            print("globals()", repr(initial_globals))
            print("locals()", repr(initial_locals))
            print(initial_python)
            raise
        self.initial_state = {x: initial_locals.pop(mangle(x)) for x in self.states}

    def _compile_breakpoint_block(self, time_step):
        input_variables = list(self.breakpoint_block.arguments)
        initial_scope_carryover = []
        for variable, initial_value in self.initial_scope.items():
            if variable in input_variables:
                input_variables.remove(variable)
                initial_scope_carryover.append(variable, initial_value)
        for arg in input_variables:
            if arg not in self._pointers:
                raise ValueError("Mishandled argument: \"%s\"."%arg)
        arguments = sorted(map(mangle, self._pointers))
        preamble = []
        preamble.append("import math")
        preamble.append("import numba.cuda")
        preamble.append("def BREAKPOINT(_locations_, %s):"%", ".join(arguments))
        preamble.append("    _index_ = numba.cuda.grid(1)")
        preamble.append("    if _index_ >= _locations_.shape[0]: return")
        preamble.append("    _location_ = _locations_[_index_]")
        for variable_value_pair in initial_scope_carryover:
            preamble.append("    %s = %s"%variable_value_pair)
        if not self.use_units: time_step *= 1000 # Convert from NEUWONs seconds to NEURONs milliseconds.
        preamble.append("    time_step = "+str(time_step))
        for variable, pointer in self._pointers.items():
            if not pointer.read: continue
            index = "_index_" if pointer.dtype else "_location_"
            factor = str(pointer.NEURON_conversion_factor())
            preamble.append("    %s = %s[%s] * %s"%(variable, mangle(variable), index, factor))
        py = self.breakpoint_block.to_python("    ")
        py = "\n".join(preamble) + "\n" + py
        if True: print(py)
        breakpoint_globals = {}
        exec(py, breakpoint_globals)
        self._cuda_advance = numba.cuda.jit(breakpoint_globals["BREAKPOINT"])

    def new_instance(self, time_step, location, geometry, scale=1):
        data = dict(self.initial_state)
        for name, (value, units) in self.surface_area_parameters.items():
            sa = geometry.surface_areas[location] * scale
            if not self.use_units: sa *= 10000 # Convert from NEUWONs m^2 to NEURONs cm^2.
            data[name] = value * sa
        return data

    def advance(self, time_step, locations, **pointers):
        threads = 64
        blocks = (locations.shape[0] + (threads - 1)) // threads
        self._cuda_advance[blocks,threads](locations,
                *(ptr for name, ptr in sorted(pointers.items())))

class CodeBlock:
    def __init__(self, file, AST):
        if hasattr(AST, "name"):
            self.name = AST.name.get_node_name()
        elif AST.is_breakpoint_block():
            self.name = "BREAKPOINT"
        elif AST.is_initial_block():
            self.name = "INITIAL"
        else:
            self.name = None
        self.derivative = AST.is_derivative_block()
        self.statements = []
        AST = getattr(AST, "statement_block", AST)
        for stmt in AST.statements:
            self.statements.extend(file._parse_statement(stmt))
        # TODO: Move conserve statements to the end of the block, where they
        # belong. I think the nmodl library is moving them around...
        pass

        # Move assignments to conductances to the end of the block, where they
        # belong. This is needed because the nmodl library inserts conductance
        # hints and associated statements at the beginning of the block.
        self.statements.sort(key=lambda stmt: bool(
                isinstance(stmt, AssignStatement)
                and stmt.pointer and stmt.pointer.conductance))
        self._gather_arguments(file)

    def _gather_arguments(self, file):
        """ Sets arguments and assigned lists. """
        self.arguments = set()
        self.assigned = set()
        for stmt in self.statements:
            if isinstance(stmt, AssignStatement):
                for symbol in stmt.rhs.free_symbols:
                    if symbol.name not in self.assigned:
                        self.arguments.add(symbol.name)
                self.assigned.add(stmt.lhsn)
            elif isinstance(stmt, IfStatement):
                for symbol in stmt.arguments:
                    if symbol not in self.assigned:
                        self.arguments.add(symbol)
                self.assigned.update(stmt.assigned)
            elif isinstance(stmt, SolveStatement):
                target_block = file.derivative_blocks[stmt.block]
                for symbol in target_block.arguments:
                    if symbol not in self.assigned:
                        self.arguments.add(symbol)
                self.assigned.update(target_block.assigned)
            elif isinstance(stmt, ConserveStatement): pass
            else: raise NotImplementedError(stmt)
        # Remove the arguments which are implicit / always given.
        self.arguments.discard("time_step")
        for x in file.states: self.arguments.discard(x)
        for x in file.surface_area_parameters: self.arguments.discard(x)
        self.arguments = sorted(self.arguments)
        self.assigned = sorted(self.assigned)

    def _solve(self, method):
        """ Replace differential equation statements with their analytic solution. """
        # TODO: Do what the "nmodl" library does: If method is "cnexp" then do
        # "exact" solver if able, and fall back to Crank-Nicolson if it fails.
        for stmt in self.statements:
            if isinstance(stmt, IfStatement):
                stmt._solve(method)
            elif isinstance(stmt, AssignStatement) and stmt.derivative:
                if method == "euler":
                    stmt.solve_foward_euler()
                elif method == "derivimplicit":
                    stmt.solve_backward_euler()
                elif method == "cnexp":
                    stmt.solve_crank_nicholson()
                elif method == "exact":
                    stmt.solve_exact()
                else: raise NotImplementedError(method)

    def to_python(self, indent=""):
        py = ""
        for stmt in self.statements:
            py += stmt.to_python(indent)
        return py.rstrip() + "\n"

class IfStatement:
    def __init__(self, file, AST):
        self.condition = file._parse_expression(AST.condition)
        self.main_block = CodeBlock(file, AST.statement_block)
        self.elif_blocks = [CodeBlock(file, block) for block in AST.elseifs]
        assert(not self.elif_blocks) # TODO: Unimplemented.
        self.else_block = CodeBlock(file, AST.elses)
        self._gather_arguments()

    def _gather_arguments(self):
        """ Sets arguments and assigned lists. """
        self.arguments = set()
        self.assigned = set()
        for symbol in self.condition.free_symbols:
            self.arguments.add(symbol.name)
        self.arguments.update(self.main_block.arguments)
        self.assigned.update(self.main_block.assigned)
        for block in self.elif_blocks:
            self.arguments.update(block.arguments)
            self.assigned.update(block.assigned)
        self.arguments.update(self.else_block.arguments)
        self.assigned.update(self.else_block.assigned)

    def _solve(self, method):
        self.main_block._solve(method)
        for block in self.elif_blocks:
            block._solve(method)
        self.else_block._solve(method)

    def to_python(self, indent):
        py = indent + "if %s:\n"%pycode(self.condition)
        py += self.main_block.to_python(indent + "    ")
        assert(not self.elif_blocks) # TODO: Unimplemented.
        py += indent + "else:\n"
        py += self.else_block.to_python(indent + "    ")
        return py

class AssignStatement:
    def __init__(self, lhsn, rhs, derivative=False, pointer=None):
        self.lhsn = str(lhsn) # Left hand side name.
        self.rhs  = rhs       # Right hand side.
        self.derivative = bool(derivative)
        self.pointer = pointer # Associated with the left hand side.
        if self.pointer: assert(self.pointer.write)

    def to_python(self,  indent=""):
        assert(not self.derivative)
        if not isinstance(self.rhs, str):
            self.rhs = pycode(self.rhs.simplify())
        if self.pointer:
            array_access = "[_index_]" if self.pointer.dtype else "[_location_]"
            eq = " = " if self.pointer.dtype else " += "
            lhs = mangle(self.lhsn) + array_access
            py = indent + lhs + eq + self.rhs + "\n"
            if self.pointer.read:
                py += indent + self.lhsn + " = " + lhs + "\n"
            return py
        else:
            return indent + self.lhsn + " = " + self.rhs + "\n"


    def solve_foward_euler(self):
        self.rhs = sympy.Symbol(self.lhsn) + self.rhs * sympy.Symbol("time_step")
        self.derivative = False

    def solve_backward_euler(self):
        1/0 # TODO: Proof read this method!
        state, deriv  = self.lhsn, self.rhs
        current_state = sympy.Symbol(self.lhsn)
        future_state  = sympy.Symbol("Future" + self.lhsn)
        implicit_deriv = self.rhs.subs(state, future_state)
        delta = implicit_deriv * sympy.Symbol("time_step")
        delta = sympy.cancel(delta)
        eq = sympy.Eq(future_state, current_state + delta)
        backwards_euler = sympy.solve(eq, future_state)
        assert(len(backwards_euler) == 1)
        self.rhs = backwards_euler.pop()
        self.derivative = False

    def solve_crank_nicholson(self):
        init_state      = sympy.Symbol(self.lhsn)
        next_state      = sympy.Symbol("Future" + self.lhsn)
        dt              = sympy.Symbol("time_step")
        implicit_deriv  = self.rhs.subs(init_state, next_state)
        eq = sympy.Eq(next_state, init_state + implicit_deriv * dt / 2)
        backward_euler = sympy.solve(eq, next_state)
        assert(len(backward_euler) == 1)
        self.rhs = backward_euler.pop() * 2 - init_state
        self.derivative = False

    def solve_exact(self):
        dt = sympy.Symbol("time_step")
        state = sympy.Function(self.lhsn)(dt)
        deriv = self.rhs.subs(sympy.Symbol(self.lhsn), state)
        eq = sympy.Eq(state.diff(dt), deriv)
        self.rhs = sympy.dsolve(eq, state)
        C1 = sympy.solve(self.rhs.subs(state, self.lhsn).subs(dt, 0), "C1")[0]
        self.rhs = self.rhs.subs("C1", C1).rhs
        self.derivative = False

class ConserveStatement:
    def __init__(self, file, AST):
        conserved_expr = file._parse_expression(AST.expr) - sympy.Symbol(AST.react.name.get_node_name())
        self.states = sorted(str(x) for x in conserved_expr.free_symbols)
        # Assume that the conserve statement was once in the form: sum-of-states = constant-value
        sum_symbol = sympy.Symbol("_CONSERVED_SUM")
        assumed_form = sum_symbol
        for symbol in conserved_expr.free_symbols:
            assumed_form = assumed_form - symbol
        sum_solution = sympy.solvers.solve(sympy.Eq(conserved_expr, assumed_form), sum_symbol)
        assert(len(sum_solution) == 1)
        self.conserve_sum = sum_solution[0].evalf()

    def to_python(self, indent):
        py  = indent + "_CORRECTION_FACTOR = %s / (%s)\n"%(str(self.conserve_sum), " + ".join(self.states))
        for x in self.states:
            py += indent + x + " *= _CORRECTION_FACTOR\n"
        return py


# TODO: What are the units on atol? How does the timestep factor into it?

# TODO: Make this accept a list of pointers instead of "input_ranges"

# TODO: Convert kinetics into a function.

class KineticModel:
    def __init__(self, time_step, input_pointers, num_states, kinetics,
        conserve_sum=False,
        atol=1e-3):
        # Save and check the arguments.
        self.time_step = float(time_step)
        self.kinetics = kinetics
        self.input_ranges = np.array(input_ranges, dtype=Real)
        self.input_ranges.sort(axis=1)
        self.lower, self.upper = zip(*self.input_ranges)
        self.num_inputs = len(self.input_pointers)
        self.num_states = int(num_states)
        self.conserve_sum = float(conserve_sum) if conserve_sum else None
        self.atol = float(atol)
        assert(isinstance(self.kinetics, Callable))
        assert(len(self.input_ranges.shape) == 2 and self.input_ranges.shape[1] == 2)
        assert(self.num_inputs > 0)
        assert(self.num_states > 0)
        assert(self.atol > 0)
        # Determine how many interpolation points to use.
        self.grid_size = np.full((self.num_inputs,), 2)
        self._compute_interpolation_grid()
        while self._estimate_min_accuracy() >= self.atol:
            self.grid_size += 1
            self.grid_size *= 2
            self._compute_interpolation_grid()
        self.data = cp.array(self.data)

    def _compute_impulse_response_matrix(self, inputs):
        A = np.zeros([self.num_states] * 2, dtype=float)
        for src, dst, coef, func in self.kinetics:
            if func is not None:
                A[dst, src] += coef * func(*inputs)
            else:
                A[dst, src] += coef
        return scipy.linalg.expm(A * self.time_step)

    def _compute_interpolation_grid(self):
        """ Assumes self.grid_size is already set. """
        grid_range = np.subtract(self.upper, self.lower)
        grid_range[grid_range == 0] = 1
        self.grid_factor = np.subtract(self.grid_size, 1) / grid_range
        self.data = np.empty(list(self.grid_size) + [self.num_states]*2, dtype=Real)
        # Visit every location on the new interpolation grid.
        grid_axes = [list(enumerate(np.linspace(*args, dtype=float)))
                    for args in zip(self.lower, self.upper, self.grid_size)]
        for inputs in itertools.product(*grid_axes):
            index, inputs = zip(*inputs)
            self.data[index] = self._compute_impulse_response_matrix(inputs)

    def _estimate_min_accuracy(self):
        atol = 0
        num_points = np.product(self.grid_size)
        num_test_points = max(int(round(num_points / 10)), 100)
        for _ in range(num_test_points):
            inputs = np.random.uniform(self.lower, self.upper)
            exact = self._compute_impulse_response_matrix(inputs)
            interp = self._interpolate_impulse_response_matrix(inputs)
            atol = max(atol, np.max(np.abs(exact - interp)))
        return atol

    def _interpolate_impulse_response_matrix(self, inputs):
        assert(len(inputs) == self.num_inputs)
        inputs = np.array(inputs, dtype=Real)
        assert(all(inputs >= self.lower) and all(inputs <= self.upper)) # Bounds check the inputs.
        # Determine which grid box the inputs are inside of.
        inputs = self.grid_factor * np.subtract(inputs, self.lower)
        lower_idx = np.array(np.floor(inputs), dtype=int)
        upper_idx = np.array(np.ceil(inputs), dtype=int)
        upper_idx = np.minimum(upper_idx, self.grid_size - 1) # Protect against floating point error.
        # Prepare to find the interpolation weights, by finding the distance
        # from the input point to each corner of its grid box.
        inputs -= lower_idx
        corner_weights = [np.subtract(1, inputs), inputs]
        # Visit each corner of the grid box and accumulate the results.
        irm = np.zeros([self.num_states]*2, dtype=Real)
        for corner in itertools.product(*([(0,1)] * self.num_inputs)):
            idx = np.choose(corner, [lower_idx, upper_idx])
            weight = np.product(np.choose(corner, corner_weights))
            irm += weight * np.squeeze(self.data[idx])
        return irm

    def advance(self, inputs, states):
        numba.cuda.synchronize()
        assert(len(inputs) == self.num_inputs)
        assert(len(states.shape) == 2 and states.shape[1] == self.num_states)
        assert(states.dtype == Real)
        for l, u, x in zip(self.lower, self.upper, inputs):
            assert(x.shape[0] == states.shape[0])
            assert(cp.all(cp.logical_and(x >= l, x <= u))) # Bounds check the inputs.
        if self.num_inputs == 1:
            scratch = cp.zeros(states.shape, dtype=Real)
            threads = 64
            blocks = (states.shape[0] + (threads - 1)) // threads
            _1d[blocks,threads](inputs[0], states, scratch,
                self.lower[0], self.grid_size[0], self.grid_factor[0], self.data)
        else:
            raise TypeError("KineticModel is unimplemented for more than 1 input dimension.")
        numba.cuda.synchronize()
        # Enforce the invariant sum of states.
        if self.conserve_sum is not None:
            threads = 64
            blocks = (states.shape[0] + (threads - 1)) // threads
            _conserve_sum[blocks,threads](states, self.conserve_sum)
            numba.cuda.synchronize()

@numba.cuda.jit()
def _1d(inputs, states, scratch, input_lower_bound, grid_size, grid_factor, data):
    index = numba.cuda.grid(1)
    if index >= states.shape[0]:
        return
    inpt = inputs[index]
    state = states[index]
    accum = scratch[index]
    # Determine which grid box the inputs are inside of.
    inpt = (inpt - input_lower_bound) * grid_factor
    lower_idx = int(math.floor(inpt))
    upper_idx = int(math.ceil(inpt))
    upper_idx = min(upper_idx, grid_size - 1) # Protect against floating point error.
    inpt -= lower_idx
    # Visit each corner of the grid box and accumulate the results.
    _weighted_matrix_vector_multiplication(
        1 - inpt, data[lower_idx], state, accum)
    _weighted_matrix_vector_multiplication(
        inpt, data[upper_idx], state, accum)
    for i in range(len(state)):
        state[i] = accum[i]

@numba.cuda.jit(device=True)
def _weighted_matrix_vector_multiplication(w, m, v, results):
    """ Computes: results += weight * (matrix * vector)

    Arguments:
        [w]eight
        [m]atrix
        [v]ector
        results - output accumulator """
    l = len(v)
    for r in range(l):
        dot = 0
        for c in range(l):
            dot += m[r, c] * v[c]
        results[r] += w * dot

@numba.cuda.jit()
def _conserve_sum(states, target_sum):
    index = numba.cuda.grid(1)
    if index >= states.shape[0]:
        return
    state = states[index]
    accumulator = 0.
    num_states = len(state)
    for i in range(num_states):
        accumulator += state[i]
    correction_factor = target_sum / accumulator
    for i in range(num_states):
        state[i] *= correction_factor


pycode = lambda x: sympy.printing.pycode(x, user_functions={"abs": "abs"})
insert_indent = lambda i, s: i + "\n".join(i + l for l in s.split("\n"))

mangle = lambda x: "_" + x
demangle = lambda x: x[1:]
mangle2 = lambda x: "_" + x + "_"
demangle2 = lambda x: x[1:-1]

def _exec_wrapper(python, globals_, locals_):
    globals_["math"] = math
    try: exec(python, globals_, locals_)
    except:
        for noshow in ("__builtins__", "math"):
            if noshow in globals_: globals_.pop(noshow)
        print("Error while exec'ing the following python code:")
        print(python)
        print("globals():", repr(globals_))
        print("locals():", repr(locals_))
        raise

if __name__ == "__main__":
    for name, (nmodl_file_path, kw_args) in sorted(library.items()):
        print("ATTEMPTING:", name)
        new_mechanism = NmodlMechanism(nmodl_file_path, **kw_args)

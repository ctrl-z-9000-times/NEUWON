""" Internal representation for NMODL files.

This module reads and processes NMODL files into convenient data structures.
This uses the "nmodl" python library to parse NMODL files into abstract syntax
trees (AST). Then this module converts the AST into python and SymPy objects.
These classes save all of the special symbols, such as the program entry points
and the state variables. Finally, it solves any differential equations. """

import nmodl
from nmodl import dsl, ast, symtab
from nmodl.dsl import visitor
ANT = ast.AstNodeType
import os.path
import itertools
import copy
import re
from collections.abc import Callable, Iterable, Mapping
import neuwon
from neuwon import Real
import sympy
import sympy.printing.pycode
import numpy as np
import math
import numba.cuda

default_parameters = {
    "celsius": (neuwon.celsius, "degC")
}

pycode = lambda x: sympy.printing.pycode(x, user_functions={"abs": "abs"})

class NmodlMechanism(neuwon.Mechanism):
    def __init__(self, filename, method_override=False, parameter_overrides={}):
        self.filename = os.path.normpath(str(filename))
        with open(self.filename, 'rt') as f:
            nmodl_text = f.read()
        # Parse the NMDOL file into an abstract syntax tree (AST).
        AST = dsl.NmodlDriver().parse_string(nmodl_text)
        visitor.ConstantFolderVisitor().visit_program(AST)
        symtab.SymtabVisitor().visit_program(AST)
        visitor.InlineVisitor().visit_program(AST)
        visitor.SympyConductanceVisitor().visit_program(AST)
        visitor.KineticBlockVisitor().visit_program(AST)
        # Helpful debugging printouts:
        if False: print(dsl.to_nmodl(AST))
        if False: print(AST.get_symbol_table())
        if False: ast.view(AST)
        self._set_ast(AST)
        self._check_for_unsupported()
        self._gather_documentation()
        self._gather_io()
        self._gather_parameters(default_parameters, parameter_overrides)
        self.states = [v.get_name() for v in
                self.symbols.get_variables_with_properties(symtab.NmodlType.state_var)]
        self._gather_functions(method_override)
        self._discard_ast()
        self._run_initial_block()

    def _set_ast(self, AST):
        """ Sets visitor, lookup, and symbols. """
        self.visitor = visitor.AstLookupVisitor()
        self.lookup = lambda n: self.visitor.lookup(AST, n)
        symtab.SymtabVisitor().visit_program(AST)
        self.symbols = AST.get_symbol_table()

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
            raise ValueError("FUNCTION_TABLE's are not allowed.")
        if self.lookup(ANT.VERBATIM):
            raise ValueError("VERBATIM's are not allowed.")
        # TODO: No support for Pointer!
        # TODO: No support for Independent!
        # TODO: No support for Nonlinear!
        # TODO: No support for Include!

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

    def _gather_io(self):
        """ Sets write_currents and write_conductances. """
        self.write_currents = []
        self.write_conductances = []
        self.write_species = []
        for x in self.lookup(ANT.CONDUCTANCE_HINT):
            self.write_conductances.append(x.conductance.get_node_name())
            ion = x.ion.get_node_name() if x.ion else None
            self.write_species.append(ion.title())
        for x in self.lookup(ANT.WRITE_ION_VAR):
            self.write_currents.append(x.name.get_node_name())
        for x in self.lookup(ANT.NONSPECIFIC_CUR_VAR):
            self.write_currents.append(x.name.get_node_name())

    def required_species(self):
        return self.write_species

    def _gather_parameters(self, default_parameters, parameter_overrides):
        """ Sets parameters & surface_area_parameters.

        The surface area parameters are special because each segment of neuron
        has its own surface area and so their actual values are different for
        each instance of the mechanism. They are not inlined directly into the
        source code, instead they are stored alongside the state variables and
        accessed at run time. """
        self.parameters     = dict(default_parameters)
        parameter_overrides = dict(parameter_overrides)
        for assign in self.lookup(ANT.PARAM_ASSIGN):
            name = str(self.visitor.lookup(assign, ANT.NAME)[0].get_node_name())
            value = self.visitor.lookup(assign, [ANT.INTEGER, ANT.DOUBLE])
            units = self.visitor.lookup(assign, ANT.UNIT)
            if value:
                value = float(value[0].eval())
                if name in parameter_overrides:
                    value = parameter_overrides.pop(name)
            else:
                continue
            if units:
                units = units[0].get_node_name()
            else:
                units = None
            self.parameters[name] = (value, units)
        if parameter_overrides:
            extra_parameters = "%s"%", ".join(str(x) for x in parameter_overrides.keys())
            raise ValueError("Invalid parameter overrides: %s."%extra_parameters)
        # TODO: Convert all units to prefix-less units. 
        self.surface_area_parameters = []
        for name, (value, units) in self.parameters.items():
            if units is not None and "/cm2" in units:
                value = value * 100 * 100
                self.surface_area_parameters.append((name, value, units))
        for name, value, units in self.surface_area_parameters: self.parameters.pop(name)

    def _gather_functions(self, method_override):
        """ Sets initial_block, breakpoint_block, and derivative_blocks. """
        self.method_override = str(method_override) if method_override else False
        assert(self.method_override in (False, "euler", "derivimplicit", "cnexp", "exact"))
        self.derivative_blocks = {}
        for x in self.lookup(ANT.DERIVATIVE_BLOCK):
            name = x.name.get_node_name()
            self.derivative_blocks[name] = CodeBlock(self, x)
        assert(len(self.derivative_blocks) <= 1) # Otherwise unimplemented.
        self.initial_block = CodeBlock(self, self.lookup(ANT.INITIAL_BLOCK).pop())
        self.breakpoint_block = CodeBlock(self, self.lookup(ANT.BREAKPOINT_BLOCK).pop())

    def _parse_statement(self, AST):
        if AST.is_unit_state(): return []
        if AST.is_local_list_statement(): return []
        if AST.is_conductance_hint(): return []
        if AST.is_if_statement(): return [IfStatement(self, AST)]
        if AST.is_conserve(): return [ConserveStatement(self, AST)]
        if AST.is_expression_statement():
            AST = AST.expression
        else:
            help(AST); 1/0 # Unrecognized syntax.
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
            lhsn = AST.lhs.name.get_node_name()
            assert(AST.op.eval() == "=")
            if lhsn in self.write_currents:
                return []
            return [AssignStatement(lhsn, self._parse_expression(AST.rhs),
                    derivative=is_derivative,
                    accumulate=lhsn in self.write_conductances)]
        help(AST); 1/0 # Unrecognized syntax.

    def _parse_expression(self, AST):
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
            else: print(op); 1/0 # Unrecognized syntax.
        elif AST.is_unary_expression():
            op = AST.op.eval()
            if op == "-": return - self._parse_expression(AST.expression)
            else: print(op); 1/0 # Unrecognized syntax.
        elif AST.is_function_call():
            name = AST.name.get_node_name()
            if name == "fabs": name = "abs"
            args = [self._parse_expression(x) for x in AST.arguments]
            return sympy.Function(name)(*args)
        elif AST.is_double_unit():
            # TODO: Deal with AST.unit.
            return self._parse_expression(AST.value)
        elif AST.is_integer(): return sympy.Integer(AST.eval())
        elif AST.is_double():  return sympy.Float(AST.eval(), 18)
        else:
            help(AST); 1/0 # Unrecognized syntax.

    def instance_dtype(self):
        dtype = {"state": (Real, len(self.states))}
        if self.surface_area_parameters:
            dtype["surface_area_parameters"] = (Real, len(self.surface_area_parameters))
        return dtype

    def _run_initial_block(self, initial_assumptions={"v": -70e-3}):
        """ Use pythons built-in "exec" function to run the INITIAL_BLOCK.
        Sets: initial_state and initial_scope. """
        initial_globals = {'math': math}
        self.initial_scope = initial_locals = {}
        for arg in self.initial_block.arguments:
            if arg not in initial_assumptions: raise ValueError("Missing initial value for \"%s\"."%arg)
            initial_globals[arg] = initial_assumptions[arg]
        initial_state = 1/len(self.states) # TODO: Determine this from the conserve statement!
        # TODO: If there are not conserve statements to constrain the initial state then assume zero init.
        for x in self.states: initial_locals[x] = initial_state
        initial_python = self.initial_block.to_python()
        try:
            exec(initial_python, initial_globals, initial_locals)
        except:
            print("Error while exec'ing the following python code:")
            print("globals()", repr(initial_globals))
            print("locals()", repr(initial_locals))
            print(initial_python)
            raise
        self.initial_state = [initial_locals.pop(x) for x in self.states]

    def new_instance(self, time_step, location, geometry, scale=1):
        data = {"state": self.initial_state}
        if self.surface_area_parameters:
            sa = geometry.surface_areas[location] * scale
            data["surface_area_parameters"] = [value * sa
                    for name, value, units in self.surface_area_parameters]
        return data

    def set_time_step(self, time_step):
        self._compile_breakpoint_block(time_step)

    def _compile_breakpoint_block(self, time_step):
        for arg, initial_value in self.initial_scope.items():
            if arg in self.breakpoint_block.arguments:
                self.breakpoint_block.arguments.remove(arg)
                self.breakpoint_block.statements.insert(0,
                        AssignStatement(arg, sympy.Float(initial_value, 18)))
        py = self.breakpoint_block.to_python("    ")
        preamble =  ""
        preamble += "def BREAKPOINT(locations, instances, surface_area_parameters, %s, %s):\n"%(
                ", ".join(self.breakpoint_block.arguments),
                ", ".join(self.write_conductances))
        preamble += "    _index = numba.cuda.grid(1)\n"
        preamble += "    if _index >= instances.shape[0]: return\n"
        preamble += "    _location = locations[_index]\n"
        for arg in self.breakpoint_block.arguments:
            if arg == "v":
                py = re.sub("\\b"+arg+"\\b", arg+"[_index]*1000", py)
            else:
                print(arg); 1/0 # Unimplemented.
        for idx, state in enumerate(self.states):
            py = re.sub("\\b"+state+"\\b", "instances[_index][%d]"%idx, py)
        for idx, (name, _, _) in enumerate(self.surface_area_parameters):
            py = re.sub("\\b"+name+"\\b", "surface_area_parameters[_index][%d]"%idx, py)
        for g in self.write_conductances:
            py = re.sub("\\b"+g+"\\b", g+"[_location]", py)
        py = re.sub("\\btime_step\\b", str(time_step*1000), py)
        py = preamble + py
        for g in self.write_conductances:
            if g not in self.breakpoint_block.assigned:
                for idx, (name, value, units) in enumerate(self.surface_area_parameters):
                    if name == g:
                        py += "    "+g+"[_location] += surface_area_parameters[_index][%d]\n"%idx
        exec(py, globals())
        self._cuda_advance = numba.cuda.jit(BREAKPOINT)

    def advance(self, locations, instances, time_step, reaction_inputs, reaction_outputs):
        surface_area_parameters = instances["surface_area_parameters"]
        states = instances["state"]
        threads = 128
        blocks = (states.shape[0] + (threads - 1)) // threads
        star_args = []
        for arg in self.breakpoint_block.arguments:
            if arg == "v":
                star_args.append(reaction_inputs.v)
            else:
                print(arg); 1/0 # Unimplemented.
        for out in self.write_species:
            star_args.append(getattr(reaction_outputs.conductances, out))
        self._cuda_advance[blocks,threads](locations, states, surface_area_parameters, *star_args)

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
        # Move assignments to conductances to the end of the block, where they
        # belong. This is needed because the nmodl library inserts conductance
        # hints and associated statements at the begining of the block.
        self.statements.sort(key=lambda stmt: isinstance(stmt, AssignStatement) and stmt.accumulate)
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
        # Remove the arguments which are implicit / always given.
        self.arguments.discard("time_step")
        for x in file.states: self.arguments.discard(x)
        for x, v, u in file.surface_area_parameters: self.arguments.discard(x)
        self.arguments = sorted(self.arguments)
        self.assigned = sorted(self.assigned)

    def _solve(self, method):
        """ Replace differential equation statements with their analytic solution. """
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
    def __init__(self, lhsn, rhs, derivative=False, accumulate=False):
        self.lhsn = str(lhsn)
        self.rhs = rhs
        self.derivative = bool(derivative)
        self.accumulate = bool(accumulate) # TODO: Consider making this name more descriptive... maybe "conductance"?

    def to_python(self,  indent=""):
        assert(not self.derivative)
        self.rhs = self.rhs.simplify()
        if self.accumulate:
            return indent + self.lhsn + " += " + pycode(self.rhs) + "\n"
        else:
            return indent + self.lhsn + " = " + pycode(self.rhs) + "\n"

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

if __name__ == "__main__":
    # print(dsl.list_examples())
    # m = NmodlMechanism("neuwon/nmodl_library/hh.mod", method_override="exact")
    # m = NmodlMechanism("neuwon/nmodl_library/Balbi2017/Nav11_a.mod")
    m = NmodlMechanism("neuwon/nmodl_library/Kv-kinetic-models/hbp-00009_Kv1.1/hbp-00009_Kv1.1__13States_temperature2/hbp-00009_Kv1.1__13States_temperature2_Kv11.mod")
    # print(m.instance_dtype())

""" Private module. """
__all__ = []

import copy
import itertools
import neuwon.units
import nmodl
import nmodl.ast
import nmodl.dsl
import nmodl.symtab
import os.path
import sympy

ANT = nmodl.ast.AstNodeType

class NmodlParser:
    """ Attributes: filename, visitor, lookup, and symbols.

    Keep all references to the "nmodl" library separate from the main classes
    for clean & easy deletion. The nmodl library is implemented in C++ and as
    such does not support some critical python features: objects returned from
    the nmodl library do not support copying or pickling. """
    def __init__(self, nmodl_filename):
        """ Parse the NMDOL file into an abstract syntax tree (AST). """
        self.filename = str(nmodl_filename)
        with open(self.filename, 'rt') as f: nmodl_text = f.read()
        AST = nmodl.dsl.NmodlDriver().parse_string(nmodl_text)
        nmodl.dsl.visitor.ConstantFolderVisitor().visit_program(AST)
        nmodl.symtab.SymtabVisitor().visit_program(AST)
        nmodl.dsl.visitor.InlineVisitor().visit_program(AST)
        nmodl.dsl.visitor.KineticBlockVisitor().visit_program(AST)
        self.visitor = nmodl.dsl.visitor.AstLookupVisitor()
        self.lookup = lambda n: self.visitor.lookup(AST, n)
        nmodl.symtab.SymtabVisitor().visit_program(AST)
        self.symbols = AST.get_symbol_table()
        # Helpful debugging printouts:
        if False: print(self.to_nmodl(AST))
        if False: print(AST.get_symbol_table())
        if False: nmodl.ast.view(AST)

    to_nmodl = nmodl.dsl.to_nmodl

    def gather_documentation(self):
        """ Returns triple of (name, title, and description).
        This assumes that the first block comment is the primary documentation. """
        x = self.lookup(ANT.SUFFIX)
        if x: name = x[0].name.get_node_name()
        else: name = os.path.split(self.filename)[1] # TODO: Split extension too?
        title = self.lookup(ANT.MODEL)
        title = title[0].title.eval().strip() if title else ""
        if title.startswith(name + ".mod"):
            title = title[len(name + ".mod"):].strip()
        if title: title = title[0].title() + title[1:] # Capitalize the first letter.
        comments = self.lookup(ANT.BLOCK_COMMENT)
        description = comments[0].statement.eval() if comments else ""
        return (name, title, description)

    def gather_states(self):
        """ Returns sorted list the names of the states. """
        return sorted(v.get_name() for v in
                self.symbols.get_variables_with_properties(nmodl.symtab.NmodlType.state_var))

    def gather_units(self):
        units = copy.deepcopy(neuwon.units.builtin_units)
        for AST in self.lookup(ANT.UNIT_DEF):
            units.add_unit(AST.unit1.name.eval(), AST.unit2.name.eval())
        return units

    def gather_parameters(self):
        """ Returns dictionary of name -> (value, units). """
        parameters = {}
        for assign_stmt in self.lookup(ANT.PARAM_ASSIGN):
            name  = str(self.visitor.lookup(assign_stmt, ANT.NAME)[0].get_node_name())
            value = self.visitor.lookup(assign_stmt, [ANT.INTEGER, ANT.DOUBLE])
            units = self.visitor.lookup(assign_stmt, ANT.UNIT)
            value = float(value[0].eval())        if value else None
            units = str(units[0].get_node_name()) if units else None
            parameters[name] = (value, units)
        return parameters

    def gather_code_blocks(self):
        blocks = {}
        for AST in self.lookup(ANT.DERIVATIVE_BLOCK):
            name = AST.name.get_node_name()
            code_block = self.parse_code_block(AST)
            blocks[name] = code_block
            code_block.gather_arguments(blocks)
        blocks['INITIAL']    = self.parse_code_block(self.lookup(ANT.INITIAL_BLOCK).pop())
        blocks['BREAKPOINT'] = self.parse_code_block(self.lookup(ANT.BREAKPOINT_BLOCK).pop())
        blocks['INITIAL']   .gather_arguments(blocks)
        blocks['BREAKPOINT'].gather_arguments(blocks)
        return blocks

    @classmethod
    def parse_code_block(cls, AST):
        return CodeBlock(AST)

    @classmethod
    def parse_statement(cls, AST):
        """ Returns a list of Statement objects. """
        original = AST
        if AST.is_unit_state():             return []
        if AST.is_local_list_statement():   return []
        if AST.is_conductance_hint():       return []
        if AST.is_if_statement(): return [IfStatement(AST)]
        if AST.is_conserve():     return [ConserveStatement(AST)]
        if AST.is_expression_statement():
            AST = AST.expression
        if AST.is_solve_block(): return [SolveStatement(AST)]
        if AST.is_statement_block():
            return list(itertools.chain.from_iterable(
                    cls.parse_statement(stmt) for stmt in AST.statements))
        is_derivative = AST.is_diff_eq_expression()
        if is_derivative: AST = AST.expression
        if AST.is_binary_expression():
            assert(AST.op.eval() == "=")
            lhsn = AST.lhs.name.get_node_name()
            return [AssignStatement(lhsn, NmodlParser.parse_expression(AST.rhs),
                    derivative = is_derivative,)]
        # TODO: Catch procedure calls and raise an explicit error, instead of
        # just saying "unrecognised syntax". Procedure calls must be inlined by
        # the nmodl library.
        # TODO: Get line number from AST and include it in error message.
        raise ValueError("Unrecognized syntax at %s."%NmodlParser.to_nmodl(original))

    @classmethod
    def parse_expression(cls, AST):
        """ Returns a SymPy expression. """
        if AST.is_wrapped_expression() or AST.is_paren_expression():
            return cls.parse_expression(AST.expression)
        if AST.is_integer():  return sympy.Integer(AST.eval())
        if AST.is_double():   return sympy.Float(AST.eval(), 18)
        if AST.is_name():     return sympy.symbols(AST.get_node_name(), real=True)
        if AST.is_var_name(): return sympy.symbols(AST.name.get_node_name(), real=True)
        if AST.is_unary_expression():
            op = AST.op.eval()
            if op == "-": return - cls.parse_expression(AST.expression)
            raise ValueError("Unrecognized syntax at %s."%cls.to_nmodl(AST))
        if AST.is_binary_expression():
            op = AST.op.eval()
            lhs = cls.parse_expression(AST.lhs)
            rhs = cls.parse_expression(AST.rhs)
            if op == "+": return lhs + rhs
            if op == "-": return lhs - rhs
            if op == "*": return lhs * rhs
            if op == "/": return lhs / rhs
            if op == "^": return lhs ** rhs
            if op == "<": return lhs < rhs
            if op == ">": return lhs > rhs
            raise ValueError("Unrecognized syntax at %s."%cls.to_nmodl(AST))
        if AST.is_function_call():
            name = AST.name.get_node_name()
            args = [cls.parse_expression(x) for x in AST.arguments]
            if name == "fabs":  return sympy.Abs(*args)
            if name == "exp":   return sympy.exp(*args)
            else: return sympy.Function(name)(*args)
        if AST.is_double_unit():
            return cls.parse_expression(AST.value)
        raise ValueError("Unrecognized syntax at %s."%cls.to_nmodl(AST))

class CodeBlock:
    def __init__(self, AST):
        if hasattr(AST, "name"):        self.name = AST.name.get_node_name()
        elif AST.is_breakpoint_block(): self.name = "BREAKPOINT"
        elif AST.is_initial_block():    self.name = "INITIAL"
        else:                           self.name = None
        self.derivative = AST.is_derivative_block()
        self.statements = []
        AST = getattr(AST, "statement_block", AST)
        for stmt in AST.statements:
            self.statements.extend(NmodlParser.parse_statement(stmt))
        self.conserve_statements = [x for x in self if isinstance(x, ConserveStatement)]

    def __iter__(self):
        """ Yields all Statements contained in this block. """
        for stmt in self.statements:
            yield stmt
            if isinstance(stmt, IfStatement):
                for x in stmt.main_block:
                    yield x
                for block in stmt.elif_blocks:
                    for x in block:
                        yield x
                for x in stmt.else_block:
                    yield x

    def map(self, f):
        """ Argument f is function f(Statement) -> [Statement,]"""
        mapped_statements = []
        for stmt in self.statements:
            if isinstance(stmt, IfStatement):
                stmt.main_block.map(f)
                for block in stmt.elif_blocks:
                    block.map(f)
                stmt.else_block.map(f)
            mapped_statements.extend(f(stmt))
        self.statements = mapped_statements

    def gather_arguments(self, code_blocks):
        """ Sets arguments and assigned lists. """
        self.arguments = set()
        self.assigned  = set()
        for stmt in iter(self):
            if isinstance(stmt, AssignStatement):
                for symbol in stmt.rhs.free_symbols:
                    if symbol.name not in self.assigned:
                        self.arguments.add(symbol.name)
                self.assigned.add(stmt.lhsn)
            elif isinstance(stmt, IfStatement):
                for symbol in stmt.condition.free_symbols:
                    self.arguments.add(symbol.name)
                for block in [stmt.main_block] + stmt.elif_blocks + [stmt.else_block]:
                    block.gather_arguments(code_blocks)
                    self.arguments.update(block.arguments)
                    self.assigned.update(block.assigned)
            elif isinstance(stmt, SolveStatement):
                target_block = code_blocks[stmt.block]
                for symbol in target_block.arguments:
                    if symbol not in self.assigned:
                        self.arguments.add(symbol)
                self.assigned.update(target_block.assigned)
            elif isinstance(stmt, ConserveStatement): pass
            else: raise NotImplementedError(stmt)
        self.arguments = sorted(self.arguments)
        self.assigned  = sorted(self.assigned)

class IfStatement:
    def __init__(self, AST):
        self.condition = NmodlParser.parse_expression(AST.condition)
        self.main_block = CodeBlock(AST.statement_block)
        self.elif_blocks = [CodeBlock(block) for block in AST.elseifs]
        assert(not self.elif_blocks) # TODO: Unimplemented.
        self.else_block = CodeBlock(AST.elses)

class AssignStatement:
    def __init__(self, lhsn, rhs, derivative=False):
        self.lhsn = str(lhsn) # Left hand side name.
        self.rhs  = rhs       # Right hand side.
        self.derivative = bool(derivative)
        self.pointer = None # Associated with the left hand side.

    def __repr__(self):
        s = self.lhsn + " = " + str(self.rhs)
        if self.derivative: s = "'" + s
        if self.pointer: s += "  (%s)"%str(self.pointer)
        return s

    def solve(self):
        """ Solve this differential equation in-place. """
        assert(self.derivative)
        if False: print("SOLVE:   ", 'd/dt', self.lhsn, "=", self.rhs)
        self._solve_sympy()
        # self._solve_crank_nicholson()
        # try:
        # except Exception as x: eprint("Warning Sympy solver failed: "+str(x))
        # try: ; return
        # except Exception as x: eprint("Warning Crank-Nicholson solver failed: "+str(x))
        self.derivative = False
        if False: print("SOLUTION:", self.lhsn, '=', self.rhs)

    def _solve_sympy(self):
        from sympy import Symbol, Function, Eq
        dt    = Symbol("time_step")
        lhsn  = code_gen.mangle2(self.lhsn)
        state = Function(lhsn)(dt)
        deriv = self.rhs.subs(Symbol(self.lhsn), state)
        eq    = Eq(state.diff(dt), deriv)
        ics   = {state.subs(dt, 0): Symbol(lhsn)}
        self.rhs = sympy.dsolve(eq, state, ics=ics).rhs.simplify()
        self.rhs = self.rhs.subs(Symbol(lhsn), Symbol(self.lhsn))

    def _sympy_solve_ode(self, use_pade_approx=False):
        """ Analytically integrate this derivative equation.

        Optionally, the analytic result can be expanded in powers of dt,
        and the (1,1) Pade approximant to the solution returned.
        This approximate solution is correct to second order in dt.

        Raises an exception if the ODE is too hard or if sympy fails to solve it.

        Copyright (C) 2018-2019 Blue Brain Project. This method was part of the
        NMODL library distributed under the terms of the GNU Lesser General Public License.
        """
        # Only try to solve ODEs that are not too hard.
        ode_properties_require_all = {"separable"}
        ode_properties_require_one_of = {
            "1st_exact",
            "1st_linear",
            "almost_linear",
            "nth_linear_constant_coeff_homogeneous",
            "1st_exact_Integral",
            "1st_linear_Integral",
        }

        x = sympy.Symbol(self.lhsn)
        dxdt = self.rhs
        x, dxdt = _sympify_diff_eq(diff_string, vars)
        # Set up differential equation d(x(t))/dt = ...
        # Where the function x_t = x(t) is substituted for the symbol x.
        # The dependent variable is a function of t.
        t = sp.Dummy("t", real=True, positive=True)
        x_t = sp.Function("x(t)", real=True)(t)
        diffeq = sp.Eq(x_t.diff(t), dxdt.subs({x: x_t}))

        # For simple linear case write down solution in preferred form:
        dt = sp.symbols("time_step", real=True, positive=True)
        solution = None
        c1 = dxdt.diff(x).simplify()
        if c1 == 0:
            # Constant equation:
            # x' = c0
            # x(t+dt) = x(t) + c0 * dt
            solution = (x + dt * dxdt).simplify()
        elif c1.diff(x) == 0:
            # Linear equation:
            # x' = c0 + c1*x
            # x(t+dt) = (-c0 + (c0 + c1*x(t))*exp(c1*dt))/c1
            c0 = (dxdt - c1 * x).simplify()
            solution = (-c0 / c1).simplify() + (c0 + c1 * x).simplify() * sp.exp(
                c1 * dt
            ) / c1
        else:
            # Otherwise try to solve ODE with sympy:
            # First classify ODE, if it is too hard then exit.
            ode_properties = set(sp.classify_ode(diffeq))
            assert ode_properties.issuperset(ode_properties_require_all), "ODE too hard"
            assert ode_properties.intersection(ode_properties_require_one_of), "ODE too hard"
            # Try to find analytic solution, with initial condition x_t(t=0) = x
            # (note dsolve can return a list of solutions, in which case this currently fails)
            solution = sp.dsolve(diffeq, x_t, ics={x_t.subs({t: 0}): x})
            # evaluate solution at x(dt), extract rhs of expression
            solution = solution.subs({t: dt}).rhs.simplify()

    def pade_approx(self):
        """
        (1,1) order Pade approximant, correct to 2nd order in dt,
        constructed from the coefficients of 2nd order Taylor expansion.

        Copyright (C) 2018-2019 Blue Brain Project. This method was part of the
        NMODL library distributed under the terms of the GNU Lesser General Public License.
        """
        1/0 # unimplemtned
        taylor_series = sp.Poly(sp.series(solution, dt, 0, 3).removeO(), dt)
        _a0 = taylor_series.nth(0)
        _a1 = taylor_series.nth(1)
        _a2 = taylor_series.nth(2)
        solution = (
            (_a0 * _a1 + (_a1 * _a1 - _a0 * _a2) * dt) / (_a1 - _a2 * dt)
        ).simplify()
        # Special case where above form gives 0/0 = NaN.
        if _a1 == 0 and _a2 == 0:
            solution = _a0

    def _solve_crank_nicholson(self):
        dt              = sympy.Symbol("time_step", real=True, positive=True)
        init_state      = sympy.Symbol(self.lhsn)
        next_state      = sympy.Symbol("Future" + self.lhsn)
        implicit_deriv  = self.rhs.subs(init_state, next_state)
        eq = sympy.Eq(next_state, init_state + implicit_deriv * dt / 2)
        backward_euler = sympy.solve(eq, next_state)
        assert len(backward_euler) == 1, backward_euler
        self.rhs = backward_euler.pop() * 2 - init_state

class SolveStatement:
    def __init__(self, AST):
        self.block = AST.block_name.get_node_name()
        self.steadystate = AST.steadystate
        if AST.method: self.method = AST.method.get_node_name()
        else: self.method = "sparse" # FIXME
        AST.ifsolerr # TODO: What does this do?
        # assert(self.block in mechanism.derivative_blocks)

class ConserveStatement:
    def __init__(self, AST):
        conserved_expr = NmodlParser.parse_expression(AST.expr) - sympy.Symbol(AST.react.name.get_node_name())
        self.states = sorted(str(x) for x in conserved_expr.free_symbols)
        # Assume that the conserve statement was once in the form: sum-of-states = constant-value
        sum_symbol = sympy.Symbol("_CONSERVED_SUM")
        assumed_form = sum_symbol
        for symbol in conserved_expr.free_symbols:
            assumed_form = assumed_form - symbol
        sum_solution = sympy.solvers.solve(sympy.Eq(conserved_expr, assumed_form), sum_symbol)
        assert(len(sum_solution) == 1)
        self.conserve_sum = sum_solution[0].evalf()

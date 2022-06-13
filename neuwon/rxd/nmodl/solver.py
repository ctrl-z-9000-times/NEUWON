from neuwon.rxd.nmodl.parser import AssignStatement, ConserveStatement
import sympy

dt = sympy.Symbol("dt", real=True, positive=True)

def sympy_solve_ode(self: AssignStatement, use_pade_approx=False):
    """ Analytically integrate this derivative equation.

    Optionally, the analytic result can be expanded in powers of dt,
    and the (1,1) Pade approximant to the solution returned.
    This approximate solution is correct to second order in dt.

    Raises an exception if the ODE is too hard or if sympy fails to solve it.

    Copyright (C) 2018-2019 Blue Brain Project. This method was part of the
    NMODL library distributed under the terms of the GNU Lesser General Public License.
    """
    assert self.derivative
    self.derivative = False
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

    x = sympy.Symbol(self.lhsn, real=True)
    dxdt = self.rhs
    # Set up differential equation d(x(t))/dt = ...
    # Where the function x_t = x(t) is substituted for the symbol x.
    # The dependent variable is a function of t.
    t = sympy.Dummy("t", real=True, positive=True)
    x_t = sympy.Function("x(t)", real=True)(t)
    diffeq = sympy.Eq(x_t.diff(t), dxdt.subs({x: x_t}))

    # For simple linear case write down solution in preferred form:
    solution = None
    c1 = dxdt.diff(x).simplify()
    if c1 == 0:
        # Constant equation:
        # x' = c0
        # x(t+dt) = x(t) + c0 * dt
        self.rhs = x + dt * dxdt
    elif c1.diff(x) == 0:
        # Linear equation:
        # x' = c0 + c1*x
        # x(t+dt) = (-c0 + (c0 + c1*x(t))*exp(c1*dt))/c1
        c0 = (dxdt - c1 * x).simplify()
        self.rhs = (-c0 / c1).simplify() + (c0 + c1 * x).simplify() * sympy.exp(
            c1 * dt
        ) / c1
    else:
        # Otherwise try to solve ODE with sympy:
        # First classify ODE, if it is too hard then exit.
        ode_properties = set(sympy.classify_ode(diffeq))
        if (not ode_properties.issuperset(ode_properties_require_all) or
            not ode_properties.intersection(ode_properties_require_one_of)):
            return False

        # Try to find analytic solution, with initial condition x_t(t=0) = x
        # (note dsolve can return a list of solutions, in which case this currently fails)
        solution = sympy.dsolve(diffeq, x_t, ics={x_t.subs({t: 0}): x})
        # evaluate solution at x(dt), extract rhs of expression
        self.rhs = solution.subs({t: dt}).rhs

    self.rhs = self.rhs.simplify()

    if use_pade_approx:
        pade_approx(self)

    return True

def pade_approx(self: AssignStatement):
    """
    (1,1) order Pade approximant, correct to 2nd order in dt,
    constructed from the coefficients of 2nd order Taylor expansion.

    Copyright (C) 2018-2019 Blue Brain Project. This method was part of the
    NMODL library distributed under the terms of the GNU Lesser General Public License.
    """
    1/0 # unimplemented
    taylor_series = sympy.Poly(sympy.series(self.rhs, dt, 0, 3).removeO(), dt)
    _a0 = taylor_series.nth(0)
    _a1 = taylor_series.nth(1)
    _a2 = taylor_series.nth(2)
    self.rhs = (
        (_a0 * _a1 + (_a1 * _a1 - _a0 * _a2) * dt) / (_a1 - _a2 * dt)
    ).simplify()
    # Special case where above form gives 0/0 = NaN.
    if _a1 == 0 and _a2 == 0:
        self.rhs = _a0

def forward_euler(self: AssignStatement):
    if sympy_solve_ode(self): return
    assert self.derivative
    self.derivative = False
    init_state      = sympy.Symbol(self.lhsn, real=True)
    self.rhs = init_state + self.rhs * dt
    self.rhs = self.rhs.simplify()

def backward_euler(self: AssignStatement):
    if sympy_solve_ode(self): return
    assert self.derivative
    self.derivative = False
    init_state      = sympy.Symbol(self.lhsn, real=True)
    next_state      = sympy.Symbol("_Future_" + self.lhsn, real=True)
    implicit_deriv  = self.rhs.subs(init_state, next_state)
    eq = sympy.Eq(next_state, init_state + implicit_deriv * dt)
    backward_euler = sympy.solve(eq, next_state)
    assert len(backward_euler) == 1, backward_euler
    self.rhs = backward_euler.pop()
    self.rhs = self.rhs.simplify()

def crank_nicholson(self: AssignStatement):
    if sympy_solve_ode(self): return
    assert self.derivative
    self.derivative = False
    init_state      = sympy.Symbol(self.lhsn, real=True)
    next_state      = sympy.Symbol("_Future_" + self.lhsn, real=True)
    implicit_deriv  = self.rhs.subs(init_state, next_state)
    eq = sympy.Eq(next_state, init_state + implicit_deriv * dt / 2)
    backward_euler = sympy.solve(eq, next_state)
    assert len(backward_euler) == 1, backward_euler
    self.rhs = backward_euler.pop() * 2 - init_state
    self.rhs = self.rhs.simplify()

def conserve_statement_solution(self: ConserveStatement):
    """ Usage: CodeBlock.map(conserve_statement_solution) """
    if not isinstance(self, ConserveStatement):
        return [self]
    if not self.states:
        return []
    true_sum = self.states[0]
    for state in self.states[1:]:
        true_sum = true_sum + state
    replacement = [AssignStatement('_CORRECTION_FACTOR', self.conserve_sum / true_sum)]
    for state in self.states:
        replacement.append(
                AssignStatement(state, '_CORRECTION_FACTOR', operation = '*='))
    return replacement

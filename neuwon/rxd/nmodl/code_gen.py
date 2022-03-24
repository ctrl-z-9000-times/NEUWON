from neuwon.rxd.nmodl.parser import (CodeBlock,
        IfStatement, AssignStatement, SolveStatement, ConserveStatement)
import math
import tempfile
import textwrap
import sympy.core.basic
import sympy.printing.pycode as sympy_to_pycode

def to_python(self, indent=""):
    """ Argument self is any parser CodeBlock or Statement. """
    py = ""
    if isinstance(self, CodeBlock):
        for stmt in self.statements:
            py += to_python(stmt, indent)
        if not self.statements:
            py += indent + "pass"
    elif isinstance(self, IfStatement):
        py += indent + "if %s:\n"%sympy_to_pycode(self.condition)
        py += to_python(self.main_block, indent + "    ")
        assert not self.elif_blocks, "Unimplemented."
        if self.else_block is not None:
            py += indent + "else:\n"
            py += to_python(self.else_block, indent + "    ")
    elif isinstance(self, AssignStatement):
        try:
            if isinstance(self.rhs, sympy.core.basic.Basic):
                self.rhs = sympy_to_pycode(self.rhs.simplify())
            else:
                self.rhs = str(self.rhs)
            py += indent + self.lhsn + self.operation + self.rhs
        except Exception:
            print("Failed at:", repr(self), flush=True)
            raise
    elif isinstance(self, ConserveStatement):
        py += indent + "pass    # Warning: ignored CONSERVE statement here."
    elif isinstance(self, SolveStatement):
        solve_block = self.block
        arguments = ', '.join(sorted(solve_block.arguments))
        py += f"{indent}{solve_block.name}({arguments})"
    elif isinstance(self, str):
        py += textwrap.indent(self, indent)
    else:
        raise NotImplementedError(type(self))
    return py.rstrip() + "\n"

def exec_string(python, globals_, locals_=None):
    if not isinstance(python, str): python = str(python)
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as f:
        f.write(python)
        f.flush()
    globals_["math"] = math
    try:
        bytecode = compile(python, f.name, mode='exec')
        exec(bytecode, globals_, locals_)
    except:
        for noshow in ("__builtins__", "math"):
            if noshow in globals_: globals_.pop(noshow)
        err_msg = "Error while exec'ing the following python program:\n" + python
        err_msg + "\nglobals(): %s"%repr(globals_)
        err_msg + "\nlocals(): %s"%repr(locals_)
        print(err_msg, flush=True)
        raise

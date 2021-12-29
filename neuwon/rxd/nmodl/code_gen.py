import math
from neuwon.rxd.nmodl.parser import (CodeBlock,
        IfStatement, AssignStatement, SolveStatement, ConserveStatement)
import tempfile

import sympy.printing.pycode as sympy_to_pycode

def to_python(self, indent="", pointers={}, accumulators=set()):
    """ Argument self is any parser CodeBlock or Statement. """
    py = ""
    if isinstance(self, CodeBlock):
        for stmt in self.statements:
            py += to_python(stmt, indent, pointers)
    elif isinstance(self, IfStatement):
        py += indent + "if %s:\n"%sympy_to_pycode(self.condition)
        py += to_python(self.main_block, indent + "    ", pointers)
        assert not self.elif_blocks, "Unimplemented."
        py += indent + "else:\n"
        py += to_python(self.else_block, indent + "    ", pointers)
    elif isinstance(self, AssignStatement):
        if not isinstance(self.rhs, str):
            try: self.rhs = sympy_to_pycode(self.rhs.simplify())
            except Exception:
                eprint("Failed at:", repr(self))
                raise
        if self.derivative:
            lhs = 'd' + self.lhsn
            return indent + lhs + " += " + self.rhs + "\n"
        return indent + self.lhsn + self.operation + self.rhs + "\n"
    else: raise NotImplementedError(type(self))
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

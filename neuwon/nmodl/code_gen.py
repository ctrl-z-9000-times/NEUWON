""" Private module. """
__all__ = []

import math
from neuwon.nmodl.parser import CodeBlock, IfStatement, AssignStatement, SolveStatement, ConserveStatement

import sympy.printing.pycode as sympy_to_pycode

def to_python(self, indent="", pointers={}, accumulators=set()):
    """ Argument self is any parser CodeBlock or Statment. """
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

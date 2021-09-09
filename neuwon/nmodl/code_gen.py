""" Private module. """
__all__ = []

import math
import sys
from neuwon.nmodl.parser import CodeBlock, IfStatement, AssignStatement, SolveStatement, ConserveStatement

# TODO: Consider making and documenting a convention regarding what gets mangled & how.
# -> mangle1 for the user's nmodl variables.
# -> mangle2 for NEUWON's internal variables.

def mangle(x):
    return "_" + x

def demangle(x):
    return x[1:]

def mangle2(x):
    return "_" + x + "_"

def demangle2(x):
    return x[1:-1]

import sympy.printing.pycode as sympy_to_pycode

def insert_indent(indent, string):
    return indent + "\n".join(indent + line for line in string.split("\n"))

def to_python(self, indent="", **kwargs):
    """ Argument self is any parser CodeBlock or Statment. """
    py = ""
    if isinstance(self, CodeBlock):
        for stmt in self.statements:
            py += to_python(stmt, indent, **kwargs)
    elif isinstance(self, IfStatement):
        py += indent + "if %s:\n"%sympy_to_pycode(self.condition)
        py += to_python(self.main_block, indent + "    ", **kwargs)
        assert not self.elif_blocks, "Unimplemented."
        py += indent + "else:\n"
        py += to_python(self.else_block, indent + "    ", **kwargs)
    elif isinstance(self, AssignStatement):
        INITIAL_BLOCK = kwargs.get("INITIAL_BLOCK", False)
        if not isinstance(self.rhs, str):
            try: self.rhs = sympy_to_pycode(self.rhs.simplify())
            except Exception:
                eprint("Failed at:", repr(self))
                raise
        if self.derivative:
            lhs = mangle('d' + self.lhsn)
            return indent + lhs + " += " + self.rhs + "\n"
        if self.pointer and not INITIAL_BLOCK:
            assert self.pointer.w, self.pointer.name + " is not a writable pointer!"
            array_access = self.pointer.write_py() + "[" + self.pointer.index_py() + "]"
            eq = " += " if self.pointer.a else " = "
            assign_local = self.lhsn + " = " if self.pointer.r and not self.pointer.a else ""
            return indent + array_access + eq + assign_local + self.rhs + "\n"
        return indent + self.lhsn + " = " + self.rhs + "\n"
    elif isinstance(self, ConserveStatement):
        py  = indent + "_CORRECTION_FACTOR = %s / (%s)\n"%(str(self.conserve_sum), " + ".join(self.states))
        for x in self.states:
            py += indent + x + " *= _CORRECTION_FACTOR\n"
        return py
    else: raise NotImplementedError(type(self))
    return py.rstrip() + "\n"

def py_exec(python, globals_, locals_=None):
    if not isinstance(python, str): python = str(python)
    globals_["math"] = math
    try: exec(python, globals_, locals_)
    except:
        for noshow in ("__builtins__", "math"):
            if noshow in globals_: globals_.pop(noshow)
        err_msg = "Error while exec'ing the following python code:\n" + python
        err_msg + "\nglobals(): %s"%repr(globals_)
        err_msg + "\nlocals(): %s"%repr(locals_)
        print(err_msg, file=sys.stdout)
        raise

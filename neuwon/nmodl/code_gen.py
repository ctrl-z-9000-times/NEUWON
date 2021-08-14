import math
import sys

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

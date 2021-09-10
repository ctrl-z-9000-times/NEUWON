from collections.abc import Callable, Iterable, Mapping
from neuwon.database import Real
from neuwon.model import Reaction
from neuwon.nmodl import code_gen, cache
from neuwon.nmodl.parser import NmodlParser, ANT, SolveStatement, AssignStatement, IfStatement
from neuwon.nmodl.pointers import PointerTable, Pointer
from neuwon.nmodl.solver import solve
import copy
import math
import numba
import numba.cuda
import numpy as np
import os.path
import sympy
import sys
import textwrap

__all__ = ["NmodlMechanism"]

def eprint(*args, **kwargs):
    """ Prints to standard error (sys.stderr). """
    print(*args, file=sys.stderr, **kwargs)


# TODO: support for arrays? - arrays should really be unrolled in an AST pass...

# TODO: Ensure that all output pointers are written to.
# surface_area_parameters = sorted(self.surface_area_parameters)
# for variable, pointer in self.pointers.items():
#     if pointer.conductance and variable not in self.breakpoint_block.assigned:
#         if variable in surface_area_parameters:
#             idx = surface_area_parameters.index(variable)
#             self.breakpoint_block.statements.append(
#                     AssignStatement(variable, variable, pointer=pointer))

# TODO: Consider renamining this to "NMODL_Factory"
class NmodlMechanism(Reaction):
    def __init__(self, filename, pointers={}, parameters={}, use_cache=True):
        """
        Argument filename is an NMODL file to load.
                The standard NMODL file name extension is ".mod"

        Argument pointers is a mapping of NMODL variable names to database
                component names.

        Argument parameters is a mapping of parameter names to custom
                floating-point values.
        """
        self.filename = os.path.abspath(str(filename))
        if use_cache and cache.try_loading(self.filename, self): pass
        else:
            try:
                parser = NmodlParser(self.filename)
                self._check_for_unsupported(parser)
                self.name, self.title, self.description = parser.gather_documentation()
                self.units = parser.gather_units()
                self.parameters = ParameterTable(parser.gather_parameters())
                self.pointers = PointerTable(self)
                self.states = parser.gather_states()
                for state in self.states:
                    self.pointers.add(state, read=(self.name+"."+state), write=(self.name+"."+state))
                blocks = parser.gather_code_blocks()
                self.initial_block = blocks['INITIAL']
                self.breakpoint_block = blocks['BREAKPOINT']
                self.derivative_blocks = {k:v for k,v in blocks.items() if v.derivative}
                self._gather_IO(parser)
                self._solve()
            except Exception:
                eprint("ERROR while loading file", self.filename)
                raise
            cache.save(self.filename, self)
        self.parameters.update(parameters, strict=True)
        self.surface_area_parameters = self.parameters.separate_surface_area_parameters()
        for param in self.surface_area_parameters:
            self.pointers.add(param, read=(self.name+"."+param))
        for var_name, kwargs in pointers.items():
            self.pointers.add(var_name, **kwargs)

    def _check_for_unsupported(self, parser):
        # TODO: support for NONLINEAR?
        # TODO: support for INCLUDE?
        # TODO: support for COMPARTMENT?
        disallow = (
            "FUNCTION_TABLE_BLOCK",
            "LON_DIFUSE",
            "NONSPECIFIC_CUR_VAR",
            "TABLE_STATEMENT",
            "VERBATIM",
        )
        for x in disallow:
            if parser.lookup(getattr(ANT, x)):
                raise ValueError("\"%s\"s are not allowed."%x)

    def get_name(self):
        return self.name

    def _gather_IO(self, parser):
        """ Determine what external data the mechanism accesses. """
        self.breakpoint_block.gather_arguments()
        self.initial_block.gather_arguments()
        all_args = self.breakpoint_block.arguments + self.initial_block.arguments
        if "v" in all_args:
            self.pointers.add("v", read="Segment.voltage")
        if "area" in all_args:
            self.pointers.add("area", read="Segment.surface_area")
        if "volume" in all_args:
            self.pointers.add("volume", read="Segment.inside_volume")
        for x in parser.lookup(ANT.USEION):
            ion = x.name.value.eval()
            # Automatically generate the variable names for this ion.
            equilibrium = ('e' + ion,)
            current     = ('i' + ion,)
            conductance = ('g' + ion,)
            inside  = (ion + 'i', ion + '_inside',)
            outside = (ion + 'o', ion + '_outside',)
            for y in x.readlist:
                var_name = y.name.value.eval()
                if var_name in equilibrium:
                    pass # Ignored, mechanisms output conductances instead of currents.
                elif var_name in inside:
                    self.pointers.add(var_name, read="Segment.inside_concentrations/%s"%ion)
                elif var_name in outside:
                    self.pointers.add(var_name, read="outside/concentrations/%s"%ion)
                else: raise ValueError("Unrecognized species READ: \"%s\"."%var_name)
            for y in x.writelist:
                var_name = y.name.value.eval()
                if var_name in current:
                    pass # Ignored, mechanisms output conductances instead of currents.
                elif var_name in conductance:
                    self.pointers.add(var_name,
                            write="Segment.conductances_%s"%ion, accumulate=True)
                elif var_name in inside:
                    self.pointers.add(var_name,
                            write="Segment.inside_delta_concentrations/%s"%ion, accumulate=True)
                elif var_name in outside:
                    self.pointers.add(var_name,
                            write="outside/delta_concentrations/%s"%ion, accumulate=True)
                else: raise ValueError("Unrecognized species WRITE: \"%s\"."%var_name)
        for x in parser.lookup(ANT.CONDUCTANCE_HINT):
            var_name = x.conductance.get_node_name()
            if var_name not in self.pointers:
                ion = x.ion.get_node_name()
                self.pointers.add(var_name, write="Segment.conductances_%s"%ion, accumulate=True)

    def _solve(self):
        """ Replace SolveStatements with the solved equations to advance the
        systems of differential equations. """
        sympy_methods = ("cnexp", "derivimplicit", "euler")
        for idx, stmt in enumerate(self.breakpoint_block):
            if not isinstance(stmt, SolveStatement): continue
            syseq_block = stmt.block
            if stmt.method in sympy_methods:
                if syseq_block.derivative:
                    for stmt in syseq_block:
                        if isinstance(stmt, AssignStatement) and stmt.derivative:
                            solve(stmt)
            elif stmt.method == "sparse":
                1/0
            bp_stmts = self.breakpoint_block.statements
            self.breakpoint_block.statements = bp_stmts[:idx] + syseq_block.statements + bp_stmts[idx+1:]

    def initialize(self, database, **builtin_parameters):
        try:
            self.parameters.update(builtin_parameters, strict=True, override=False)
            self.parameters.substitute(
                    list(self.derivative_blocks.values()) + [self.initial_block, self.breakpoint_block,],)
            self._run_initial_block(database)
            cls = self._initialize_database(database)
            self.pointers.initialize(database)
            self._compile_breakpoint_block()
            cls._cuda_advance = self._cuda_advance
            cls._advance_arguments = self.arguments
            cls._surface_area_parameters = self.surface_area_parameters
            return cls
        except Exception:
            eprint("ERROR while loading file", self.filename)
            raise

    def _run_initial_block(self, database):
        """ Use pythons built-in "exec" function to run the INITIAL_BLOCK.
        Sets: initial_state and initial_scope. """
        self.initial_scope = {x: 0 for x in self.states}
        initial_python = code_gen.to_python(self.initial_block)
        for arg in self.initial_block.arguments:
            if arg in self.parameters:
                self.initial_scope[arg] = self.parameters[arg][0]
                continue
            if arg in self.pointers:
                read = self.pointers[arg].read
                value = database.get(read).get_initial_value()
                if value is not None:
                    self.initial_scope[arg] = value
                    continue
            eprint(initial_python)
            eprint("Arguments:", self.initial_block.arguments)
            eprint("Assigned:", self.initial_block.assigned)
            raise ValueError("Missing initial value for \"%s\"."%arg)
        code_gen.py_exec(initial_python, {}, self.initial_scope)
        self.initial_state = {x: self.initial_scope.pop(x) for x in self.states}

    def _initialize_database(self, database):
        cls = database.add_class(self.name, NMODL, doc=self.description)
        cls.add_attribute("segment", dtype="Segment")
        for name in self.surface_area_parameters:
            cls.add_attribute(name, units=None) # TODO: units!
        for name in self.states:
            cls.add_attribute(name, initial_value=self.initial_state[name], units=name)
        return cls.get_instance_type()

    def _compile_breakpoint_block(self):
        # Move assignments to conductances to the end of the block, where they
        # belong. This is needed because the nmodl library inserts conductance
        # hints and associated statements at the beginning of the block.
        # self.breakpoint_block.statements.sort(key=lambda stmt: bool(
        #         isinstance(stmt, AssignStatement)
        #         and stmt.pointer and "conductance" in stmt.pointer.name))
        # 
        self.breakpoint_block.gather_arguments()
        for arg in self.breakpoint_block.arguments:
            if arg in self.pointers: pass
            elif arg in self.initial_scope:
                value = float(self.initial_scope[arg])
                self.breakpoint_block.statements.insert(0, AssignStatement(arg, value))
            else: raise ValueError("Unhandled argument: \"%s\"."%arg)
        self.breakpoint_block.gather_arguments()
        self.arguments = set()
        for ptr in self.pointers.values():
            if ptr.r: self.arguments.add((ptr.read_py(), ptr.read))
            if ptr.w: self.arguments.add((ptr.write_py(), ptr.write))
        self.arguments = sorted(self.arguments)
        index     = code_gen.mangle2("index")
        # TODO: Consider moving some of this preamble stuff to the code_gen module?
        preamble  = []
        preamble.append("import numba.cuda")
        preamble.append("def BREAKPOINT(%s):"%", ".join(py for py, db in self.arguments))
        preamble.append("    "+index+" = numba.cuda.grid(1)")
        segment  = self.pointers.get("segment",  None)
        inside   = self.pointers.get("inside",   None)
        outside  = self.pointers.get("outside",  None)
        pointers = (segment, inside, outside)
        preamble.append("    if "+index+" >= "+segment.read_py()+".shape[0]: return")
        for ptr in pointers:
            if ptr is None: continue
            preamble.append("    %s = %s[%s]"%(
                code_gen.mangle2(ptr.name), ptr.read_py(), ptr.index_py()))
        for variable, ptr in sorted(self.pointers.items()):
            if not ptr.r: continue
            if ptr in pointers: continue
            stmt = "    %s = %s[%s]"%(ptr.name, ptr.read_py(), ptr.index_py())
            preamble.append(stmt)
        py = code_gen.to_python(self.breakpoint_block, "    ", self.pointers)
        py = "\n".join(preamble) + "\n" + py
        breakpoint_globals = {
            # code_gen.mangle(name): km.advance for name, km in self.kinetic_models.items()
        }
        code_gen.py_exec(py, breakpoint_globals)
        self._cuda_advance = numba.cuda.jit(breakpoint_globals["BREAKPOINT"])

class NMODL(Reaction):
    __slots__ = ()
    def __init__(self, segment, scale=1.0):
        self.segment = segment
        sa = float(scale) * self.segment.surface_area
        for name, (value, units) in self._surface_area_parameters.items():
            param = getattr(self, name)
            x_factor = (1e-2 * 1e-2) / (1e-6 * 1e-6) # Convert from NEUWONs um^2 to NEURONs cm^2.
            setattr(self, name, value * sa * x_factor)

    @classmethod
    def advance(cls):
        db = cls.get_database_class().get_database()
        segment_data = db.get_data(cls.get_name() + ".segment")
        if not len(segment_data): return
        args = (db.get_data(db_access) for arg_name, db_access in cls._advance_arguments)
        threads = 64
        blocks = (segment_data.shape[0] + (threads - 1)) // threads
        cls._cuda_advance[blocks,threads](*args)

class ParameterTable(dict):
    """ Dictionary mapping from nmodl parameter name to pairs of (value, units). """

    builtin_parameters = {
        "celsius": (None, "degC"),
        "time_step": (None, "ms"),
    }

    def __init__(self, parameters):
        dict.__init__(self)
        self.update(self.builtin_parameters)
        self.update(parameters)

    def update(self, parameters, strict=False, override=True):
        for name, value in parameters.items():
            name = str(name)
            if not isinstance(value, Iterable): value = (value, None)
            value, units = value
            value = float(value) if value is not None else None
            units = str(units)   if units is not None else None
            if name in self:
                old_value, old_units = self[name]
                if units is None: units = old_units
                elif old_units is not None and (strict or not override):
                    assert units == old_units, "Parameter \"%s\" units changed."%name
                if old_value is not None and not override:
                    value = old_value
            elif strict: raise ValueError("Invalid parameter override \"%s\"."%name)
            self[name] = (value, units)
        return self

    def separate_surface_area_parameters(self):
        """ Returns surface_area_parameters, Modifies parameters (self).

        The surface area parameters are special because each segment of neuron
        has its own surface area and so their actual values are different for
        each instance of the mechanism. They are not in-lined directly into the
        source code, instead they are stored alongside the state variables and
        accessed at run time. 
        """
        surface_area_parameters = {}
        for name, (value, units) in list(self.items()):
            if units and "/cm2" in units:
                surface_area_parameters[name] = self.pop(name)
        return surface_area_parameters

    def substitute(self, blocks):
        substitutions = {}
        for name, (value, units) in self.items():
            if value is None:
                continue
            if name == "time_step":
                symbol = sympy.Symbol(name, real=True, positive=True)
            else:
                symbol = sympy.Symbol(name, real=True)
            substitutions[symbol] = value
        for block in blocks:
            for stmt in block:
                if isinstance(stmt, AssignStatement):
                    stmt.rhs = stmt.rhs.subs(substitutions)
                elif isinstance(stmt, IfStatement):
                    stmt.condition = stmt.condition.subs(substitutions)

import sympy
import numba.cuda
import numba
import numpy as np
import math
import os.path
import itertools
import copy
from zlib import crc32
import pickle
from collections.abc import Callable, Iterable, Mapping
from neuwon.database import Real, Entity, eprint
from neuwon.model import Reaction, Model, Segment
from neuwon.nmodl_parser import _NmodlParser, ANT
from scipy.linalg import expm

def eprint(*args, **kwargs):
    """ Prints to standard error (sys.stderr). """
    print(*args, file=sys.stderr, **kwargs)

class NmodlMechanism(Reaction):
    def __init__(self, filename, pointers={}, parameter_overrides={}, use_cache=True):
        """
        Argument filename is an NMODL file to load.
            The standard NMODL file name extension is ".mod"

        Argument pointers is a mapping of NMODL variable names to database
            component names.

        Argument parameter_overrides is a mapping of parameter names to custom
            floating-point values.
        """
        self.filename = os.path.abspath(str(filename))
        if use_cache and _cache.try_loading(self.filename, self): pass
        else:
            try:
                parser = _NmodlParser(self.filename)
                self._check_for_unsupported(parser)
                self._name, self.title, self.description = parser.gather_documentation()
                self.units = parser.gather_units()
                self.parameters = _Parameters().update(parser.gather_parameters())
                self.states = parser.gather_states()
                self.pointers = {}
                self._gather_functions(parser)
                self._gather_IO(parser)
                self._solve()
            except Exception:
                eprint("ERROR while loading file", self.filename)
                raise
            _cache.save(self)
        self.parameters.update(parameter_overrides, strict=True)
        self.surface_area_parameters = self.parameters.separate_surface_area_parameters()
        for var_name, kwargs in pointers.items(): _Pointer.update(self, var_name, **kwargs)

    def initialize(self, database):
        try:
            self.parameters.gather_builtin_parameters(database, self.pointers)
            self.parameters.substitute(itertools.chain(
                    (self.initial_block, self.breakpoint_block,),
                    self.derivative_blocks.values(),))
            # self._solve() # TODO: only solve for kinetic models here.
            # self.kinetic_models = {}
            # for name, block in self.derivative_functions.items():
            #     self.kinetic_models[name] = KineticModel(1/0)
            self._run_initial_block(database)
            self._initialize_database(database)
            self._compile_breakpoint_block(database)
            for ptr in self.pointers.values():
                if ptr.r: database.access(ptr.read)
                if ptr.w: database.access(ptr.write)
        except Exception:
            eprint("ERROR while loading file", self.filename)
            raise

    def _check_for_unsupported(self, parser):
        # TODO: support for NONLINEAR?
        # TODO: support for INCLUDE?
        # TODO: support for COMPARTMENT?
        # TODO: sanity check for breakpoint block.
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

    def name(self):
        return self._name

    def _gather_functions(self, parser):
        """ Process all blocks of code which contain imperative instructions.
        Sets: initial_block, breakpoint_block, derivative_blocks. """
        self.derivative_blocks = {AST.name.get_node_name(): _CodeBlock(self, AST)
                                    for AST in parser.lookup(ANT.DERIVATIVE_BLOCK)}
        self.initial_block    = _CodeBlock(self, parser.lookup(ANT.INITIAL_BLOCK).pop())
        self.breakpoint_block = _CodeBlock(self, parser.lookup(ANT.BREAKPOINT_BLOCK).pop())

    def _parse_statement(self, AST):
        """ Returns a list of Statement objects. """
        original = AST
        if AST.is_unit_state():             return []
        if AST.is_local_list_statement():   return []
        if AST.is_conductance_hint():       return []
        if AST.is_if_statement(): return [_IfStatement(self, AST)]
        if AST.is_conserve():     return [_ConserveStatement(self, AST)]
        if AST.is_expression_statement():
            AST = AST.expression
        if AST.is_solve_block(): return [_SolveStatement(self, AST)]
        if AST.is_statement_block():
            return list(itertools.chain.from_iterable(
                    self._parse_statement(stmt) for stmt in AST.statements))
        is_derivative = AST.is_diff_eq_expression()
        if is_derivative: AST = AST.expression
        if AST.is_binary_expression():
            assert(AST.op.eval() == "=")
            lhsn = AST.lhs.name.get_node_name()
            return [_AssignStatement(lhsn, _NmodlParser.parse_expression(AST.rhs),
                    derivative = is_derivative,)]
        # TODO: Catch procedure calls and raise an explicit error, instead of
        # just saying "unrecognised syntax". Procedure calls must be inlined by
        # the nmodl library.
        # TODO: Get line number from AST and include it in error message.
        raise ValueError("Unrecognized syntax at %s."%_NmodlParser.to_nmodl(original))

    def _gather_IO(self, parser):
        """ Determine what external data the mechanism accesses. """
        all_args = self.breakpoint_block.arguments + self.initial_block.arguments
        if "v" in all_args:
            _Pointer.update(self, "v", read="membrane/voltages")
        if "area" in all_args:
            _Pointer.update(self, "area", read="membrane/surface_areas")
        if "volume" in all_args:
            _Pointer.update(self, "volume", read="membrane/inside/volumes")
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
                    _Pointer.update(self, var_name, read="membrane/inside/concentrations/%s"%ion)
                elif var_name in outside:
                    _Pointer.update(self, var_name, read="outside/concentrations/%s"%ion)
                else: raise ValueError("Unrecognized species READ: \"%s\"."%var_name)
            for y in x.writelist:
                var_name = y.name.value.eval()
                if var_name in current:
                    raise NotImplementedError(var_name)
                elif var_name in conductance:
                    _Pointer.update(self, var_name,
                            write="membrane/conductances/%s"%ion, accumulate=True)
                elif var_name in inside:
                    _Pointer.update(self, var_name,
                            write="membrane/inside/delta_concentrations/%s"%ion, accumulate=True)
                elif var_name in outside:
                    _Pointer.update(self, var_name,
                            write="outside/delta_concentrations/%s"%ion, accumulate=True)
                else: raise ValueError("Unrecognized species WRITE: \"%s\"."%var_name)
        for x in parser.lookup(ANT.CONDUCTANCE_HINT):
            var_name = x.conductance.get_node_name()
            if var_name not in self.pointers:
                ion = x.ion.get_node_name()
                _Pointer.update(self, var_name, write="membrane/conductances/%s"%ion, accumulate=True)

    def _solve(self):
        """ Replace SolveStatements with the solved equations to advance the
        systems of differential equations.

        Sets solved_blocks. """
        self.solved_blocks = {}
        sympy_methods = ("cnexp", "derivimplicit", "euler")
        self.breakpoint_block.map((lambda stmt: self._sympy_solve(stmt.block).statements
                if isinstance(stmt, _SolveStatement) and stmt.method in sympy_methods else [stmt]))
        self.breakpoint_block.map((lambda stmt: _LinearSystem(self, stmt.block)
                if isinstance(stmt, _SolveStatement) and stmt.method == "sparse" else [stmt]))

    def _sympy_solve(self, block_name):
        if block_name in self.solved_blocks: return self.solved_blocks[block_name]
        block = self.derivative_blocks[block_name]
        self.solved_blocks[block_name] = block = copy.deepcopy(block)
        for stmt in block:
            if isinstance(stmt, _AssignStatement) and stmt.derivative:
                # if stmt.lhsn in self.pointers and self.pointers[stmt.lhsn].accumulate:
                #     stmt.derivative = False # Main model will integrate, after summing derivatives across all reactions.
                #     continue
                stmt.solve()
        return block

    def _compile_derivative_blocks(self):
        """ Replace the derivative_blocks with compiled functions in the form:
                f(state_vector, **block.arguments) ->  Δstate_vector/Δt
        """
        self.derivative_functions = {}
        solve_statements = {stmt.block: stmt
                for stmt in self.breakpoint_block if isinstance(stmt, _SolveStatement)}
        for name, block in self.derivative_blocks.items():
            if name not in solve_statements: continue
            if solve_statements[name].method == "sparse":
                self.derivative_functions[name] = self._compile_derivative_block(block)

    def _compile_derivative_block(self, block):
        """ Returns function in the form:
                f(state_vector, **block.arguments) -> derivative_vector """
        block = copy.deepcopy(block)
        globals_ = {}
        locals_ = {}
        py = "def derivative(%s, %s):\n"%(_CodeGen.mangle2("state"), ", ".join(block.arguments))
        for idx, name in enumerate(self.states):
            py += "    %s = %s[%d]\n"%(name, _CodeGen.mangle2("state"), idx)
        for name in self.states:
            py += "    %s = 0\n"%_CodeGen.mangle('d' + name)
        block.map(lambda x: [] if isinstance(x, _ConserveStatement) else [x])
        py += block.to_python(indent="    ")
        py += "    return [%s]\n"%", ".join(_CodeGen.mangle('d' + x) for x in self.states)
        _CodeGen.py_exec(py, globals_, locals_)
        return numba.njit(locals_["derivative"])

    def _compute_propagator_matrix(self, block, time_step, kwargs):
        1/0
        f = self.derivative_functions[block]
        n = len(self.states)
        A = np.zeros((n,n))
        for i in range(n):
            state = np.array([0. for x in self.states])
            state[i] = 1
            A[:, i] = f(state, **kwargs)
        return expm(A * time_step)

    def _run_initial_block(self, database):
        """ Use pythons built-in "exec" function to run the INITIAL_BLOCK.
        Sets: initial_state and initial_scope. """
        globals_ = {
            "solve_steadystate": self.solve_steadystate,
        }
        self.initial_block.gather_arguments(self)
        for arg in self.initial_block.arguments:
            try: globals_[arg] = database.get_initial_value(self.pointers[arg].read)
            except KeyError: raise ValueError("Missing initial value for \"%s\"."%arg)
        self.initial_scope = {x: 0 for x in self.states}
        initial_python = self.initial_block.to_python(INITIAL_BLOCK=True)
        _CodeGen.py_exec(initial_python, globals_, self.initial_scope)
        self.initial_state = {x: self.initial_scope.pop(x) for x in self.states}

    def solve_steadystate(self, block, args):
        # First generate a state which satisfies the CONSERVE constraints.
        states = {x: 0 for x in self.states}
        conserved_states = set()
        for block_name, block in self.derivative_functions.items():
            for stmt in block.conserve_statements:
                initial_value = stmt.conserve_sum / len(stmt.states)
                for x in stmt.states:
                    if x in conserved_states:
                        raise ValueError(
                            "Unsupported: states can not be CONSERVED more than once. State: \"%s\"."%x)
                    else: conserved_states.add(x)
                for x in stmt.states: states[x] = initial_value
        if block in self.derivative_functions:
            dt = 1000 * 60 * 60 * 24 * 7 # One week in ms.
            irm = self._compute_propagator_matrix(block, dt, args)
            states = [states[name] for name in self.states] # Convert to list.
            states = irm.dot(states)
            states = {name: states[index] for index, name in enumerate(self.states)} # Convert to dictionary.
        else:
            1/0 # TODO: run the simulation until the state stops changing.
        return states

    def _initialize_database(self, database):
        # TODO: The pointers should really be updated as soon as the states block is parsed.
        database.add_archetype(self.name(), doc=self.description)
        database.add_attribute(self.name() + "/insertions", dtype="membrane")
        for name in self.surface_area_parameters:
            path = self.name() + "/data/" + name
            database.add_attribute(path, units=None) # TODO: Keep track of the units!
            _Pointer.update(self, name, read=path)
        for name in self.states:
            path = self.name() + "/data/" + name
            database.add_attribute(path, initial_value=self.initial_state[name], units=name)
            _Pointer.update(self, name, read=path, write=path)

    def _compile_breakpoint_block(self, database):
        # Link assignment to their pointers.
        for stmt in self.breakpoint_block:
            if isinstance(stmt, _AssignStatement):
                stmt.pointer = self.pointers.get(stmt.lhsn, None)
        # Move assignments to conductances to the end of the block, where they
        # belong. This is needed because the nmodl library inserts conductance
        # hints and associated statements at the beginning of the block.
        self.breakpoint_block.statements.sort(key=lambda stmt: bool(
                isinstance(stmt, _AssignStatement)
                and stmt.pointer and "conductance" in stmt.pointer.name))
        # 
        self.breakpoint_block.gather_arguments(self)
        initial_scope_carryover = []
        for arg in self.breakpoint_block.arguments:
            if arg in self.pointers: pass
            elif arg in self.initial_scope:
                initial_scope_carryover.append((arg, self.initial_scope[arg]))
            else: raise ValueError("Unhandled argument: \"%s\"."%arg)
        self.arguments = set()
        for ptr in self.pointers.values():
            if ptr.r: self.arguments.add((ptr.read_py, ptr.read))
            if ptr.w: self.arguments.add((ptr.write_py, ptr.write))
        self.arguments = sorted(self.arguments)
        index     = _CodeGen.mangle2("index")
        preamble  = []
        preamble.append("import numba.cuda")
        preamble.append("def BREAKPOINT(%s):"%", ".join(py for py, db in self.arguments))
        preamble.append("    "+index+" = numba.cuda.grid(1)")
        membrane = self.pointers.get("membrane", None)
        inside   = self.pointers.get("inside",   None)
        outside  = self.pointers.get("outside",  None)
        pointers = (membrane, inside, outside)
        preamble.append("    if "+index+" >= "+membrane.read_py+".shape[0]: return")
        for ptr in pointers:
            if ptr is None: continue
            preamble.append("    %s = %s[%s]"%(
                _CodeGen.mangle2(ptr.name), ptr.read_py, ptr.index_py))
        for variable, ptr in sorted(self.pointers.items()):
            if not ptr.r: continue
            if ptr in pointers: continue
            stmt = "    %s = %s[%s]"%(ptr.name, ptr.read_py, ptr.index_py)
            preamble.append(stmt)
        for variable_value_pair in initial_scope_carryover:
            preamble.append("    %s = %s"%variable_value_pair)
        py = self.breakpoint_block.to_python("    ")
        py = "\n".join(preamble) + "\n" + py
        breakpoint_globals = {
            # _CodeGen.mangle(name): km.advance for name, km in self.kinetic_models.items()
        }
        _CodeGen.py_exec(py, breakpoint_globals)
        self._cuda_advance = numba.cuda.jit(breakpoint_globals["BREAKPOINT"])

    def new_instances(self, database, locations, scale=1):
        if isinstance(database, Model): database = database.db
        locations = list(locations)
        for i, x in enumerate(locations):
            if isinstance(x, Segment):
                locations[i] = x = x.index
            elif isinstance(x, Entity):
                assert(x.archetype.name == "membrane")
                locations[i] = x = x.index
            assert(isinstance(x, int))
        ent_idx = database.create_entity(self.name(), len(locations), return_entity=False)
        ent_idx = np.array(ent_idx, dtype=np.int)
        database.access(self.name() + "/insertions")[ent_idx] = locations
        surface_areas = database.access("membrane/surface_areas")[locations]
        for name, (value, units) in self.surface_area_parameters.items():
            param = database.access(self.name() + "/data/" + name)
            x = 10000 # Convert from NEUWONs m^2 to NEURONs cm^2.
            param[ent_idx] = value * scale * surface_areas * x
        return ent_idx

    def advance(self, access):
        # for name, km in self.kinetic_models.items():
        #     1/0
        threads = 64
        locations = access(self.name() + "/insertions")
        if not len(locations): return
        blocks = (locations.shape[0] + (threads - 1)) // threads
        self._cuda_advance[blocks,threads](*(access(db) for py, db in self.arguments))

class _Parameters(dict):
    """ Dictionary mapping from nmodl parameter name to pairs of (value, units). """

    _builtin_parameters = {
        "celsius": (None, "degC"),
        "time_step": (None, "ms"),
    }

    def __init__(self, parameters=_builtin_parameters):
        dict.__init__(self)
        self.update(parameters)

    def update(self, parameters, strict=False):
        for name, value in parameters.items():
            name = str(name)
            if not isinstance(value, Iterable): value = (value, None)
            value, units = value
            value = float(value) if value is not None else None
            units = str(units)   if units is not None else None
            if name in self:
                old_value, old_units = self[name]
                if units is None: units = old_units
                elif strict and old_units is not None:
                    assert units == old_units, "Parameter \"%s\" units changed."%name
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

    def gather_builtin_parameters(self, database, pointers):
        """ Fill in missing parameter values from the database. """
        for name in self._builtin_parameters:
            builtin_value = float(database.access(name))
            given_value, units = self[name]
            if given_value is None: self[name] = (builtin_value, units)
        for name, ptr in list(pointers.items()):
            if ptr.r:
                try: value = database.access(ptr.read)
                except KeyError: continue
                if isinstance(value, float):
                    self.update({ptr.name: value})
                    ptr.read = None
                    if not ptr.mode: del pointers[name]

    def substitute(self, blocks):
        substitutions = []
        for name, (value, units) in self.items():
            if value is None: continue
            substitutions.append((name, value))
        for block in blocks:
            for stmt in block:
                if isinstance(stmt, _AssignStatement):
                    stmt.rhs = stmt.rhs.subs(substitutions)

class _Pointer:
    @staticmethod
    def update(mechanism, name, read=None, write=None, accumulate=None):
        """ Factory method for _Pointer objects.

        Argument name is an nmodl varaible name.
        Arguments read & write are a database access paths.
        Argument accumulate must be given at the same time as the "write" argument.
        """
        if name in mechanism.pointers:
            self = mechanism.pointers[name]
            if read is not None:
                read = str(read)
                if self.r and self.read != read:
                    eprint("Warning: Pointer override: %s read changed from '%s' to '%s'"%(
                            self.name, self.read, read))
                self.read = read
            if write is not None:
                write = str(write)
                if self.w and self.write != write:
                    eprint("Warning: Pointer override: %s write changed from '%s' to '%s'"%(
                            self.name, self.write, write))
                self.write = write
                self.accumulate = bool(accumulate)
            else: assert accumulate is None
        else:
            mechanism.pointers[name] = self = _Pointer(mechanism, name, read, write, accumulate)
        archetype = self.archetype
        if (archetype not in mechanism.pointers) and (archetype != mechanism.name()):
            if   archetype == "membrane": component = mechanism.name() + "/insertions"
            elif archetype == "inside":   component = "membrane/inside"
            elif archetype == "outside":  component = "membrane/outside"
            else: raise NotImplementedError(archetype)
            _Pointer.update(mechanism, archetype, read=component)
        return self

    def __init__(self, mechanism, name, read, write, accumulate):
        self.nmodl_name = mechanism.name()
        self.name  = str(name)
        self.read  = str(read)  if read  is not None else None
        self.write = str(write) if write is not None else None
        self.accumulate = bool(accumulate)

    @property
    def r(self): return self.read is not None
    @property
    def w(self): return self.write is not None
    @property
    def a(self): return self.accumulate
    @property
    def mode(self):
        return (('r' if self.r else '') +
                ('w' if self.w else '') +
                ('a' if self.a else ''))
    def __repr__(self):
        args = []
        if self.r: args.append("read='%s'"%self.read)
        if self.w: args.append("write='%s'"%self.write)
        if self.a: args.append("accumulate")
        return self.name + " = _Pointer(%s)"%', '.join(args)
    @property
    def archetype(self):
        component = self.read or self.write
        if component is None:                       return None
        elif component.startswith(self.nmodl_name): return self.nmodl_name
        elif component.startswith("membrane"):      return "membrane"
        elif component.startswith("inside"):        return "inside"
        elif component.startswith("outside"):       return "outside"
        else: raise Exception("Unrecognised archetype: " + component)
    @property
    def index(self):
        if   self.archetype == "membrane":      return "membrane"
        elif self.archetype == "inside":        return "inside"
        elif self.archetype == "outside":       return "outside"
        elif self.archetype == self.nmodl_name: return "index"
        else: raise NotImplementedError(self.archetype)
    @property
    def read_py(self):
        if self.r:
            if self.w and self.read != self.write:
                return _CodeGen.mangle('read_' + self.name)
            return _CodeGen.mangle(self.name)
    @property
    def write_py(self):
        if self.w:
            if self.r and self.read != self.write:
                return _CodeGen.mangle('write_' + self.name)
            return _CodeGen.mangle(self.name)
    @property
    def index_py(self):
        return _CodeGen.mangle2(self.index)

class _CodeBlock:
    def __init__(self, mechanism, AST):
        if hasattr(AST, "name"):        self.name = AST.name.get_node_name()
        elif AST.is_breakpoint_block(): self.name = "BREAKPOINT"
        elif AST.is_initial_block():    self.name = "INITIAL"
        else:                           self.name = None
        self.derivative = AST.is_derivative_block()
        top_level = self.derivative or self.name in ("BREAKPOINT", "INITIAL")
        self.statements = []
        AST = getattr(AST, "statement_block", AST)
        for stmt in AST.statements:
            self.statements.extend(mechanism._parse_statement(stmt))
        self.conserve_statements = [x for x in self if isinstance(x, _ConserveStatement)]
        if top_level: self.gather_arguments(mechanism)

    def gather_arguments(self, mechanism):
        """ Sets arguments and assigned lists. """
        self.arguments = set()
        self.assigned = set()
        for stmt in self.statements:
            if isinstance(stmt, _AssignStatement):
                for symbol in stmt.rhs.free_symbols:
                    if symbol.name not in self.assigned:
                        self.arguments.add(symbol.name)
                self.assigned.add(stmt.lhsn)
            elif isinstance(stmt, _IfStatement):
                stmt.gather_arguments(mechanism)
                for symbol in stmt.arguments:
                    if symbol not in self.assigned:
                        self.arguments.add(symbol)
                self.assigned.update(stmt.assigned)
            elif isinstance(stmt, _SolveStatement):
                target_block = mechanism.derivative_blocks[stmt.block]
                for symbol in target_block.arguments:
                    if symbol not in self.assigned:
                        self.arguments.add(symbol)
                self.assigned.update(target_block.assigned)
            elif isinstance(stmt, _ConserveStatement): pass
            else: raise NotImplementedError(stmt)
        self.arguments = sorted(self.arguments)
        self.assigned = sorted(self.assigned)

    def __iter__(self):
        for stmt in self.statements:
            if isinstance(stmt, _IfStatement):
                for x in stmt: yield x
            else: yield stmt

    def map(self, f):
        """ Argument f is function f(Statement) -> [Statement,]"""
        mapped_statements = []
        for stmt in self.statements:
            if isinstance(stmt, _IfStatement):
                stmt.map(f)
            mapped_statements.extend(f(stmt))
        self.statements = mapped_statements

    def to_python(self, indent="", **kwargs):
        py = ""
        for stmt in self.statements:
            py += stmt.to_python(indent, **kwargs)
        return py.rstrip() + "\n"

class _IfStatement:
    def __init__(self, mechanism, AST):
        self.condition = _NmodlParser.parse_expression(AST.condition)
        self.main_block = _CodeBlock(mechanism, AST.statement_block)
        self.elif_blocks = [_CodeBlock(mechanism, block) for block in AST.elseifs]
        assert(not self.elif_blocks) # TODO: Unimplemented.
        self.else_block = _CodeBlock(mechanism, AST.elses)

    def gather_arguments(self, mechanism):
        """ Sets arguments and assigned lists. """
        self.arguments = set()
        self.assigned = set()
        for symbol in self.condition.free_symbols:
            self.arguments.add(symbol.name)
        self.main_block.gather_arguments(mechanism)
        self.arguments.update(self.main_block.arguments)
        self.assigned.update(self.main_block.assigned)
        for block in self.elif_blocks:
            block.gather_arguments(mechanism)
            self.arguments.update(block.arguments)
            self.assigned.update(block.assigned)
        self.else_block.gather_arguments(mechanism)
        self.arguments.update(self.else_block.arguments)
        self.assigned.update(self.else_block.assigned)

    def __iter__(self):
        for x in self.main_block: yield x
        for block in self.elif_blocks:
            for x in block: yield x
        for x in self.else_block: yield x

    def map(self, f):
        """ Argument f is function f(Statement) -> [Statement,]"""
        self.main_block.map(f)
        for block in self.elif_blocks: block.map(f)
        self.else_block.map(f)

    def to_python(self, indent, **kwargs):
        py = indent + "if %s:\n"%_CodeGen.sympy_to_pycode(self.condition)
        py += self.main_block.to_python(indent + "    ", **kwargs)
        assert(not self.elif_blocks) # TODO: Unimplemented.
        py += indent + "else:\n"
        py += self.else_block.to_python(indent + "    ", **kwargs)
        return py

class _AssignStatement:
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

    def to_python(self,  indent="", INITIAL_BLOCK=False, **kwargs):
        if not isinstance(self.rhs, str):
            try: self.rhs = _CodeGen.sympy_to_pycode(self.rhs.simplify())
            except Exception:
                eprint("Failed at:", repr(self))
                raise
        if self.derivative:
            lhs = _CodeGen.mangle('d' + self.lhsn)
            return indent + lhs + " += " + self.rhs + "\n"
        if self.pointer and not INITIAL_BLOCK:
            assert self.pointer.w, self.pointer.name + " is not a writable pointer!"
            array_access = self.pointer.write_py + "[" + self.pointer.index_py + "]"
            eq = " += " if self.pointer.a else " = "
            assign_local = self.lhsn + " = " if self.pointer.r and not self.pointer.a else ""
            return indent + array_access + eq + assign_local + self.rhs + "\n"
        return indent + self.lhsn + " = " + self.rhs + "\n"

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
        lhsn  = _CodeGen.mangle2(self.lhsn)
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

class _SolveStatement:
    def __init__(self, mechanism, AST):
        self.block = AST.block_name.get_node_name()
        self.steadystate = AST.steadystate
        if AST.method: self.method = AST.method.get_node_name()
        else: self.method = "sparse" # FIXME
        AST.ifsolerr # TODO: What does this do?
        assert(self.block in mechanism.derivative_blocks)
        # arguments = mechanism.derivative_blocks[self.block].arguments
        # if self.steadystate:
        #     states_var = _CodeGen.mangle2("states")
        #     index_var  = _CodeGen.mangle2("index")
        #     self.py = states_var + " = solve_steadystate('%s', {%s})\n"%(self.block,
        #             ", ".join("'%s': %s"%(x, x) for x in arguments))
        #     for x in mechanism.states:
        #         self.py += "%s[%s] = %s['%s']\n"%(_CodeGen.mangle(x), index_var, states_var, x)
        # else:
        #     self.py = self.block + "(%s)\n"%(
        #                 ", ".join(arguments))

    def to_python(self, indent, **kwargs):
        1/0
        if self.steadystate: return insert_indent(indent, self._steadystate_callback)
        else:
            return indent + 1/0

class _LinearSystem:
    def __init__(self, mechanism, name):
        1/0

    def to_python(self, indent, **kwargs):
        1/0

class _ConserveStatement:
    def __init__(self, mechanism, AST):
        conserved_expr = _NmodlParser.parse_expression(AST.expr) - sympy.Symbol(AST.react.name.get_node_name())
        self.states = sorted(str(x) for x in conserved_expr.free_symbols)
        # Assume that the conserve statement was once in the form: sum-of-states = constant-value
        sum_symbol = sympy.Symbol("_CONSERVED_SUM")
        assumed_form = sum_symbol
        for symbol in conserved_expr.free_symbols:
            assumed_form = assumed_form - symbol
        sum_solution = sympy.solvers.solve(sympy.Eq(conserved_expr, assumed_form), sum_symbol)
        assert(len(sum_solution) == 1)
        self.conserve_sum = sum_solution[0].evalf()

    def to_python(self, indent, **kwargs):
        py  = indent + "_CORRECTION_FACTOR = %s / (%s)\n"%(str(self.conserve_sum), " + ".join(self.states))
        for x in self.states:
            py += indent + x + " *= _CORRECTION_FACTOR\n"
        return py


# TODO: What are the units on atol? How does the timestep factor into it?

# TODO: Make this accept a list of pointers instead of "input_ranges"

# TODO: Convert kinetics into a function.

# TODO: Use impulse response integration method in place of sparse solver...
#       TODO: How to dispatch to it?
#       TODO: Update the kinetic model to use Derivative function instead of sparse deriv equations.

# TODO: Write function to check that derivative_functions are Linear &
# time-invariant. Put this in the KineticModel class.

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

class _cache:
    @staticmethod
    def _dir_and_file(filename):
        cache_dir = os.path.abspath(".nmodl_cache")
        cache_file = os.path.join(cache_dir, "%X.pickle"%crc32(bytes(filename, 'utf8')))
        return (cache_dir, cache_file)

    @staticmethod
    def try_loading(filename, obj):
        """ Returns True on success, False indicates that no changes were made to the object. """
        cache_dir, cache_file = _cache._dir_and_file(filename)
        if not os.path.exists(cache_file): return False
        nmodl_ts  = os.path.getmtime(filename)
        cache_ts  = os.path.getmtime(cache_file)
        python_ts = os.path.getmtime(__file__)
        if nmodl_ts > cache_ts: return False
        if python_ts > cache_ts: return False
        try:
            with open(cache_file, 'rb') as f: data = pickle.load(f)
        except Exception as err:
            eprint("Warning: nmodl cache read failed:", str(err))
            return False
        obj.__dict__.update(data)
        return True

    @staticmethod
    def save(obj):
        cache_dir, cache_file = _cache._dir_and_file(obj.filename)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f: pickle.dump(obj.__dict__, f)
        except Exception as x:
            eprint("Warning: cache error", str(x))

class _CodeGen:
    # TODO: Consider making and documenting a convention regarding what gets mangled & how.
    # -> mangle1 for the user's nmodl variables.
    # -> mangle2 for NEUWON's internal variables.
    def mangle(x):      return "_" + x
    def demangle(x):    return x[1:]
    def mangle2(x):     return "_" + x + "_"
    def demangle2(x):   return x[1:-1]

    import sympy.printing.pycode as sympy_to_pycode

    def insert_indent(indent, string):
        return indent + "\n".join(indent + line for line in string.split("\n"))

    def py_exec(python, globals_, locals_=None):
        if False: print(python)
        globals_["math"] = math
        try: exec(python, globals_, locals_)
        except:
            for noshow in ("__builtins__", "math"):
                if noshow in globals_: globals_.pop(noshow)
            eprint("Error while exec'ing the following python code:")
            eprint(python)
            eprint("globals():", repr(globals_))
            eprint("locals():", repr(locals_))
            raise

# TODO: support for arrays? - arrays should really be unrolled in an AST pass...

# TODO: Ensure that all output pointers are written to.
# surface_area_parameters = sorted(self.surface_area_parameters)
# for variable, pointer in self.pointers.items():
#     if pointer.conductance and variable not in self.breakpoint_block.assigned:
#         if variable in surface_area_parameters:
#             idx = surface_area_parameters.index(variable)
#             self.breakpoint_block.statements.append(
#                     _AssignStatement(variable, variable, pointer=pointer))

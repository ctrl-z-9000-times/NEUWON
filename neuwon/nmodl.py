import nmodl
import nmodl.ast
import nmodl.dsl
import nmodl.symtab
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
from neuwon.database import Real, Index, Entity
import neuwon.units
from neuwon.model import Reaction, Model, Segment
from scipy.linalg import expm
import sys

def eprint(*args, **kwargs): print(*args, file=sys.stderr, **kwargs)

ANT = nmodl.ast.AstNodeType

# TODO: Initial state. Mostly works...  Need code to run a simulation until it
# reaches a steady state, given the solved system.

# TODO: support for arrays? - arrays should really be unrolled in an AST pass...

# # Ensure that all output pointers are written to.
# surface_area_parameters = sorted(self.surface_area_parameters)
# for variable, pointer in self.pointers.items():
#     if pointer.conductance and variable not in self.breakpoint_block.assigned:
#         if variable in surface_area_parameters:
#             idx = surface_area_parameters.index(variable)
#             self.breakpoint_block.statements.append(
#                     _AssignStatement(variable, variable, pointer=pointer))

_builtin_parameters = {
    "celsius": (None, "degC"),
    "time_step": (None, "ms"),
}

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
                with open(self.filename, 'rt') as f: nmodl_text = f.read()
                self._parse_text(nmodl_text)
                self._check_for_unsupported()
                self._gather_documentation()
                self._gather_units()
                self._gather_parameters()
                self.states = sorted(v.get_name() for v in
                        self.symbols.get_variables_with_properties(nmodl.symtab.NmodlType.state_var))
                self.pointers = {}
                self._gather_functions()
                self._gather_IO()
                self._discard_ast()
                self._solve()
            except Exception:
                eprint("ERROR while loading file", self.filename)
                raise
            _cache.save(self)
        self.pointers.update({nmodl_name: _Pointer(self, database_name, mode)
                for nmodl_name, (database_name, mode) in pointers.items()})
        self._apply_parameter_overrides(parameter_overrides)
        self._separate_surface_area_parameters()

    def _parse_text(self, nmodl_text):
        """ Parse the NMDOL file into an abstract syntax tree (AST).
        Sets visitor, lookup, and symbols. """
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
        if False: print(nmodl.dsl.to_nmodl(AST))
        if False: print(AST.get_symbol_table())
        if False: nmodl.ast.view(AST)

    def _discard_ast(self):
        """ Cleanup. Delete all references to the nmodl library.

        The nmodl library is implemented in C++ and as such does not
        automatically support some python features. In specific: objects
        returned from the nmodl library do not support copying or pickling. """
        del self.visitor
        del self.lookup
        del self.symbols

    def _check_for_unsupported(self):
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
            if self.lookup(getattr(ANT, x)):
                raise ValueError("\"%s\"s are not allowed."%x)

    def _gather_documentation(self):
        """ Sets name, title, and description.
        This assumes that the first block comment is the primary documentation. """
        x = self.lookup(ANT.SUFFIX)
        if x: self._name = x[0].name.get_node_name()
        else: self._name = os.path.split(self.filename)[1] # TODO: Split extension too?
        title = self.lookup(ANT.MODEL)
        self.title = title[0].title.eval().strip() if title else ""
        if self.title.startswith(self._name + ".mod"):
            self.title = self.title[len(self._name + ".mod"):].strip()
        if self.title: self.title = self.title[0].title() + self.title[1:] # Capitalize the first letter.
        comments = self.lookup(ANT.BLOCK_COMMENT)
        self.description = comments[0].statement.eval() if comments else ""

    def name(self):
        return self._name

    def _gather_units(self):
        """
        Sets flag "use_units" which determines how to deal with differences
        between NEURONs and NEUWONs unit systems:
          * True: modify the mechanism to use NEUWONs unit system.
          * False: convert the I/O into NEURON unit system.
        """
        self.units = copy.deepcopy(neuwon.units.builtin_units)
        for AST in self.lookup(ANT.UNIT_DEF):
            self.units.add_unit(AST.unit1.name.eval(), AST.unit2.name.eval())
        self.use_units = not any(x.eval() == "UNITSOFF" for x in self.lookup(ANT.UNIT_STATE))
        self.use_units = False # TODO, either implement this or delete it...
        if not self.use_units: eprint("Warning: UNITSOFF detected.")

    def _gather_parameters(self):
        """ Sets parameters. """
        self.parameters = dict(_builtin_parameters)
        for assign in self.lookup(ANT.PARAM_ASSIGN):
            name  = str(self.visitor.lookup(assign, ANT.NAME)[0].get_node_name())
            value = self.visitor.lookup(assign, [ANT.INTEGER, ANT.DOUBLE])
            units = self.visitor.lookup(assign, ANT.UNIT)
            value = float(value[0].eval())   if value else None
            units = units[0].get_node_name() if units else None
            self.parameters[name] = (value, units)

    def _gather_functions(self):
        """ Process all blocks of code which contain imperative instructions.
        Sets: initial_block, breakpoint_block, derivative_blocks. """
        self.derivative_blocks = {AST.name.get_node_name(): _CodeBlock(self, AST)
                                    for AST in self.lookup(ANT.DERIVATIVE_BLOCK)}
        self.initial_block    = _CodeBlock(self, self.lookup(ANT.INITIAL_BLOCK).pop())
        self.breakpoint_block = _CodeBlock(self, self.lookup(ANT.BREAKPOINT_BLOCK).pop())

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
            return [_AssignStatement(lhsn, self._parse_expression(AST.rhs),
                    derivative = is_derivative,)]
        # TODO: Catch procedure calls and raise an explicit error, instead of
        # just saying "unrecognised syntax". Procedure calls must be inlined by
        # the nmodl library.
        # TODO: Get line number from AST and include it in error message.
        raise ValueError("Unrecognized syntax at %s."%nmodl.dsl.to_nmodl(original))

    def _parse_expression(self, AST):
        """ Returns a SymPy expression. """
        if AST.is_wrapped_expression() or AST.is_paren_expression():
            return self._parse_expression(AST.expression)
        if AST.is_integer():  return sympy.Integer(AST.eval())
        if AST.is_double():   return sympy.Float(AST.eval(), 18)
        if AST.is_name():     return sympy.symbols(AST.get_node_name())
        if AST.is_var_name(): return sympy.symbols(AST.name.get_node_name())
        if AST.is_unary_expression():
            op = AST.op.eval()
            if op == "-": return - self._parse_expression(AST.expression)
            raise ValueError("Unrecognized syntax at %s."%nmodl.dsl.to_nmodl(AST))
        if AST.is_binary_expression():
            op = AST.op.eval()
            lhs = self._parse_expression(AST.lhs)
            rhs = self._parse_expression(AST.rhs)
            if op == "+": return lhs + rhs
            if op == "-": return lhs - rhs
            if op == "*": return lhs * rhs
            if op == "/": return lhs / rhs
            if op == "^": return lhs ** rhs
            if op == "<": return lhs < rhs
            if op == ">": return lhs > rhs
            raise ValueError("Unrecognized syntax at %s."%nmodl.dsl.to_nmodl(AST))
        if AST.is_function_call():
            name = AST.name.get_node_name()
            args = [self._parse_expression(x) for x in AST.arguments]
            if name == "fabs":  return sympy.Abs(*args)
            if name == "exp":   return sympy.exp(*args)
            else: return sympy.Function(name)(*args)
        if AST.is_double_unit():
            if self.use_units:
                factor, dimensions = self.units.standardize(AST.unit.name.eval())
            else: factor = 1
            return self._parse_expression(AST.value) * factor
        raise ValueError("Unrecognized syntax at %s."%nmodl.dsl.to_nmodl(AST))

    def _gather_IO(self):
        """ Determine what external data the mechanism accesses. """
        if "v" in self.breakpoint_block.arguments: self._add_pointer("v", "membrane/voltages", 'r')
        for x in self.lookup(ANT.USEION):
            ion = x.name.value.eval()
            # Automatically generate the variable names for this ion.
            equilibrium = 'e' + ion
            current = 'i' + ion
            conductance = 'g' + ion
            inside = ion + 'i'
            outside = ion + 'o'
            for y in x.readlist:
                variable = y.name.value.eval()
                if variable == equilibrium:
                    pass # Ignored, mechanisms output conductances instead of currents.
                elif variable == inside:
                    self._add_pointer(variable, "membrane/inside/%s/concentrations"%ion, 'r')
                elif variable == outside:
                    self._add_pointer(variable, "membrane/outside/%s/concentrations"%ion, 'r')
                else: raise ValueError("Unrecognized ion READ: \"%s\"."%variable)
            for y in x.writelist:
                variable = y.name.value.eval()
                if variable == current:
                    raise NotImplementedError(variable)
                elif variable == conductance:
                    self._add_pointer(variable, "membrane/conductances/%s"%ion, 'a')
                elif variable == inside:
                    self._add_pointer(variable, "membrane/inside/%s/release_rates"%ion, 'a')
                elif variable == outside:
                    self._add_pointer(variable, "membrane/outside/%s/release_rates"%ion, 'a')
                else: raise ValueError("Unrecognized ion WRITE: \"%s\"."%variable)
        for x in self.lookup(ANT.CONDUCTANCE_HINT):
            variable = x.conductance.get_node_name()
            if variable not in self.pointers:
                ion = x.ion.get_node_name()
                self._add_pointer(variable, "membrane/conductances/%s"%ion, 'a')

    def _add_pointer(self, name, pointer, mode):
        """ Argument name is an nmodl varaible name.
            Argument pointer is a database access path. """
        assert name not in self.pointers
        self.pointers[name] = _Pointer(self, pointer, mode)

    def _solve(self):
        """ Replace SolveStatements with the solved equations to advance the
        systems of differential equations.

        Sets solved_blocks. """
        self.solved_blocks = {}
        sympy_methods = ("euler", "cnexp")
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
                stmt.solve()
        return block

    def _apply_parameter_overrides(self, parameter_overrides):
        for name, value in parameter_overrides.items():
            if name in self.parameters:
                old_value, units = self.parameters[name]
                self.parameters[name] = (value, units)
            else: raise ValueError("Invalid parameter override \"%s\"."%name)

    def _separate_surface_area_parameters(self):
        """ Sets surface_area_parameters, Modifies parameters.

        The surface area parameters are special because each segment of neuron
        has its own surface area and so their actual values are different for
        each instance of the mechanism. They are not in-lined directly into the
        source code, instead they are stored alongside the state variables and
        accessed at run time. 
        """
        self.surface_area_parameters = {}
        for name, (value, units) in list(self.parameters.items()):
            if units and "/cm2" in units:
                self.surface_area_parameters[name] = self.parameters.pop(name)

    def initialize(self, database):
        try:
            self._gather_builtin_parameters(database)
            self._substitute_parameters(database)
            # self._solve() # TODO: only solve for kinetic models here.
            # self.kinetic_models = {}
            # for name, block in self.derivative_functions.items():
            #     self.kinetic_models[name] = KineticModel(1/0)
            self._run_initial_block(database)
            self._initialize_database(database)
            self._compile_breakpoint_block(database)
            for ptr in self.pointers.values(): database.access(ptr.name)
        except Exception:
            eprint("ERROR while loading file", self.filename)
            raise

    def _gather_builtin_parameters(self, database):
        for name in _builtin_parameters:
            builtin_value = database.access(name)
            given_value, units = self.parameters[name]
            if name == "time_step": builtin_value *= 1000 # Convert from NEUWONs seconds to NEURONs milliseconds.
            if given_value is None:
                self.parameters[name] = (builtin_value, units)

    def _substitute_parameters(self, database):
        substitutions = []
        for name, (value, units) in self.parameters.items():
            if value is None: continue
            substitutions.append((name, value))
        for block in itertools.chain(
                    (self.initial_block, self.breakpoint_block,),
                    self.derivative_blocks.values(),):
            for stmt in block:
                if isinstance(stmt, _AssignStatement):
                    stmt.rhs = stmt.rhs.subs(substitutions)

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
            try: globals_[arg] = database.initial_value(self.pointers[arg].name)
            except KeyError: raise ValueError("Missing initial value for \"%s\"."%arg)
        self.initial_scope = {x: 0 for x in self.states}
        initial_python = self.initial_block.to_python(context="INITIAL_BLOCK")
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
        database.add_archetype(self.name(), doc=self.description)
        database.add_attribute(self.name() + "/insertions", dtype="membrane")
        for name in self.surface_area_parameters:
            path = self.name() + "/data/" + name
            database.add_attribute(path)
            self._add_pointer(name, path, 'r')
        for name in self.states:
            path = self.name() + "/data/" + name
            database.add_attribute(path, initial_value=self.initial_state[name])
            self._add_pointer(name, path, 'rw')

    def _compile_breakpoint_block(self, database):
        # Link assignment to their pointers.
        for stmt in self.breakpoint_block:
            if isinstance(stmt, _AssignStatement):
                stmt.pointer = self.pointers.get(stmt.lhsn, None)
        # 
        for x in self.pointers.values():
            # TODO: rename mangle2(location) to mangle2(membrane)
            if   x.archetype == "membrane":  x.index = _CodeGen.mangle2("location")
            elif x.archetype == "inside":    x.index = _CodeGen.mangle2("inside")
            elif x.archetype == "outside":   x.index = _CodeGen.mangle2("outside")
            elif x.archetype == self.name(): x.index = _CodeGen.mangle2("index")
            else: raise NotImplementedError(x.archetype)
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
        arguments = sorted(_CodeGen.mangle(name) for name in  self.pointers)
        locations = _CodeGen.mangle2("locations")
        location  = _CodeGen.mangle2("location")
        index     = _CodeGen.mangle2("index")
        preamble  = []
        preamble.append("import numba.cuda")
        preamble.append("def BREAKPOINT("+locations+", %s):"%", ".join(arguments))
        preamble.append("    "+index+" = numba.cuda.grid(1)")
        preamble.append("    if "+index+" >= "+locations+".shape[0]: return")
        preamble.append("    "+location+" = "+locations+"["+index+"]")
        for variable_value_pair in initial_scope_carryover:
            preamble.append("    %s = %s"%variable_value_pair)
        for variable, pointer in self.pointers.items():
            if not pointer.read: continue
            if pointer.name == "membrane/voltages": factor = 1000 # From NEUWONs volts to NEURONs millivolts.
            else:                                   factor = 1
            preamble.append("    %s = %s[%s] * %s"%(variable, _CodeGen.mangle(variable), pointer.index, factor))
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

    def advance(self, access):
        # for name, km in self.kinetic_models.items():
        #     1/0
        threads = 64
        locations = access(self.name() + "/insertions")
        if not len(locations): return
        pointers = {name: access(ptr.name) for name, ptr in self.pointers.items()}
        blocks = (locations.shape[0] + (threads - 1)) // threads
        self._cuda_advance[blocks,threads](locations,
                *(ptr for name, ptr in sorted(pointers.items())))

class _Pointer:
    def __init__(self, nmodl, name, mode='rw', archetype=None):
        self.name = str(name)
        self.mode = str(mode)
        assert self.mode in ('r', 'w', 'rw', 'a')
        if   archetype is not None:              self.archetype = str(archetype)
        elif self.name.startswith("membrane"):   self.archetype = "membrane"
        elif self.name.startswith("inside"):     self.archetype = "inside"
        elif self.name.startswith("outside"):    self.archetype = "outside"
        elif self.name.startswith(nmodl.name()): self.archetype = nmodl.name()
        else: raise Exception("Unrecognised archetype: " + self.name)
    @property
    def read(self):
        return 'r' in self.mode
    @property
    def write(self):
        return self.accumulate or 'w' in self.mode
    @property
    def accumulate(self):
        return 'a' in self.mode
    def __repr__(self):
        return "_Pointer('%s', '%s')"%(self.name, self.mode)

class _CodeBlock:
    def __init__(self, nmodl, AST):
        if hasattr(AST, "name"):        self.name = AST.name.get_node_name()
        elif AST.is_breakpoint_block(): self.name = "BREAKPOINT"
        elif AST.is_initial_block():    self.name = "INITIAL"
        else:                           self.name = None
        self.derivative = AST.is_derivative_block()
        top_level = self.derivative or self.name in ("BREAKPOINT", "INITIAL")
        self.statements = []
        AST = getattr(AST, "statement_block", AST)
        for stmt in AST.statements:
            self.statements.extend(nmodl._parse_statement(stmt))
        self.conserve_statements = [x for x in self if isinstance(x, _ConserveStatement)]
        if top_level: self.gather_arguments(nmodl)

    def gather_arguments(self, nmodl):
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
                stmt.gather_arguments(nmodl)
                for symbol in stmt.arguments:
                    if symbol not in self.assigned:
                        self.arguments.add(symbol)
                self.assigned.update(stmt.assigned)
            elif isinstance(stmt, _SolveStatement):
                target_block = nmodl.derivative_blocks[stmt.block]
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
    def __init__(self, nmodl, AST):
        self.condition = nmodl._parse_expression(AST.condition)
        self.main_block = _CodeBlock(nmodl, AST.statement_block)
        self.elif_blocks = [_CodeBlock(nmodl, block) for block in AST.elseifs]
        assert(not self.elif_blocks) # TODO: Unimplemented.
        self.else_block = _CodeBlock(nmodl, AST.elses)

    def gather_arguments(self, nmodl):
        """ Sets arguments and assigned lists. """
        self.arguments = set()
        self.assigned = set()
        for symbol in self.condition.free_symbols:
            self.arguments.add(symbol.name)
        self.main_block.gather_arguments(nmodl)
        self.arguments.update(self.main_block.arguments)
        self.assigned.update(self.main_block.assigned)
        for block in self.elif_blocks:
            block.gather_arguments(nmodl)
            self.arguments.update(block.arguments)
            self.assigned.update(block.assigned)
        self.else_block.gather_arguments(nmodl)
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
        # self.pointer = pointer # Associated with the left hand side.

    def to_python(self,  indent="", context=None, **kwargs):
        if not isinstance(self.rhs, str):
            try: self.rhs = _CodeGen.sympy_to_pycode(self.rhs.simplify())
            except Exception:
                print("Failed at:", self.lhsn, "=", repr(self.rhs))
                raise
        if self.derivative:
            lhs = _CodeGen.mangle('d' + self.lhsn)
            return indent + lhs + " += " + self.rhs + "\n"
        if context != "INITIAL_BLOCK" and self.pointer:
            assert self.pointer.write, self.pointer.name
            lhs = _CodeGen.mangle(self.lhsn) + "[" + self.pointer.index + "]"
            eq = " += " if self.pointer.accumulate else " = "
            py = indent + lhs + eq + self.rhs + "\n"
            if self.pointer.read:
                py += indent + self.lhsn + " = " + lhs + "\n"
            return py
        return indent + self.lhsn + " = " + self.rhs + "\n"

    def solve(self):
        """ Solve this differential equation in-place. """
        assert(self.derivative)
        self.derivative = False
        try: self._solve_sympy(); return
        except Exception: pass
        try: self._solve_crank_nicholson(); return
        except Exception: pass
        self._solve_foward_euler()

    def _solve_sympy(self):
        dt    = sympy.Symbol("time_step", real=True, positive=True)
        state = sympy.Function(self.lhsn)(dt)
        deriv = self.rhs.subs(sympy.Symbol(self.lhsn), state)
        eq    = sympy.Eq(state.diff(dt), deriv)
        self.rhs = sympy.dsolve(eq, state)
        # TODO: Look up how to give the initial values to sympy to get rid of the constants.
        C1 = sympy.solve(self.rhs.subs(state, self.lhsn).subs(dt, 0), "C1")[0]
        self.rhs = self.rhs.subs("C1", C1).rhs

    def _solve_crank_nicholson(self):
        dt              = sympy.Symbol("time_step", real=True, positive=True)
        init_state      = sympy.Symbol(self.lhsn)
        next_state      = sympy.Symbol("Future" + self.lhsn)
        implicit_deriv  = self.rhs.subs(init_state, next_state)
        eq = sympy.Eq(next_state, init_state + implicit_deriv * dt / 2)
        backward_euler = sympy.solve(eq, next_state)
        assert(len(backward_euler) == 1)
        self.rhs = backward_euler.pop() * 2 - init_state

    def _solve_foward_euler(self):
        dt = sympy.Symbol("time_step", real=True, positive=True)
        self.rhs = sympy.Symbol(self.lhsn) + self.rhs * dt

class _SolveStatement:
    def __init__(self, nmodl, AST):
        self.block = AST.block_name.get_node_name()
        self.steadystate = AST.steadystate
        if AST.method: self.method = AST.method.get_node_name()
        else: self.method = "sparse" # FIXME
        AST.ifsolerr # TODO: What does this do?
        assert(self.block in nmodl.derivative_blocks)
        # arguments = nmodl.derivative_blocks[self.block].arguments
        # if self.steadystate:
        #     states_var = _CodeGen.mangle2("states")
        #     index_var  = _CodeGen.mangle2("index")
        #     self.py = states_var + " = solve_steadystate('%s', {%s})\n"%(self.block,
        #             ", ".join("'%s': %s"%(x, x) for x in arguments))
        #     for x in nmodl.states:
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
    def __init__(self, nmodl, name):
        1/0

    def to_python(self, indent, **kwargs):
        1/0

class _ConserveStatement:
    def __init__(self, nmodl, AST):
        conserved_expr = nmodl._parse_expression(AST.expr) - sympy.Symbol(AST.react.name.get_node_name())
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
            eprint("Error: loading nmodl from cache:", str(err))
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

if __name__ == "__main__":
    for name, (nmodl_file_path, kwargs) in sorted(library.items()):
        print("ATTEMPTING:", name)
        new_mechanism = NmodlMechanism(nmodl_file_path, **kwargs)

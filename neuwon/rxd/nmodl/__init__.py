from collections.abc import Callable, Iterable, Mapping
from neuwon.database import Real, Compute
from neuwon.rxd.mechanisms import OmnipresentMechanism, LocalMechanismSpecification, LocalMechanismInstance
from . import code_gen, cache, solver
from .parser import (NmodlParser, ANT,
        SolveStatement,
        AssignStatement,
        IfStatement,
        ConserveStatement)
import math
import numbers
import numpy as np
import os.path
import re
import sympy
import sys

__all__ = ["NMODL"]

def eprint(*args, **kwargs):
    """ Prints to standard error (sys.stderr). """
    print(*args, file=sys.stderr, flush=True, **kwargs)


# TODO: support for arrays? - arrays should really be unrolled in an AST pass...

# TODO: Move assignments to conductances to the end of the breakpoint block,
# where they belong. This is needed because the nmodl library inserts
# conductance hints and associated statements at the beginning of the block.
# self.breakpoint_block.statements.sort(key=lambda stmt: bool(
#         isinstance(stmt, AssignStatement)
#         and stmt.pointer and "conductance" in stmt.pointer.name))


class NMODL:
    def __init__(self, filename, parameters={}, use_cache=True):
        """
        Argument filename is an NMODL file to load.
                The standard NMODL file name extension is ".mod"

        Argument parameters is a mapping of parameter names to custom
                floating-point values.
        """
        self.filename = os.path.abspath(str(filename))
        if use_cache and cache.try_loading(self.filename, self): pass
        else:
            try:
                parser = NmodlParser(self.filename)
                self._check_for_unsupported(parser)
                self.nmodl_name, self.point_process,  self.title, self.description = parser.gather_documentation()
                self.parameters = ParameterTable(parser.gather_parameters(), self.nmodl_name)
                self.states = parser.gather_states()
                blocks = parser.gather_code_blocks()
                self.all_blocks = list(blocks.values())
                self.initial_block = blocks['INITIAL']
                self.breakpoint_block = blocks['BREAKPOINT']
                self.derivative_blocks = {k:v for k,v in blocks.items() if v.derivative}
                self._gather_IO(parser)
                if self.omnipresent:    self.__class__ = OmnipresentNmodlMechanism
                else:                   self.__class__ = LocalNmodlMechanismSpecification
                self._solve()
                self._fixup_breakpoint_IO()
            except Exception:
                eprint("ERROR while loading file", self.filename)
                raise
            cache.save(self.filename, self)
        self.parameters.update(parameters, strict=True)
        self.instance_parameters = self.parameters.split_instance_parameters(self.point_process)

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

    def _gather_IO(self, parser):
        """
        Determine what external data the mechanism accesses.

        Sets attributes: "pointers", "accumulators", "other_mechanisms_", and "omnipresent".
        """
        self.pointers = {}
        self.accumulators = set()
        self.other_mechanisms_ = []
        self.omnipresent = False # TODO!
        for state in self.states:
            self.pointers[state] = f'self.{state}'
        self.breakpoint_block.gather_arguments()
        self.initial_block.gather_arguments()
        all_args = self.breakpoint_block.arguments + self.initial_block.arguments
        if "v" in all_args:
            self.pointers["v"] = "self.segment.voltage"
        if "area" in all_args:
            self.pointers["area"] = "self.segment.surface_area"
        if "volume" in all_args:
            self.pointers["volume"] = "self.segment.inside_volume"
        for x in parser.lookup(ANT.USEION):
            ion = x.name.value.eval()
            # Automatically generate the variable names for this ion.
            equilibrium = ('e' + ion, ion + '_equilibrium',)
            current     = ('i' + ion, ion + '_current',)
            conductance = ('g' + ion, ion + '_conductance',)
            inside      = (ion + 'i', ion + '_inside',)
            outside     = (ion + 'o', ion + '_outside',)
            for y in x.readlist:
                var_name = y.name.value.eval()
                if var_name in equilibrium:
                    pass # Ignored, mechanisms output conductances instead of currents.
                elif var_name in inside:
                    self.pointers[var_name] = "self.segment.inside_concentrations/%s"
                elif var_name in outside:
                    self.pointers[var_name] = "self.outside/concentrations/%s"
                else:
                    raise ValueError(f"Unrecognized USEION READ: \"{var_name}\".")
            for y in x.writelist:
                var_name = y.name.value.eval()
                if var_name in current:
                    pass # Ignored, mechanisms output conductances instead of currents.
                elif var_name in conductance:
                    self.pointers[var_name] = f"self.segment.{ion}_conductance"
                    self.accumulators.add(var_name)
                elif var_name in inside:
                    self.pointers[var_name] = "self.segment.inside_delta_concentrations/%s"%ion
                    self.accumulators.add(var_name)
                elif var_name in outside:
                    self.pointers[var_name] = "outside/delta_concentrations/%s"
                    self.accumulators.add(var_name)
                else: raise ValueError(f"Unrecognized USEION WRITE: \"{var_name}\".")
        for x in parser.lookup(ANT.CONDUCTANCE_HINT):
            var_name = x.conductance.get_node_name()
            if var_name not in self.pointers:
                ion = x.ion.get_node_name()
                self.pointers[var_name] = f"Segment.{ion}_conductance"
                self.accumulators.add(var_name)
        for x in parser.lookup(ANT.POINTER_VAR):
            name = x.get_node_name()
            self.other_mechanisms_.append(name)
            self.pointers[name] = f"self.{name}.magnitude"
        self.other_mechanisms_ = tuple(sorted(self.other_mechanisms_))

    def _solve(self):
        """
        Replace SolveStatements with the solved equations to advance the systems
        of differential equations.
        """
        sympy_methods = ("cnexp", "derivimplicit", "euler")
        while True:
            for idx, stmt in enumerate(self.breakpoint_block):
                if not isinstance(stmt, SolveStatement): continue
                syseq_block = stmt.block
                if stmt.method in sympy_methods:
                    if syseq_block.derivative:
                        for stmt in syseq_block:
                            if isinstance(stmt, AssignStatement) and stmt.derivative:
                                solver.solve(stmt)
                elif stmt.method == "sparse":
                    1/0
                # Splice the solved block into the breakpoint block.
                bp_stmts = self.breakpoint_block.statements
                self.breakpoint_block.statements = bp_stmts[:idx] + syseq_block.statements + bp_stmts[idx+1:]
                break
            else:
                break

    def _fixup_breakpoint_IO(self):
        for stmt in self.breakpoint_block:
            if isinstance(stmt, AssignStatement):
                if stmt.lhsn in self.accumulators:
                    stmt.operation = '+='
        self.breakpoint_block.map(lambda stmt: stmt.simple_solution()
                                if isinstance(stmt, ConserveStatement) else [stmt])
        self.breakpoint_block.substitute(self.pointers)

    def initialize(self, rxd_model, name):
        database = rxd_model.get_database()
        builtin_parameters = {
                "time_step": rxd_model.get_time_step(),
                "celsius":   rxd_model.get_celsius(),
        }
        self.name = str(name)
        try:
            self.parameters.update(builtin_parameters, strict=True, override=False)
            self._run_initial_block(database)
            self._compile_breakpoint_block()
            if self.omnipresent:
                return self._initialize_omnipresent_mechanism_class(database)
            else:
                return self._initialize_local_mechanism_class(database)
        except Exception:
            eprint("ERROR while loading file", self.filename)
            raise

    def _run_initial_block(self, database):
        """ Use pythons built-in "exec" function to run the INITIAL_BLOCK.
        Sets: initial_state and initial_scope. """
        self.parameters.substitute(self.initial_block)
        initial_pointer_values = {}
        for arg in self.initial_block.arguments:
            if arg in self.pointers:
                db_access = self.pointers[arg]
                db_access = re.sub(r'^self\.segment\.', 'Segment.', db_access)
                db_access = re.sub(r'^self\.inside\.',  'Inside.',  db_access)
                db_access = re.sub(r'^self\.outside\.', 'Outside.', db_access)
                value = database.get(db_access).get_initial_value()
                if value is not None:
                    initial_pointer_values[arg] = value
        self.initial_block.substitute(initial_pointer_values)
        self.initial_block.gather_arguments()
        self.initial_scope = {self.pointers[x]: 0.0 for x in self.states}
        missing_arguments = set(self.initial_block.arguments) - set(self.initial_scope)
        if missing_arguments:
            raise ValueError(f"Missing initial values for {', '.join(missing_arguments)}.")
        initial_python = code_gen.to_python(self.initial_block)
        code_gen.exec_string(initial_python, {}, self.initial_scope)
        self.initial_state = {x: self.initial_scope.pop(x) for x in self.states}

    def _compile_breakpoint_block(self):
        # 
        self.parameters.substitute(self.breakpoint_block)
        # 
        magnitude = sympy.Symbol('self.magnitude')
        for name, (value, units) in self.instance_parameters.items():
            self.breakpoint_block.statements.insert(0, AssignStatement(name, value * magnitude))
        # 
        for arg in self.breakpoint_block.arguments:
            if arg in self.pointers: continue
            elif arg in self.initial_scope:
                value = float(self.initial_scope[arg])
                self.breakpoint_block.statements.insert(0, AssignStatement(arg, value))
        # 
        self.advance_pycode = (
                "@Compute\n"
                "def advance(self):\n"
                + code_gen.to_python(self.breakpoint_block, "    ", self.pointers))
        globals_ = {
                'Compute': Compute,
        }
        code_gen.exec_string(self.advance_pycode, globals_)
        self.advance_bytecode = globals_['advance']

    def _initialize_omnipresent_mechanism_class(self, database):
        1/0 # TODO

        # NOTES:
        # 
        # Omnipresent mechanisms are used to implement chemical reactions.
        # 
        # Omnipresent mechanisms must have no state variables, instead they use
        # species concentrations and (possibly?) the standard segment attributes.
        # 
        # All omnipresent mechanisms must be associated to a single entity type:
        #       -> Intracellular
        #       -> Extracellular
        #       -> Segment
        # And their advance method will be an anonymous method over all
        # instances of the associated entity.

    def _initialize_local_mechanism_class(self, database):
        mechanism_superclass = type(self.name, (LocalMechanismInstance,), {
            '__slots__': (),
            '__init__':             NMODL._instance__init__,
            '_point_process':       self.point_process,
            '_other_mechanisms':    self.other_mechanisms_,
            'advance':              self.advance_bytecode,
            '_advance_pycode':      self.advance_pycode,
        })
        mech_data = database.add_class(self.name, mechanism_superclass, doc=self.description)
        mech_data.add_attribute("segment", dtype="Segment")
        if self.point_process:
            mech_data.add_attribute("magnitude", 1.0,
                    doc="") # TODO?
        else:
            mech_data.add_attribute("magnitude",
                    units = database.get('Segment.surface_area').get_units(),
                    doc="") # TODO?
        for name in self.states:
            mech_data.add_attribute(name, initial_value=self.initial_state[name], units=name)
        for name in self.other_mechanisms_:
            mech_data.add_attribute(name, dtype=name)
        return mech_data.get_instance_type()

    @staticmethod
    def _instance__init__(self, segment, magnitude=1.0, *other_mechanisms):
        """ Insert this mechanism onto the given segment. """
        self.segment = segment
        if self._point_process:
            self.magnitude = magnitude
        else:
            self.magnitude = magnitude * segment.surface_area
        assert len(self._other_mechanisms) == len(other_mechanisms)
        for name, ref in zip(self._other_mechanisms, other_mechanisms):
            setattr(self, name, ref)

class ParameterTable(dict):
    """ Dictionary mapping from nmodl parameter name to pairs of (value, units). """

    builtin_parameters = {
        "celsius": (None, "degC"),
        "time_step": (None, "ms"),
    }

    def __init__(self, parameters, mechanism_name):
        dict.__init__(self)
        self.mechanism_name = mechanism_name
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

    def get_instance_parameters(self, point_process):
        """ Returns the instance_parameters.

        The surface area parameters are special because each segment of neuron
        has its own surface area and so their actual values are different for
        each instance of the mechanism. They are not in-lined directly into the
        source code, instead they are stored alongside the state variables and
        accessed at run time. 
        """
        parameters = {}
        for name, (value, units) in self.items():
            if units:
                if point_process:
                    if "/" + self.mechanism_name in units:
                        parameters[name] = (value, units)
                else:
                    if "/cm2" in units:
                        # Convert from NEUWONs um^2 to NEURONs cm^2.
                        value *= (1e-6 * 1e-6) / (1e-2 * 1e-2)
                        units  = units.replace("/cm2", '')
                        parameters[name] = (value, units)
        return parameters

    def split_instance_parameters(self, point_process):
        """ Removes and returns the instance_parameters. """
        parameters = self.get_instance_parameters(point_process)
        for name in parameters:
            self.pop(name)
        return parameters

    def substitute(self, block):
        substitutions = {}
        for name, (value, units) in self.items():
            if value is None:
                continue
            if name == "time_step":
                name = solver.dt
            substitutions[name] = value
        block.substitute(substitutions)

class OmnipresentNmodlMechanism(NMODL, OmnipresentMechanism):
    pass

class LocalNmodlMechanismSpecification(NMODL, LocalMechanismSpecification):
    def other_mechanisms(self):
        return self.other_mechanisms_

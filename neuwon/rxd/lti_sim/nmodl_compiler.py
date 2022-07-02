"""
Frontend for reading the users NMODL files, extracting key information, and
evaluating the DERIVATIVE blocks contained in them.
"""

import math
import os.path
import re
import nmodl.ast
import nmodl.dsl
import nmodl.symtab
from .inputs import (Input, LinearInput, LogarithmicInput)

ANT = nmodl.ast.AstNodeType

class NMODL_Compiler:
    def __init__(self, nmodl_filename, inputs, temperature):
        self.nmodl_filename = os.path.abspath(str(nmodl_filename))
        self._setup_parser()
        self._gather_name()
        self._gather_states()
        self._gather_parameters()
        self._gather_conserve_statements()
        nmodl.dsl.visitor.KineticBlockVisitor().visit_program(self.AST)
        self._gather_code_blocks()
        self._gather_inputs(inputs)
        self._compile_derivative_block(temperature)

    def _setup_parser(self):
        with open(self.nmodl_filename, 'rt') as f: nmodl_text = f.read()
        self.AST = nmodl.dsl.NmodlDriver().parse_string(nmodl_text)
        nmodl.symtab.SymtabVisitor().visit_program(self.AST)
        nmodl.dsl.visitor.InlineVisitor().visit_program(self.AST)
        self.visitor = nmodl.dsl.visitor.AstLookupVisitor()
        self.lookup  = lambda n: self.visitor.lookup(self.AST, n)
        self.symbols = self.AST.get_symbol_table()
        # Helpful debugging readouts:
        if False: print(nmodl.dsl.to_nmodl(self.AST))
        if False: print(self.AST.get_symbol_table())
        if False: nmodl.ast.view(self.AST)

    def _gather_name(self):
        x = self.lookup(ANT.SUFFIX)
        if x: self.name = x[0].name.get_node_name()
        else: self.name = os.path.splitext(os.path.split(self.filename)[1])[0]

    def _gather_states(self):
        states = self.symbols.get_variables_with_properties(nmodl.symtab.NmodlType.state_var)
        self.state_names = sorted(x.get_name() for x in states)
        self.num_states = len(self.state_names)

    def _gather_parameters(self):
        self.parameters = {}
        for stmt in self.lookup(ANT.PARAM_ASSIGN):
            value = stmt.value
            if value is None:
                continue
            name = stmt.name.get_node_name()
            self.parameters[name] = float(value.eval())

    def _gather_conserve_statements(self):
        conserve_statements = self.lookup(ANT.CONSERVE)
        if not conserve_statements:
            self.conserve_sum = None
            return
        elif len(conserve_statements) > 1:
            raise ValueError("Multiple CONSERVE statements are not supported.")
        stmt    = conserve_statements[0]
        states  = nmodl.dsl.to_nmodl(stmt.react).split('+')
        if set(states) != set(self.state_names) or not stmt.expr.is_number():
            raise ValueError('CONSERVE statement must be in the form: sum-of-all-states = number.')
        self.conserve_sum = float(stmt.expr.eval())

    def _gather_code_blocks(self):
        derivative_blocks = self.lookup(ANT.DERIVATIVE_BLOCK)
        assert len(derivative_blocks) == 1
        self.derivative_block = CodeBlock(derivative_blocks[0])
        self.initial_block = CodeBlock(self.lookup(ANT.INITIAL_BLOCK)[0])

    def _gather_inputs(self, inputs):
        # 
        input_symbols = []
        if self.derivative_block.reads_symbol("v"):
            input_symbols.append("v")
        for stmt in self.lookup(ANT.USEION):
            ion = stmt.name.value.eval()
            for x in stmt.readlist:
                var_name = x.name.value.eval()
                if   var_name == ion + 'i': input_symbols.append(var_name)
                elif var_name == ion + 'o': input_symbols.append(var_name)
        for x in self.lookup(ANT.POINTER_VAR):
            input_symbols.append(x.get_node_name())
        for x in self.lookup(ANT.BBCORE_POINTER_VAR):
            input_symbols.append(x.get_node_name())
        # Check argument "inputs".
        if not inputs:
            inputs = self._get_default_inputs(input_symbols)
        assert all(isinstance(inp, Input) for inp in inputs)
        # Match up the expected inputs with the given "Input" data structures.
        inputs = {inp.name: inp for inp in inputs}
        try:
            self.inputs = [inputs[name] for name in sorted(input_symbols)]
            assert len(inputs) == len(input_symbols)
        except (KeyError, AssertionError):
            expected_inputs = ' & '.join(input_symbols)
            received_inputs = ' & '.join(inputs.keys())
            raise ValueError(f'Invalid inputs, expected {expected_inputs} got {received_inputs}')
        self.num_inputs = len(self.inputs)
        self.input_names = [inp.name for inp in self.inputs]
        # Make aliases "input1", "input2", etc.
        for inp_idx, inp in enumerate(self.inputs):
            setattr(self, f"input{inp_idx+1}", inp)

    def _get_default_inputs(self, input_symbols):
        inputs = []
        for name in sorted(input_symbols):
            if name == "v":
                inputs.append(LinearInput(name, -120, 120))
            else:
                inputs.append(LogarithmicInput(name, 0, 1000))
        return inputs

    def _compile_derivative_block(self, temperature):
        scope = {'celsius': float(temperature)} # Allow NMODL file to override temperature.
        scope.update(self.parameters)
        pycode = self.initial_block.to_python()
        arguments = [inp.name for inp in self.inputs] + self.state_names
        pycode += f"def {self.name}_derivative_({', '.join(arguments)}):\n"
        for state in self.state_names:
            pycode += f"    __d_{state} = 0.0\n"
        pycode += self.derivative_block.to_python("    ")
        pycode += f"    return [{', '.join(f'__d_{state}' for state in self.state_names)}]\n\n"
        _exec_string(pycode, scope)
        self.derivative = scope[f"{self.name}_derivative_"]
        self.temperature = scope['celsius']

    @classmethod
    def _parse_statement(cls, AST):
        """ Returns a list of Statement objects. """
        original = AST
        if AST.is_expression_statement():   AST = AST.expression
        if AST.is_wrapped_expression():     AST = AST.expression
        if AST.is_unit_state():             return []
        if AST.is_local_list_statement():   return []
        if AST.is_conductance_hint():       return []
        if AST.is_conserve():               return []
        if AST.is_solve_block():            return []
        if AST.is_if_statement():           return [IfStatement(AST)]
        if AST.is_statement_block():
            statements = []
            for stmt in AST.statements:
                statements.extend(cls._parse_statement(stmt))
            return statements
        is_derivative = AST.is_diff_eq_expression()
        if is_derivative: AST = AST.expression
        if AST.is_binary_expression():
            assert AST.op.eval() == "="
            lhsn = AST.lhs.name.get_node_name()
            return [AssignStatement(lhsn, NMODL_Compiler._parse_expression(AST.rhs),
                    derivative = is_derivative,)]
        raise ValueError("Unsupported syntax at %s."%nmodl.dsl.to_nmodl(original))

    @classmethod
    def _parse_expression(cls, AST):
        """ Returns a string of python code. """
        if AST.is_var_name():
            AST = AST.name
        if AST.is_name():
            return AST.get_node_name()
        if AST.is_double_unit():
            AST = AST.value
        if AST.is_number():
            return AST.eval()
        if AST.is_wrapped_expression() or AST.is_paren_expression():
            return f'({cls._parse_expression(AST.expression)})'
        if AST.is_unary_expression():
            op   = AST.op.eval()
            expr = cls._parse_expression(AST.expression)
            return f'({op} {expr})'
        if AST.is_binary_expression():
            lhs = cls._parse_expression(AST.lhs)
            rhs = cls._parse_expression(AST.rhs)
            op  = AST.op.eval()
            if op == '^':
                op = '**'
            return f'({lhs} {op} {rhs})'
        if AST.is_function_call():
            name = AST.name.get_node_name()
            args = ', '.join(cls._parse_expression(x) for x in AST.arguments)
            if name == 'fabs':  name = 'math.abs'
            if name == 'exp':   name = 'math.exp'
            return f'{name}({args})'
        raise ValueError("Unsupported syntax at %s."%nmodl.dsl.to_nmodl(AST))

class CodeBlock:
    def __init__(self, AST):
        self.statements = []
        AST = getattr(AST, "statement_block", AST)
        for stmt in AST.statements:
            self.statements.extend(NMODL_Compiler._parse_statement(stmt))

    def reads_symbol(self, symbol):
        regex = rf'\b{symbol}\b'
        for stmt in self.statements:
            if isinstance(stmt, AssignStatement):
                if re.search(regex, stmt.rhs):
                    return True
            elif isinstance(stmt, IfStatement):
                if re.search(regex, stmt.condition):
                    return True
                for block in stmt.blocks:
                    if block.reads_symbol(symbol):
                        return True
        return False

    def to_python(self, indent=""):
        py = ""
        for stmt in self.statements:
            py += stmt.to_python(indent)
        if not self.statements:
            py += indent + "pass\n"
        return py

class IfStatement:
    def __init__(self, AST):
        self.condition   = NMODL_Compiler._parse_expression(AST.condition)
        self.main_block  = CodeBlock(AST.statement_block)
        self.elif_blocks = [CodeBlock(block) for block in AST.elseifs]
        self.else_block  = CodeBlock(AST.elses) if AST.elses is not None else None
        self.blocks      = [self.main_block] + self.elif_blocks
        if self.else_block is not None:
            self.blocks.append(self.else_block)

    def to_python(self, indent=""):
        py  = indent + "if %s:\n"%self.condition
        py += self.main_block.to_python(indent + "    ")
        assert not self.elif_blocks, "Unimplemented."
        if self.else_block is not None:
            py += indent + "else:\n"
            py += self.else_block.to_python(indent + "    ")
        return py

class AssignStatement:
    def __init__(self, lhsn, rhs, derivative=False):
        self.lhsn = str(lhsn) # Left hand side name.
        self.rhs  = str(rhs) # Right hand side.
        self.derivative = bool(derivative)

    def to_python(self, indent=""):
        if self.derivative:
            return f'{indent}__d_{self.lhsn} = {self.rhs}\n'
        else:
            return f'{indent}{self.lhsn} = {self.rhs}\n'

def _exec_string(pycode, globals_):
    globals_.setdefault("math", math)
    try: exec(pycode, globals_)
    except:
        err_msg  = "Error while executing the following python program:\n"
        err_msg += pycode
        err_msg += "\n"
        err_msg += "END OF PROGRAM.\n"
        err_msg += "Global Variables:\n"
        for name, value in sorted(globals_.items()):
            if name == "__builtins__": value = '...'
            err_msg += f"\t{name.ljust(15)} {value}\n"
        print(err_msg, flush=True)
        raise

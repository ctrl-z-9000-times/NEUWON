import itertools
import nmodl
import nmodl.ast
import nmodl.dsl
import nmodl.symtab
import os.path
import sympy
import textwrap

ANT = nmodl.ast.AstNodeType

class NmodlParser:
    """ Attributes: filename, visitor, lookup, and symbols.

    Keep all references to the "nmodl" library separate from the main classes
    for clean & easy deletion. The nmodl library is implemented in C++ and as
    such does not support some critical python features: objects returned from
    the nmodl library do not support copying or pickling. """
    def __init__(self, nmodl_filename):
        """ Parse the NMDOL file into an abstract syntax tree (AST). """
        self.filename = str(nmodl_filename)
        with open(self.filename, 'rt') as f: nmodl_text = f.read()
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
        if False: print(self.to_nmodl(AST))
        if False: print(AST.get_symbol_table())
        if False: nmodl.ast.view(AST)

    to_nmodl = nmodl.dsl.to_nmodl

    def gather_documentation(self):
        """ Returns tuple of (name, point_process, title, and description).
        This assumes that the first block comment is the primary documentation. """
        x = self.lookup(ANT.SUFFIX)
        if x: name = x[0].name.get_node_name()
        else: name = os.path.split(self.filename)[1] # TODO: Split extension too?
        if x: point_process = (x[0].type.get_node_name() == "POINT_PROCESS")
        else: point_process = False
        title = self.lookup(ANT.MODEL)
        title = title[0].title.eval().strip() if title else ""
        comments = self.lookup(ANT.BLOCK_COMMENT)
        description = comments[0].statement.eval().strip() if comments else ""
        description = textwrap.dedent(description)
        return (name, point_process, title, description)

    def gather_states(self):
        """ Returns sorted list the names of the states. """
        return sorted(v.get_name() for v in
                self.symbols.get_variables_with_properties(nmodl.symtab.NmodlType.state_var))

    def gather_parameters(self):
        """ Returns dictionary of name -> (value, units). """
        parameters = {}
        for assign_stmt in self.lookup(ANT.PARAM_ASSIGN):
            name  = str(self.visitor.lookup(assign_stmt, ANT.NAME)[0].get_node_name())
            value = self.visitor.lookup(assign_stmt, [ANT.INTEGER, ANT.DOUBLE])
            units = self.visitor.lookup(assign_stmt, ANT.UNIT)
            value = float(value[0].eval())        if value else None
            units = str(units[0].get_node_name()) if units else None
            parameters[name] = (value, units)
        return parameters

    def gather_code_blocks(self):
        blocks = {}
        for AST in self.lookup(ANT.DERIVATIVE_BLOCK):
            name = AST.name.get_node_name()
            code_block = self.parse_code_block(AST)
            blocks[name] = code_block
        initial_block = self.lookup(ANT.INITIAL_BLOCK)
        if initial_block:
            blocks['INITIAL'] = self.parse_code_block(initial_block.pop())
        else:
            blocks['INITIAL'] = EmptyCodeBlock('INITIAL')
        blocks['BREAKPOINT'] = self.parse_code_block(self.lookup(ANT.BREAKPOINT_BLOCK).pop())
        for b in blocks.values():
            for stmt in b:
                if isinstance(stmt, SolveStatement):
                    stmt.block = blocks[stmt.block]
        return blocks

    @classmethod
    def parse_code_block(cls, AST):
        return CodeBlock(AST)

    @classmethod
    def parse_statement(cls, AST):
        """ Returns a list of Statement objects. """
        original = AST
        if AST.is_unit_state():             return []
        if AST.is_local_list_statement():   return []
        if AST.is_conductance_hint():       return []
        if AST.is_if_statement(): return [IfStatement(AST)]
        if AST.is_conserve():     return [ConserveStatement(AST)]
        if AST.is_expression_statement():
            AST = AST.expression
        if AST.is_wrapped_expression():
            AST = AST.expression
        if AST.is_solve_block(): return [SolveStatement(AST)]
        if AST.is_statement_block():
            return list(itertools.chain.from_iterable(
                    cls.parse_statement(stmt) for stmt in AST.statements))
        is_derivative = AST.is_diff_eq_expression()
        if is_derivative: AST = AST.expression
        if AST.is_binary_expression():
            assert(AST.op.eval() == "=")
            lhsn = AST.lhs.name.get_node_name()
            return [AssignStatement(lhsn, NmodlParser.parse_expression(AST.rhs),
                    derivative = is_derivative,)]
        original = NmodlParser.to_nmodl(original)
        # TODO: Get line number from AST and include it in error message.
        if AST.is_function_call():
            raise ValueError(f"Function call not inlined: {original}")
        raise ValueError("Unrecognized syntax at %s."%original)

    @classmethod
    def parse_expression(cls, AST):
        """ Returns a SymPy expression. """
        if AST.is_wrapped_expression() or AST.is_paren_expression():
            return cls.parse_expression(AST.expression)
        if AST.is_integer():  return sympy.Integer(AST.eval())
        if AST.is_double():   return sympy.Float(AST.eval(), 18)
        if AST.is_name():     return sympy.Symbol(AST.get_node_name(), real=True)
        if AST.is_var_name(): return sympy.Symbol(AST.name.get_node_name(), real=True)
        if AST.is_unary_expression():
            op = AST.op.eval()
            if op == "-": return - cls.parse_expression(AST.expression)
            raise ValueError("Unrecognized syntax at %s."%cls.to_nmodl(AST))
        if AST.is_binary_expression():
            op = AST.op.eval()
            lhs = cls.parse_expression(AST.lhs)
            rhs = cls.parse_expression(AST.rhs)
            if op == "+": return lhs + rhs
            if op == "-": return lhs - rhs
            if op == "*": return lhs * rhs
            if op == "/": return lhs / rhs
            if op == "^": return lhs ** rhs
            if op == "<": return lhs < rhs
            if op == ">": return lhs > rhs
            if op == "<=": return lhs <= rhs
            if op == ">=": return lhs >= rhs
            raise ValueError("Unrecognized syntax at %s."%cls.to_nmodl(AST))
        if AST.is_function_call():
            name = AST.name.get_node_name()
            args = [cls.parse_expression(x) for x in AST.arguments]
            if name == "fabs":  return sympy.Abs(*args)
            if name == "exp":   return sympy.exp(*args)
            else: return sympy.Function(name)(*args)
        if AST.is_double_unit():
            return cls.parse_expression(AST.value)
        raise ValueError("Unrecognized syntax at %s."%cls.to_nmodl(AST))

class CodeBlock:
    def __init__(self, AST):
        if hasattr(AST, "name"):        self.name = AST.name.get_node_name()
        elif AST.is_breakpoint_block(): self.name = "BREAKPOINT"
        elif AST.is_initial_block():    self.name = "INITIAL"
        else:                           self.name = None
        self.derivative = AST.is_derivative_block()
        self.statements = []
        AST = getattr(AST, "statement_block", AST)
        for stmt in AST.statements:
            self.statements.extend(NmodlParser.parse_statement(stmt))
        self.conserve_statements = [x for x in self if isinstance(x, ConserveStatement)]

    def __iter__(self):
        """ Yields all Statements contained in this block. """
        for stmt in self.statements:
            yield stmt
            if isinstance(stmt, IfStatement):
                for block in stmt.blocks:
                    for x in block:
                        yield x

    def map(self, f):
        """ Argument f is function f(Statement) -> [Statement,]"""
        mapped_statements = []
        for stmt in self.statements:
            if isinstance(stmt, IfStatement):
                for block in stmt.blocks:
                    block.map(f)
            mapped_statements.extend(f(stmt))
        self.statements = mapped_statements

    def substitute(self, substitutions: dict):
        substitution_symbols = {
                sympy.Symbol(k, real=True) if isinstance(k, str) else k :
                sympy.Symbol(v, real=True) if isinstance(v, str) else v
                for k, v in substitutions.items()}
        for stmt in self:
            if isinstance(stmt, AssignStatement):
                stmt.rhs = stmt.rhs.subs(substitution_symbols)
                if stmt.lhsn in substitutions:
                    stmt.lhsn = substitutions[stmt.lhsn]
            elif isinstance(stmt, IfStatement):
                stmt.condition = stmt.condition.subs(substitution_symbols)

    def gather_arguments(self):
        """ Sets arguments and assigned lists. """
        self.arguments = set()
        self.assigned  = set()
        def read_symbol(symbol_name):
            if symbol_name not in self.assigned:
                self.arguments.add(symbol_name)
        def write_symbol(symbol_name):
            self.assigned.add(symbol_name)
        for stmt in iter(self):
            if isinstance(stmt, AssignStatement):
                for symbol in stmt.rhs.free_symbols:
                    read_symbol(symbol.name)
                write_symbol(stmt.lhsn)
            elif isinstance(stmt, IfStatement):
                for symbol in stmt.condition.free_symbols:
                    read_symbol(symbol.name)
                for block in stmt.blocks:
                    block.gather_arguments()
                    for name in block.arguments:    read_symbol(name)
                    for name in block.assigned:     write_symbol(name)
            elif isinstance(stmt, SolveStatement):
                target_block = stmt.block
                target_block.gather_arguments()
                for name in target_block.arguments: read_symbol(name)
                for name in target_block.assigned:  write_symbol(name)
            elif isinstance(stmt, ConserveStatement): pass
            else: raise NotImplementedError(stmt)
        self.arguments = sorted(self.arguments)
        self.assigned  = sorted(self.assigned)

class EmptyCodeBlock(CodeBlock):
    def __init__(self, name):
        self.name = str(name)
        self.derivative = False
        self.statements = []
        self.conserve_statements = []

class IfStatement:
    def __init__(self, AST):
        self.condition   = NmodlParser.parse_expression(AST.condition)
        self.main_block  = CodeBlock(AST.statement_block)
        self.elif_blocks = [CodeBlock(block) for block in AST.elseifs]
        self.else_block  = CodeBlock(AST.elses) if AST.elses is not None else None
        self.blocks      = [self.main_block] + self.elif_blocks
        if self.else_block is not None:
            self.blocks.append(self.else_block)

class AssignStatement:
    def __init__(self, lhsn, rhs, derivative=False, operation='='):
        self.lhsn = str(lhsn) # Left hand side name.
        if isinstance(rhs, str):
            rhs = sympy.sympify(rhs)
        self.rhs = rhs # Right hand side.
        self.derivative = bool(derivative)
        self.operation = str(operation)

    def __repr__(self):
        lhsn = self.lhsn
        if self.derivative:
            lhsn = lhsn + "'"
        s = f'{lhsn} {self.operation} {str(self.rhs)}'
        return s

class SolveStatement:
    def __init__(self, AST):
        # The "block" attribute gets rewritten by the method gather_code_blocks().
        self.block = AST.block_name.get_node_name()
        self.steadystate = AST.steadystate
        if AST.method:
            self.method = AST.method.get_node_name()
        else:
            self.method = "exact"
        AST.ifsolerr # TODO: What does this do?

class ConserveStatement:
    def __init__(self, AST):
        conserved_expr = NmodlParser.parse_expression(AST.expr) - sympy.Symbol(AST.react.name.get_node_name())
        self.states = list(sympy.Symbol(str(x), real=True) for x in conserved_expr.free_symbols)
        # Assume that the conserve statement was once in the form: sum-of-states = constant-value
        sum_symbol = sympy.Symbol("_CONSERVED_SUM")
        assumed_form = sum_symbol
        for symbol in conserved_expr.free_symbols:
            assumed_form = assumed_form - symbol
        sum_solution = sympy.solvers.solve(sympy.Eq(conserved_expr, assumed_form), sum_symbol)
        assert(len(sum_solution) == 1)
        self.conserve_sum = sum_solution[0].evalf()

    def simple_solution(self):
        if not self.states:
            return []
        true_sum = self.states[0]
        for state in self.states[1:]:
            true_sum = true_sum + state
        replacement = [AssignStatement('_CORRECTION_FACTOR', self.conserve_sum / true_sum)]
        for state in self.states:
            replacement.append(
                    AssignStatement(state, '_CORRECTION_FACTOR', operation = '*='))
        return replacement

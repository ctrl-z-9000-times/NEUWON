""" Private module. """
import copy
import neuwon.units
import nmodl
import nmodl.ast
import nmodl.dsl
import nmodl.symtab
import sympy
import os.path

ANT = nmodl.ast.AstNodeType

class _NmodlParser:
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
        """ Returns triple of (name, title, and description).
        This assumes that the first block comment is the primary documentation. """
        x = self.lookup(ANT.SUFFIX)
        if x: name = x[0].name.get_node_name()
        else: name = os.path.split(self.filename)[1] # TODO: Split extension too?
        title = self.lookup(ANT.MODEL)
        title = title[0].title.eval().strip() if title else ""
        if title.startswith(name + ".mod"):
            title = title[len(name + ".mod"):].strip()
        if title: title = title[0].title() + title[1:] # Capitalize the first letter.
        comments = self.lookup(ANT.BLOCK_COMMENT)
        description = comments[0].statement.eval() if comments else ""
        return (name, title, description)

    def gather_states(self):
        """ Returns sorted list the names of the states. """
        return sorted(v.get_name() for v in
                self.symbols.get_variables_with_properties(nmodl.symtab.NmodlType.state_var))

    def gather_units(self):
        units = copy.deepcopy(neuwon.units.builtin_units)
        for AST in self.lookup(ANT.UNIT_DEF):
            units.add_unit(AST.unit1.name.eval(), AST.unit2.name.eval())
        return units

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

    @classmethod
    def parse_expression(cls, AST):
        """ Returns a SymPy expression. """
        if AST.is_wrapped_expression() or AST.is_paren_expression():
            return cls.parse_expression(AST.expression)
        if AST.is_integer():  return sympy.Integer(AST.eval())
        if AST.is_double():   return sympy.Float(AST.eval(), 18)
        if AST.is_name():     return sympy.symbols(AST.get_node_name(), real=True)
        if AST.is_var_name(): return sympy.symbols(AST.name.get_node_name(), real=True)
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

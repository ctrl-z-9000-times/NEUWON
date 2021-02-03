# What technology am I targetting now? CUDA!
# Anything cuda will work, as long as I can call it from python. the baeuty of
# the system is that it delegates the *entire* compute to the mechanism, simply
# passing them the CUDA buffers. I could write a rust API which did the dispatch
# to a rust cuda kernel? Or I could do python with numba, or C with cupy.

# Be sure to split the logic into multiple classes for the front-end (NMODL) and
# back-end (GPU-Implementation).

# Do not spent extra time implementing NMODL features.

import nmodl
from nmodl import dsl, ast, symtab
from nmodl.dsl import visitor
ANT = ast.AstNodeType
import os.path
import neuwon
from neuwon import Real

class NmodlFile:
    """ Read and process an NMODL file into an easier to use data structure. """
    def __init__(self, filename):
        self.filename = os.path.normpath(str(filename))
        with open(self.filename, 'rt') as f:
            nmodl_text = f.read()
        # Parse the NMDOL file into AST.
        AST = dsl.NmodlDriver().parse_string(nmodl_text)
        self.set_ast(AST) # Sets: visitor, lookup, symbols
        self.check_for_unsupported()
        self.gather_documentation() # Sets: name, title, description
        self.gather_io()
        self.scope_stack = [{}]
        self.gather_parameters() # Sets: parameters, surface_area_parameters
        self.states = [v.get_name() for v in
                self.symbols.get_variables_with_properties(symtab.NmodlType.state_var)]
        # Pre-Process the AST.
        visitor.ConstantFolderVisitor().visit_program(AST)
        visitor.InlineVisitor().visit_program(AST)
        self.set_ast(AST)
        # Determine which functions are still in use, after inlining most of them.
        self.functions_called = set(x.name.get_node_name() for x in self.lookup(ANT.FUNCTION_CALL))
        self.solve_blocks = set(x.block_name.get_node_name() for x in self.lookup(ANT.SOLVE_BLOCK))
        self.functions_called.update(self.solve_blocks)
        # Process conserve statements and determine the initial state.
        self.conserve = []
        for x in self.lookup(ANT.CONSERVE):
            variables = self.visitor.lookup(x.react, ANT.REACT_VAR_NAME)
            variables = [v.get_node_name() for v in variables]
            assert(all(v in self.states for v in variables))
            self.conserve.append((variables, x.expr.eval()))
        self.initial_state = {s: 0.0 for s in self.states}
        for (variables, value) in self.conserve:
            for v in variables:
                assert(self.initial_state[v] == 0.0) # State variable is CONSERVEd multiple times.
                self.initial_state[v] = value / len(variables)
        # Solve the differential equations.
        visitor.KineticBlockVisitor().visit_program(AST)
        visitor.SympySolverVisitor().visit_program(AST)
        self.set_ast(AST)
        for x in self.lookup(ANT.DERIVATIVE_BLOCK):
            self.visit_derivative_block(x)
        for x in self.lookup(ANT.FUNCTION_BLOCK):
            self.func_name = x.name.get_node_name()
            self.visit_function_block(x)
            del self.func_name
        for x in self.lookup(ANT.PROCEDURE_BLOCK):
            self.visit_function_block(x)

        # Helpful routines:
        # print(dsl.to_nmodl(AST))
        # ast.view(AST)
        # print(self.symbols)
        # help(visitor.AstVisitor)
        # help(ast.AstNodeType)

    def set_ast(self, AST):
        self.visitor = visitor.AstLookupVisitor()
        self.lookup = lambda n: self.visitor.lookup(AST, n)
        symtab.SymtabVisitor().visit_program(AST)
        self.symbols = AST.get_symbol_table()

    def check_for_unsupported(self):
        if self.lookup(ANT.FUNCTION_TABLE_BLOCK):
            raise ValueError("FUNCTION_TABLE's are not allowed.")
        # TODO: No support for Pointer!
        # TODO: No support for Independent!
        # TODO: No support for Nonlinear!
        # TODO: No support for Include!
        # TODO: No support for VERBATIM!
        # TODO: No support for solver methods SPARSE or EULER!
        for x in self.lookup(ANT.SOLVE_BLOCK):
            pass

    def gather_documentation(self):
        self.name = os.path.split(self.filename)[1]
        for x in self.lookup(ANT.SUFFIX):
            self.name = x.name.get_node_name()
        title = self.lookup(ANT.MODEL)
        if not title:
            self.title = ""
        else:
            self.title = title[0].title.eval().strip()
            if self.title.startswith(self.name + ".mod"):
                self.title = self.title[len(self.name + ".mod"):].strip()
            if self.title:
                self.title = self.title[0].title() + self.title[1:] # Capitalize the first letter.
        # This assumes that the first block comment is the primary documentation.
        block_comments = self.lookup(ANT.BLOCK_COMMENT)
        if block_comments:
            self.description = block_comments[0].statement.eval()
        else:
            self.description = ""

    def gather_io(self):
        self.read_concentrations = [x.name.get_node_name() for x in self.lookup(ANT.READ_ION_VAR)]
        self.write_currents = [x.name.get_node_name() for x in self.lookup(ANT.WRITE_ION_VAR)]
        self.nonspecific_currents = [x.name.get_node_name() for x in self.lookup(ANT.NONSPECIFIC_CUR_VAR)]
        self.write_conductances = {} # Conductance variable -> rust substitution.
        for current in self.write_currents:
            ion = current.lstrip('i')
            self.write_conductances['g' + ion] = "__r_out.%s_g[__inst.location]"%ion
            if ion not in self.species:
                raise ValueError("Unrecognized ion species: "+ion)
        self.write_currents.extend(self.nonspecific_currents)

    def gather_parameters(self):
        self.parameters = {}
        self.surface_area_parameters = []
        for assign in self.lookup(ANT.PARAM_ASSIGN):
            name = str(self.visitor.lookup(assign, ANT.NAME)[0].get_node_name())
            value = self.visitor.lookup(assign, [ANT.INTEGER, ANT.DOUBLE])
            units = self.visitor.lookup(assign, ANT.UNIT)
            if not value:
                continue
            value = float(value[0].eval())
            if units:
                units = units[0].get_node_name()
                if "/cm2" in units:
                    self.surface_area_parameters.append(name)
            else:
                units = None
            self.parameters[name] = (value, units)
            self.scope_stack[-1][name] = (value, units)
        for init in self.lookup(ANT.INITIAL_BLOCK):
            for x in self.visitor.lookup(init, ANT.EXPRESSION_STATEMENT):
                if isinstance(x.expression, ast.BinaryExpression) and x.expression.op.eval() == "=":
                    lhsn = str(x.expression.lhs.name.get_node_name())
                    if lhsn in self.states:
                        continue
                    self.scope_stack[-1][lhsn] = (x.expression.rhs.accept(self), None)



class NmodlMechanism(neuwon.Mechanism):
    def __init__(self, filename):
        self.file = NmodlFile(filename)

    def instance_dtype(self):
        return {
            "state": ("Real", len(self.file.states)),
            "parameters": ("Real", len(self.surface_area_parameters)),
        }

    def new_instance(self, time_step, location, geometry, *args):
        1/0
        return {
            "state": initial_state,
            "parameters": parameter___todo * geometry.surface_areas[location],
        }

    def advance(self, locations, instances, time_step, reaction_inputs, reaction_outputs):
        self.scope_stack.push({})
        breakpoint_ = self.lookup(ANT.BREAKPOINT_BLOCK)
        assert(len(breakpoint_) == 1)
        breakpoint_[0].accept(self)
        # Write any output variables which have not yet been written to.
        for s in self.states:
            if not self.is_assigned(s):
                self += "__inst.state." + s + " = next_" + s + ";\n"
        # Check for parameters with the same name as the conductance.
        for g in self.write_conductances:
            if not self.is_assigned(g):
                self += self.write_conductances[g] + " += __inst." + g + ";\n"
        for g in self.nonspecific_conductances:
            if not self.is_assigned(g):
                for specific_g, scalar in self.nonspecific_conductances[g]:
                    self += specific_g + " += " + str(float(scalar)) + " * __inst." + g + ";\n"
        self.scope_stack.pop()
        self += "}\n"
        self.indent -= 1
        self += "}\n\n"

    def visit_solve_block(self, node):
        name = node.block_name.get_node_name()
        self += name + "(__inst, __r_in);\n"

    def visit_derivative_block(self, node):
        self += "fn " + str(node.name.get_node_name())
        self += "(__inst: &mut Instance, __r_in: &ReactionInputs) {\n"
        node.statement_block.accept(self)
        for s in self.states:
            self.assigned[0].append(s)
            self += "__inst.state." + s + " = next_" + s + ";\n"
        for (variables, value) in self.conserve:
            self += "\n"
            variables = ["__inst.state." + v for v in variables]
            self += "let __correction: f64 = (%s) / (%s);\n"%(value, " + ".join(variables))
            for v in variables:
                self += v + " *= __correction;\n"
        self += "}\n\n"

    def visit_diff_eq_expression(self, node):
        expr = node.expression
        assert(isinstance(expr, ast.BinaryExpression) and expr.op.eval() == "=")
        lhsn = expr.lhs.name.get_node_name()
        assert(lhsn in self.states)
        rhs = expr.rhs
        self += "let next_%s = "%lhsn
        rhs.accept(self)
        self += ";\n"

    def visit_local_var(self, node):
        name = node.name.get_node_name()
        if not self.is_assigned(name):
            self.assigned[-1].append(name)
            self += "let mut %s: f64;\n"%name

    def visit_expression_statement(self, node):
        expr = node.expression
        if isinstance(expr, ast.BinaryExpression) and expr.op.eval() == "=":
            #
            # Visit Assignment Statement
            #
            rhs = expr.rhs
            lhsn = expr.lhs.name.get_node_name()
            if lhsn == getattr(self, "func_name", object()):
                self += "return "
                rhs.accept(self)
                self += ";\n"
            # Do not write any currents.
            elif lhsn in self.write_currents:
                pass
            # Accumulate conductances.
            elif lhsn in self.write_conductances:
                self.assigned[0].append(lhsn)
                self += self.write_conductances[lhsn]
                self += " += "
                rhs.accept(self)
                self += ";\n"
            elif lhsn in self.nonspecific_conductances:
                self.assigned[0].append(lhsn)
                for specific_g, scalar in self.nonspecific_conductances[lhsn]:
                    self += specific_g + " += " + str(float(scalar)) + " * "
                    rhs.accept(self)
                    self += ";\n"
            elif lhsn in self.states:
                self.assigned[-1].append("next_" + lhsn)
                self += "let next_" + lhsn + " = "
                rhs.accept(self)
                self += ";\n"
            # First assignment to variable in this scope.
            elif not self.is_assigned(lhsn):
                self.assigned[-1].append(lhsn)
                self += "let mut "
                node.visit_children(self)
                self += ";\n"
            else:
                node.visit_children(self)
                self += ";\n"
        else:
            node.visit_children(self)
            self += ";\n"

    def visit_var_name(self, node):
        name = node.name.get_node_name()
        if name in self.states:
            self += "__inst.state." + name
        elif name == "v":
            self += "__r_in.voltages[__inst.location]"
        elif name in self.surface_area_parameters:
            self += "__inst." + name
        elif name in self.species:
            x = self.species[name]
            if x.is_extracellular() and not x.is_intracellular():
                self += "__r_in." + name + "_extra[__inst.location]"
            else:
                1/0 # Unimplemented.
        else:
            self += name
        self += " "

    def visit_statement_block(self, node):
        self.indent += 1
        node.visit_children(self)
        self.indent -= 1

    def visit_function_block(self, node):
        name = node.name.get_node_name()
        if name not in self.functions_called:
            return
        self += "pub fn " + name + "("
        if name in self.solve_blocks:
            self += "__inst: &mut Instance, __r_in: &ReactionInputs, "
        for p in node.parameters:
            p.accept(self)
            self += ": f64, "
        self += ") -> f64 {\n"
        self.assigned.append([])
        node.statement_block.accept(self)
        self.assigned.pop()
        self += "    return 0.0;\n"
        self += "}\n\n"

    def is_assigned(self, name):
        return any(name in scope for scope in self.assigned)

    def visit_binary_expression(self, node):
        op = node.op.eval()
        if op == "^":
            self += "f64::powf("
            node.lhs.accept(self)
            self += ", "
            node.rhs.accept(self)
            self += ")"
        else:
            node.lhs.accept(self)
            self += f"{op} "
            node.rhs.accept(self)

    def visit_function_call(self, node):
        name = node.name.get_node_name()
        functions = {
            # NMODL name -> Rust name
            "exp": "f64::exp",
            "fabs": "f64::abs",
            "pow": "f64::powf",
        }
        if name in functions:
            self += functions[name]
        else:
            self += name
        self += "("
        for a in node.arguments:
            a.accept(self)
            self += ", "
        self += ") "

    def visit_name(self, node):
        self += node.get_node_name() + " "

    def visit_if_statement(self, node):
        self += "if "
        node.condition.accept(self)
        self += " {\n"
        self.assigned.append([])
        node.get_statement_block().accept(self)
        self.assigned.pop()
        self += "}\n"
        for n in node.elseifs:
            n.accept(self)
        if node.elses:
            node.elses.accept(self)

    def visit_else_if_statement(self, node):
        self += "else if {\n"
        self.assigned.append([])
        node.get_statement_block().accept(self)
        self.assigned.pop()
        self += "}\n"

    def visit_else_statement(self, node):
        self += "else {\n"
        self.assigned.append([])
        node.get_statement_block().accept(self)
        self.assigned.pop()
        self += "}\n"

    def visit_paren_expression(self, node):
        self += "("
        node.visit_children(self)
        self += ") "

    def visit_wrapped_expression(self, node):
        node.visit_children(self)

    def visit_binary_operator(self, node):
        self += node.eval() + " "

    def visit_unary_operator(self, node):
        self += node.eval()

    def visit_integer(self, node):
        self += str(int(node.eval())) + " "

    def visit_double(self, node):
        self += str(float(node.eval())) + "_f64 "

    def visit_conserve(self, node):
        pass

    def visit_conductance_hint(self, node):
        pass

    def visit_table_statement(self, node):
        pass

if __name__ == "__main__":
    # print(dsl.list_examples())
    hh = dsl.load_example('hh.mod')
    self = Mechanism(hh)
    print(self.text)

from collections.abc import Callable, Iterable, Mapping
from neuwon.database.database import DB_Class, DB_Object
from neuwon.database.data_components import ClassAttribute, Attribute, SparseMatrix
from neuwon.database.doc import Documentation
from neuwon.database.dtypes import *
import ast
import copy
import inspect
import io
import numba
import re
import uncompyle6

# uncompyle6.code_deparse(compile(ast_node, "filename", mode='exec'))

# TODO: Rename this file to something other than the builtin's name "callable"
#       Maybe function.py

class Function(Documentation):
    def __init__(self, function):
        assert isinstance(function, Callable)
        if isinstance(function, Function):
            function = function.function
        Documentation.__init__(self, function.__name__, inspect.getdoc(function))
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def _disassemble_function(self):
        self.filename  = inspect.getsourcefile(self.function)
        self.signature = inspect.signature(self.function)
        self.body_text = io.StringIO()
        uncompyle6.code_deparse(self.function.__code__, out=self.body_text)
        self.body_text = self.body_text.getvalue()
        self.body_ast  = ast.parse(self.body_text, self.filename)
        self.globals   = self.function.__globals__

class Method(Function):
    def __init__(self, function):
        Function.__init__(self, function)
        self._disassemble_function()
        self.self_variable = next(iter(self.signature.parameters))
        self.db_class = None
        self.is_jited = False

    def _register_method(self, db_class, add_attr=True):
        assert isinstance(db_class, DB_Class)
        assert self.db_class is None
        self.db_class = db_class
        if add_attr:
            assert self.name not in self.db_class.components
            assert self.name not in self.db_class.methods
            self.db_class.methods[self.name] = self
            setattr(self.db_class.instance_type, self.name, lambda inst=None: self.__call__(inst))

    def jit(self):
        transformer         = _OOP_to_SoA(self)
        self.function       = numba.njit(transformer.function)
        self.db_arguments   = transformer.arguments
        self.is_jited       = True

    def __call__(self, instances=None, *args, **kwargs):
        """
        Argument instances is one of:
                * A single instance,
                * A range of instances,
                * An iterable of instances (or their unstable indexes),
                * None, in which case this method is called on all instances.
        """
        assert self.db_class is not None
        if not self.is_jited: self.jit()
        db_args = [self.db_class.get_data(x) for x in self.db_arguments]
        if instances is None:
            instances = range(0, len(self.db_class))
        if isinstance(instances, range):
            for idx in instances:
                self.function(idx, *db_args, *args, **kwargs)
        elif isinstance(instances, Iterable):
            1/0
        else:
            assert isinstance(instances, self.db_class.instance_type)
            return self.function(instances._idx, *db_args, *args, **kwargs)


class _OOP_to_SoA(ast.NodeTransformer):
    def __init__(self, method):
        ast.NodeTransformer.__init__(self)
        self.method     = method
        self.loads      = set()
        self.stores     = set()
        self.body_ast   = self.visit(method.body_ast)
        self.loads      = sorted(self.loads, key=lambda x: x.get_name())
        self.stores     = sorted(self.stores, key=lambda x: x.get_name())
        self.arguments  = sorted(set(self.loads + self.stores), key=lambda x: x.get_name())
        self.prepend_loads()
        self.append_stores()
        self.assemble_function()

    def local_name(self, attribute):
        return f"{attribute.get_class().get_name()}_{attribute.get_name()}"

    def arg_name(self, attribute):
        return f"{attribute.get_class().get_name()}_{attribute.get_name()}_array"

    def visit_Attribute(self, node):
        # Visit the syntax: "value.attr"
        value   = node.value
        ctx     = node.ctx
        access  = [node.attr]
        while isinstance(value, ast.Attribute):
            access.append(value.attr)
            value = value.value
        access = tuple(reversed(access))
        if isinstance(value, ast.Name):
            if value.id != self.method.self_variable:
                return node
        else: raise NotImplementedError(type(value))
        # Get all of the data components for this access.
        components  = []
        db_class    = self.method.db_class
        for ptr in access:
            components.append(db_class.get(ptr))
            db_class = components[-1].reference
        # Always load the data, regardless of the ctx flag, because augmenting
        # assignment is labeled as just a store when in fact it is both a load
        # and a store. The compiler should optimize out the unused loads.
        self.loads.update(components)
        if isinstance(ctx, ast.Store):
            self.stores.update(components)
        if isinstance(ctx, ast.Del):
            raise TypeError("Can not 'del' database attributes.")
        # Replace instance attribute access with a local variable.
        new_node = ast.Name(id=self.local_name(components[-1]), ctx=ctx)
        return ast.copy_location(new_node, node)

    def prepend_loads(self):
        load_stmts = []
        for attr in self.loads:
            if isinstance(attr, ClassAttribute):
                load_stmts.append(f"{self.local_name(attr)} = {self.arg_name(attr)}")
            elif isinstance(attr, Attribute):
                load_stmts.append(f"{self.local_name(attr)} = {self.arg_name(attr)}[_idx]")
            else: raise NotImplementedError(type(attr))
        load_ast = ast.parse("\n".join(load_stmts))
        self.body_ast.body = load_ast.body + self.body_ast.body

    def append_stores(self):
        store_stmts = []
        for attr in self.stores:
            store_stmts.append(f"{self.arg_name(attr)}[_idx] = {self.local_name(attr)}")
        store_ast = ast.parse("\n".join(store_stmts))
        self.body_ast.body.extend(store_ast.body)

    def assemble_function(self):
        arguments = ['_idx']
        arguments.extend(self.arg_name(x) for x in self.arguments)
        signature = re.subn(rf'\b{self.method.self_variable}\b',
                            ', '.join(arguments),
                            str(self.method.signature), count = 1)[0]

        template            = f"def {self.method.name}{signature}:\n pass\n"
        module_ast          = ast.parse(template)
        function_ast        = module_ast.body[0]
        function_ast.body   = self.body_ast.body
        module_scope        = self.method.globals
        exec(compile(module_ast, self.method.filename, mode='exec'), module_scope)
        self.function       = module_scope[self.method.name]

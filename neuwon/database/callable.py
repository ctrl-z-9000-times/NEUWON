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
        self.first_call = True

    def _register_method(self, db_class, add_attr=True):
        assert isinstance(db_class, DB_Class)
        assert self.db_class is None
        self.db_class = db_class
        self.qualname = f'{self.db_class.name}.{self.name}'
        if add_attr:
            assert self.name not in self.db_class.components
            assert self.name not in self.db_class.methods
            self.db_class.methods[self.name] = self
            setattr(self.db_class.instance_type, self.name, lambda inst=None: self.__call__(inst))

    def __call__(self, instances=None, *args, **kwargs):
        """
        Argument instances is one of:
                * A single instance,
                * A range of instances,
                * An iterable of instances (or their unstable indexes),
                * None, in which case this method is called on all instances.
        """
        assert self.db_class is not None
        if self.first_call: self._jit()
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

    def _jit(self):
        transformer         = _OOP_to_SoA(self)
        self.function       = numba.njit(transformer.function)
        self.db_arguments   = transformer.arguments
        self.first_call     = False


class _OOP_to_SoA(ast.NodeTransformer):
    def __init__(self, method):
        ast.NodeTransformer.__init__(self)
        assert isinstance(method, Method)
        self.method         = method
        self.method_calls   = set()
        self.loads          = set()
        self.stores         = set()
        self.body_ast       = self.visit(method.body_ast)
        self.inline_methods()
        self.loads          = sorted(self.loads, key=lambda x: x.get_name())
        self.stores         = sorted(self.stores, key=lambda x: x.get_name())
        self.arguments      = sorted(set(self.loads + self.stores), key=lambda x: x.get_name())
        self.prepend_loads()
        self.append_stores()
        self.assemble_function()
        self.compile_function()

    def local_name(self, attribute):
        return attribute.qualname.replace('.', '_')

    def global_name(self, attribute):
        if isinstance(attribute, ClassAttribute):
            return self.local_name(attribute)
        elif isinstance(attribute, Attribute):
            return f"{self.local_name(attribute)}_array"
        else: raise NotImplementedError(type(attribute))

    def visit_Attribute(self, node):
        # Visit the syntax: "value.attr"
        value   = node.value
        ctx     = node.ctx
        # Make a list of all names in the chain of attributes.
        access  = [node.attr]
        while isinstance(value, ast.Attribute):
            access.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            access.append(value.id)
        else: raise NotImplementedError(type(value))
        access = tuple(reversed(access))
        # 
        if access[0] != self.method.self_variable:
            return node
        if isinstance(ctx, ast.Del):
            raise TypeError("Can not 'del' database attributes.")
        # Get all of the database components for this access.
        db_class = self.method.db_class
        for ptr in access[1:]:
            component = db_class.get(ptr)
            if isinstance(ctx, ast.Store):
                self.stores.add(component)
            if isinstance(component, Method):
                self.method_calls.add(component)
            else:
                # Always load the data, regardless of the ctx flag, because
                # augmenting assignment is labeled as a store but implicitly
                # requires a load. The compiler should optimize out the unused
                # loads.
                self.loads.add(component)
            db_class = getattr(component, 'reference', False)
        # Replace instance attribute access with a local variable.
        new_node = ast.Name(id=self.local_name(component), ctx=ctx)
        return ast.copy_location(new_node, node)

    def inline_methods(self):
        for method in self.method_calls:
            method = _MethodInMethod(method)
            self.stores.update(method.stores)
            self.loads.update(method.loads)
            self.body_ast.body.insert(0, method.function_ast)

    def prepend_loads(self):
        load_stmts = []
        for attr in self.loads:
            if isinstance(attr, ClassAttribute):
                pass # local_name == global_name
            elif isinstance(attr, Attribute):
                load_stmts.append(f"{self.local_name(attr)} = {self.global_name(attr)}[_idx]")
            else: raise NotImplementedError(type(attr))
        load_ast = ast.parse("\n".join(load_stmts), filename=self.method.filename)
        self.body_ast.body = load_ast.body + self.body_ast.body

    def append_stores(self):
        store_stmts = []
        for attr in self.stores:
            if isinstance(attr, Attribute):
                store_stmts.append(f"{self.global_name(attr)}[_idx] = {self.local_name(attr)}")
            else:
                raise TypeError(f"Can not assign to '{attr.qualname}' in this context.")
        store_ast = ast.parse("\n".join(store_stmts), filename=self.method.filename)
        self.body_ast.body.extend(store_ast.body)

    def get_arguments(self):
        arguments = ['_idx']
        arguments.extend(self.global_name(x) for x in self.arguments)
        return arguments

    def assemble_function(self):
        signature = re.subn(rf'\b{self.method.self_variable}\b',
                            ', '.join(self.get_arguments()),
                            str(self.method.signature), count = 1)[0]
        template                = f"def {self.method.name}{signature}:\n pass\n"
        module_ast              = ast.parse(template, filename=self.method.filename)
        self.function_ast       = module_ast.body[0]
        self.function_ast.body  = self.body_ast.body
        self.module_scope       = self.method.globals
        self.module_ast         = ast.fix_missing_locations(module_ast)

    def compile_function(self):
        print(self.function_ast.body)
        exec(compile(self.module_ast, self.method.filename, mode='exec'), self.module_scope)
        self.function = self.module_scope[self.method.name]


class _MethodInMethod(_OOP_to_SoA):

    def prepend_loads(self):
        # Declare variables as non-local's instead of reading from an array.
        load_stmts = []
        for attr in self.loads:
            load_stmts.append(f"nonlocal {self.local_name(attr)}")
        load_ast = ast.parse("\n".join(load_stmts), filename=self.method.filename)
        self.body_ast.body = load_ast.body + self.body_ast.body

    def append_stores(self):
        pass

    def get_arguments(self):
        return []

    def assemble_function(self):
        super().assemble_function()
        self.function_ast.name = self.local_name(self.method)

    def compile_function(self):
        pass

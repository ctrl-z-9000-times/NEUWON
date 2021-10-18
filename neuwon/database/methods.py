from collections.abc import Callable, Iterable, Mapping
from neuwon.database.database import DB_Class, DB_Object
from neuwon.database.data_components import ClassAttribute, Attribute, SparseMatrix
from neuwon.database.doc import Documentation
from neuwon.database.dtypes import *
from neuwon.database.memory_spaces import host
import ast
import copy
import inspect
import io
import numba
import re
import uncompyle6

# uncompyle6.code_deparse(compile(ast_node, "filename", mode='exec'))

# import astor
# print(type(self), "Compiling AST ...")
# print(astor.to_source(self.module_ast))

class Function(Documentation):
    def __init__(self, function):
        assert isinstance(function, Callable)
        if isinstance(function, Function): function = function.original
        Documentation.__init__(self, function.__name__, inspect.getdoc(function))
        self.original   = function
        self.host_jit   = None
        self.jit_cache  = {}

    def __call__(self, *args, **kwargs):
        if self.host_jit is None:
            self.host_jit = self._jit(host)
        return self.host_jit(*args, **kwargs)

    def _jit(self, target):
        cached = self.jit_cache.get(target, None)
        if cached is not None: return cached
        if isinstance(self, Method):
            jit_method          = _JIT_Method(self.db_class, self, target)
            function            = jit_method.function
            self.db_arguments   = jit_method.arguments
        elif isinstance(self, Function):
            function = _JIT_Function(self.original, target).function
        else: raise NotImplementedError(type(self))
        self.jit_cache[target] = function
        return function

class Method(Function):
    def __init__(self, function):
        Function.__init__(self, function)
        self.db_class = None

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
            # TODO: This should use a property so that it can show documentation!
            #       And maybe then I wont need the lambda?

    def __call__(self, instances=None, *args, **kwargs):
        """
        Argument instances is one of:
                * A single instance,
                * A range of instances,
                * An iterable of instances (or their unstable indexes),
                * None, in which case this method is called on all instances.
        """
        assert self.db_class is not None
        target = self.db_class.database.memory_space
        function = self._jit(target)

        db_args = [self.db_class.get_data(x) for x in self.db_arguments]
        if instances is None:
            instances = range(0, len(self.db_class))
        if isinstance(instances, range):
            for idx in instances:
                function(idx, *db_args, *args, **kwargs)
        elif isinstance(instances, Iterable):
            1/0
        else:
            assert isinstance(instances, self.db_class.instance_type)
            return function(instances._idx, *db_args, *args, **kwargs)

class _JIT_Function:
    """ Breakout a function into all of its constituent parts. """
    def __init__(self, function, target):
        assert isinstance(function, Callable)
        assert not isinstance(function, Function)
        self.original  = function
        self.target    = target
        self.name      = self.original.__name__.replace('<', '_').replace('>', '_')
        self.filename  = inspect.getsourcefile(self.original)
        self.signature = inspect.signature(self.original)
        self.body_text = io.StringIO()
        uncompyle6.code_deparse(self.original.__code__, out=self.body_text)
        self.body_text = self.body_text.getvalue()
        self.body_ast  = ast.parse(self.body_text, self.filename)
        nonlocals, globals, builtins, unbound = inspect.getclosurevars(self.original)
        self.closure = {}
        self.closure.update(builtins)
        self.closure.update(globals)
        self.closure.update(nonlocals)

        self.assemble_function()
        self.function = target.jit_wrapper(self.py_function)

    def jit_closure(self):
        closure = dict(self.closure)
        for name, value in closure.items():
            if isinstance(value, Function):
                closure[name] = _JIT_Function(value.original, self.target).function
        return closure

    def assemble_function(self):
        template                = f"def {self.name}{self.signature}:\n pass\n"
        module_ast              = ast.parse(template, filename=self.filename)
        self.function_ast       = module_ast.body[0]
        self.function_ast.body  = self.body_ast.body
        self.module_ast         = ast.fix_missing_locations(module_ast)
        closure                 = self.jit_closure()
        exec(compile(self.module_ast, self.filename, mode='exec'), closure)
        self.py_function        = closure[self.name]

class _JIT_Method(_JIT_Function, ast.NodeTransformer):
    def __init__(self, db_class, method, target):
        _JIT_Function.__init__(self, method.original, target)
        ast.NodeTransformer.__init__(self)
        self.db_class       = db_class
        self.method         = method
        self.self_var       = next(iter(self.signature.parameters))
        self.method_calls   = set()
        self.loads          = set()
        self.stores         = set()
        self.body_ast       = self.visit(copy.deepcopy(self.body_ast))
        self.inline_methods()
        self.loads          = sorted(self.loads, key=lambda x: x.get_name())
        self.stores         = sorted(self.stores, key=lambda x: x.get_name())
        self.arguments      = sorted(set(self.loads + self.stores), key=lambda x: x.get_name())
        self.prepend_loads()
        self.append_stores()
        self.assemble_method()
        self.compile_method()
        self.function = self.target.jit_wrapper(self.py_function)

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
        if access[0] != self.self_var:
            return node
        if isinstance(ctx, ast.Del):
            raise TypeError("Can not 'del' database attributes.")
        # Get all of the database components for this access.
        db_class = self.db_class
        for ptr in access[1:]:
            try:
                component = db_class.get(ptr)
            except AttributeError as err:
                if hasattr(db_class.get_instance_type(), ptr):
                    raise AttributeError('Is that method missing its "@Method" decorator?') from err
                else:
                    raise err
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
            method = _MethodInMethod(self.db_class, method, self.target)
            self.stores.update(method.stores)
            self.loads.update(method.loads)
            self.body_ast.body.insert(0, method.function_ast)
            # TODO: Include the method's closure into this one.

    def prepend_loads(self):
        load_stmts = []
        for attr in self.loads:
            if isinstance(attr, ClassAttribute):
                pass # local_name == global_name
            elif isinstance(attr, Attribute):
                load_stmts.append(f"{self.local_name(attr)} = {self.global_name(attr)}[_idx]")
            else: raise NotImplementedError(type(attr))
            if attr.reference:
                var = self.local_name(attr)
                load_stmts.append(f'if {var} == {NULL}: {var} = None')
        load_ast = ast.parse("\n".join(load_stmts), filename=self.filename)
        self.body_ast.body = load_ast.body + self.body_ast.body

    def append_stores(self):
        # TODO: Also insert the stores infront of any explicit return stmt's.
        store_stmts = []
        for attr in self.stores:
            if isinstance(attr, Attribute):
                store_stmts.append(f"{self.global_name(attr)}[_idx] = {self.local_name(attr)}")
            else:
                raise TypeError(f"Can not assign to '{attr.qualname}' in this context.")
        store_ast = ast.parse("\n".join(store_stmts), filename=self.filename)
        self.body_ast.body.extend(store_ast.body)

    def get_arguments(self):
        arguments = ['_idx']
        arguments.extend(self.global_name(x) for x in self.arguments)
        return arguments

    def assemble_method(self):
        signature = re.subn(rf'\b{self.self_var}\b',
                            ', '.join(self.get_arguments()),
                            str(self.signature), count = 1)[0]
        template                = f"def {self.name}{signature}:\n pass\n"
        module_ast              = ast.parse(template, filename=self.filename)
        self.function_ast       = module_ast.body[0]
        self.function_ast.body  = self.body_ast.body
        self.module_ast         = ast.fix_missing_locations(module_ast)

    def compile_method(self):
        closure = self.jit_closure()
        exec(compile(self.module_ast, self.filename, mode='exec'), closure)
        self.py_function = closure[self.name]

class _MethodInMethod(_JIT_Method):
    """
    This class is used to inline methods by nesting them like:
    >>> def foo(_idx, ...):
    >>>     self_x = ...
    >>>     def bar():
    >>>         nonlocal self_x
    >>>         self_x += 1
    >>>     bar()
    """
    def prepend_loads(self):
        # Declare variables as non-local's instead of reading from an array.
        load_stmts = []
        for attr in self.loads:
            load_stmts.append(f"nonlocal {self.local_name(attr)}")
        load_ast = ast.parse("\n".join(load_stmts), filename=self.filename)
        self.body_ast.body = load_ast.body + self.body_ast.body

    def append_stores(self):
        pass

    def get_arguments(self):
        return []

    def assemble_method(self):
        super().assemble_method()
        self.function_ast.name = self.local_name(self.method)

    def compile_method(self):
        pass

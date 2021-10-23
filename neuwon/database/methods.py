from collections.abc import Callable, Iterable, Mapping
from neuwon.database.database import DB_Class, DB_Object
from neuwon.database.data_components import ClassAttribute, Attribute, SparseMatrix
from neuwon.database.doc import Documentation
from neuwon.database.dtypes import *
from neuwon.database.memory_spaces import host
import ast
import inspect
import io
import itertools
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
            jit_method          = _JIT_Method(self.db_class, self.original, target)
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
        assert self.db_class is None, 'This method is already registered with a DB_Class!'
        self.db_class = db_class
        self.qualname = f'{self.db_class.name}.{self.name}'
        if add_attr:
            assert self.name not in self.db_class.components
            assert self.name not in self.db_class.methods
            self.db_class.methods[self.name] = self
            _getter = lambda instance=None, *args, **kwargs: self.__call__(instance, *args, **kwargs)
            _getter.__name__ = self.name
            _getter.__doc__ = self.doc
            setattr(self.db_class.instance_type, self.name, _getter)

    def __call__(self, instance=None, *args, **kwargs):
        """
        Argument instance can be:
                * A single instance,
                * An iterable of instances (or their unstable indexes),
                * None, in which case this method is called on all instances.

        All additional arguments are passed through to the users method.
        """
        assert self.db_class is not None
        target = self.db_class.database.memory_space
        function = self._jit(target)
        db_args = [self.db_class.get_data(x) for x in self.db_arguments]
        if isinstance(instance, self.db_class.instance_type):
            return function(instance._idx, *db_args, *args, **kwargs)
        if instance is None:
            instance = range(0, len(self.db_class))
        return [function(idx, *db_args, *args, **kwargs) for idx in instance]

class _JIT_Function:
    def __init__(self, function, target):
        assert isinstance(function, Callable)
        assert not isinstance(function, Function)
        self.original  = function
        self.target    = target
        # Breakout the function into all of its constituent parts.
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
        # 
        self.assemble_function()
        self.function = target.jit_wrapper(self.py_function)

    def jit_closure(self):
        """
        Returns a copy of this function's closure with all captured Functions
        replaced by their JIT'ed versions.
        """
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

################################################################################

class _ReferenceRewriter(ast.NodeTransformer):
    def __init__(self, db_class, reference_name, body_ast, top_level=True):
        ast.NodeTransformer.__init__(self)
        self.db_class       = db_class
        self.reference_name = str(reference_name)
        self.body_ast       = body_ast
        self.loads          = []
        self.stores         = []
        self.methods        = set()
        self.db_arguments   = set()
        self.body_ast       = self.visit(self.body_ast)

        # Find all references which are exposed via this class and recursively rewrite them.
        for db_attr in itertools.chain(self.loads, self.stores):
            if db_attr.reference:
                rr = _ReferenceRewriter(db_attr.reference, self.local_name(db_attr),
                                        self.body_ast, top_level=False)
            self.body_ast = rr.body_ast
            self.loads.update(rr.loads)
            self.stores.update(rr.stores)
            self.methods.update(rr.methods)

        self.db_arguments = sorted(set(self.global_name(db_attribute)
                            for db_attribute in itertools.chain(self.loads, self.stores)))

        if top_level:
            self.prepend_loads()
            self.append_stores()

    def local_name(self, db_attribute):
        if (isinstance(db_attribute, ClassAttribute) or
            isinstance(db_attribute, Method)):
            return self.global_name(db_attribute)
        else:
            return f'{self.reference_name}_{db_attribute.name}'

    def global_name(self, db_attribute):
        return '_%s_' % db_attribute.qualname.replace('.', '_')

    def visit_Attribute(self, node):
        # Visit the syntax: "value.attr"
        value = node.value
        if not isinstance(value, ast.Name): return node
        value = value.id
        if value != self.reference_name: return node
        ctx = node.ctx
        if isinstance(ctx, ast.Del):
            raise TypeError("Can not 'del' database attributes.")
        # Get the database component for this access.
        try:
            db_attribute = self.db_class.get(node.attr)
        except AttributeError as err:
            if hasattr(self.db_class.get_instance_type(), node.attr):
                raise AttributeError('Is that method missing its "@Method" decorator?') from err
            else:
                raise err
        # 
        if isinstance(ctx, ast.Store):
            self.stores.add(db_attribute)
        if isinstance(db_attribute, Method):
            self.method_calls.add(db_attribute)
        else:
            # Always load the data, regardless of the ctx flag, because
            # augmenting assignment is labeled as a store but implicitly
            # requires a load. The compiler should optimize out the unused
            # loads.
            self.loads.add(db_attribute)
        # Replace this attribute access with a local variable.
        new_name = self.local_name(db_attribute)
        return ast.copy_location(ast.Name(id=new_name, ctx=ctx), node)

    def visit_Call(self, node):
        """
        Replace: pointer.method(*args, **kwargs)
        With:    global_name(method)(local_name(pointer), *args, **kwargs)
        """
        # TODO!!!

        return node

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
        store_stmts = []
        for attr in self.stores:
            if isinstance(attr, Attribute):
                store_stmts.append(f"{self.global_name(attr)}[_idx] = {self.local_name(attr)}")
            else:
                raise TypeError(f"Can not assign to '{attr.qualname}' in this context.")
        store_ast = ast.parse("\n".join(store_stmts), filename=self.filename)
        # Insert the stores in front of any explicit return statements.
        for node in ast.walk(self.body_ast):
            body = getattr(node, 'body', [])
            for idx, return_stmt in enumerate(body):
                if not isinstance(return_stmt, ast.Return): continue
                for store_stmt in reversed(store_ast.body):
                    body.insert(idx, store_stmt)
                break
        # Append the stores to the end of the method.
        self.body_ast.body.extend(store_ast.body)


################################################################################

class _JIT_Method(_JIT_Function):
    def __init__(self, db_class, function, target):
        _JIT_Function.__init__(self, function, target)
        self.db_class   = db_class
        self.self_var   = next(iter(self.signature.parameters))
        rr = _ReferenceRewriter(self.db_class, self.self_var, self.body_ast)

        for method in rr.methods:
            self.closure[method.name] = method._jit(target)

        # TODO: Rename this to 'db_arguments' for consistancy with the rest of the module.
        self.arguments = sorted(set(self.loads + self.stores), key=lambda x: x.get_name())

        self.assemble_method()
        self.compile_method()
        self.function = self.target.jit_wrapper(self.py_function)

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

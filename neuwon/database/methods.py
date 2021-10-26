from collections.abc import Callable, Iterable, Mapping
from neuwon.database.database import DB_Class, DB_Object
from neuwon.database.data_components import ClassAttribute, Attribute, SparseMatrix
from neuwon.database.doc import Documentation
from neuwon.database.dtypes import *
from neuwon.database.memory_spaces import host
import ast
import collections
import inspect
import io
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
        self._jit_cache = {}

    def __call__(self, *args, **kwargs):
        return self._jit(host)(*args, **kwargs)

    def _jit(self, target):
        cached = self._jit_cache.get(target, None)
        if cached is not None: return cached
        if isinstance(self, Method):
            assert self.db_class is not None, 'Method not registered!'
            jit_data            = _JIT(self.original, target, self.db_class)
            function            = jit_data.function
            self.db_arguments   = jit_data.db_arguments
        elif isinstance(self, Function):
            function = _JIT(self.original, target).function
        else: raise NotImplementedError(type(self))
        self._jit_cache[target] = function
        return function

class Method(Function):
    def __init__(self, function):
        Function.__init__(self, function)
        self.db_class = None

    def _register_method(self, db_class, add_attr=True):
        assert isinstance(db_class, DB_Class)
        assert self.db_class is None, 'This method is already registered with a DB_Class!'
        self.db_class = db_class
        self.db = self.db_class.get_database()
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
        db_args = [db_attr.get_data() for _, db_attr in self.db_arguments]
        if isinstance(instance, self.db_class.instance_type):
            return function(instance._idx, *db_args, *args, **kwargs)
        if instance is None:
            instance = range(0, len(self.db_class))
        return [function(idx, *db_args, *args, **kwargs) for idx in instance]

class _JIT:
    def __init__(self, function, target, method_db_class=None):
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
        # Replace all Functions captured in this closure with their JIT'ed versions.
        for name, value in self.closure.items():
            if isinstance(value, Function):
                self.closure[name] = _JIT(value.original, self.target).function
        # Transform and then reassemble the function.
        if method_db_class: self._rewrite_method(method_db_class)
        self.assemble_function()
        self.function = target.jit_wrapper(self.py_function)

    def _rewrite_method(self, db_class):
        self_var = next(iter(self.signature.parameters))
        rr = _ReferenceRewriter(db_class, self_var, self.body_ast)
        self.db_arguments = sorted(rr.db_arguments.items(), key=lambda pair: pair[1].qualname)
        parameters = list(self.signature.parameters.values())
        for arg_name, db_attr in reversed(self.db_arguments):
            parameters.insert(1, inspect.Parameter(arg_name, inspect.Parameter.POSITIONAL_OR_KEYWORD))
        self.signature = self.signature.replace(parameters=parameters)
        self.body_ast = rr.body_ast
        for name, method in rr.method_calls.items():
            self.closure[name] = method._jit(self.target)

    def assemble_function(self):
        template                = f"def {self.name}{self.signature}:\n pass\n"
        module_ast              = ast.parse(template, filename=self.filename)
        self.function_ast       = module_ast.body[0]
        self.function_ast.body  = self.body_ast.body
        self.module_ast         = ast.fix_missing_locations(module_ast)
        exec(compile(self.module_ast, self.filename, mode='exec'), self.closure)
        self.py_function        = self.closure[self.name]

class _ReferenceRewriter(ast.NodeTransformer):
    def __init__(self, db_class, reference_name, body_ast):
        ast.NodeTransformer.__init__(self)
        self.db_class       = db_class
        self.reference_name = str(reference_name)
        self.db_arguments   = {} # Name -> DB-Attribute
        self.method_calls   = {} # Name -> Method-Instance
        self.body_ast       = self.visit(body_ast)
        # Find all references which are exposed via this class and recursively rewrite them.
        for name, db_attribute in list(self.db_arguments.items()):
            if db_attribute.reference:
                rr = _ReferenceRewriter(db_attribute.reference, name, self.body_ast)
                self.db_arguments.update(rr.db_arguments)
                self.method_calls.update(rr.method_calls)
                self.body_ast = rr.body_ast

    def visit_Attribute(self, node):
        """ Visit the syntax: "value.attr" """
        # Filter for references to this db_class.
        value = node.value
        if isinstance(value, ast.Name):
            if value.id != self.reference_name:
                return node
        elif isinstance(value, ast.Attribute):
            if getattr(value, 'db_reference', False):
                if value.db_reference != self.db_class:
                    return node
            else:
                return node
        else:
            return node
        # Get the Load/Store/Del context flag.
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
        # Replace this attribute access.
        qualname = '_%s_' % db_attribute.qualname.replace('.', '_')
        if isinstance(db_attribute, Method):
            self.method_calls[qualname] = db_attribute
            replacement = ast.Name(id=qualname, ctx=ctx)
            replacement.db_method   = db_attribute
            replacement.db_instance = value
        else:
            self.db_arguments[qualname] = db_attribute
            if isinstance(db_attribute, ClassAttribute):
                replacement = ast.Name(id=qualname, ctx=ctx)
            elif isinstance(db_attribute, Attribute):
                replacement = ast.Subscript(
                                value=ast.Name(id=qualname, ctx=ast.Load()),
                                slice=ast.Index(value=value), ctx=ctx)
            else: raise NotImplementedError(type(db_attribute))
            replacement.db_reference = db_attribute.reference
        return ast.copy_location(replacement, node)

    def visit_Call(self, node):
        """
        Fix-up method calls to include 'self' and the db_arguments.
        """
        # First transform the attribute access from a method call into a global
        # function call. This also stores the instance / 'self' variable inside
        # the AST node.
        self.generic_visit(node)
        db_method = getattr(node.func, 'db_method', None)
        if db_method is None:
            return node
        node.args.insert(0, node.func.db_instance)
        # Insert the method's db_arguments.
        if not hasattr(db_method, 'db_arguments'):
            db_method._jit(host)
        self.db_arguments.update(db_method.db_arguments)
        for arg_name, db_attr in reversed(db_method.db_arguments):
            node.args.insert(1, ast.Name(id=arg_name, ctx=ast.Load()))
        return node

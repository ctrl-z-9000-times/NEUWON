from collections.abc import Callable, Iterable, Mapping
from neuwon.database.database import DB_Class, DB_Object
from neuwon.database.doc import Documentation
from neuwon.database.dtypes import *
import ast
import inspect
import io
import numba
import textwrap
import uncompyle6

class Function(Documentation):
    def __init__(self, function):
        assert isinstance(function, Callable)
        if isinstance(function, Function):
            function = function.function
        Documentation.__init__(self, function.__name__, inspect.getdoc(function))
        self.function = function

    def _inspect_function(self):
        self.filename  = inspect.getsourcefile(self.function)
        self.signature = inspect.signature(self.function)
        self.body_text = io.StringIO()
        uncompyle6.code_deparse(self.function.__code__, out=self.body_text)
        self.body_text = self.body_text.getvalue()
        self.body_ast  = ast.parse(self.body_text, self.filename)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

class Method(Function):
    def __init__(self, function):
        Function.__init__(self, function)
        self.db_class = None
        self.is_jited = False
        self.read = []
        self.write = []

        # self.host_function = numba.njit(function)
        # self.cuda_function = numba.cuda.jit(function)

        # Resolve the function's signature into real DB components. But
        # actually, do this in a mtheod, and defer doing it until called so
        # that methods can be added at any time, as long as thier strings
        # resolve into components at GO time.
        # self.arguments = 1/0
        # self.returns = 1/0

    def _register_method(self, db_class):
        assert isinstance(db_class, DB_Class)
        assert self.db_class is None
        self.db_class = db_class
        assert self.name not in self.db_class.components
        assert self.name not in self.db_class.methods
        self.db_class.methods[self.name] = self
        setattr(self.db_class.instance_type, self.name, lambda inst=None: self.__call__(inst))

    def jit(self):
        self._inspect_function()
        ast = _RemoveSelf(self).visit(self.body_ast)
        f = compile(ast, self.filename, mode='exec')
        print("")
        uncompyle6.code_deparse(f)
        print("")

    def __call__(self, instances=None, *args, **kwargs):
        """
        Argument instances is one of:
                * A single instance,
                * A range of instances,
                * An iterable of instances (or their unstable indexes),
                * None, in which case this method is called on all instances.
        """
        if not self.is_jited: self.jit()
        if instances is None:
            instances = range(0, len(self.db_class))
        if isinstance(instances, range):
            for idx in instances:
                self.function(self.db_class.index_to_object(idx), *args, **kwargs)
        elif isinstance(instances, Iterable):
            1/0
        else:
            assert isinstance(instances, self.db_class.instance_type)
            return self.function(instances, *args, **kwargs)


class _RemoveSelf(ast.NodeTransformer):
    """
    Rewrite a method's AST to not use the 'self' argument, instead using the
    database backend and an index to access all data & methods.
    """
    def __init__(self, method):
        self.method = method
        ast.NodeTransformer.__init__(self)

    def visit_Attribute(self, node):
        value = node.value
        if isinstance(value, ast.Name):
            if value.id != "self": return node
        else:
            print(value)
            1/0
        attr = node.attr
        local_name = "self_" + attr

        if node.ctx == ast.Load():
            self.method.read.add(attr)
        elif node.ctx == ast.Store():
            self.method.write.add(attr)
        elif node.ctx == ast.Del():
            raise TypeError("Can not 'del' this here.") # TODO: put lineno in error message.

        return ast.copy_location(ast.Name(id=local_name, ctx=node.ctx), node)

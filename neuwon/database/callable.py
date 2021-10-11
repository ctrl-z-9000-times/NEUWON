from collections.abc import Callable, Iterable, Mapping
from neuwon.database.database import DB_Class, DB_Object
from neuwon.database.data_components import ClassAttribute, Attribute, SparseMatrix
from neuwon.database.doc import Documentation
from neuwon.database.dtypes import *
import ast
import inspect
import io
import numba
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
        self.load = []
        self.store = []
        transform = _OOP_to_SoA(self)
        self.body_ast = transform.body_ast
        transform.loads
        transform.stores
        1/0

        f = compile(self.body_ast, self.filename, mode='exec')


        print("")
        uncompyle6.code_deparse(f)
        print("")
        1/0
        self.is_jited = True

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


class _OOP_to_SoA(ast.NodeTransformer):
    """
    Rewrite a method's AST to not use the 'self' argument, instead using the
    database backend and an index to access all data & methods.
    """
    def __init__(self, method):
        ast.NodeTransformer.__init__(self)
        self.method     = method
        self.loads      = []
        self.stores     = []
        self.body_ast   = self.visit(method.body_ast)
        self.loads      = sorted(set(self.loads))
        self.stores     = sorted(set(self.stores))
        self.loads      = [method.db_class.get(x) for x in self.loads]
        self.stores     = [method.db_class.get(x) for x in self.stores]
        self.prepend_loads()
        self.append_stores()
        self.rewrite_signature()

    def local_name(self, attribute):
        return f"{self.method.self_variable}_{attribute}"

    def arg_name(self, attribute):
        return f"{self.method.self_variable}_{attribute}_array"

    def visit_Attribute(self, node):
        value = node.value
        attr  = node.attr
        if isinstance(value, ast.Name):
            if value.id != self.method.self_variable:
                return node
        else: raise NotImplementedError(type(value))

        if   node.ctx == ast.Load():  self.loads.add(attr)
        elif node.ctx == ast.Store(): self.stores.add(attr)
        elif node.ctx == ast.Del():
            raise TypeError("Can not 'del' database attributes.")

        return ast.copy_location(ast.Name(id=self.local_name(attr), ctx=node.ctx), node)

    def prepend_loads(self):
        load_stmts = []
        for attr in self.loads:
            name = attr.get_name()
            if isinstance(attr, ClassAttribute):
                load_stmts.append(f"{self.local_name(name)} = {self.arg_name(name)}")
            else:
                load_stmts.append(f"{self.local_name(name)} = {self.arg_name(name)}[_index]")
        load_ast = ast.parse("\n".join(load_stmts))
        self.body_ast.body = load_ast.body + self.body_ast.body

    def append_stores(self):
        store_stmts = []
        for attr in self.stores:
            store_stmts.append(f"{self.arg_name(attr)}[_index] = {self.local_name(attr)}")
        store_ast = ast.parse("\n".join(store_stmts))
        self.body_ast.body.extend(store_ast.body)

    def rewrite_signature(self):
        """ fixup the parameters to include the loaded and stored data vectors. """
        self.signature = self.method.signature
        arguments = sorted(set(self.loads + self.stores))
        for attr in arguments:
            self.signature = self.signature.replace(1/0)
        1/0

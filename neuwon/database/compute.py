from collections.abc import Callable, Iterable, Mapping
from neuwon.database.database import Database, DB_Class, DB_Object
from neuwon.database.data_components import ClassAttribute, Attribute, SparseMatrix
from neuwon.database.doc import Documentation
from neuwon.database.dtypes import *
from neuwon.database.memory_spaces import host, cuda
import ast
import collections
import inspect
import io
import textwrap
import uncompyle6
import numba
import numba.cuda
import numpy as np


# TODO: CUDA!

# And also retval, and i think that these two things need to be done by the same shim?
# 
# Yes, because both need to see the instances list, as opposed to the normal JIT
# function which only sees the 'self' index.



# THought, a quick AST search could check for missing return-type annotations,
# and give a helpful error message...


# IDEAS:
# 
#   Database.add_function()
#       Register an anonymous function.
#       This allows users to define & change code at run time.
#       This will also be needed for processing the type-annotations of functions.
#       Note: this will broadcast its inputs like numpy does
#               Can broadcast function over array of pointers, for method-like behavior.
# 
#   Coordinate Kernels for sparse matrixes.
#       User writes func accepting (row, col, val) and returning the new value.
#       Alternatively, I could allow matrix access inside of the methods.
#           But that's generally a bad design, since it encourages for-loops
#           inside of the compute kernel, esp for GPUs.
# 
#   Return arrays, instead of lists.
#       Will need to determine the return value's dtype. (or accept an annotation?)
#       Replace return stmt's with a write to the array, and pass the array in to the function.
#           ** This strategy should work with both host and cuda code.
# 
#   Special case for @compute on __init__, add class method `batch_init(num, *,**)`.
# 
# TODO:
#       NULL Pointers become integers, not None. Document this somewhere?

def _print_pycode(f):
    """ Decompile and print a python function, for debugging purposes. """
    signature = inspect.signature(f)
    body_text = io.StringIO()
    uncompyle6.code_deparse(f.__code__, out=body_text)
    body_text = body_text.getvalue()
    body_text = textwrap.indent(body_text, ' '*4)
    print(f'def {f.__name__}{signature}:\n{body_text}\n')

class Compute(Documentation):
    """
    A decorator for functions and methods which are executed by the database.

    The database will apply appropriate optimizations to the given callable,
    including Just-In-Time compilation with numba.

    Functions may call other functions which are marked with this decorator.

    Internally this uses numba to compile the python code into binary machine
    code. All limitations and caveats of `numba.jit(nopython=True)` also apply
    to this decorator.

    If applied to methods then special caveats exist:
    All data access must be written as single syntactic expressions.
        -> give an example of how to access data (and how NOT to do it too)


    TODO: DOCS!
    """
    def __init__(self, function):
        if isinstance(function, Compute): function = function.original
        assert isinstance(function, Callable)
        Documentation.__init__(self, function.__name__, inspect.getdoc(function))
        self.original = function

    def __call__(self, *args, **kwargs):
        return self.original(*args, **kwargs)

    def _register_method(self, db_class, add_attr=True):
        assert isinstance(db_class, DB_Class)
        self = Compute(self)
        self.db_class = db_class
        self.qualname = f'{self.db_class.name}.{self.name}'
        self._jit_cache = {}
        if add_attr:
            assert self.name not in self.db_class.components
            assert self.name not in self.db_class.methods
            self.db_class.methods[self.name] = self
            wrapper = lambda instance=None, *args, **kwargs: self._call(instance, *args, **kwargs)
            wrapper.__name__ = self.name
            wrapper.__doc__  = self.doc
            setattr(self.db_class.instance_type, self.name, wrapper)
        return self

    def _call(self, instance, *args, **kwargs):
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
        if instance is None:
            instance = range(0, len(self.db_class))
        if isinstance(instance, self.db_class.instance_type):
            instance = instance._idx
        if isinstance(instance, int):
            instance = range(instance, instance + 1)
            single_input = True
        else:
            single_input = False

        if target is host:
            _print_pycode(function)
            instance = np.array(instance, dtype=Pointer)
            retval = function(instance, *db_args, *args, **kwargs)
            if single_input and retval is not None:
                return retval[0]
            else:
                return retval
        elif target is cuda:
            threadsperblock = 32
            blockspergrid = (len(instance) + (threadsperblock - 1)) // threadsperblock
            function[blockspergrid, threadsperblock](*db_args, *args, **kwargs)

    def _jit(self, target):
        cached = self._jit_cache.get(target, None)
        if cached is not None: return cached
        assert self.db_class is not None, "Method not registered with a 'DB_Class'!"
        jit_data = _JIT(self.original, target, self.db_class, entry_point=True)
        self._jit_cache[target] = function = jit_data.entry_point
        self.db_arguments       = jit_data.db_arguments
        return function

class _JIT:
    def __init__(self, function, target,
                db_something: 'Database or DB_Class',
                entry_point=False):
        assert isinstance(function, Callable)
        assert not isinstance(function, Compute)
        self.original  = function
        self.target    = target
        if isinstance(db_something, DB_Class):
            self.db_class = db_something
            self.database = self.db_class.get_database()
        elif isinstance(db_something, Database):
            self.db_class = None
            self.database = db_something
        self.entry_point = bool(entry_point)
        self.deconstruct_function(self.original)
        # Transform and then reassemble the function.
        self.db_arguments = {}
        if self.is_method():
            self.rewrite_method_self()
        self.rewrite_annotated_references()
        self.rewrite_functions()
        if self.target is cuda:
            self.rewrite_cuda_self()
        self.assemble_function()
        if self.entry_point:
            self.write_entry_point()

    def deconstruct_function(self, function):
        """ Breakout the function into all of its constituent parts. """
        self.name      = function.__name__.replace('<', '_').replace('>', '_')
        self.filename  = inspect.getsourcefile(function)
        self.signature = inspect.signature(function)
        self.body_text = io.StringIO()
        uncompyle6.code_deparse(function.__code__, out=self.body_text)
        self.body_text = self.body_text.getvalue()
        self.body_ast  = ast.parse(self.body_text, self.filename)
        nonlocals, globals, builtins, unbound = inspect.getclosurevars(function)
        self.closure = {}
        self.closure.update(builtins)
        self.closure.update(globals)
        self.closure.update(nonlocals)
        self.parameters = []
        parameters_iter = iter(self.signature.parameters.items())
        if self.is_method(): next(parameters_iter)
        for name, parameter in parameters_iter:
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                self.parameters.append(f'{name}={name}')
            else:
                self.parameters.append(name)
        self.parameters = ', '.join(self.parameters)

    def is_method(self):
        return self.db_class is not None

    def is_function(self):
        return not self.is_method()

    def rewrite_method_self(self):
        self.self_variable = next(iter(self.signature.parameters))
        self.rewrite_reference(self.db_class, self.self_variable)

    def rewrite_annotated_references(self):
        parameters = list(self.signature.parameters.values())
        if self.is_method():
            parameters = parameters[1:]
        for p in parameters:
            if p.annotation == inspect.Parameter.empty:
                continue
            try:
                db_class = self.database.get_class(p.annotation)
            except KeyError:
                continue
            self.rewrite_reference(db_class, p.name)

    def rewrite_reference(self, ref_class, ref_name):
        rr = _ReferenceRewriter(ref_class, ref_name, self.body_ast)
        self.body_ast = rr.body_ast
        self.db_arguments.update(rr.db_arguments)
        for name, method in rr.method_calls.items():
            jit_data = _JIT(method.original, self.target, ref_class)
            self.closure[name] = jit_data.jit_function

    def rewrite_functions(self):
        """ Replace all functions captured in this closure with their JIT'ed versions. """
        for name, value in self.closure.items():
            if not isinstance(value, Compute):
                continue
            called_func_jit     = _JIT(value.original, self.target, self.database)
            self.closure[name]  = called_func_jit.jit_function
            self.db_arguments.update(called_func_jit.db_arguments)
            # Insert this function's db_arguments into all calls to it.
            self.body_ast = _FuncCallRewriter(name, called_func_jit).visit(self.body_ast)

    def assemble_function(self):
        # Give the db_arguments a stable ordering and fixup the functions signature.
        self.db_arguments = sorted(self.db_arguments.items(),
                                    key=lambda pair: pair[1].qualname)
        parameters = list(self.signature.parameters.values())
        start = 1 if self.is_method() else 0
        for arg_name, db_attr in reversed(self.db_arguments):
            parameters.insert(start, inspect.Parameter(arg_name,
                                        inspect.Parameter.POSITIONAL_OR_KEYWORD))
        self.signature = self.signature.replace(parameters=parameters)
        # Assemble a new AST for the function.
        template                = f"def {self.name}{self.signature}:\n pass\n"
        module_ast              = ast.parse(template, filename=self.filename)
        self.function_ast       = module_ast.body[0]
        self.function_ast.body  = self.body_ast.body
        self.module_ast         = ast.fix_missing_locations(module_ast)
        exec(compile(self.module_ast, self.filename, mode='exec'), self.closure)
        self.py_function        = self.closure[self.name]
        if True: _print_pycode(self.py_function)
        # Apply JIT compilation to the function.
        if self.target is host:
            self.jit_function = numba.njit(self.py_function)
        elif self.target is host:
            self.jit_function = numba.cuda.jit(self.py_function, device=True)

    def write_entry_point(self):
        return_type = self.signature.return_annotation
        if return_type == inspect.Signature.empty:
            self.return_type = return_type = None
        else:
            self.return_type = return_type = np.dtype(return_type)
        exec_scope = {
                'function': self.jit_function,
                'return_type': return_type,
                'np': np,
                'numba': numba,
        }
        arguments = ''.join(f'{arg_name}, ' for arg_name, db_attr in self.db_arguments)
        arguments += self.parameters
        if self.target is host:
            if return_type is None:
                py_code = f'''
                @numba.njit()
                def entry_point(instances, {arguments}):
                    for index in numba.prange(len(instances)):
                        self = instances[index]
                        function(self, {arguments})
                '''
            else:
                py_code = f'''
                @numba.njit()
                def njit_entry_point(instances, return_array, {arguments}):
                    for index in numba.prange(len(instances)):
                        self = instances[index]
                        return_array[self] = function(self, {arguments})
                @numba.jit()
                def entry_point(instances, {arguments}):
                    return_array = np.zeros(len(instances), dtype=return_type)
                    njit_entry_point(instances, return_array, {arguments})
                    return return_array
                '''
        elif self.target is cuda:
            1/0
            @numba.cuda.jit()
            def _cuda_call_shim(instances, *args, **kwargs):
                index = cuda.grid(1)
                if index < instances.shape[0]:
                    self = instances[index]
                    call_inner(self, *args, **kwargs)

        exec(textwrap.dedent(py_code), exec_scope)
        self.entry_point = exec_scope['entry_point']

class _ReferenceRewriter(ast.NodeTransformer):
    def __init__(self, db_class, reference_name, body_ast):
        ast.NodeTransformer.__init__(self)
        self.db_class       = db_class
        self.reference_name = str(reference_name)
        self.db_arguments   = {} # Name -> DB-Attribute
        self.method_calls   = {} # Name -> Conpute-Instance
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
        ignore_node = False
        if isinstance(value, ast.Name):
            if value.id != self.reference_name:
                ignore_node = True
        elif isinstance(value, ast.Subscript):
            if getattr(value, 'db_reference', False):
                if value.db_reference != self.db_class:
                    ignore_node = True
            else:
                ignore_node = True
        else:
            ignore_node = True
        if ignore_node:
            self.generic_visit(node)
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
                raise AttributeError('Is that method missing its "@Compute" decorator?') from err
            else:
                raise err
        # Replace this attribute access.
        qualname = '_%s_' % db_attribute.qualname.replace('.', '_')
        if isinstance(db_attribute, Compute):
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
        # Filter for methods that the database recognizes.
        db_method = getattr(node.func, 'db_method', None)
        if db_method is None:
            return node
        # Pass the instance as the methods 'self' argument.
        node.args.insert(0, node.func.db_instance)
        # Insert the method's db_arguments.
        if not hasattr(db_method, 'db_arguments'):
            db_method._jit(host)
        self.db_arguments.update(db_method.db_arguments)
        for arg_name, db_attr in reversed(db_method.db_arguments):
            node.args.insert(1, ast.Name(id=arg_name, ctx=ast.Load()))
        # Remove these tags so that this call will not be re-processed in any
        # subsequent AST pass.
        node.func.db_method   = None
        node.func.db_instance = None
        return node

class _FuncCallRewriter(ast.NodeTransformer):
    """ Inserts a function's db_arguments into all calls to it. """
    def __init__(self, func_name, func_jit):
        ast.NodeTransformer.__init__(self)
        self.func_name = func_name
        self.func_jit  = func_jit

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == self.func_name:
            for arg_name, db_attr in reversed(self.func_jit.db_arguments):
                node.args.insert(0, ast.Name(id=arg_name, ctx=ast.Load()))
        self.generic_visit(node)
        return node

from collections.abc import Callable, Iterable, Mapping
from .database import Database, DB_Class, DB_Object
from .data_components import ClassAttribute, Attribute, SparseMatrix
from .doc import Documentation
from .dtypes import *
from .memory_spaces import host, cuda
import ast
import inspect
import numba
import numpy
import re
import textwrap

# IDEAS:
# 
#   Coordinate Kernels for sparse matrixes.
#       User writes func accepting (row, col, val) and returning the new value.
#       Alternatively, I could allow matrix access inside of the methods.
#           But that's generally a bad design, since it encourages for-loops
#           inside of the compute kernel, esp for GPUs.
# 
# TODO:
# 
#   Allow returning pointers.
#       Both from function to methods, and from methods to the user.
#       Start with a few test cases.
# 
#   Special case for @compute on __init__, add class method `batch_init(num, *,**)`.
#       Note: python does not like @decorators on __init__.
#       Instead I should make my own magic "init" method.
#       Different name, custom implementation, mutually exclusive with "__init__".
# 
#   Allow numeric type annotations on function signatures.
#       If these are correctly given to numba, it will enforce them.

class Compute(Documentation):
    """
    A decorator for functions and methods to be executed by the database.

    This uses numba to compile the python code into binary machine code.
    All limitations and caveats of "numba.njit" also apply to this decorator.

    Notes:
        Functions may call other functions which are marked with this decorator.

        Variables containing DB_Objects must be annotated with the name of the
        object's DB_Class. For example:
            >>> class Foo
            >>>     @Compute
            >>>     def bar(self, other: "Foo"):
            >>>         ptr: "Foo" = self.my_pointer

        @Compute represents NULL pointers as the maximum integer "neuwon.database.NULL".
        As opposed to the object-oriented API which represents them as "None".
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
        wrapper = lambda instance=None, *args, **kwargs: self._call(instance, *args, **kwargs)
        wrapper.__name__ = self.name
        wrapper.__doc__  = self.doc
        if add_attr:
            assert self.name not in self.db_class.components
            assert self.name not in self.db_class.methods
            self.db_class.methods[self.name] = self
            setattr(self.db_class.instance_type, self.name, wrapper)
        return wrapper

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
        if len(instance) == 0:
            return []
        if isinstance(instance, range):
            assert instance.step == 1, 'range.step > 1 is unimplemented'

        retval = function(instance, *db_args, *args, **kwargs)
        if single_input and retval is not None:
            return retval[0]
        else:
            return retval

    def _jit(self, target):
        cached = self._jit_cache.get(target, None)
        if cached is not None: return cached
        assert self.db_class is not None, "Method not registered with a DB_Class"
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
            self.rewrite_self_argument()
        self.rewrite_annotated_arguments()
        self.rewrite_annotated_assignments()
        self.rewrite_functions()
        self.assemble_function()
        if self.entry_point:
            self.check_for_missing_return_type_annotation()
            self.write_entry_point()

    def deconstruct_function(self, function):
        """ Breakout the function into all of its constituent parts. """
        self.name      = function.__name__.replace('<', '_').replace('>', '_')
        self.filename  = inspect.getsourcefile(function)
        assert self.filename is not None
        self.lineno    = function.__code__.co_firstlineno
        self.signature = inspect.signature(function)
        self.func_text = textwrap.dedent(inspect.getsource(function))
        module_ast     = ast.parse(self.func_text, self.filename)
        for _ in range(self.lineno): ast.increment_lineno(module_ast)
        self.body_ast  = module_ast.body[0]
        nonlocals, globals_, builtins, unbound = inspect.getclosurevars(function)
        self.closure = {}
        self.closure.update(builtins)
        self.closure.update(globals_)
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
        if self.signature.return_annotation == inspect.Signature.empty:
            self.return_type = None
        else:
            self.return_type = numpy.dtype(self.signature.return_annotation)

    def is_method(self):
        return self.db_class is not None

    def is_function(self):
        return not self.is_method()

    def check_for_missing_return_type_annotation(self):
        if not self.entry_point:
            return
        if self.return_type is not None:
            return
        for node in ast.walk(self.body_ast):
            if isinstance(node, ast.Return):
                if node.value is None:
                    continue
                elif (isinstance(node.value, ast.Constant) or
                      isinstance(node.value, ast.NameConstant)):
                    if node.value.value is None:
                        continue
                raise TypeError(f"Method '{self.name}' is missing return type annotations")

    def rewrite_self_argument(self):
        self.self_variable = next(iter(self.signature.parameters))
        self.rewrite_reference(self.db_class, self.self_variable)

    def rewrite_annotated_arguments(self):
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

    def rewrite_annotated_assignments(self):
        self.local_types = {}
        self.local_alloc = {}
        for node in ast.walk(self.body_ast):
            if isinstance(node, ast.AnnAssign):
                if not node.simple: continue
                ref_name = node.target.id
                anno = node.annotation
                if isinstance(anno, ast.Str):
                    ref_class = anno.s
                elif isinstance(anno, ast.Constant) and isinstance(anno.value, str):
                    ref_class = anno.value
                else:
                    continue
                try:
                    ref_class = self.database.get_class(ref_class)
                except KeyError:
                    self.process_local_variable_annotation(ref_name, ref_class)
                    continue
                self.rewrite_reference(ref_class, ref_name)

    def process_local_variable_annotation(self, var_name, dtype):
        size = 0
        if m := re.match(r'^[Aa]lloc\((.*)\)\s*$', dtype):
            size, dtype = m.groups()[0].split(',')
            size  = int(size.strip())
            dtype = dtype.strip()
        if   dtype == 'Real':    dtype = Real
        elif dtype == 'Pointer': dtype = Pointer
        else:                    dtype = numpy.dtype(dtype)
        numba_type = numba.from_dtype(dtype)
        if size:
            self.local_alloc[var_name] = (numba_type, size)
        else:
            self.local_types[var_name] = numba_type

    def rewrite_reference(self, ref_class, ref_name):
        rr = _ReferenceRewriter(ref_class, ref_name, self.body_ast, self.target)
        self.body_ast = rr.body_ast
        self.db_arguments.update(rr.db_arguments)
        self.closure.update(rr.method_calls)

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
        for arg_name, db_attr in self.db_arguments:
            parameters.insert(start, inspect.Parameter(arg_name,
                                        inspect.Parameter.POSITIONAL_OR_KEYWORD))
            start += 1
        for var_name, (dtype, size) in self.local_alloc.items():
            parameters.insert(start, inspect.Parameter(var_name,
                                        inspect.Parameter.POSITIONAL_OR_KEYWORD))
            start += 1
        self.signature = self.signature.replace(parameters=parameters)
        # Strip out the type annotations BC they're not needed for this step and
        # they can cause errors: especially if they're complex types EG "np.dtype".
        signature = _remove_annotations(self.signature)
        # Assemble a new AST for the function.
        template            = f"def {self.name}{signature}:\n pass\n"
        module_ast          = ast.parse(template, filename=self.filename)
        function_ast        = module_ast.body[0]
        function_ast.body   = self.body_ast.body
        for _ in range(self.lineno): ast.increment_lineno(function_ast)
        self.module_ast     = ast.fix_missing_locations(module_ast)
        exec(compile(self.module_ast, self.filename, mode='exec'), self.closure)
        self.py_function    = self.closure[self.name]
        # Apply JIT compilation to the function.
        if self.target is host:
            self.jit_function = host.jit_module.njit(self.py_function, locals=self.local_types)
        elif self.target is cuda:
            self.jit_function = cuda.jit_module.jit(self.py_function, device=True, locals=self.local_types)

    def write_entry_point(self):
        exec_scope = {
                'function': self.jit_function,
                'return_type': self.return_type,
                'numba': numba,
                'numpy': numpy,
        }
        db_args    = ''.join(f'{arg_name}, ' for arg_name, db_attr in self.db_arguments)
        array_args = ''.join(f'{var_name}, ' for var_name in self.local_alloc)
        arguments  = db_args + self.parameters
        inner_args = db_args + array_args + self.parameters

        if self.target is host:
            array_alloc = ''.join(f'numpy.empty({sz}, dtype=numpy.{dt}), ' for dt,sz in self.local_alloc.values())

            if self.return_type is None:
                py_code = f'''
                    def entry_point(instances, {arguments}):
                        ({array_args}) = ({array_alloc})
                        if isinstance(instances, range):
                            range_entry_point(instances.start, instances.stop, {inner_args})
                        else:
                            array_entry_point(instances, {inner_args})
                    @numba.njit()
                    def range_entry_point(start, stop, {inner_args}):
                        for self in numba.prange(start, stop):
                            function(self, {inner_args})
                    @numba.njit()
                    def array_entry_point(instances, {inner_args}):
                        for index in numba.prange(len(instances)):
                            self = instances[index]
                            function(self, {inner_args})
                    '''
            else:
                py_code = f'''
                    def entry_point(instances, {arguments}):
                        ({array_args}) = ({array_alloc})
                        return_array = numpy.empty(len(instances), dtype=return_type)
                        if isinstance(instances, range):
                            range_entry_point(instances.start, instances.stop,
                                              return_array, {inner_args})
                        else:
                            array_entry_point(instances, return_array, {inner_args})
                        return return_array
                    @numba.njit()
                    def range_entry_point(start, stop, return_array, {inner_args}):
                        for index in numba.prange(stop - start):
                            self = start + index
                            return_array[index] = function(self, {inner_args})
                    @numba.njit()
                    def array_entry_point(instances, return_array, {inner_args}):
                        for index in numba.prange(len(instances)):
                            self = instances[index]
                            return_array[index] = function(self, {inner_args})
                    '''
        elif self.target is cuda:
            exec_scope['cuda'] = cuda.jit_module
            array_alloc = ''.join(f'numba.cuda.local.array({sz}, numba.{dt}), ' for dt,sz in self.local_alloc.values())

            if self.return_type is None:
                py_code = f'''
                    def entry_point(instances, {arguments}):
                        n_inst = len(instances)
                        threads = 32
                        blocks = (n_inst + (threads - 1)) // threads
                        if isinstance(instances, range):
                            range_entry_point[blocks, threads](
                                    instances.start, n_inst, {arguments})
                        else:
                            array_entry_point[blocks, threads](instances, {arguments})
                    @cuda.jit()
                    def range_entry_point(start, range_len, {arguments}):
                        index = cuda.grid(1)
                        if index < range_len:
                            self = start + index
                            ({array_args}) = ({array_alloc})
                            function(self, {inner_args})
                    @cuda.jit()
                    def array_entry_point(instances, {arguments}):
                        index = cuda.grid(1)
                        if index < len(instances):
                            self = instances[index]
                            ({array_args}) = ({array_alloc})
                            function(self, {inner_args})
                    '''
            else:
                py_code = f'''
                    import cupy
                    def entry_point(instances, {arguments}):
                        return_array = cupy.empty(len(instances), dtype=return_type)
                        n_inst = len(instances)
                        threads = 32
                        blocks = (n_inst + (threads - 1)) // threads
                        if isinstance(instances, range):
                            range_entry_point[blocks, threads](
                                    instances.start, n_inst, return_array, {arguments})
                        else:
                            array_entry_point[blocks, threads](instances, return_array, {arguments})
                        return return_array
                    @cuda.jit()
                    def range_entry_point(start, range_len, return_array, {arguments}):
                        index = cuda.grid(1)
                        if index < range_len:
                            self = start + index
                            ({array_args}) = ({array_alloc})
                            return_array[index] = function(self, {inner_args})
                    @cuda.jit()
                    def array_entry_point(instances, return_array, {arguments}):
                        index = cuda.grid(1)
                        if index < len(instances):
                            self = instances[index]
                            ({array_args}) = ({array_alloc})
                            return_array[index] = function(self, {inner_args})
                    '''
        exec(textwrap.dedent(py_code), exec_scope)
        self.entry_point = exec_scope['entry_point']

class _ReferenceRewriter(ast.NodeTransformer):
    def __init__(self, db_class, reference_name, body_ast, target):
        # print(f'Rewriting references to {reference_name} ({db_class.get_name()}).')
        ast.NodeTransformer.__init__(self)
        self.db_class       = db_class
        self.reference_name = str(reference_name)
        self.target         = target
        self.db_arguments   = {} # Name -> DB-Attribute
        self.method_calls   = {} # Name -> jit_function
        self.body_ast       = self.visit(body_ast)
        # Find all references which are exposed via this class and recursively rewrite them.
        for name, db_attribute in list(self.db_arguments.items()):
            if db_attribute.reference is not False:
                rr = _ReferenceRewriter(db_attribute.reference, name, self.body_ast, self.target)
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
            if getattr(value, 'db_reference', None) is not False:
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
            jit_data = _JIT(db_attribute.original, self.target, db_attribute.db_class)
            self.method_calls[qualname] = jit_data.jit_function
            replacement = ast.Name(id=qualname, ctx=ctx)
            replacement.db_method       = db_attribute
            replacement.db_arguments    = jit_data.db_arguments
            replacement.db_instance     = value
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
        if not hasattr(node.func, 'db_method'):
            return node
        # Pass the instance as the methods 'self' argument.
        node.args.insert(0, node.func.db_instance)
        # Insert the method's db_arguments.
        self.db_arguments.update(node.func.db_arguments)
        for arg_name, db_attr in reversed(node.func.db_arguments):
            node.args.insert(1, ast.Name(id=arg_name, ctx=ast.Load()))
        # Remove these tags so that this call will not be re-processed in any
        # subsequent AST pass.
        del node.func.db_method
        del node.func.db_arguments
        del node.func.db_instance
        return ast.fix_missing_locations(node)

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
        return ast.fix_missing_locations(node)

def _remove_annotations(signature) -> 'signature':
    signature = signature.replace(return_annotation=inspect.Signature.empty)
    signature = signature.replace(parameters=(
                    p.replace(annotation=inspect.Parameter.empty)
                        for p in signature.parameters.values()))
    return signature

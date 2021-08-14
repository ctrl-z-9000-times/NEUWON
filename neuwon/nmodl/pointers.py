""" Private module. """
__all__ = []

from neuwon.nmodl import code_gen

class PointerTable(dict):
    def __init__(self, mechanism):
        super().__init__()
        self.mech_name = mechanism.name()

    def add(self, name, read=None, write=None, accumulate=None):
        """ Factory method for Pointer objects.

        Argument name is an nmodl varaible name.
        Arguments read & write are a database access paths.
        Argument accumulate must be given at the same time as the "write" argument.
        """
        if name in self:
            ptr = self[name]
            if read is not None:
                read = str(read)
                if ptr.r and ptr.read != read:
                    eprint("Warning: Pointer override: %s read changed from '%s' to '%s'"%(
                            ptr.name, ptr.read, read))
                ptr.read = read
            if write is not None:
                write = str(write)
                if ptr.w and ptr.write != write:
                    eprint("Warning: Pointer override: %s write changed from '%s' to '%s'"%(
                            ptr.name, ptr.write, write))
                ptr.write = write
                ptr.accumulate = bool(accumulate)
            else: assert accumulate is None
        else:
            self[name] = ptr = Pointer(self.mech_name, name, read, write, accumulate)
        return ptr

    def initialize(self, database):
        for ptr in self.values():
            ptr.initialize(database)
            if ptr.target_class == self.mech_name: continue
            elif ptr.target_class not in self:
                if   ptr.target_class == "Segment": read = self.mech_name + ".insertion"
                elif ptr.target_class == "Inside":  read = "Segment.inside"
                elif ptr.target_class == "Outside": read = "Segment.outside"
                else: raise NotImplementedError(ptr.target_class)
                self.add(ptr.target_class, read=read)

    def verify_pointers_exist(self, database):
        for ptr in self.values():
            if ptr.r: database.get_component(ptr.read)
            if ptr.w: database.get_component(ptr.write)

class Pointer:
    def __init__(self, mech_name, name, read, write, accumulate):
        self.mech_name = mech_name
        self.name  = str(name)
        self.read  = str(read)  if read  is not None else None
        self.write = str(write) if write is not None else None
        self.accumulate = bool(accumulate)

    def initialize(self, database):
        component = self.read or self.write
        if component is None:
            self.target_class = None
        else:
            component = database.get_component(component)
            self.target_class = component.get_class().get_name()

    @property
    def r(self): return self.read is not None
    @property
    def w(self): return self.write is not None
    @property
    def a(self): return self.accumulate

    @property
    def mode(self):
        return (('r' if self.r else '') +
                ('w' if self.w else '') +
                ('a' if self.a else ''))

    def __repr__(self):
        args = []
        if self.r: args.append("read='%s'"%self.read)
        if self.w: args.append("write='%s'"%self.write)
        if self.a: args.append("accumulate")
        return self.name + " = Pointer(%s)"%', '.join(args)

    @property
    def read_py(self):
        """ Python variable name. """
        if self.r:
            if self.w and self.read != self.write:
                return CodeGen.mangle('read_' + self.name)
            return CodeGen.mangle(self.name)

    @property
    def write_py(self):
        """ Python variable name. """
        if self.w:
            if self.r and self.read != self.write:
                return CodeGen.mangle('write_' + self.name)
            return CodeGen.mangle(self.name)

    @property
    def index_py(self):
        """ Python variable name. """
        if self.target_class == self.mech_name:
            index_var = "index"
        else:
            index_var = self.target_class.lower()
        return CodeGen.mangle2(index_var)

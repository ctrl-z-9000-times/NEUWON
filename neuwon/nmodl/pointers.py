""" Private module. """
__all__ = []

from neuwon.nmodl import code_gen

class PointerTable(dict):
    """
    The PointerTable contains all references to persistant memory that the
    mechanism uses.

    PointerTable[NMODL_Variable_Name] -> Pointer
    """
    def __init__(self, mechanism):
        super().__init__()
        self.mech_name = mechanism.get_name()

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
        for ptr in list(self.values()):
            target_class = ptr.get_target_class()
            if target_class == self.mech_name: continue
            elif target_class not in self:
                if   target_class == "Segment": read = self.mech_name + ".segment"
                elif target_class == "Inside":  read = "Segment.inside"
                elif target_class == "Outside": read = "Segment.outside"
                else: raise NotImplementedError(target_class)
                self.add(target_class.lower(), read=read)
        self._verify_pointers_exist(database)

    def _verify_pointers_exist(self, database):
        for ptr in self.values():
            if ptr.r: database.get_component(ptr.read)
            if ptr.w: database.get_component(ptr.write)

    def sorted_values(self):
        return [ptr for (name, ptr) in sorted(self.items())]

    def __repr__(self):
        return "{\t" + ",\n\t".join(repr(ptr) for ptr in self.sorted_values()) + "}"

class Pointer:
    def __init__(self, mech_name, name, read, write, accumulate):
        self.mech_name = mech_name
        self.name  = str(name)
        self.read  = str(read)  if read  is not None else None
        self.write = str(write) if write is not None else None
        self.accumulate = bool(accumulate)

    def get_target_class(self):
        component = self.read or self.write
        if component is None:
            return None
        else:
            return component.partition('.')[0]

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
        if self.r: args.append(f"read='{self.read}'")
        if self.w: args.append(f"write='{self.write}'")
        if self.a: args.append("accumulate")
        return f"{self.name} = Pointer({', '.join(args)})"

    def read_py(self):
        """ Python variable name. """
        if self.r:
            if self.w and self.read != self.write:
                return code_gen.mangle('read_' + self.name)
            return code_gen.mangle(self.name)

    def write_py(self):
        """ Python variable name. """
        if self.w:
            if self.r and self.read != self.write:
                return code_gen.mangle('write_' + self.name)
            return code_gen.mangle(self.name)

    def index_py(self):
        """ Python variable name. """
        target_class = self.get_target_class()
        if target_class == self.mech_name:
            index_var = "index"
        else:
            index_var = target_class.lower()
        return code_gen.mangle2(index_var)

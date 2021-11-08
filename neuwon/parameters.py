from collections.abc import Callable, Iterable, Mapping
from numbers import Number
from pprint import pformat

class Parameters(dict):
    """
    Parameter dictionaries may contain strings, numbers, lists, and dicts.
    """
    def __init__(self, parameters):
        for name, value in parameters.items():
            name = str(name)
            self[name] = Parameters._clean(value)

    @staticmethod
    def _clean(value) -> 'value':
        if   isinstance(value, str):      return value
        elif isinstance(value, Number):   return value
        elif isinstance(value, Mapping):  return Parameters(value)
        elif isinstance(value, Iterable): return list(map(Parameters._clean, value))
        else: raise ValueError(f"Bad parameter value: '{value}'")

    def __repr__(self):
        return pformat(dict(self))

    def update_with_defaults(self, default_parameters):
        """
        Merge the given default_parameters into this dictionary, without
        overwritting any existing entries.
        """
        for name, default_value in default_parameters.items():
            if name not in self:
                self[name] = default_value
            else:
                parameter_value = self[name]
                if isinstance(default_value, Mapping) and not isinstance(parameter_value, Mapping):
                    raise ValueError(f'Expected parameter {name} to be a dictionary!')
                if isinstance(parameter_value, Parameters):
                    if not isinstance(default_value, Mapping):
                        raise ValueError(f'Expected parameter {name} to be a value, not a dictionary!')
                    parameter_value.update_with_defaults(default_value)

    @classmethod
    def combine(cls, parents) -> 'Parameters':
        1/0

    def mutate(self) -> 'self':
        1/0
        return self

    @staticmethod
    def traverse(node):
        if isinstance(node, str):
            return []
        elif isinstance(node, Mapping):
            return node.values()
        elif isinstance(node, Iterable):
            return node
        else:
            return []





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
        return pformat(self)

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





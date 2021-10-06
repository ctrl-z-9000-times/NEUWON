import numpy as _np

Real    = _np.dtype('f8')
epsilon = _np.finfo(Real).eps
Pointer = _np.dtype('u4')
NULL    = _np.iinfo(Pointer).max

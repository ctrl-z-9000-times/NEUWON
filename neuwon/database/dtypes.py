import numpy as np

Real    = np.dtype('f8')
epsilon = np.finfo(Real).eps
Pointer = np.dtype('u4')
NULL    = np.iinfo(Pointer).max

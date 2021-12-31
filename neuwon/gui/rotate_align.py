import numpy as np
import math

def magnitude(vector):
    return math.sqrt(vector.dot(vector))

def rotate_align(a, b):
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    a = a / magnitude(a)
    b = b / magnitude(b)
    v = np.cross(a, b)
    c = np.dot(a, b) # Cosine of angle
    if c == 0:
        raise NotImplementedError
    vx = np.array([ # Skew symmetric cross-product matrix
        [0,    -v[2], v[1]],
        [v[2],  0,   -v[0]],
        [-v[1], v[0], 0]])
    return np.eye(3) + vx + np.matmul(vx, vx) / (1.0 + c)

from neuwon.gui.rotate_align import rotate_align, magnitude
import numpy as np
import pytest

def test_random_vectors():
    for trial in range(1000):
        v1 = np.random.uniform(-100, 100, size=3) # Reference vector
        v2 = np.random.uniform(-100, 100, size=3) # Target vector
        rot_matrix = rotate_align(v2, v1)
        v3 = v1.dot(rot_matrix)
        v3 = np.squeeze(np.asarray(v3))
        v2_norm = v2 / magnitude(v2)
        assert magnitude(v1) == pytest.approx(magnitude(v3))
        v3_norm = v3 / magnitude(v3)
        assert v3_norm[0] == pytest.approx(v2_norm[0])
        assert v3_norm[1] == pytest.approx(v2_norm[1])
        assert v3_norm[2] == pytest.approx(v2_norm[2])

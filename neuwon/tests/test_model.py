from neuwon.model import *
import pytest

@pytest.mark.skip
def test_model():
    m = Model(.1)
    m.advance()

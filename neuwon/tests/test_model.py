from neuwon.model import *
import pytest

def test_model():
    m = Model(.1)
    m.advance()

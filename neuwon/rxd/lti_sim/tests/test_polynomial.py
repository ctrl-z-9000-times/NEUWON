from lti_sim.inputs import *
from lti_sim.polynomial import *
import pytest

def test_cubic_polynomial():
    v = LinearInput('v', -100, 100)
    cubic = PolynomialForm([v], 3)
    assert cubic.degree == 3
    assert cubic.terms  == ((0,), (1,), (2,), (3,))
    print(cubic)

    cubic_copy = PolynomialForm([v], cubic)
    assert cubic_copy.terms         == cubic.terms
    assert cubic_copy.num_terms     == 4
    assert cubic_copy.num_var_terms == 3

    irregular = PolynomialForm([v], [[0], [2], [3]])
    assert irregular.degree        == 3
    assert irregular.num_terms     == 3
    assert irregular.num_var_terms == 2
    print(irregular)

def test_hash():
    i1 = LinearInput('i1', -100, 100)
    i2 = LinearInput('i2', -100, 100)
    cubic_a = PolynomialForm([i1], 3)
    cubic_b = PolynomialForm([i1], [[3],[2],[1],[0]])
    assert cubic_a == cubic_b
    assert hash(cubic_a) == hash(cubic_b)
    # Test polynomial with different independent variable.
    cubic_c = PolynomialForm([i2], 3)
    assert cubic_a != cubic_c
    assert hash(cubic_a) != hash(cubic_c)
    # Test different polynomials are not equal.
    irregular = PolynomialForm([i1], [[0], [2], [3]])
    assert len(set((cubic_a, cubic_a, irregular))) == 2
    # Test polynomial with two independent variables.
    quad2d_a = PolynomialForm([i1,i2], [[0,0], [2,0], [0,2]])
    quad2d_b = PolynomialForm([i1,i2], [[0,0], [0,2], [2,0]])
    assert quad2d_a == quad2d_b
    assert len(set((quad2d_a, quad2d_b))) == 1

def test_suggestions():
    i1 = LinearInput('i1', -100, 100)
    i2 = LinearInput('i2', -100, 100)

    cubic = PolynomialForm([i1], 3)
    assert cubic._get_perimeter() == [(3,)]
    assert cubic == cubic._from_perimeter(cubic.inputs, [(3,)])
    assert cubic.suggest_add() == [PolynomialForm([i1], 4)]
    assert cubic.suggest_remove() == [PolynomialForm([i1], 2)]

    const = PolynomialForm([i1, i2], [[0,0]])
    perimeter = {(0,0)}
    assert set(const._get_perimeter()) == perimeter
    assert const == const._from_perimeter(const.inputs, perimeter)
    assert set(const.suggest_add()) == {PolynomialForm([i1, i2], [[0,0], [0,1]]),
                                        PolynomialForm([i1, i2], [[0,0], [1,0]])}
    assert const.suggest_remove() == []

    linear1 = PolynomialForm([i1, i2], [[0,0], [0,1], [1,0]])
    perimeter = {(1,0), (0,1)}
    assert set(linear1._get_perimeter()) == perimeter
    assert linear1 == linear1._from_perimeter(linear1.inputs, perimeter)
    assert set(linear1.suggest_add()) == {
            PolynomialForm([i1, i2], [[0,0], [0,1], [1,0], [2,0]]),
            PolynomialForm([i1, i2], [[0,0], [0,1], [1,0], [0,2]]),
            PolynomialForm([i1, i2], [[0,0], [0,1], [1,0], [1,1]]),
    }
    assert set(linear1.suggest_remove()) == {PolynomialForm([i1, i2], [[0,0], [0,1]]),
                                             PolynomialForm([i1, i2], [[0,0], [1,0]])}

    linear2 = PolynomialForm([i1, i2], [[0,0], [0,1], [1,0], [1,1]])
    perimeter = {(1,0), (0,1), (1,1)}
    assert set(linear2._get_perimeter()) == perimeter
    assert linear2 == linear2._from_perimeter(linear2.inputs, perimeter)
    for poly in linear2.suggest_add():
        print(str(poly))
    assert set(linear2.suggest_add()) == {
            PolynomialForm([i1, i2], [[0,0], [0,1], [1,0], [1,1], [2,0]]),
            PolynomialForm([i1, i2], [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]]),
            PolynomialForm([i1, i2], [[0,0], [0,1], [1,0], [1,1], [0,2]]),
            PolynomialForm([i1, i2], [[0,0], [0,1], [1,0], [1,1], [0,2], [1,2]]),
    }
    assert set(linear2.suggest_remove()) == {PolynomialForm([i1, i2], [[0,0], [0,1], [1,0]])}

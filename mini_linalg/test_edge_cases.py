import pytest
from mini_linalg.core import Vector, Matrix


def test_empty_vector():
    v = Vector([])
    assert v.size() == 0
    assert v.data == []


def test_single_element_vector():
    v = Vector([5.0])
    assert v.size() == 1


def test_empty_matrix():
    m = Matrix([])
    assert m.rows() == 0
    assert m.cols() == 0


def test_single_element_matrix():
    m = Matrix([[7.0]])
    assert m.rows() == 1
    assert m.cols() == 1


def test_matrix_vector_negative_numbers():
    m = Matrix([[-1.0, 2.0], [3.0, -4.0]])
    v = Vector([2.0, 3.0])
    result = m.mat_vec_mult(v)
    assert result.data == [4.0, -6.0]


def test_matrix_vector_zero():
    m = Matrix([[0.0, 0.0], [0.0, 0.0]])
    v = Vector([5.0, 10.0])
    result = m.mat_vec_mult(v)
    assert result.data == [0.0, 0.0]


def test_matrix_multiplication_by_zero():
    a = Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = Matrix([[0.0, 0.0], [0.0, 0.0]])
    result = a.mat_mat_mult(b)
    assert result.data == [[0.0, 0.0], [0.0, 0.0]]


def test_matrix_multiplication_single_element():
    a = Matrix([[3.0]])
    b = Matrix([[4.0]])
    result = a.mat_mat_mult(b)
    assert result.data == [[12.0]]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.0], 1),
        ([1.0, 2.0], 2),
        ([1.0, 2.0, 3.0], 3),
    ],
)
def test_vector_sizes_parametrized(values, expected):
    assert Vector(values).size() == expected
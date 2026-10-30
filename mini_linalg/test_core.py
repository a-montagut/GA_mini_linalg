import pytest
from mini_linalg.core import Vector, Matrix


def test_vector_creation():
    v = Vector([1.0, 2.0, 3.0])
    assert v.data == [1.0, 2.0, 3.0]


def test_vector_size():
    v = Vector([1.0, 2.0, 3.0])
    assert v.size() == 3


def test_vector_repr():
    v = Vector([1.0, 2.0])
    assert repr(v) == "Vector([1.0, 2.0])"


def test_matrix_creation():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert m.data == [[1.0, 2.0], [3.0, 4.0]]


def test_matrix_rows_cols():
    m = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert m.rows() == 2
    assert m.cols() == 3


def test_matrix_repr():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert repr(m) == "Matrix([[1.0, 2.0], [3.0, 4.0]])"


def test_mat_vec_mult():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    v = Vector([5.0, 6.0])
    result = m.mat_vec_mult(v)
    assert result.data == [17.0, 39.0]


def test_mat_vec_mult_non_square():
    m = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v = Vector([1.0, 1.0, 1.0])
    result = m.mat_vec_mult(v)
    assert result.data == [6.0, 15.0]


def test_mat_vec_mult_dimension_mismatch():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    v = Vector([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Dimension mismatch"):
        m.mat_vec_mult(v)


def test_mat_mat_mult():
    a = Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = Matrix([[5.0, 6.0], [7.0, 8.0]])
    result = a.mat_mat_mult(b)
    assert result.data == [[19.0, 22.0], [43.0, 50.0]]


def test_mat_mat_mult_non_square():
    a = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = Matrix([[1.0], [2.0], [3.0]])
    result = a.mat_mat_mult(b)
    assert result.data == [[14.0], [32.0]]


def test_mat_mat_mult_identity():
    a = Matrix([[1.0, 0.0], [0.0, 1.0]])
    b = Matrix([[5.0, 6.0], [7.0, 8.0]])
    result = a.mat_mat_mult(b)
    assert result.data == [[5.0, 6.0], [7.0, 8.0]]


def test_mat_mat_mult_dimension_mismatch():
    a = Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    with pytest.raises(ValueError, match="Dimension mismatch"):
        a.mat_mat_mult(b)

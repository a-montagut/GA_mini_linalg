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


@pytest.mark.parametrize(
    "matrix_data, vector_data",
    [
        ([[1.0, 2.0]], [1.0]),
        ([[1.0, 2.0]], [1.0, 2.0, 3.0]),
        ([[1.0, 2.0, 3.0]], [1.0, 2.0]),
    ],
)
def test_mat_vec_dimension_mismatch_cases(matrix_data, vector_data):
    m = Matrix(matrix_data)
    v = Vector(vector_data)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        m.mat_vec_mult(v)


@pytest.mark.parametrize(
    "left_data, right_data",
    [
        ([[1.0, 2.0]], [[1.0]]),
        ([[1.0]], [[1.0], [2.0]]),
        ([[1.0, 2.0, 3.0]], [[1.0], [2.0]]),
    ],
)
def test_mat_mat_dimension_mismatch_cases(left_data, right_data):
    a = Matrix(left_data)
    b = Matrix(right_data)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        a.mat_mat_mult(b)

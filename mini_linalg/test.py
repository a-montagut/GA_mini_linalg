import pytest

from mini_linalg.implementation.matrix import (
    Matrix,
    batch_matvec,
    batch_matmul
)
from mini_linalg.implementation.vector import Vector



def test_empty_vector():
    v = Vector([])
    assert len(v) == 0
    assert v.data.tolist() == []


def test_single_element_vector():
    v = Vector([5.0])
    assert len(v) == 1


def test_empty_matrix():
    m = Matrix([])
    assert m.data.size == 0


def test_single_element_matrix():
    m = Matrix([[7.0]])
    assert m.shape == (1, 1)


def test_matrix_vector_negative_numbers():
    m = Matrix([[-1.0, 2.0], [3.0, -4.0]])
    v = Vector([2.0, 3.0])
    result = m.mat_vec(v)
    assert result.data.tolist() == [4.0, -6.0]


def test_matrix_vector_zero():
    m = Matrix([[0.0, 0.0], [0.0, 0.0]])
    v = Vector([5.0, 10.0])
    result = m.mat_vec(v)
    assert result.data.tolist() == [0.0, 0.0]


def test_matrix_multiplication_by_zero():
    a = Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = Matrix([[0.0, 0.0], [0.0, 0.0]])
    result = a.mat_mul(b)
    assert result.data.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_matrix_multiplication_single_element():
    a = Matrix([[3.0]])
    b = Matrix([[4.0]])
    result = a.mat_mul(b)
    assert result.data.tolist() == [[12.0]]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.0], 1),
        ([1.0, 2.0], 2),
        ([1.0, 2.0, 3.0], 3),
    ],
)
def test_vector_sizes_parametrized(values, expected):
    assert len(Vector(values)) == expected


def test_vector_creation():
    v = Vector([1.0, 2.0, 3.0])
    assert v.data.tolist() == [1.0, 2.0, 3.0]


def test_vector_size():
    v = Vector([1.0, 2.0, 3.0])
    assert len(v) == 3


def test_vector_repr():
    v = Vector([1.0, 2.0])
    assert repr(v) == "Vector([1.0, 2.0])"


def test_matrix_creation():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert m.data.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_matrix_shape():
    m = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert m.shape == (2, 3)


def test_matrix_repr():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert repr(m) == "Matrix([[1.0, 2.0], [3.0, 4.0]])"


def test_mat_vec():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    v = Vector([5.0, 6.0])
    result = m.mat_vec(v)
    assert result.data.tolist() == [17.0, 39.0]


def test_mat_vec_non_square():
    m = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v = Vector([1.0, 1.0, 1.0])
    result = m.mat_vec(v)
    assert result.data.tolist() == [6.0, 15.0]


def test_mat_vec_dimension_mismatch():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    v = Vector([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        m.mat_vec(v)


def test_mat_mul():
    a = Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = Matrix([[5.0, 6.0], [7.0, 8.0]])
    result = a.mat_mul(b)
    assert result.data.tolist() == [[19.0, 22.0], [43.0, 50.0]]


def test_mat_mul_non_square():
    a = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = Matrix([[1.0], [2.0], [3.0]])
    result = a.mat_mul(b)
    assert result.data.tolist() == [[14.0], [32.0]]


def test_mat_mul_identity():
    a = Matrix([[1.0, 0.0], [0.0, 1.0]])
    b = Matrix([[5.0, 6.0], [7.0, 8.0]])
    result = a.mat_mul(b)
    assert result.data.tolist() == [[5.0, 6.0], [7.0, 8.0]]


def test_mat_mul_dimension_mismatch():
    a = Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    with pytest.raises(ValueError):
        a.mat_mul(b)


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

    with pytest.raises(ValueError):
        m.mat_vec(v)


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

    with pytest.raises(ValueError):
        a.mat_mul(b)


def test_batch_matvec():

    matrices = [
        Matrix([[1, 2], [3, 4]]),
        Matrix([[2, 0], [0, 2]])
    ]

    vectors = [
        Vector([1, 1]),
        Vector([3, 4])
    ]

    results = batch_matvec(matrices, vectors)

    assert results[0].data.tolist() == [3, 7]
    assert results[1].data.tolist() == [6, 8]


def test_batch_matmul():

    matrices_a = [
        Matrix([[1, 2], [3, 4]]),
        Matrix([[1, 0], [0, 1]])
    ]

    matrices_b = [
        Matrix([[5, 6], [7, 8]]),
        Matrix([[2, 3], [4, 5]])
    ]

    results = batch_matmul(matrices_a, matrices_b)

    assert results[0].data.tolist() == [[19, 22], [43, 50]]
    assert results[1].data.tolist() == [[2, 3], [4, 5]]


def test_batch_matvec_dimension_mismatch():

    matrices = [Matrix([[1, 2], [3, 4]])]
    vectors = [Vector([1, 2, 3])]

    with pytest.raises(ValueError):
        batch_matvec(matrices, vectors)


def test_batch_matmul_dimension_mismatch():

    matrices_a = [Matrix([[1, 2], [3, 4]])]
    matrices_b = [Matrix([[1, 2], [3, 4], [5, 6]])]

    with pytest.raises(ValueError):
        batch_matmul(matrices_a, matrices_b)


def test_batch_size_mismatch():

    matrices = [Matrix([[1, 2], [3, 4]])]
    vectors = [
        Vector([1, 2]),
        Vector([3, 4])
    ]

    with pytest.raises(ValueError):
        batch_matvec(matrices, vectors)


def test_empty_batches():

    assert batch_matvec([], []) == []
    assert batch_matmul([], []) == []
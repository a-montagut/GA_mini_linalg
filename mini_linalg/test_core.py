import pytest

from mini_linalg.core import (
    Vector,
    Matrix,
    batch_matvec,
    batch_matmul
)


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

    assert results[0].data == [3, 7]
    assert results[1].data == [6, 8]


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

    assert results[0].data == [[19, 22], [43, 50]]
    assert results[1].data == [[2, 3], [4, 5]]


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


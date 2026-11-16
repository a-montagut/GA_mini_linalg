import numpy as np
from .vector import Vector


class Matrix:
    def __init__(self, data: list[list[float]]):
        self.data = np.array(data, dtype=float)

    def __repr__(self) -> str:
        return f"Matrix({self.data.tolist()})"

    def rows(self) -> int:
        return len(self.data)

    def cols(self) -> int:
        return self.data.shape[1] if self.data.ndim == 2 else 0

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __getitem__(self, index):
        result = self.data[index]
        if isinstance(result, np.ndarray):
            return result.tolist()
        return float(result)

    def __add__(self, other: "Matrix") -> "Matrix":
        if self.data.shape != other.data.shape:
            raise ValueError("Matrix dimensions must match")
        return Matrix((self.data + other.data).tolist())

    def __sub__(self, other: "Matrix") -> "Matrix":
        if self.data.shape != other.data.shape:
            raise ValueError("Matrix dimensions must match")
        return Matrix((self.data - other.data).tolist())

    def __mul__(self, scalar: float) -> "Matrix":
        return Matrix((self.data * scalar).tolist())

    def mat_mul(self, other: "Matrix") -> "Matrix":
        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError(
                f"Cannot multiply ({self.data.shape[0]}x{self.data.shape[1]}) "
                f"by ({other.data.shape[0]}x{other.data.shape[1]})"
            )
        return Matrix(np.matmul(self.data, other.data).tolist())

    def mat_vec(self, vec: "Vector") -> "Vector":
        if self.data.shape[1] != len(vec.data):
            raise ValueError("Matrix columns must match Vector length")
        return Vector(np.matmul(self.data, vec.data).tolist())

    def mat_vec_mult(self, v: Vector) -> Vector:
        if self.cols() != v.size():
            raise ValueError(
                f"Dimension mismatch: matrix has {self.cols()} columns "
                f"but vector has {v.size()} elements"
            )
        result = []
        for row in self.data:
            s = 0.0
            for a, b in zip(row, v.data):
                s += a * b
            result.append(s)
        return Vector(result)

    def mat_mat_mult(self, other: "Matrix") -> "Matrix":
        if self.cols() != other.rows():
            raise ValueError(
                f"Dimension mismatch: left matrix has {self.cols()} columns "
                f"but right matrix has {other.rows()} rows"
            )
        result = []
        for row in self.data:
            new_row = []
            for j in range(other.cols()):
                s = 0.0
                for k in range(self.cols()):
                    s += row[k] * other.data[k][j]
                new_row.append(s)
            result.append(new_row)
        return Matrix(result)

    def transpose(self) -> "Matrix":
        return Matrix(self.data.T.tolist())


def batch_matvec(
    matrices: list[Matrix],
    vectors: list[Vector]
) -> list[Vector]:
    """
    Perform pairwise matrix-vector multiplication.

    Each matrix is multiplied by its corresponding vector.
    """

    if len(matrices) != len(vectors):
        raise ValueError("Batch sizes must match.")

    results = []

    for matrix, vector in zip(matrices, vectors):
        result = matrix.mat_vec(vector)
        results.append(result)

    return results


def batch_matmul(
    matrices_a: list[Matrix],
    matrices_b: list[Matrix]
) -> list[Matrix]:
    """
    Perform pairwise matrix-matrix multiplication.

    Each matrix in the first batch is multiplied
    by its corresponding matrix in the second batch.
    """

    if len(matrices_a) != len(matrices_b):
        raise ValueError("Batch sizes must match.")

    results = []

    for A, B in zip(matrices_a, matrices_b):
        result = A.mat_mul(B)
        results.append(result)

    return results

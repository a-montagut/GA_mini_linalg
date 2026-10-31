import numpy as np


class Vector:
    def __init__(self, data: list[float]):
        self.data = np.array(data, dtype=float)

    def __repr__(self) -> str:
        return f"Vector({self.data.tolist()})"

    def __len__(self) -> int:
        return len(self.data)

    def size(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> float:
        return float(self.data[index])

    def __add__(self, other: "Vector") -> "Vector":
        if len(self.data) != len(other.data):
            raise ValueError("Vector dimensions must match")
        return Vector((self.data + other.data).tolist())

    def __sub__(self, other: "Vector") -> "Vector":
        if len(self.data) != len(other.data):
            raise ValueError("Vector dimensions must match")
        return Vector((self.data - other.data).tolist())

    def __mul__(self, scalar: float) -> "Vector":
        return Vector((self.data * scalar).tolist())

    def dot(self, other: "Vector") -> float:
        if len(self.data) != len(other.data):
            raise ValueError("Vector dimensions must match")
        return float(np.dot(self.data, other.data))

    def norm(self) -> float:
        return float(np.linalg.norm(self.data))


class Matrix:
    def __init__(self, data: list[list[float]]):
        self.data = np.array(data, dtype=float)

    def __repr__(self) -> str:
        return f"Matrix({self.data.tolist()})"

    def rows(self) -> int:
        return len(self.data)

    def cols(self) -> int:
        return len(self.data[0]) if self.data else 0

    @property
    def shape(self) -> tuple[int, int]:
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

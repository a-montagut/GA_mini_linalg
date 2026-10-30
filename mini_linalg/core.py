class Vector:
    def __init__(self, data: list[float]):
        self.data = list(data)

    def __repr__(self) -> str:
        return f"Vector({self.data})"

    def size(self) -> int:
        return len(self.data)


class Matrix:
    def __init__(self, data: list[list[float]]):
        self.data = [list(r) for r in data]

    def __repr__(self) -> str:
        return f"Matrix({self.data})"

    def rows(self) -> int:
        return len(self.data)

    def cols(self) -> int:
        return len(self.data[0]) if self.data else 0

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

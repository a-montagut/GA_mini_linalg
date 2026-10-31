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

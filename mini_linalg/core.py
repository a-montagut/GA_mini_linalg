class Vector :
  def __init__ ( self , data : list [ float ]) :
    self . data = list ( data )
  def __repr__ ( self ) -> str :
    return f" Vector ({ self . data })"
    
class Matrix :
  def __init__ ( self , data : list [ list [ float ]]) :
    self . data = [ list (r) for r in data ]
  def __repr__ ( self ) -> str :
    return f" Matrix ({ self . data })"

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

        rows = matrix.data
        v = vector.data

        if not rows or any(len(row) != len(v) for row in rows):
            raise ValueError("Incompatible matrix and vector dimensions.")

        result = [
            sum(a * b for a, b in zip(row, v))
            for row in rows
        ]

        results.append(Vector(result))

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

        a = A.data
        b = B.data

        if not a or not b:
            raise ValueError("Matrices cannot be empty.")

        if any(len(row) != len(a[0]) for row in a):
            raise ValueError("Matrix A must be rectangular.")

        if any(len(row) != len(b[0]) for row in b):
            raise ValueError("Matrix B must be rectangular.")

        if len(a[0]) != len(b):
            raise ValueError("Incompatible matrix dimensions.")

        # Transpose B to simplify multiplication
        b_transposed = list(zip(*b))

        result = [
            [
                sum(x * y for x, y in zip(row, col))
                for col in b_transposed
            ]
            for row in a
        ]

        results.append(Matrix(result))

    return results


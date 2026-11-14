"""Kernels and theoretical models for the Roofline homework.

Both operations are implemented with NumPy (which calls the BLAS library
NumPy was built against, OpenBLAS here). The output arrays are allocated
once outside the timed region and passed through ``out=`` so that the
timed region contains only the arithmetic.
"""
import numpy as np


# ---------------------------------------------------------------------------
# 1. Operations
# ---------------------------------------------------------------------------
def init_arrays(N, dtype, op):
    """Allocate and initialise the operands with constant values.

    Returns a dict with the inputs and a pre-allocated output array.
    """
    A = np.full((N, N), 1.0, dtype=dtype)
    if op == "matvec":
        x = np.full(N, 3.0, dtype=dtype)
        y = np.empty(N, dtype=dtype)
        return {"A": A, "x": x, "out": y}
    if op == "matmul":
        B = np.full((N, N), 2.0, dtype=dtype)
        C = np.empty((N, N), dtype=dtype)
        return {"A": A, "B": B, "out": C}
    raise ValueError(op)


def matvec(A, x, out):
    """y = A x  (BLAS level 2, GEMV)."""
    return np.matmul(A, x, out=out)


def matmul(A, B, out):
    """C = A B  (BLAS level 3, GEMM)."""
    return np.matmul(A, B, out=out)


def run(op, arrays):
    if op == "matvec":
        return matvec(arrays["A"], arrays["x"], arrays["out"])
    return matmul(arrays["A"], arrays["B"], arrays["out"])


def check(op, arrays, N):
    """Correctness check with a tolerance (never exact equality).

    With the constant initialisation every entry of the result is known:
    y_i = sum_j 1*3 = 3N and C_ij = sum_k 1*2 = 2N.
    """
    expected = 3.0 * N if op == "matvec" else 2.0 * N
    out = arrays["out"]
    rtol = 10 * np.finfo(out.dtype).eps * N  # loose bound for N-term sums
    return bool(np.allclose(out, expected, rtol=rtol, atol=0.0))


# ---------------------------------------------------------------------------
# 2. Theoretical FLOP / byte / arithmetic-intensity models
# ---------------------------------------------------------------------------
def flops(op, N):
    """Floating-point operations (one multiply + one add per term).

    matvec: N rows x (N mult + N-1 add) = 2N^2 - N  ~ 2N^2
    matmul: N^2 entries x (N mult + N-1 add) = 2N^3 - N^2 ~ 2N^3
    The usual 2N^2 / 2N^3 convention is used (BLAS counts FMA as 2 FLOP).
    """
    return 2.0 * N**2 if op == "matvec" else 2.0 * N**3


def bytes_min(op, N, s):
    """Compulsory memory traffic (each array moved once, perfect reuse).

    matvec: read A (N^2) + read x (N) + write y (N)      -> (N^2 + 2N) s
    matmul: read A, B (2N^2) + write C (N^2)            -> 3N^2 s
    s = bytes per element (4 for float32, 8 for float64).
    """
    if op == "matvec":
        return (N**2 + 2.0 * N) * s
    return 3.0 * N**2 * s


def bytes_noreuse(op, N, s):
    """Pessimistic traffic: every operand is re-read from memory for every
    multiply-add (no cache reuse at all). Used only as a lower bound on I.

    matvec: A_ij and x_j for each of N^2 terms + write y -> (2N^2 + N) s
    matmul: A_ik and B_kj for each of N^3 terms + write C -> (2N^3 + N^2) s
    """
    if op == "matvec":
        return (2.0 * N**2 + N) * s
    return (2.0 * N**3 + N**2) * s


def intensity(op, N, s, model="min"):
    b = bytes_min(op, N, s) if model == "min" else bytes_noreuse(op, N, s)
    return flops(op, N) / b

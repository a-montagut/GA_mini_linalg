"""Benchmark matrix-vector and matrix-matrix multiplication (float32/float64).

Usage
-----
    python benchmark.py                 # all BLAS threads (default)
    python benchmark.py --threads 1     # single-threaded BLAS
    python benchmark.py --quick         # small sizes, for a fast test

Writes results/bench_<tag>.csv, which plot_results.py reads.
"""
import argparse
import os

# The number of BLAS threads must be fixed BEFORE NumPy is imported.
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--threads", type=int, default=0,
                    help="BLAS threads (0 = library default, all cores)")
parser.add_argument("--quick", action="store_true", help="small sizes only")
args = parser.parse_args()
if args.threads > 0:
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)

import time

import numpy as np

import kernels

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

DTYPES = {"float32": np.float32, "float64": np.float64}

# Each point is repeated until both limits are met, so that short calls are
# still timed over a reasonable amount of work.
MIN_REPS = 5
MIN_TIME = 0.5      # accumulated timed seconds per point
MAX_REPS = 5000


def sizes(op):
    """Problem sizes, growing by a factor of about sqrt(2)."""
    if op == "matvec":
        n = [128, 181, 256, 362, 512, 724, 1024, 1448, 2048, 2896,
             4096, 5793, 8192, 11585, 16384]
        return n[:8] if args.quick else n
    n = [64, 91, 128, 181, 256, 362, 512, 724, 1024, 1448, 2048, 2896, 4096]
    return n[:7] if args.quick else n


def time_kernel(op, arrays):
    """Time repeated calls. Allocation and initialisation are NOT timed:
    the operands and the output array exist before the first call."""
    for _ in range(2):                      # warm-up (caches, thread pool)
        kernels.run(op, arrays)

    times = []
    while (len(times) < MIN_REPS or sum(times) < MIN_TIME) \
            and len(times) < MAX_REPS:
        t0 = time.perf_counter()
        kernels.run(op, arrays)
        times.append(time.perf_counter() - t0)
    return np.array(times)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for op in ("matvec", "matmul"):
        for N in sizes(op):
            # both precisions back-to-back so they see the same machine state
            for name, dtype in DTYPES.items():
                arrays = kernels.init_arrays(N, dtype, op)   # not timed
                times = time_kernel(op, arrays)
                correct = kernels.check(op, arrays, N)

                t = float(np.median(times))
                flops = kernels.flops(op, N)
                nbytes = kernels.bytes_min(op, N, np.dtype(dtype).itemsize)
                # Every value is formatted to a fixed width, so the columns
                # line up when the file is read. The times keep 6 significant
                # digits: the clock resolves nanoseconds, the rest is noise.
                rows.append({
                    "op": f"{op:>6}", "dtype": f"{name:>7}",
                    "N": f"{N:>5d}", "reps": f"{len(times):>5d}",
                    "t_median_s": f"{t:>12.6e}",
                    "t_min_s": f"{times.min():>12.6e}",
                    "AI_min": f"{flops / nbytes:>9.4f}",
                    "GFLOPs_median": f"{flops / t / 1e9:>13.2f}",
                    "GBs_median": f"{nbytes / t / 1e9:>10.2f}",
                    "correct": f"{str(correct):>7}",
                })
                print(f"{op:6s} {name} N={N:6d} reps={len(times):5d} "
                      f"t={t:.3e}s  {flops / t / 1e9:8.2f} GFLOP/s  "
                      f"{nbytes / t / 1e9:8.2f} GB/s  ok={correct}", flush=True)
                del arrays

    tag = f"{args.threads}t" if args.threads > 0 else "allt"
    path = os.path.join(RESULTS, f"bench_{tag}.csv")
    with open(path, "w", newline="\n") as f:
        # each column name is padded to the width of its own values
        f.write(",".join(f"{name:>{len(value)}}"
                         for name, value in rows[0].items()) + "\n")
        for row in rows:
            f.write(",".join(row.values()) + "\n")
    print("Saved", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()

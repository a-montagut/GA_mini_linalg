"""Benchmark matrix-vector and matrix-matrix multiplication (float32/float64).

Usage
-----
    python benchmark.py                 # all BLAS threads (default)
    python benchmark.py --threads 1     # single-threaded BLAS
    python benchmark.py --quick         # small sizes, for a fast test

Writes results/bench_<T>t.csv and results/machine_<T>t.json.
"""
import argparse
import os
import sys

# The number of BLAS threads must be fixed BEFORE NumPy is imported.
_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--threads", type=int, default=0,
                help="BLAS threads (0 = library default, all logical cores)")
_p.add_argument("--quick", action="store_true", help="small sizes only")
_p.add_argument("--min-time", type=float, default=0.5,
                help="minimum accumulated timed seconds per point")
_p.add_argument("--min-reps", type=int, default=5,
                help="minimum timed repetitions per point")
ARGS = _p.parse_args()
if ARGS.threads > 0:
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(ARGS.threads)

import csv
import json
import platform
import time

import numpy as np

import kernels

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

DTYPES = {"float32": np.float32, "float64": np.float64}


def sizes(op, quick):
    """Roughly geometric (factor sqrt 2) problem sizes."""
    if op == "matvec":
        n = [128, 181, 256, 362, 512, 724, 1024, 1448, 2048, 2896,
             4096, 5793, 8192, 11585, 16384]
        return n[:8] if quick else n
    n = [64, 91, 128, 181, 256, 362, 512, 724, 1024, 1448, 2048,
         2896, 4096]
    return n[:7] if quick else n


def time_kernel(op, arrays, min_time, min_reps, max_reps=5000):
    """Time repeated calls. Allocation/initialisation are NOT timed:
    the operands and the output array exist before the first call."""
    for _ in range(2):                      # warm-up (caches, thread pool)
        kernels.run(op, arrays)
    times = []
    total = 0.0
    while (len(times) < min_reps or total < min_time) and len(times) < max_reps:
        t0 = time.perf_counter()
        kernels.run(op, arrays)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        total += t1 - t0
    return np.array(times)


def stream_probe(n_elem=2**25, reps=10):
    """STREAM-like measurement of the attainable DRAM bandwidth (float64).

    copy: b = a   (read 8 B + write 8 B per element)
    add : c = a+b (read 16 B + write 8 B per element)
    Arrays of 256 MB each, far larger than the 8 MB L3 cache.
    """
    a = np.full(n_elem, 1.0)
    b = np.full(n_elem, 2.0)
    c = np.empty(n_elem)
    res = {}
    for name, fn, nbytes in (
        ("copy", lambda: np.copyto(c, a), 16 * n_elem),
        ("add", lambda: np.add(a, b, out=c), 24 * n_elem),
    ):
        fn()
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            t.append(time.perf_counter() - t0)
        res[name + "_GBs"] = nbytes / min(t) / 1e9
    return res


def machine_info():
    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "blas_threads_requested": ARGS.threads,
    }
    try:
        cfg = np.show_config(mode="dicts")
        info["blas"] = cfg["Build Dependencies"]["blas"]["name"]
        info["blas_version"] = cfg["Build Dependencies"]["blas"]["version"]
    except Exception:  # pragma: no cover - older NumPy
        pass
    return info


def main():
    os.makedirs(RESULTS, exist_ok=True)
    tag = f"{ARGS.threads}t" if ARGS.threads > 0 else "allt"
    info = machine_info()
    print("Machine:", json.dumps(info, indent=1))
    print("STREAM-like bandwidth probe ...")
    info["stream"] = stream_probe()
    print(info["stream"])

    rows = []
    for op in ("matvec", "matmul"):
        for N in sizes(op, ARGS.quick):
            # both precisions back-to-back so they see the same machine state
            for dname, dtype in DTYPES.items():
                s = np.dtype(dtype).itemsize
                arrays = kernels.init_arrays(N, dtype, op)   # not timed
                t = time_kernel(op, arrays, ARGS.min_time, ARGS.min_reps)
                ok = kernels.check(op, arrays, N)
                t_med = float(np.median(t))
                t_min = float(t.min())
                F = kernels.flops(op, N)
                Q = kernels.bytes_min(op, N, s)
                row = {
                    "op": op, "dtype": dname, "N": N, "reps": len(t),
                    "t_median_s": t_med, "t_min_s": t_min,
                    "t_std_s": float(t.std()),
                    "flops": F, "bytes_min": Q,
                    "AI_min": F / Q,
                    "GFLOPs_median": F / t_med / 1e9,
                    "GFLOPs_best": F / t_min / 1e9,
                    "GBs_median": Q / t_med / 1e9,
                    "GBs_best": Q / t_min / 1e9,
                    "correct": ok,
                }
                rows.append(row)
                print(f"{op:6s} {dname} N={N:6d} reps={len(t):5d} "
                      f"t={t_med:.3e}s  {row['GFLOPs_median']:8.2f} GFLOP/s  "
                      f"{row['GBs_median']:8.2f} GB/s  ok={ok}", flush=True)
                del arrays

    csv_path = os.path.join(RESULTS, f"bench_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RESULTS, f"machine_{tag}.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("Saved", os.path.relpath(csv_path, HERE))


if __name__ == "__main__":
    main()

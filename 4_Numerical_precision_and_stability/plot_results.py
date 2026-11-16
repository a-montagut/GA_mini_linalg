"""Plots for the Roofline homework.

Reads the CSV files written by benchmark.py and produces, for each run:

    fig1_time_<tag>      execution time vs problem size, with N^2 / N^3 lines
    fig2_perf_<tag>      performance (GFLOP/s) and effective bandwidth (GB/s)
    fig3_roofline_<tag>  the Roofline diagram

Usage:  python plot_results.py
"""
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

OPS = {"matvec": "matrix-vector", "matmul": "matrix-matrix"}
DTYPES = ["float32", "float64"]
COLOR = {"matvec": "tab:blue", "matmul": "tab:red"}
LINE = {"float32": "-", "float64": "--"}
MARKER = {"float32": "o", "float64": "s"}
EXPONENT = {"matvec": 2, "matmul": 3}   # work is 2N^2 and 2N^3 FLOP


def read_results(path):
    """Read one benchmark CSV into a list of dictionaries, sorted by N."""
    rows = []
    with open(path) as f:
        # skipinitialspace drops the padding that aligns the columns
        for row in csv.DictReader(f, skipinitialspace=True):
            for key in ("N", "t_median_s", "GFLOPs_median", "GBs_median",
                        "AI_min"):
                row[key] = float(row[key])
            rows.append(row)
    rows.sort(key=lambda r: r["N"])
    return rows


def select(rows, op, dtype):
    """The rows of one operation and one precision."""
    return [r for r in rows if r["op"] == op and r["dtype"] == dtype]


def column(rows, name):
    return np.array([r[name] for r in rows])


def ceilings(rows):
    """The two Roofline limits, taken from the measurements themselves.

    B_mem is the bandwidth of the largest matrix-vector case: that matrix is
    far too large for any cache, so the data comes from main memory and the
    operation is purely memory-bound. P_peak is the best performance of the
    run, reached by matrix-matrix multiplication, which is compute-bound.
    Both are *attainable* limits, lower than the theoretical ones of the
    machine; the gap to a point is the performance it leaves unused.
    """
    biggest = max(r["N"] for r in rows if r["op"] == "matvec")
    b_mem = max(r["GBs_median"] for r in rows
                if r["op"] == "matvec" and r["N"] == biggest)
    return b_mem, max(r["GFLOPs_median"] for r in rows)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, name + ".png"), dpi=200)
    plt.close(fig)


def plot_time(rows, tag):
    """Execution time vs N, compared with the theoretical N^2 / N^3 growth."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for op in OPS:
        for dtype in DTYPES:
            sel = select(rows, op, dtype)
            ax.loglog(column(sel, "N"), column(sel, "t_median_s"),
                      LINE[dtype] + MARKER[dtype], color=COLOR[op], ms=4,
                      label=f"{OPS[op]}, {dtype}")

    # Reference slopes, anchored above the largest size of each operation.
    for op in OPS:
        p = EXPONENT[op]
        sel = select(rows, op, "float64")
        N = column(sel, "N")
        t = 2 * sel[-1]["t_median_s"] * (N / N[-1]) ** p
        ax.loglog(N, t, ":", color="gray", lw=1)
        ax.text(N[-1], t[-1] * 1.2, f"$\\propto N^{p}$", color="gray",
                ha="right", fontsize=9)

    ax.set_xlabel("problem size N")
    ax.set_ylabel("time per call [s]")
    ax.set_title(f"Execution time ({tag})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, f"fig1_time_{tag}")


def plot_performance(rows, tag):
    """Performance and effective bandwidth as the problem grows."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for op in OPS:
        for dtype in DTYPES:
            sel = select(rows, op, dtype)
            style = LINE[dtype] + MARKER[dtype]
            label = f"{OPS[op]}, {dtype}"
            ax1.loglog(column(sel, "N"), column(sel, "GFLOPs_median"), style,
                       color=COLOR[op], ms=4, label=label)
            ax2.loglog(column(sel, "N"), column(sel, "GBs_median"), style,
                       color=COLOR[op], ms=4, label=label)

    b_mem, p_peak = ceilings(rows)
    ax1.axhline(p_peak, color="k", ls=":", lw=1,
                label=f"best measured, {p_peak:.0f} GFLOP/s")
    ax1.set_ylim(top=p_peak * 2)
    ax1.set_ylabel("performance [GFLOP/s]")
    ax1.set_title("(a) performance")

    ax2.axhline(b_mem, color="k", ls=":", lw=1,
                label=f"main memory, {b_mem:.0f} GB/s")
    ax2.set_ylabel("effective bandwidth [GB/s]")
    ax2.set_title("(b) memory traffic")

    for ax in (ax1, ax2):
        ax.set_xlabel("problem size N")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Performance and bandwidth ({tag})")
    save(fig, f"fig2_perf_{tag}")


def plot_roofline(rows, tag):
    """Roofline: measured performance against the two hardware limits."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    b_mem, p_peak = ceilings(rows)
    ridge = p_peak / b_mem
    I = np.logspace(-2, 3, 200)
    ax.loglog(I, np.minimum(p_peak, b_mem * I), "k-", lw=1.5,
              label=f"measured roofline ({b_mem:.0f} GB/s, "
                    f"{p_peak:.0f} GFLOP/s)")
    ax.axvline(ridge, color="gray", ls=":", lw=1)
    ax.text(ridge * 1.15, p_peak * 1.5,
            f"ridge point\nI = {ridge:.1f} FLOP/byte", fontsize=8, color="gray")

    for op in OPS:
        for dtype in DTYPES:
            sel = select(rows, op, dtype)
            ax.loglog(column(sel, "AI_min"), column(sel, "GFLOPs_median"),
                      MARKER[dtype], ls="none", color=COLOR[op], ms=5,
                      mfc="none", label=f"{OPS[op]}, {dtype}")

    ax.set_xlabel("arithmetic intensity I [FLOP/byte]")
    ax.set_ylabel("performance [GFLOP/s]")
    ax.set_title(f"Roofline model ({tag})")
    ax.set_ylim(0.1, p_peak * 5)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, f"fig3_roofline_{tag}")


def print_summary(rows, tag):
    """Short table of the measurements, one line per point."""
    print(f"\nResults for {tag}")
    print(f"{'operation':14s} {'dtype':8s} {'N':>6s} {'time [s]':>11s} "
          f"{'GFLOP/s':>9s} {'GB/s':>8s} {'I [FLOP/B]':>11s}")
    for op in OPS:
        for dtype in DTYPES:
            for r in select(rows, op, dtype):
                print(f"{OPS[op]:14s} {dtype:8s} {int(r['N']):6d} "
                      f"{r['t_median_s']:11.3e} {r['GFLOPs_median']:9.2f} "
                      f"{r['GBs_median']:8.2f} {r['AI_min']:11.2f}")

    failed = [r for r in rows if r["correct"] != "True"]
    if failed:
        print(f"WARNING: {len(failed)} points failed the correctness check")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RESULTS, "bench_*.csv")))
    if not files:
        raise SystemExit("No results found. Run benchmark.py first.")

    for path in files:
        tag = os.path.basename(path)[len("bench_"):-len(".csv")]
        rows = read_results(path)
        plot_time(rows, tag)
        plot_performance(rows, tag)
        plot_roofline(rows, tag)
        print_summary(rows, tag)

    print(f"\nFigures written to {os.path.relpath(FIGURES, HERE)}/")


if __name__ == "__main__":
    main()

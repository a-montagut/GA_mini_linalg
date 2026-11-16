"""Generate the performance plots from results/bench_*.csv.

Usage:  python plot_results.py
Writes figures/*.pdf and figures/*.png and prints a summary table.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import kernels

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

# ---------------------------------------------------------------------------
# Hardware model (Intel Core i7-8650U, Kaby Lake R, 4 cores / 8 threads,
# AVX2 with 2 FMA units per core, dual-channel DDR4-2400).
# Edit these values if you run on a different machine.
# ---------------------------------------------------------------------------
CORES = 4
F_BASE = 1.9e9                   # base clock [Hz]
F_TURBO = 4.2e9                  # maximum turbo clock [Hz]
FLOP_PER_CYCLE = {"float64": 16,  # 2 FMA x 4 doubles x 2 FLOP
                  "float32": 32}  # 2 FMA x 8 floats  x 2 FLOP
# The clock varies between base and turbo with power and temperature, so the
# compute ceiling is a band: P_PEAK (base clock) .. P_TURBO (max turbo).
P_PEAK = {d: CORES * F_BASE * f / 1e9 for d, f in FLOP_PER_CYCLE.items()}
P_TURBO = {d: CORES * F_TURBO * f / 1e9 for d, f in FLOP_PER_CYCLE.items()}
B_THEO = 2 * 8 * 2.4e9 / 1e9     # 2 channels x 8 B x 2400 MT/s = 38.4 GB/s

NAME = {"matvec": "matrix-vector", "matmul": "matrix-matrix"}
COL = {"matvec": "#1f77b4", "matmul": "#d62728"}
MRK = {"float32": "o", "float64": "s"}
LS = {"float32": "-", "float64": "--"}

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "legend.fontsize": 7.5, "figure.dpi": 150})


def load(tag):
    p = os.path.join(RES, f"bench_{tag}.csv")
    if not os.path.exists(p):
        return None, None
    df = pd.read_csv(p)
    with open(os.path.join(RES, f"machine_{tag}.json")) as f:
        info = json.load(f)
    return df, info


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_time(df, df1):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for op in ("matvec", "matmul"):
        for d in ("float32", "float64"):
            s = df[(df.op == op) & (df.dtype == d)]
            ax[0].loglog(s.N, s.t_median_s, LS[d], marker=MRK[d], ms=3,
                         color=COL[op], label=f"{NAME[op]} {d}")
    # reference slopes anchored at the largest size
    for op, p in (("matvec", 2), ("matmul", 3)):
        s = df[(df.op == op) & (df.dtype == "float64")]
        N = np.array([s.N.min(), s.N.max()], float)
        t_ref = s.t_median_s.iloc[-1] * (N / N[-1]) ** p
        ax[0].loglog(N, t_ref * 2.5, ":", color="gray", lw=1)
        ax[0].text(N[-1] * 0.45, t_ref[-1] * 0.6, rf"$\propto N^{p}$",
                   color="gray", fontsize=8)
    ax[0].set_xlabel("N"); ax[0].set_ylabel("time per call [s]")
    ax[0].set_title("(a) execution time (median)")
    ax[0].legend(ncol=1, loc="upper left")

    for op in ("matvec", "matmul"):
        for frame, ls, lab in ((df, "-", "all threads"), (df1, "--", "1 thread")):
            if frame is None:
                continue
            a = frame[(frame.op == op) & (frame.dtype == "float32")].set_index("N")
            b = frame[(frame.op == op) & (frame.dtype == "float64")].set_index("N")
            r = b.t_median_s / a.t_median_s
            ax[1].semilogx(r.index, r.values, ls, marker="o", ms=3,
                           color=COL[op], label=f"{NAME[op]}, {lab}")
    ax[1].axhline(2.0, color="k", lw=0.8, ls=":")
    ax[1].text(70, 2.05,
               "ideal 2x", fontsize=7)
    ax[1].set_xlabel("N"); ax[1].set_ylabel(r"$t_{64}/t_{32}$")
    ax[1].set_title("(b) float32 speed-up over float64")
    ax[1].legend(loc="upper left", ncol=2, fontsize=6.5)
    ax[1].set_ylim(0.5, 5.8)
    fig.tight_layout()
    save(fig, "fig1_time_speedup")


def fig_perf(df, df1, B_meas):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for op in ("matvec", "matmul"):
        for d in ("float32", "float64"):
            s = df[(df.op == op) & (df.dtype == d)]
            ax[0].loglog(s.N, s.GFLOPs_median, LS[d], marker=MRK[d], ms=3,
                         color=COL[op], label=f"{NAME[op]} {d}")
            ax[1].loglog(s.N, s.GBs_median, LS[d], marker=MRK[d], ms=3,
                         color=COL[op], label=f"{NAME[op]} {d}")
            if df1 is not None:
                s1 = df1[(df1.op == op) & (df1.dtype == d)]
                ax[0].loglog(s1.N, s1.GFLOPs_median, LS[d], color=COL[op],
                             alpha=0.3, lw=1)
    for d, c in (("float64", "gray"), ("float32", "k")):
        ax[0].axhspan(P_PEAK[d], P_TURBO[d], color=c, alpha=0.08, lw=0)
        ax[0].axhline(P_PEAK[d], color=c, ls=":", lw=0.9)
        ax[0].text(70, P_PEAK[d] * 1.07,
                   f"peak {d}: {P_PEAK[d]:.0f} (base) - {P_TURBO[d]:.0f} (turbo)",
                   fontsize=6.5, color=c)
    ax[0].set_xlabel("N"); ax[0].set_ylabel("GFLOP/s")
    ax[0].set_title("(a) performance (faint: 1 thread)")
    ax[0].legend(loc="lower right", ncol=2, fontsize=6)
    ax[0].set_ylim(0.4, P_TURBO["float32"] * 1.5)
    hb = [ax[1].axhline(B_THEO, color="k", ls=":", lw=0.9,
                        label=f"main memory, theoretical {B_THEO:.1f} GB/s"),
          ax[1].axhline(B_meas, color="gray", ls="-.", lw=0.9,
                        label=f"main memory, measured {B_meas:.1f} GB/s")]
    ax[1].legend(handles=hb, loc="lower left", fontsize=6.5)
    for d, s in (("float32", 4), ("float64", 8)):
        ax[1].axvline(np.sqrt(CACHE_BYTES / s), color="tab:green", lw=0.8,
                      ls=LS[d], alpha=0.7)
    ax[1].text(np.sqrt(CACHE_BYTES / 4) * 1.1, 1.3, "A exceeds cache\n(f64 | f32)",
               fontsize=7, color="tab:green", rotation=90)
    ax[1].set_xlabel("N"); ax[1].set_ylabel("GB/s (compulsory traffic)")
    ax[1].set_title("(b) effective bandwidth")
    fig.tight_layout()
    save(fig, "fig2_perf_bandwidth")


def fig_roofline(df, B_meas, P_meas):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    I = np.logspace(-2, 3, 400)
    for k, d in enumerate(("float32", "float64")):
        a = ax[k]
        lo = np.minimum(P_PEAK[d], B_THEO * I)
        hi = np.minimum(P_TURBO[d], B_THEO * I)
        a.fill_between(I, lo, hi, color="k", alpha=0.08, lw=0)
        a.loglog(I, lo, "k-", lw=1.4,
                 label=f"theoretical, base clock ({B_THEO:.1f} GB/s, "
                       f"{P_PEAK[d]:.0f} GF/s)")
        a.loglog(I, hi, "k--", lw=1.0,
                 label=f"theoretical, max turbo ({P_TURBO[d]:.0f} GF/s)")
        a.loglog(I, np.minimum(P_meas[d], B_meas * I), "-.", color="gray", lw=1.2,
                 label=f"measured ({B_meas:.1f} GB/s, {P_meas[d]:.0f} GF/s)")
        B_cache = df[df.op == "matvec"].GBs_median.max()
        a.loglog(I[I < P_PEAK[d] / B_cache], B_cache * I[I < P_PEAK[d] / B_cache],
                 ":", color="tab:green", lw=1.2,
                 label=f"matrix-vector from cache ({B_cache:.0f} GB/s)")
        mv = df[(df.op == "matvec") & (df.dtype == d)]
        incache = mv.bytes_min <= CACHE_BYTES
        a.scatter(mv.AI_min[incache], mv.GFLOPs_median[incache], marker="o",
                  s=18, facecolors="none", edgecolors=COL["matvec"], zorder=3,
                  label="matrix-vector, data fits in cache")
        a.scatter(mv.AI_min[~incache], mv.GFLOPs_median[~incache], marker="o",
                  s=18, color=COL["matvec"], zorder=3,
                  label="matrix-vector, data in main memory")
        mm = df[(df.op == "matmul") & (df.dtype == d)]
        a.scatter(mm.AI_min, mm.GFLOPs_median, marker="^", s=18,
                  color=COL["matmul"], zorder=3, label="matrix-matrix")
        for N in (mm.N.min(), mm.N.max()):
            r = mm[mm.N == N].iloc[0]
            a.annotate(f"N={N}", (r.AI_min, r.GFLOPs_median), fontsize=6.5,
                       xytext=(0, -11), textcoords="offset points", ha="center")
        ridge = P_PEAK[d] / B_THEO
        a.axvline(ridge, color="k", lw=0.6, ls=":")
        a.text(ridge * 1.1, 900, f"ridge {ridge:.1f}", fontsize=7)
        a.set_xlabel("arithmetic intensity I [FLOP/byte]")
        a.set_title(f"({'ab'[k]}) Roofline, {d}")
        a.set_xlim(1e-2, 2e3); a.set_ylim(0.1, 1500)
        a.legend(loc="lower right", fontsize=5.3)
    ax[0].set_ylabel("performance [GFLOP/s]")
    fig.tight_layout()
    save(fig, "fig3_roofline")


# ---------------------------------------------------------------------------
# Human-readable tables
# ---------------------------------------------------------------------------
# Size of the CPU cache (last level, 8 MB L3 on the i7-8650U, shared by all
# cores). Data larger than this must stream from main memory on every call.
CACHE_BYTES = 8 * 2**20
LEVELS = ("cache", "main memory")
EXPECTED_P = {"matvec": 2, "matmul": 3}
DTYPES = ("float32", "float64")


def fmt_time(t):
    if t < 1e-3:
        return f"{t * 1e6:.1f} us"
    if t < 1:
        return f"{t * 1e3:.2f} ms"
    return f"{t:.2f} s"


def fmt_size(b):
    for unit, k in (("GB", 2**30), ("MB", 2**20), ("KB", 2**10)):
        if b >= k:
            return f"{b / k:.1f} {unit}"
    return f"{b:.0f} B"


def fmt_count(x):
    for unit, k in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if x >= k:
            return f"{x / k:.1f} {unit}"
    return f"{x:.0f}"


def cache_level(nbytes):
    """'cache' if the whole working set fits in the cache, else 'main memory'."""
    return "cache" if nbytes <= CACHE_BYTES else "main memory"


def pick(frame, op, d, N=None):
    s = frame[(frame.op == op) & (frame.dtype == d)]
    return s if N is None else s[s.N == N].iloc[0]


def fit_p(s):
    """Slope of log(time) vs log(N): time ~ N^p."""
    if len(s) < 2:
        return float("nan")
    return float(np.polyfit(np.log(s.N), np.log(s.t_median_s), 1)[0])


def render(title, headers, rows, notes=(), md=False):
    if md:
        out = [f"### {title}", "",
               "| " + " | ".join(headers) + " |",
               "|" + "|".join("---:" for _ in headers) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        out += [""] + [f"> {n}  " for n in notes]
        return "\n".join(out) + "\n"
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: " | ".join(c.rjust(w[i]) for i, c in enumerate(cells))
    sep = "-+-".join("-" * wi for wi in w)
    out = ["", title, "=" * len(sep), line(headers), sep]
    out += [line(r) for r in rows]
    out += [sep] + [f"  {n}" for n in notes]
    return "\n".join(out)


def fmt_n(N):
    return f"{int(N):,}"


def fmt_gflops(x):
    return f"{x:.2f} GFLOP/s"


def fmt_gbs(x):
    return f"{x:.1f} GB/s"


def fmt_spread(r):
    """Run-to-run spread: how much slower the median call is than the best one."""
    return f"{(r.t_median_s - r.t_min_s) / r.t_median_s * 100:.0f} %"


def scaling_table(df, op, label):
    """Table 1: does time grow exactly like the amount of work?"""
    p = EXPECTED_P[op]
    rows, prev = [], {}
    for N in sorted(pick(df, op, "float32").N):
        row = [fmt_n(N), fmt_count(kernels.flops(op, N)) + "FLOP"]
        for d in DTYPES:
            r = pick(df, op, d, N)
            t = r.t_median_s
            g = f"x{t / prev[d]:.2f}" if d in prev else "-"
            row += [fmt_time(t), g, fmt_gflops(r.GFLOPs_median)]
            prev[d] = t
        rows.append(row)
    headers = ["N", "work", "f32 time", "f32 growth", "f32 performance",
               "f64 time", "f64 growth", "f64 performance"]
    notes = [f"N grows by sqrt(2) per row: if time ~ N^{p}, time grows "
             f"x{np.sqrt(2) ** p:.2f} per row and performance stays flat.",
             "At small N growth is below that: the fixed cost of each call "
             "dominates the arithmetic."]
    return (f"TABLE 1 ({NAME[op]}, {label}): execution time vs problem size", headers,
            rows, notes)


def exponent_table(frames):
    """Table 2: fitted exponent per regime -> proof of N^2 / N^3 scaling."""
    rows = []
    for label, frame in frames:
        for op in ("matvec", "matmul"):
            for d in DTYPES:
                s = pick(frame, op, d)
                if op == "matvec":
                    data = kernels.bytes_min(op, s.N, np.dtype(d).itemsize)
                    regimes = (("data fits in cache", s[data <= CACHE_BYTES]),
                               ("data in main memory", s[data > CACHE_BYTES]))
                else:
                    regimes = (("N >= 256", s[s.N >= 256]),)
                for name, sub in regimes:
                    rows.append([NAME[op], d, label, name,
                                 f"{fmt_n(sub.N.min())} - {fmt_n(sub.N.max())}",
                                 f"{fit_p(sub):.2f}", str(EXPECTED_P[op])])
    headers = ["operation", "dtype", "threads", "regime", "N range",
               "fitted p", "expected p"]
    notes = ["Least-squares fit of log(time) = p log(N) + c inside each regime.",
             "p close to the expected value = time grows linearly with the work "
             "(2N^2 or 2N^3 FLOP)."]
    return "TABLE 2: fitted scaling exponent (time ~ N^p)", headers, rows, notes


def cache_table(df, df1, d):
    """Table 3: matrix-vector bandwidth as the matrix outgrows the cache."""
    s_ = np.dtype(d).itemsize
    rows = []
    for N in sorted(pick(df, "matvec", d).N):
        data = kernels.bytes_min("matvec", N, s_)
        r_all = pick(df, "matvec", d, N)
        row = [fmt_n(N), fmt_size(data), cache_level(data)]
        if df1 is not None:
            r1 = pick(df1, "matvec", d, N)
            row += [fmt_gbs(r1.GBs_median), fmt_gflops(r1.GFLOPs_median),
                    fmt_spread(r1)]
        row += [fmt_gbs(r_all.GBs_median), fmt_gflops(r_all.GFLOPs_median),
                fmt_spread(r_all)]
        rows.append(row)
    headers = ["N", "data (A+x+y)", "data lives in"]
    if df1 is not None:
        headers += ["bandwidth 1 thr", "performance 1 thr", "spread 1 thr"]
    headers += ["bandwidth all thr", "performance all thr", "spread all thr"]
    notes = [f"'cache' = the whole working set fits in the "
             f"{fmt_size(CACHE_BYTES)} CPU cache; 'main memory' = it must stream "
             "from main memory on every call.",
             "spread = (median time - best time) / median time. Small spread = "
             "reproducible point; large spread = noisy, trust it less."]
    return (f"TABLE 3 ({d}): matrix-vector speed, cache vs main memory",
            headers, rows, notes)


def cache_summary(df, df1):
    """Table 4: one line per memory level, the condensed cache proof."""
    frame, label = (df1, "1 thread") if df1 is not None else (df, "all threads")
    rows = []
    best = {}
    stats = {}
    for d in DTYPES:
        s = pick(frame, "matvec", d)
        lvl = s.bytes_min.map(cache_level)
        stats[d] = s.groupby(lvl).GBs_median.median()
        best[d] = stats[d].max()
    for name in LEVELS:
        if not any(name in stats[d].index for d in DTYPES):
            continue
        row = [name]
        for d in DTYPES:
            v = stats[d].get(name)
            row += ["-", "-"] if v is None else [fmt_gbs(v), f"{v / best[d] * 100:.0f} %"]
        rows.append(row)
    headers = ["data lives in", "f32 bandwidth", "f32 vs best",
               "f64 bandwidth", "f64 vs best"]
    notes = [f"Median matrix-vector bandwidth ({label}) over all sizes in each group.",
             f"Theoretical main-memory bandwidth B_mem: {B_THEO:.1f} GB/s."]
    return (f"TABLE 4: cache vs main memory, matrix-vector ({label})",
            headers, rows, notes)


def print_tables(df, df1):
    frames = [("1", df1)] if df1 is not None else []
    frames.append(("all", df))
    main, main_label = (df1, "1 thread") if df1 is not None else (df, "all threads")
    tables = [scaling_table(main, "matvec", main_label),
              scaling_table(main, "matmul", main_label),
              exponent_table(frames),
              cache_table(df, df1, "float32"),
              cache_table(df, df1, "float64"),
              cache_summary(df, df1)]
    for t in tables:
        print(render(*t))
    md_path = os.path.join(RES, "tables.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark tables\n\n")
        f.write("\n".join(render(*t, md=True) for t in tables))
    print(f"\nTables also written to {os.path.relpath(md_path, HERE)}")


def main():
    os.makedirs(FIG, exist_ok=True)
    df, info = load("allt")
    df1, info1 = load("1t")
    # Measured main-memory bandwidth: best bandwidth-probe result (copy/add of
    # 256 MB arrays). It is NOT taken from the matrix-vector results, so the
    # matrix-vector points are judged against an independent ceiling.
    B_meas = max(info["stream"].values())
    if info1 is not None:
        B_meas = max(B_meas, *info1["stream"].values())
    P_meas = {d: df[(df.op == "matmul") & (df.dtype == d)].GFLOPs_median.max()
              for d in ("float32", "float64")}

    fig_time(df, df1)
    fig_perf(df, df1, B_meas)
    fig_roofline(df, B_meas, P_meas)

    peak = lambda P: " | ".join(f"{d} {fmt_gflops(P[d])}" for d in DTYPES)
    print(f"Theoretical: bandwidth {fmt_gbs(B_THEO)} | peak (base clock) "
          f"{peak(P_PEAK)} | peak (max turbo) {peak(P_TURBO)}")
    print(f"Measured:    bandwidth {fmt_gbs(B_meas)} | best {peak(P_meas)}")
    for label, inf in (("all threads", info), ("1 thread", info1)):
        if inf is not None:
            s = inf["stream"]
            print(f"STREAM probe ({label}): copy {fmt_gbs(s['copy_GBs'])} | "
                  f"add {fmt_gbs(s['add_GBs'])}")
    if not (df.correct.all() and (df1 is None or df1.correct.all())):
        print("WARNING: some results failed the correctness check")
    print_tables(df, df1)


if __name__ == "__main__":
    main()

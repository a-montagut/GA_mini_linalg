# 4. Numerical precision and stability: exploring the Roofline model

Benchmark of dense matrix-vector multiplication (`y = A x`) and
matrix-matrix multiplication (`C = A B`) with NumPy in `float32` (single
precision) and `float64` (double precision), interpreted with the Roofline
model. NumPy hands both operations to the BLAS library (routines GEMV and
GEMM of OpenBLAS).

## Contents

| File | Purpose |
|---|---|
| `kernels.py` | The two operations, array initialisation, correctness check, and the theoretical FLOP / byte / arithmetic-intensity models |
| `benchmark.py` | Timing driver: sweeps the problem size `N` for both precisions and writes `results/bench_<tag>.csv` |
| `bandwidth_probe.py` | STREAM-like copy and add on arrays far larger than the cache: measures the main memory bandwidth *B_mem* |
| `plot_results.py` | Builds the figures in `figures/` (time vs N, GFLOP/s and GB/s, Roofline) and prints a summary of every measured point |
| `results/` | The measurements, one CSV per run: `bench_allt.csv` (all BLAS threads) and `bench_1t.csv` (one thread) |
| `figures/` | Generated plots, one set of three per run (`.png`, 200 dpi) |
| `context/` | The homework statement and the lecture slides, kept locally for reference. **Not committed** (listed in `.gitignore`); nothing in the project depends on it |

## Requirements

Python 3.10+ with `numpy` and `matplotlib`:

```bash
pip install numpy matplotlib
```

## Reproducing the results

```bash
python benchmark.py                # all BLAS threads  -> results/bench_allt.csv
python benchmark.py --threads 1    # one BLAS thread   -> results/bench_1t.csv
python bandwidth_probe.py          # measured main memory bandwidth B_mem
python plot_results.py             # figures/*.png, one set per run
```

`python benchmark.py --quick` runs only the small sizes (a few seconds) to
check that everything works. The number of timed repetitions per point is set
by `MIN_REPS` and `MIN_TIME` at the top of `benchmark.py` (at least 5
repetitions and 0.5 s of accumulated timed work).

A full run takes a few minutes per thread setting and needs about 3.5 GB of
free memory (the largest matrix-vector case is a 16384 x 16384 `float64`
matrix).

**Run both settings back-to-back, on mains power, with other applications
closed.** On a laptop the clock moves with power, temperature and background
load, so results vary between runs: over five all-threads runs the best
`float32` matrix-matrix result ranged from 174 to 312 GFLOP/s. The
single-thread run varies much less. Any number quoted in the report text
must be updated after a new run.

## Terminology

The terms follow the lecture slides:

* **arithmetic intensity** *I*: floating-point operations performed per byte
  moved between the processor and memory, `I = FLOP / bytes`.
* **cache / main memory**: data that fits in the last-level CPU cache is
  reused at high speed; larger working sets must stream from main memory on
  every call. The cache size depends on the machine, so this README does not
  fix a number for it.
* **memory-bound / compute-bound**: performance limited by *B_mem* times *I*,
  or by the peak arithmetic rate *P_peak*. The crossing of the two limits is
  the **ridge point** *I_ridge = P_peak / B_mem*.

## Reading the results

`python plot_results.py` reads every `results/bench_*.csv`, writes one set of
figures per run and prints a summary of all measured points (redirect it to a
file if you want to keep it).

| Figure | Shows |
|---|---|
| `fig1_time_<tag>` | time per call vs N, with the theoretical `N^2` and `N^3` slopes drawn as reference lines |
| `fig2_perf_<tag>` | performance in GFLOP/s and effective bandwidth in GB/s vs N, with the *P_peak* and *B_mem* ceilings |
| `fig3_roofline_<tag>` | measured points on the Roofline, with the ridge point marked |

`<tag>` is `allt` (all BLAS threads) or `1t` (one thread), depending on which
runs you have done. The all-threads run shows the highest performance; the
single-thread run is the cleaner evidence for scaling and for the cache
effect, because one core has no thread start-up cost and the smallest
run-to-run spread.

Columns of the printed summary: `N` (A is N x N), the median `time` per call,
`GFLOP/s` (work / time, with work = 2N^2 for matrix-vector and 2N^3 for
matrix-matrix), `GB/s` (compulsory traffic / time; the actual traffic can be
higher) and the arithmetic intensity `I`.

### Which figure answers which homework question

| Question | Evidence |
|---|---|
| Q1: How does time scale with N? | `fig1`: the measured curves run parallel to the `N^2` / `N^3` reference lines |
| Q2: matrix-vector vs matrix-matrix (arithmetic intensity, data reuse) | `fig3` and the `I` column: matrix-vector sits on the memory roof, matrix-matrix far to the right of the ridge point |
| Q3: float64 vs float32 | both precisions are drawn in every figure; compare the two curves |
| Q4: Unexpected results | small-N points in `fig1` and the drop of matrix-vector bandwidth in `fig2b` |

### What the measurements show

* **Scaling.** Above the smallest sizes, time follows the amount of work:
  matrix-vector grows like `N^2` and matrix-matrix like `N^3`, so performance
  (`fig2a`) flattens out. For small N each call has a fixed cost of a few
  microseconds, larger than the arithmetic itself, so the curves are flatter
  than the reference slopes there.
* **Data reuse.** Matrix-vector reads each entry of A once and uses it in a
  single multiply-add, so `I` is 0.25 FLOP/byte (float64) or 0.5 (float32):
  memory-bound, pinned to the sloped part of the Roofline. Matrix-matrix
  reuses every entry N times, so `I` grows with N and the points move into
  the compute-bound region.
* **Cache vs main memory.** Matrix-vector performance falls as the matrix
  outgrows the cache, gradually rather than in one step; the in-cache points
  exceed the main-memory ceiling in `fig2b`, which is exactly what a cache
  is for. Matrix-matrix shows no such drop: BLAS splits it into blocks that
  fit in cache and reuses each block many times.
* **float32 vs float64.** At large N float32 is about 2x faster, for two
  different reasons: matrix-vector moves half the bytes (memory-bound), and
  matrix-matrix fits twice as many values in a SIMD register (compute-bound).

## Measurement methodology

* Arrays are created with `np.full` (constant values, as in the statement)
  **before** timing; outputs are pre-allocated and written with `out=`, so
  neither allocation nor initialisation is inside the timed region.
* Two untimed warm-up calls per configuration (caches, BLAS thread pool).
* Each point is repeated until at least `MIN_REPS` calls and `MIN_TIME`
  seconds have been timed with `time.perf_counter`; the **median** is
  reported (the minimum is stored too).
* `float32` and `float64` are run back-to-back for each `N` so that both see
  the same thermal/clock state of the machine.
* Results are checked against the exact answer (`y_i = 3N`, `C_ij = 2N`)
  with a tolerance-based comparison, never exact equality.
* GFLOP/s = `2N^2 / t` (matrix-vector) or `2N^3 / t` (matrix-matrix).
  Effective bandwidth uses the *compulsory* traffic (each array moved once):
  `(N^2 + 2N) s` (matrix-vector) and `3 N^2 s` (matrix-matrix), with
  `s` = 4 or 8 bytes.

## Where the Roofline limits come from

Nothing has to be configured: both limits are taken from the run itself.

* *B_mem* is the effective bandwidth of the **largest** matrix-vector case.
  That matrix is far too large for any cache, so the data really does come
  from main memory and the operation is purely memory-bound.
* *P_peak* is the **best performance** of the run, which matrix-matrix
  multiplication reaches at large N, where it is compute-bound.

These are *attainable* limits, lower than the theoretical peak of the
machine. That matches the warning in the statement not to assume the
measurements reach the theoretical Roofline: here the roof is what the
hardware actually delivered, and the vertical gap between a point and the
roof is the performance that operation leaves unused.

Two consequences worth discussing in the report:

* The largest matrix-vector points sit *on* the sloped roof by construction;
  that is the definition of the roof, not a result.
* Small matrix-vector cases sit *above* it, because their data fits in cache
  and the cache is faster than main memory. The Roofline drawn here is the
  main-memory one.

`python bandwidth_probe.py` gives an independent check on *B_mem*: it copies
and adds arrays far larger than the cache and prints the bandwidth reached.
Comparing it with the *B_mem* in the figures is a good way to discuss the
difference between *estimated* and *actual* memory traffic.

Results vary between runs, especially on a laptop, where the clock moves with
power and temperature, so the ceilings move a little from run to run too.

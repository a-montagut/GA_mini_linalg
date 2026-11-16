"""Measure the main memory bandwidth with a STREAM-like copy and add.

The arrays are much larger than any CPU cache, so every call has to move the
data from main memory. The result is a measured value for B_MEM, the memory
ceiling of the Roofline in plot_results.py.

Usage:  python bandwidth_probe.py
"""
import time

import numpy as np

N = 2**25        # 32 M doubles = 256 MB per array (needs ~0.8 GB in total)
REPS = 10

a = np.full(N, 1.0)
b = np.full(N, 2.0)
c = np.empty(N)

# arrays moved per call: copy reads a and writes c, add reads a and b
# and writes c
for name, call, moved in (("copy  c = a", lambda: np.copyto(c, a), 2),
                          ("add   c = a + b", lambda: np.add(a, b, out=c), 3)):
    call()                                   # warm-up
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    gbs = moved * N * a.itemsize / min(times) / 1e9
    print(f"{name}: {gbs:5.1f} GB/s")

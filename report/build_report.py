"""Build the technical report in two formats from ONE source of content.

Usage:  python build_report.py
Writes Report_Roofline.tex (and compiles Report_Roofline.pdf with pdflatex,
if available) and Report_Roofline.docx, next to this script.

The text uses a tiny inline markup that both outputs understand:
    **bold**   *italic*   `code`   ^{superscript}   _{subscript}
All numbers come from results/bench_1t.csv and results/bench_allt.csv
(run of 14 Nov 2026); see results/tables.md for the full tables.
"""
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
NAME = "Report_Roofline"

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
TITLE = "Exploring the Roofline Model: matrix–vector vs. matrix–matrix multiplication"
SUBTITLE = "Scientific Programming and Large Scale Computing — Homework 4"
AUTHOR = "Aldana Montagut Barri"

DOC = [
    ("h", "Implementation and methodology"),
    ("p", "Both operations are implemented in NumPy (`kernels.py`): matrix–vector "
          "multiplication **y** = **A** **x** and matrix–matrix multiplication "
          "**C** = **A** **B** are computed with `np.matmul`, writing into a "
          "pre-allocated `out=` array. NumPy passes them to the BLAS routines GEMV "
          "and GEMM of OpenBLAS 0.3.34. The operands are initialised with constants "
          "(`A=1`, `B=2`, `x=3`) and the outputs are allocated beforehand, so "
          "neither allocation nor initialisation is timed. For every size *N* and "
          "precision, `benchmark.py` performs two untimed warm-up calls and then "
          "times calls with `time.perf_counter` until at least 5 repetitions and "
          "0.5 s of work have been collected; the median is reported. float32 and "
          "float64 are run back-to-back for each *N*, so that both see the same "
          "machine state. Every result is checked against the exact value "
          "(*y*_{i} = 3*N*, *C*_{ij} = 2*N*) with a tolerance-based comparison; all "
          "checks passed. The sweep covers *N* = 128–16384 (matrix–vector) and "
          "*N* = 64–4096 (matrix–matrix) in steps of about √2, once with one BLAS "
          "thread and once with all threads, back-to-back on 14 Nov 2026."),
    ("p", "**Two runs, two roles.** The single-thread run is the main evidence for "
          "the scaling and for the cache-versus-main-memory behaviour: one core, no "
          "thread start-up cost and the smallest run-to-run spread. The all-threads "
          "run is used for the peak performance and the Roofline plots. Every "
          "number below states which run it comes from."),
    ("p", "**Machine.** Dell Latitude 7490 on mains power: Intel Core i7-8650U "
          "(4 cores/8 threads, 1.9 GHz base, up to 4.2 GHz turbo, 15 W TDP, AVX2 "
          "with two FMA units per core), dual-channel DDR4-2400, Windows 11, "
          "Python 3.14, NumPy 2.5. In this report **cache** means the 8 MB "
          "last-level (L3) cache shared by the four cores, and **main memory** "
          "means the DDR4 memory behind it. A problem *fits in cache* when its "
          "whole working set (**A**, **x** and **y**) is at most 8 MB."),
    ("p", "*Theoretical roofline.* *P*_{peak} = 4 cores × *f* × 16 (float64) or "
          "32 (float32) FLOP/cycle. The clock *f* is not fixed: it moves between "
          "the 1.9 GHz base and the 4.2 GHz maximum turbo depending on power, "
          "temperature and load, so the compute ceiling is a band: "
          "121.6–268.8 GFLOP/s (float64) and 243.2–537.6 GFLOP/s (float32). "
          "*B*_{mem} = 2 channels × 8 B × 2.4 GT/s = 38.4 GB/s. The ridge points "
          "*I*_{ridge} = *P*_{peak}/*B*_{mem} are 3.2–7.0 (float64) and 6.3–14.0 "
          "(float32) FLOP/byte. *Measured ceilings.* A bandwidth probe that copies "
          "256 MB arrays reaches 20.5 GB/s, and the best matrix–matrix calls (all "
          "threads) reach 99.4 (float64) and 202.4 GFLOP/s (float32)."),

    ("h", "Theoretical arithmetic intensity"),
    ("p", "Let *s* be the size of one element (*s* = 4 B for float32, 8 B for "
          "float64). Each output entry is a dot product of length *N*: *N* "
          "multiplications and *N* − 1 additions, i.e. ≈ 2*N* FLOP (a fused "
          "multiply–add counts as two). The traffic estimate assumes perfect "
          "reuse: every array moves between main memory and the processor exactly "
          "once (the *compulsory* traffic)."),
    ("ul", [
        "**Matrix–vector**: *F* = 2*N*^{2}; *Q* = (*N*^{2} + 2*N*)·*s* (read **A** "
        "and **x**, write **y**); *I* = *F*/*Q* → 2/*s*. Every element of **A** is "
        "used only once, so no reuse can improve this: **A** dominates the traffic "
        "and *I* is fixed at 0.25 (float64) or 0.5 (float32) FLOP/byte, independent "
        "of *N*. Only **x** (a few kB, it stays in cache) is reused. If even **x** "
        "were re-read for every row, *Q* ≈ 2*N*^{2}*s* and *I* ≈ 1/*s*.",
        "**Matrix–matrix**: *F* = 2*N*^{3}; *Q* = 3*N*^{2}*s* (read **A**, **B**, "
        "write **C**); *I* = 2*N*/(3*s*), i.e. *N*/12 (float64) and *N*/6 "
        "(float32). Each element of **A** and **B** is used *N* times, so *I* grows "
        "with *N*. This is an upper bound: it needs all three matrices to stay in "
        "cache. For *N* = 4096 they occupy 400 MB (float64), so BLAS splits the "
        "matrices into cache-sized blocks of size *b* and re-reads each block "
        "about *N*/*b* times, giving *I* ~ *b*/*s*. With no reuse at all (both "
        "operands fetched for every multiply–add) *I* would fall to ≈ 1/*s*, the "
        "same as matrix–vector.",
    ]),
    ("table", {
        "spec": "lccccc",
        "header": ["", "FLOP", "bytes (compulsory)", "*I* float32", "*I* float64",
                   "*I* at *N* = 4096 (float32 / float64)"],
        "rows": [
            ["Matrix–vector **y** = **A** **x**", "2*N*^{2}", "(*N*^{2} + 2*N*)·*s*",
             "→ 0.5", "→ 0.25", "0.50 / 0.25"],
            ["Matrix–matrix **C** = **A** **B**", "2*N*^{3}", "3*N*^{2}·*s*",
             "*N*/6", "*N*/12", "683 / 341"],
        ],
        "caption": "Theoretical FLOP count, compulsory memory traffic and "
                   "arithmetic intensity (FLOP/byte).",
    }),
    ("p", "Compared with the ridge points, matrix–vector multiplication "
          "(*I* = 0.25–0.5 ≪ 3.2–14) is *memory-bound* for every *N*: its ceiling is "
          "*B*·*I*, i.e. 9.6 (float64) / 19.2 GFLOP/s (float32) with the theoretical "
          "bandwidth and 5.1 / 10.3 GFLOP/s with the measured 20.5 GB/s. "
          "Matrix–matrix multiplication crosses the ridge at *N* ≈ 38 (base clock) "
          "to 84 (maximum turbo), and is therefore *compute-bound* for practically "
          "all tested sizes. Halving *s* doubles *I* for both operations and "
          "doubles *P*_{peak} (twice as many values per AVX register), so the ideal "
          "float32 speed-up is 2× in both regimes."),

    ("h", "Results"),
    ("fig", "fig1_time_speedup",
     "(a) Median time per call, all threads; dotted lines show *N*^{2} and "
     "*N*^{3}. (b) Ratio *t*_{float64}/*t*_{float32} (solid: all threads, dashed: "
     "one thread)."),
    ("fig", "fig2_perf_bandwidth",
     "(a) Achieved GFLOP/s, all threads (faint lines: one thread); the shaded bands "
     "are the compute ceilings between base clock and maximum turbo. (b) Effective "
     "bandwidth computed from the compulsory traffic *Q* (all threads); vertical "
     "lines mark where **A** no longer fits in the 8 MB cache (*N* ≈ 1024 float64, "
     "1448 float32)."),
    ("table", {
        "spec": "p{7.4cm}cc",
        "header": ["", "1 thread", "all threads"],
        "rows": [
            ["Matrix–vector from cache, best bandwidth (float32 / float64)",
             "43.4 / 38.3 GB/s", "52.1 / 53.3 GB/s"],
            ["Matrix–vector from main memory, *N* ≥ 4096: bandwidth (float32 / "
             "float64)", "18.9–21.2 / 17.8–18.9 GB/s", "17.1–22.2 / 15.0–17.3 GB/s"],
            ["Matrix–vector from main memory, *N* ≥ 4096: performance (float32 / "
             "float64)", "9.5–10.6 / 4.5–4.7 GFLOP/s", "8.6–11.1 / 3.7–4.3 GFLOP/s"],
            ["Matrix–matrix, saturated performance (float32 / float64)",
             "75–91 / 38–44 GFLOP/s", "163–202 / 73–99 GFLOP/s"],
            ["Fitted exponent *p* in *t* ∝ *N*^{p}: matrix–vector from main memory "
             "(float32 / float64)", "2.21 / 2.17", "1.94 / 2.01"],
            ["Fitted exponent *p*: matrix–matrix, *N* ≥ 256 (float32 / float64)",
             "3.01 / 3.01", "2.75 / 2.94"],
        ],
        "caption": "Key measured values. Saturated matrix–matrix performance: "
                   "*N* ≥ 256 (1 thread), *N* ≥ 724 / 512 (all threads, float32 / "
                   "float64). Full tables: `results/tables.md`.",
    }),
    ("p", "**Matrix–vector** (Fig. 2, blue; Table 2). Two regimes appear. While the "
          "working set fits in cache, the effective bandwidth reaches 35–43 GB/s "
          "with one thread and up to 53 GB/s with all threads, i.e. above the "
          "theoretical main-memory bandwidth, which is possible because the data "
          "does not come from main memory. Once the matrix is larger than the "
          "cache, the bandwidth falls to a plateau of 18–21 GB/s (one thread): "
          "49–55 % of the theoretical 38.4 GB/s and close to the 20.5 GB/s of the "
          "bandwidth probe. The performance follows the bandwidth: about "
          "10 GFLOP/s (float32) and 4.5 GFLOP/s (float64) with one thread, only "
          "about 4 % of the base-clock peak. With one thread the transition is "
          "gradual rather than a single step: float64 falls from 9.4 GFLOP/s at "
          "*N* = 724 (4 MB) to 7.7 (8 MB), 6.2 (16 MB), 5.2 (32 MB) and 4.5 GFLOP/s "
          "at 128 MB. Using all four cores does not help once the data is in main "
          "memory (speed-up 0.9–1.2× for *N* ≥ 4096), because a single core already "
          "uses most of the available bandwidth."),
    ("p", "**Matrix–matrix** (red). After a ramp-up for small *N*, the performance "
          "saturates. With one thread it stays at 75–91 GFLOP/s (float32) and "
          "38–44 GFLOP/s (float64) from *N* = 256 upwards; with all threads at "
          "163–202 GFLOP/s (float32) and 73–99 GFLOP/s (float64), i.e. 67–83 % and "
          "60–82 % of the base-clock peak. Four cores give only 2.0–2.7× the "
          "single-core performance, not 4×, because the 15 W processor lowers its "
          "clock when all cores are busy. Unlike matrix–vector, matrix–matrix still "
          "gains from more cores, because it is limited by arithmetic and not by "
          "the shared memory bus. The bandwidth implied by the compulsory traffic "
          "falls as 1/*N* (from about 5 to 0.3 GB/s, Fig. 2b), far below what main "
          "memory can deliver."),
    ("fig", "fig3_roofline",
     "Roofline plots (all threads) with the theoretical ceilings at base clock "
     "(solid) and maximum turbo (dashed, shaded band in between), the measured "
     "ceiling (dash-dot), and the bandwidth reached by matrix–vector from cache "
     "(dotted). The measurements are placed at their theoretical intensity. Hollow "
     "circles: data fits in cache."),
    ("p", "**Roofline interpretation** (Fig. 3). Matrix–vector sits on the bandwidth "
          "slope, far left of the ridge. Its main-memory points (filled) lie close "
          "to the measured ceiling (some float32 points slightly above it, see "
          "Section 4), and its cache points (hollow) lie above the main-memory "
          "roof. This is expected: the actual intensity depends on the memory "
          "level considered, and when the data lives in cache the main-memory "
          "bandwidth is the wrong ceiling. Matrix–matrix sits right of the ridge "
          "and approaches the flat compute roof as *N* grows. Small products "
          "(*N* ≤ 128) lie well below both roofs: they are limited by fixed "
          "overheads (Python/NumPy call, thread start-up, packing of blocks), which "
          "the Roofline model does not include. The measurements are therefore "
          "consistent with matrix–vector being memory-bound and matrix–matrix being "
          "compute-bound. The difference is data reuse: each element of **A** is "
          "used once in matrix–vector multiplication but *N* times in matrix–matrix "
          "multiplication, which lets the latter keep its operands in registers and "
          "cache and hide the cost of memory transfers."),

    ("h", "Discussion"),
    ("p", "**1. Scaling of the execution time.** With one thread the fitted "
          "exponents are *t* ∝ *N*^{2.21} (float32) and *N*^{2.17} (float64) for "
          "matrix–vector with the data in main memory, and *N*^{3.01} for "
          "matrix–matrix in both precisions (*N* = 256–4096). This matches the "
          "expected *O*(*N*^{2}) and *O*(*N*^{3}) work: the time grows linearly with "
          "the number of FLOPs, so the performance stays constant. The "
          "matrix–vector exponent is slightly above 2 because its performance "
          "still falls between 8 and 64 MB (the gradual cache-to-main-memory "
          "transition). With all threads the exponents are 1.94 / 2.01 "
          "(matrix–vector) and 2.75 / 2.94 (matrix–matrix); the low float32 value "
          "reflects the noisier clock behaviour of that run. At small *N* the time "
          "grows more slowly than *N*^{2} (fitted 1.69–1.76 while the data fits in "
          "cache) because each call has a fixed cost of a few μs. Since "
          "matrix–vector multiplication is memory-bound, its time is really "
          "proportional to the bytes moved (*N*^{2}*s*), which also scales as "
          "*N*^{2}."),
    ("p", "**2. Matrix–vector vs. matrix–matrix.** Matrix–matrix achieves about 9× "
          "more FLOP/s with one thread (float64: ≈ 42 vs. ≈ 4.5 GFLOP/s; float32: "
          "≈ 89 vs. ≈ 10 GFLOP/s) and about 20× with all threads (float64: ≈ 90 vs. "
          "≈ 4.1 GFLOP/s; float32: ≈ 185 vs. ≈ 10 GFLOP/s). Matrix–vector has "
          "*I* ≤ 0.5 FLOP/byte because each matrix element is loaded, used once and "
          "discarded, so the core spends most of its time waiting for memory. "
          "Matrix–matrix has *I* = *O*(*N*): each loaded element takes part in *N* "
          "multiply–adds, so after blocking the traffic per FLOP is tiny and the "
          "FMA units are kept busy. With one thread, a float64 matrix–matrix "
          "product with *N* = 1024 (2.1 GFLOP) takes 54 ms, while a matrix–vector "
          "product with *N* = 11585 (0.27 GFLOP) takes 59 ms: eight times more work "
          "in slightly less time."),
    ("p", "**3. float64 → float32.** For large problems float32 is about 2× faster "
          "(Fig. 1b), so the expected benefit is observed. For matrix–matrix "
          "(compute-bound) the speed-up comes from doubling the SIMD width (8 "
          "instead of 4 values per AVX register); the measured ratio is 1.9–2.3× "
          "with one thread for *N* ≥ 256 and 2.0–2.2× with all threads for "
          "*N* ≥ 724. For matrix–vector (memory-bound) it comes from moving half as "
          "many bytes. Ideally this gives 2×, but the measured ratio is 2.1–2.8× "
          "(one thread, *N* ≥ 1024): float32 also sustains a slightly higher "
          "bandwidth (median 21.0 vs. 18.9 GB/s from main memory), and at the same "
          "*N* a float32 matrix is half as large, so it stays in cache up to a "
          "larger *N* (8 MB is reached at *N* = 1448 instead of 1024). For small "
          "*N* the ratio drops to 1.5–1.9× (one thread, *N* ≤ 256): the run time is "
          "dominated by fixed overheads that do not depend on the precision. The "
          "price of float32 is accuracy (ε ≈ 1.2·10^{−7} vs. 2.2·10^{−16}). Here it "
          "is invisible because the constant inputs give exactly representable sums "
          "(3*N*, 2*N* < 2^{24}), but in general rounding errors in long dot "
          "products grow with *N*, and with several threads the order of the "
          "additions changes, so the last digits can differ between the "
          "single-thread and the all-threads result."),
    ("p", "**4. Unexpected results.** (i) *Large run-to-run variation.* The "
          "benchmark was run five times over two days. The best all-threads "
          "float32 matrix–matrix result ranged from 174 to 312 GFLOP/s, i.e. from "
          "72 % to 128 % of the base-clock peak. A value above the base-clock peak "
          "is only possible because the CPU runs in turbo mode; the clock depends "
          "on the power plan, the temperature and the background load. This is why "
          "the compute ceiling is drawn as a band, and why the more stable "
          "single-thread run (spread below 15 % for most points) is the main "
          "evidence. The final run used the Windows *Balanced* power plan with "
          "about 20 % background CPU load from other applications. (ii) Four cores "
          "give only 2.0–2.7× for matrix–matrix: the 15 W power limit lowers the "
          "all-core clock. (iii) The main-memory bandwidth reached by "
          "matrix–vector is only 49–55 % of the theoretical 38.4 GB/s, as is "
          "usual: the theoretical value ignores memory refresh, page misses and "
          "limited prefetching per core. Some all-threads float32 points (up to "
          "22.2 GB/s) even exceed the measured 20.5 GB/s, because the probe runs on "
          "a single core and counts 16 B per copied element, although the write "
          "may cost an extra read (estimated vs. actual traffic). (iv) There is no "
          "sharp step at 8 MB: the transition spreads over 4–64 MB. The cache is "
          "shared with the operating system and other programs, hardware "
          "prefetching hides part of the main-memory latency, and part of the "
          "matrix can stay in cache between calls. (v) Isolated dips in the "
          "all-threads run near the cache size (e.g. float64 matrix–vector at "
          "*N* = 724: 7.1 GFLOP/s against 13.3 at *N* = 512, with a 41 % spread; "
          "the float64/float32 ratio of 5.2 at *N* = 1024 in Fig. 1b) are absent "
          "with one thread and come from OpenBLAS thread start-up and work "
          "splitting, plus noise. (vi) The estimated traffic used in the plots is a "
          "lower bound: the actual traffic of matrix–matrix is larger (blocks are "
          "re-read and packed), and when the matrix–vector data are in cache the "
          "traffic does not reach main memory at all. This is why the cache points "
          "can exceed the main-memory roof, and why the Roofline is used as an "
          "upper bound rather than a prediction."),
]

# ---------------------------------------------------------------------------
# Inline markup
# ---------------------------------------------------------------------------
TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`|\^\{.+?\}|_\{.+?\})")


def parse(text):
    """Split marked-up text into (kind, text) pieces."""
    out = []
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**"):
            out.append(("b", part[2:-2]))
        elif part.startswith("*"):
            out.append(("i", part[1:-1]))
        elif part.startswith("`"):
            out.append(("c", part[1:-1]))
        elif part.startswith("^{"):
            out.append(("sup", part[2:-1]))
        elif part.startswith("_{"):
            out.append(("sub", part[2:-1]))
        else:
            out.append(("t", part))
    return out


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------
LATEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
             "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
             "~": r"\textasciitilde{}", "^": r"\^{}"}
UNICODE_MATH = {"≈": r"\approx", "≪": r"\ll", "→": r"\to", "×": r"\times",
                "∝": r"\propto", "ε": r"\varepsilon", "≥": r"\geq", "≤": r"\leq",
                "√": r"\surd", "μ": r"\mu", "−": "-", "·": r"\cdot"}


def tex_escape(s):
    return "".join(LATEX_ESC.get(ch, ch) for ch in s)


def tex(text):
    res = []
    for kind, s in parse(text):
        e = tex_escape(s)
        res.append({"t": e, "b": rf"\textbf{{{e}}}", "i": rf"\emph{{{e}}}",
                    "c": rf"\texttt{{{e}}}", "sup": rf"\textsuperscript{{{e}}}",
                    "sub": rf"\textsubscript{{{e}}}"}[kind])
    return "".join(res)


def build_tex():
    uni = "\n".join(rf"\newunicodechar{{{k}}}{{\ensuremath{{{v}}}}}"
                    for k, v in UNICODE_MATH.items())
    L = [r"\documentclass[10pt,a4paper]{article}",
         r"\usepackage[margin=1.8cm]{geometry}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{lmodern,amsmath,amssymb,booktabs,graphicx,caption,enumitem}",
         r"\usepackage{newunicodechar}",
         r"\usepackage[hidelinks]{hyperref}",
         r"\graphicspath{{../figures/}}",
         r"\captionsetup{font=small,labelfont=bf}",
         r"\setlist{nosep,leftmargin=*}",
         r"\setlength{\parskip}{3pt}",
         uni,
         rf"\title{{\vspace{{-1.4cm}}{tex(TITLE)}\vspace{{-0.3cm}}}}",
         rf"\author{{{tex(SUBTITLE)} \\ \small {tex(AUTHOR)}}}",
         r"\date{\vspace{-0.6cm}}",
         r"\begin{document}", r"\maketitle", ""]
    for block in DOC:
        kind = block[0]
        if kind == "h":
            L.append(rf"\section{{{tex(block[1])}}}")
        elif kind == "p":
            L += [tex(block[1]), ""]
        elif kind == "ul":
            L.append(r"\begin{itemize}")
            L += [rf"\item {tex(it)}" for it in block[1]]
            L.append(r"\end{itemize}")
        elif kind == "table":
            t = block[1]
            L += [r"\begin{table}[h]\centering\small",
                  rf"\begin{{tabular}}{{{t['spec']}}}\toprule",
                  " & ".join(tex(c) for c in t["header"]) + r" \\\midrule"]
            L += [" & ".join(tex(c) for c in r) + r" \\" for r in t["rows"]]
            L += [r"\bottomrule\end{tabular}", rf"\caption{{{tex(t['caption'])}}}",
                  r"\end{table}", ""]
        elif kind == "fig":
            L += [r"\begin{figure}[h]\centering",
                  rf"\includegraphics[width=\textwidth]{{{block[1]}.pdf}}\vspace{{-2mm}}",
                  rf"\caption{{{tex(block[2])}}}", r"\end{figure}", ""]
    L.append(r"\end{document}")
    path = os.path.join(HERE, NAME + ".tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path


def compile_pdf():
    if shutil.which("pdflatex") is None:
        print("pdflatex not found: PDF not rebuilt")
        return
    for _ in range(2):  # twice, for figure/table numbers
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", NAME + ".tex"],
                           cwd=HERE, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit("pdflatex failed")
    for ext in (".aux", ".log", ".out"):
        p = os.path.join(HERE, NAME + ext)
        if os.path.exists(p):
            os.remove(p)
    print("Wrote", os.path.relpath(os.path.join(HERE, NAME + ".pdf"), HERE))


# ---------------------------------------------------------------------------
# Word output
# ---------------------------------------------------------------------------
def build_docx():
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt

    d = docx.Document()
    for s in d.sections:
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Cm(1.8)
    d.styles["Normal"].font.name = "Calibri"
    d.styles["Normal"].font.size = Pt(10)

    def runs(par, text, size=None, bold_all=False):
        for kind, s in parse(text):
            r = par.add_run(s)
            r.bold = True if (kind == "b" or bold_all) else None
            r.italic = True if kind == "i" else None
            if kind == "c":
                r.font.name = "Consolas"
                r.font.size = Pt(9)
            elif size:
                r.font.size = Pt(size)
            r.font.superscript = kind == "sup" or None
            r.font.subscript = kind == "sub" or None
        return par

    def caption(label, text, align):
        p = d.add_paragraph()
        p.alignment = align
        r = p.add_run(label + " ")
        r.bold = True
        r.font.size = Pt(8.5)
        runs(p, text, size=8.5)

    d.add_paragraph(TITLE, style="Title").alignment = WD_ALIGN_PARAGRAPH.CENTER
    for line in (SUBTITLE, AUTHOR):
        d.add_paragraph(line).alignment = WD_ALIGN_PARAGRAPH.CENTER

    sec = tab = fig = 0
    for block in DOC:
        kind = block[0]
        if kind == "h":
            sec += 1
            d.add_heading(f"{sec}. {block[1]}", level=1)
        elif kind == "p":
            runs(d.add_paragraph(), block[1]).alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind == "ul":
            for it in block[1]:
                p = runs(d.add_paragraph(style="List Bullet"), it)
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind == "table":
            t = block[1]
            tab += 1
            table = d.add_table(rows=1 + len(t["rows"]), cols=len(t["header"]))
            table.style = "Table Grid"
            for i, row in enumerate([t["header"]] + t["rows"]):
                for j, cell in enumerate(row):
                    p = table.cell(i, j).paragraphs[0]
                    runs(p, cell, size=9, bold_all=(i == 0))
            caption(f"Table {tab}.", t["caption"], WD_ALIGN_PARAGRAPH.CENTER)
        elif kind == "fig":
            fig += 1
            p = d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(os.path.join(FIG, block[1] + ".png"), width=Cm(17))
            caption(f"Figure {fig}.", block[2], WD_ALIGN_PARAGRAPH.JUSTIFY)
    path = os.path.join(HERE, NAME + ".docx")
    d.core_properties.author = AUTHOR
    d.core_properties.last_modified_by = AUTHOR
    d.core_properties.title = TITLE
    d.save(path)
    print("Wrote", os.path.relpath(path, HERE))


if __name__ == "__main__":
    print("Wrote", os.path.relpath(build_tex(), HERE))
    compile_pdf()
    build_docx()

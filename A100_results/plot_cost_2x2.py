"""2x2 cost figure from cost_ptyrad_mliss_A100.csv: columns = batch 1 | 32,
rows = forward+adjoint time (ms) | peak memory (GB). Five series with
ML-ISS in place of ISS + line search. Output: cost_ptyrad_2x2_mliss_A100.png"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = []
with open("/home/dnz75396/reproduce/cost_ptyrad_mliss_A100.csv") as f:
    rd = csv.reader(f); next(rd)
    for r in rd:
        rows.append((r[0], int(r[1]), int(r[2]), float(r[3]), float(r[4])))

SERIES = [
    ("multislice", "#2a78d6", "o", "multislice"),
    ("ISS, parallel", "#e8590c", "s", "ISS, parallel"),
    ("ISS, low memory (chunk 1)", "#1baf7a", "^", "ISS, low memory"),
    ("ISS + line search", "#eda100", "D", "ML-ISS"),
]
BATCHES = (1, 32)
fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.0), dpi=200)
for col, B in enumerate(BATCHES):
    for rown, (key, ylab) in enumerate([(3, "forward + adjoint (ms per batch)"),
                                        (4, "peak allocation (GB)")]):
        ax = axes[rown, col]
        for name, colr, mk, lab in SERIES:
            pts = sorted((r[2], r[key]) for r in rows if r[0] == name and r[1] == B
                         and np.isfinite(r[key]))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=colr, marker=mk, ms=6, lw=2.0, label=lab)
        # Table-2 order predictions (per probe mode), dotted, scaled to the
        # measured value at N = 64 (the least overhead-contaminated point)
        PRED = {
            "multislice": (lambda n, b: (4 * n - 2) * b, lambda n, b: n * b),
            "ISS, parallel": (lambda n, b: 2 * n * b + 2 * n + 2, lambda n, b: n * b + n),
            "ISS, low memory (chunk 1)": (lambda n, b: 2 * n * b + 2 * n + 2, lambda n, b: n),
            "ISS + line search": (lambda n, b: 4 * n * b + 3 * n + 3, lambda n, b: n * b + n),
        }
        for name, colr, mk, lab in SERIES:
            fn = PRED[name][0 if key == 3 else 1]
            pts = sorted((r[2], r[key]) for r in rows if r[0] == name and r[1] == B
                         and np.isfinite(r[key]))
            if not pts:
                continue
            xs = [q[0] for q in pts]
            anchor_x, anchor_y = pts[-1]  # uniform rule: anchor every curve at N = 64
            scale = anchor_y / fn(anchor_x, B)
            ax.plot(xs, [scale * fn(x, B) for x in xs], ":", color=colr, lw=1.4, alpha=0.9)
        ax.set_xscale("log", base=2)
        xt = sorted({r[2] for r in rows if r[1] == B})
        ax.set_xticks(xt); ax.set_xticklabels([str(x) for x in xt])
        ax.set_ylim(bottom=0)
        if rown == 0:
            ax.set_title(f"batch {B}", fontsize=12)
        if rown == 1:
            ax.set_xlabel("slices $N$", fontsize=11)
        if col == 0:
            ax.set_ylabel(ylab, fontsize=11)
        ax.grid(True, color="#e8e7de", lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
from matplotlib.lines import Line2D
TF = {
    "multislice": r"$(4N{-}2)\,|\mathcal{B}|$",
    "ISS, parallel": r"$2N|\mathcal{B}|{+}2N{+}2$",
    "ISS, low memory (chunk 1)": r"$2N|\mathcal{B}|{+}2N{+}2$",
    "ISS + line search": r"$4N|\mathcal{B}|{+}3N{+}3$",
}
FLD = {
    "multislice": r"$N|\mathcal{B}|$",
    "ISS, parallel": r"$N|\mathcal{B}|{+}N$",
    "ISS, low memory (chunk 1)": r"$N$",
    "ISS + line search": r"$N|\mathcal{B}|{+}N$",
}
blank = lambda: Line2D([], [], color="none")
hdr = lambda t: (blank(), t)
col1 = [hdr("series")] + [
    (Line2D([], [], color=c, marker=mk, ms=6, lw=2.0), lab) for _, c, mk, lab in SERIES
]
col2 = [hdr("transforms")] + [
    (Line2D([], [], color=c, ls=":", lw=1.6), TF[name]) for name, c, _, _ in SERIES
]
col3 = [hdr("stored fields")] + [(blank(), FLD[name]) for name, c, _, _ in SERIES]
entries = col1 + col2 + col3
fig.legend(
    [h for h, _ in entries], [t for _, t in entries],
    frameon=False, fontsize=9, loc="lower center", ncol=3,
    columnspacing=2.5, handlelength=1.8, handletextpad=0.6, labelspacing=0.5,
)
fig.suptitle("Forward + adjoint cost (128x128 pixel frames, 6 probe modes), A100 80GB PCIe\n"
             "dotted: Table-2 order predictions, one scale factor per curve, anchored at N = 64",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0.145, 1, 0.95])
fig.savefig("/home/dnz75396/reproduce/cost_ptyrad_2x2_mliss_A100.png", facecolor="white")
fig.savefig("/home/dnz75396/reproduce/cost_ptyrad_2x2_mliss_A100.pdf", facecolor="white")
print("saved cost_ptyrad_2x2_mliss_A100.png + .pdf")

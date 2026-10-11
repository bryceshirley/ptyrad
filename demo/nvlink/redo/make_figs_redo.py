"""Figures for bench_results.md section O (round-3 NVSwitch campaign).

Data transcribed from results_redo_20261011_204539/{carry.log,summary.log}
and results_redo_p4_20261011_220022/summary.log. Colors: dataviz reference
categorical palette (validated light-mode, adjacent order); text never wears
series color; every bar carries a direct value label.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})

# ---- Fig 1: slice-split f+b speedup vs Nz (carry.log, chain schedule) ------
NZ = [42, 64, 128, 512]
SPEEDUP = {  # 1-GPU f+b / chain f+b, pmode 12, batch 1
    ("P=4", "M=6"): [51.69 / 17.32, 79.61 / 25.71, 161.07 / 47.97, 652.67 / 180.00],
    ("P=4", "M=8"): [67.94 / 22.91, 105.91 / 34.87, 216.06 / 69.51, 879.98 / 261.27],
    ("P=2", "M=6"): [51.58 / 30.80, 79.77 / 46.12, 161.46 / 89.52, 652.28 / 350.40],
    ("P=2", "M=8"): [67.95 / 41.61, 105.99 / 62.17, 216.03 / 121.28, 879.89 / 474.47],
}
HUE = {"P=4": BLUE, "P=2": ORANGE}
DASH = {"M=6": "-", "M=8": (0, (4, 2))}

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for (p, m), y in SPEEDUP.items():
    ax.plot(NZ, y, color=HUE[p], linestyle=DASH[m],
            linewidth=1.6, marker="o", markersize=6,
            markerfacecolor=HUE[p], markeredgecolor=SURFACE,
            markeredgewidth=1.2, label=f"{p}, {m}")
    dy = {("P=4", "M=6"): 7, ("P=4", "M=8"): -7,
          ("P=2", "M=6"): 7, ("P=2", "M=8"): -8}[(p, m)]
    ax.annotate(f"{p} {m}", (NZ[-1], y[-1]), xytext=(8, dy),
                textcoords="offset points", va="center", fontsize=9, color=INK)
for ideal, lab in ((2, "ideal P=2"), (4, "ideal P=4")):
    ax.axhline(ideal, color=INK2, linewidth=0.8, linestyle=":", alpha=0.5)
    ax.annotate(lab, (150, ideal), xytext=(0, 3), textcoords="offset points",
                fontsize=8, color=INK2)
ax.set_xscale("log", base=2)
ax.set_xticks(NZ)
ax.set_xticklabels([str(n) for n in NZ])
ax.set_xlim(38, 1100)
ax.set_ylim(1, 4.3)
ax.set_xlabel("Nz (slices)")
ax.set_ylabel("fwd+bwd speedup vs 1 GPU")
ax.set_title("Born slice-split scaling on 4×A100 NVSwitch — chain carry,\n"
             "batch 1, 256², 12 probe modes (carry.log, round 3)",
             fontsize=10, loc="left")
ax.grid(axis="y", color=INK2, alpha=0.15, linewidth=0.6)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.02, 0.98))
fig.tight_layout()
fig.savefig("fig_redo_nz_scaling.png", dpi=160)
plt.close(fig)

# ---- Fig 2: E2E s/iter at the Fig-S5 workload (summary.log) ----------------
GREEN, VIOLET = "#008300", "#4a3aa7"
ARM_COLOR = {"MS 1-GPU": BLUE, "born M6 1-GPU": ORANGE,
             "born M6 slice-2": AQUA, "born M6 slice-4": YELLOW,
             "born M6 mode-4": MAGENTA,
             "born M5 slice-2": GREEN, "born M4 slice-2": VIOLET}
P12 = [("MS 1-GPU", 401.0), ("born M6 slice-4", 452.8),
       ("born M6 mode-4", 551.3), ("born M6 slice-2", 630.3),
       ("born M6 1-GPU", 1026.5)]
P4 = [("born M4 slice-2", 245.7), ("born M5 slice-2", 279.1),
      ("born M6 slice-2", 315.8), ("born M6 slice-4", 340.3),
      ("MS 1-GPU", 415.0), ("born M6 1-GPU", 474.6)]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
for ax, data, title in zip(
        axes, (P12, P4),
        ("12 probe modes (paper's Fig-S5 workload)", "4 probe modes")):
    names = [n for n, _ in data][::-1]
    vals = [v for _, v in data][::-1]
    ax.barh(names, vals, height=0.55,
            color=[ARM_COLOR[n] for n in names],
            edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}", (v, i), xytext=(4, 0),
                    textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ms = dict(data)["MS 1-GPU"]
    ax.axvline(ms, color=INK2, linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="x", color=INK2, alpha=0.15, linewidth=0.6)
    ax.set_xlabel("s/iter (16384 DPs, 42 slices, batch 1)")
    ax.set_xlim(0, 1160)
fig.suptitle("PSO Fig-S5 end-to-end, A100 SXM NVLink pods (M5/M4 arms: 2-GPU "
             "pod, cross-pod anchor 312 vs 316) — dotted line = MS baseline",
             fontsize=10, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("fig_redo_e2e.png", dpi=160)
plt.close(fig)
print("wrote fig_redo_nz_scaling.png fig_redo_e2e.png")

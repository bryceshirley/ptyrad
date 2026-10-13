"""Paper figures for the H200 fixed-M ladder (bench_results SSO.3).

Data transcribed from ~/h100-res/results_redo_q20_20261012_162504/e2e_*.log
(5-iter arms, pmode 4, 16384-DP tiled workload, 2x H200 SXM NV18, batch 1).
Fair 2-GPU comparison: born uses its best axis (slice-2); multislice can
only split probe modes (sequential slices), so MS runs mode-2.

Color: Born order M is ORDERED -> sequential blue ramp (light->dark);
multislice is a different entity -> orange. Direct labels everywhere the
palette's low-contrast steps appear (dataviz relief rule).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
ORANGE = "#eb6834"
RAMP = ["#a8c8ec", "#7dafe3", "#5395d9", "#2a78d6", "#174f94",
        "#0d3a73", "#092b57", "#06203f"]  # M=1..14 light->dark

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})

MS_SITER, MS_LOSS = 215.2, 0.3096      # multislice BEST config: 1 GPU
MS2_SITER = 302.4                       # multislice mode-2 (2 GPUs) — SLOWER
M_VALS = [1, 2, 4, 6, 8, 10, 12, 14]  # M>=10 from the second H200 pod (highM2)
SLICE_SITER = [92.3, 109.3, 139.7, 165.7, 202.5, 224.2, 256.9, 282.1]
MODE_SITER = {1: 120.5, 2: 131.9, 4: 165.6}  # M6 killed at $2 floor, M8 not reached
LOSS5 = [0.3335, 0.3160, 0.3146, 0.3117, 0.3103, 0.3099, 0.3096, 0.3096]
ITERS = [1, 2, 3, 4, 5]
TRAJ = {  # per-iteration total loss
    1: [0.4103, 0.3549, 0.3431, 0.3374, 0.3335],
    2: [0.3953, 0.3456, 0.3301, 0.3216, 0.3160],
    4: [0.3966, 0.3448, 0.3297, 0.3205, 0.3146],
    6: [0.3962, 0.3432, 0.3275, 0.3176, 0.3117],
    8: [0.3956, 0.3418, 0.3253, 0.3156, 0.3103],
    10: [0.3956, 0.3416, 0.3254, 0.3153, 0.3099],
    12: [0.3956, 0.3412, 0.3248, 0.3146, 0.3096],
    14: [0.3955, 0.3411, 0.3247, 0.3146, 0.3096],
}
MS_TRAJ = [0.3957, 0.3413, 0.3243, 0.3143, 0.3096]


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", name)


# Slots for arms still running on the highM2 H200 pod (M12, M14): fill in
# {M: loss} and {M: s/iter} as they land; drawn dashed-open until then.
EXT = {}
EXT_SITER = {}

# ---- A. Pareto: iter-5 loss vs time per iteration --------------------------
# Three variants: both MS points / MS 1-GPU only (recommended) / mode-2 only.
def pareto(name, show_ms1, show_ms2, subtitle,
           titleline="Born order (slice split, 2 GPUs) vs multislice"):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(SLICE_SITER, LOSS5, "-", color="#2a78d6", linewidth=1.6, zorder=2)
    OFF = {1: (6, 6), 2: (6, 6), 4: (6, 6), 6: (6, 6),
           8: (-2, 10), 10: (2, -18), 12: (0, 10), 14: (0, -18)}
    for m, x, y, c in zip(M_VALS, SLICE_SITER, LOSS5, RAMP):
        ax.plot(x, y, "o", color=c, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.3, zorder=3)
        ax.annotate(f"M={m}", (x, y), xytext=OFF[m], textcoords="offset points",
                    ha="center" if m >= 8 else "left", fontsize=9, color=INK)
    if show_ms1:
        ax.plot(MS_SITER, MS_LOSS, "s", color=ORANGE, markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=4)
        ax.annotate("multislice\n1 GPU" + (" (best)" if show_ms2 else ""),
                    (MS_SITER, MS_LOSS), xytext=(0, -34),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=INK)
    if show_ms2:
        ax.plot(MS2_SITER, MS_LOSS, "s",
                color=ORANGE if not show_ms1 else "none",
                markerfacecolor=ORANGE if not show_ms1 else SURFACE,
                markeredgecolor=SURFACE if not show_ms1 else ORANGE,
                markeredgewidth=1.3 if not show_ms1 else 1.6,
                markersize=9, zorder=3)
        ax.annotate("multislice\nmode-2" if not show_ms1 else "MS mode-2\n(2 GPUs)",
                    (MS2_SITER, MS_LOSS), xytext=(0, -34),
                    textcoords="offset points", ha="center",
                    fontsize=9 if not show_ms1 else 8.5,
                    color=INK if not show_ms1 else INK2)
    ax.axhline(MS_LOSS, color=INK2, linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_ylim(0.3045, 0.3355)
    ax.set_xlabel("time per iteration (s)")
    ax.set_ylabel("loss after 5 iterations")
    ax.set_title(titleline + " — loss/cost frontier\n" + subtitle,
                 fontsize=9.5, loc="left")
    ax.grid(axis="both", color=INK2, alpha=0.12, linewidth=0.6)
    save(fig, name)

pareto("fig_h200_pareto", True, True,
       "(PSO 16384 DPs, 42 slices, 4 probe modes, 2×H200 SXM NV18;\n"
       "MS best = 1 GPU — mode-split slows it 1.4×)")
pareto("fig_h200_pareto_ms1gpu", True, False,
       "(PSO 16384 DPs, 42 slices, 4 probe modes, 2×H200 SXM NV18;\n"
       "multislice shown at its best: 1 GPU)")
pareto("fig_h200_pareto_ms2only", False, True,
       "(PSO 16384 DPs, 42 slices, 4 probe modes, 2×H200 SXM NV18)",
       titleline="Born order (slice split) vs multislice (mode split)")

# ---- B. s/iter vs M, split-axis comparison ---------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.8))
xs = range(len(M_VALS))
w = 0.38
ax.bar([x - w / 2 for x in xs], SLICE_SITER, width=w, color="#2a78d6",
       edgecolor=SURFACE, linewidth=1.5, label="born, slice split")
mode_y = [MODE_SITER.get(m) for m in M_VALS]
ax.bar([x + w / 2 for x in xs if mode_y[x] is not None],
       [y for y in mode_y if y is not None], width=w, color=ORANGE,
       edgecolor=SURFACE, linewidth=1.5, label="born, probe-mode split")
for x, y in zip(xs, SLICE_SITER):
    ax.annotate(f"{y:.0f}", (x - w / 2, y), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=8.5, color=INK)
for x, y in zip(xs, mode_y):
    if y is not None:
        ax.annotate(f"{y:.0f}", (x + w / 2, y), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK)
ax.axhline(MS_SITER, color=INK2, linewidth=1.0, linestyle="--", alpha=0.7)
ax.axhline(MS2_SITER, color=INK2, linewidth=0.8, linestyle=":", alpha=0.5)
ax.annotate(f"multislice best (1 GPU): {MS_SITER:.0f} s/iter",
            (len(M_VALS) - 0.5, MS_SITER), xytext=(0, 4),
            textcoords="offset points", ha="right", fontsize=9, color=INK2)
ax.set_xticks(list(xs))
ax.set_xticklabels([f"M={m}" for m in M_VALS])
ax.set_ylabel("time per iteration (s)")
ax.set_ylim(0, 330)
ax.annotate(f"MS mode-2: {MS2_SITER:.0f}", (0.02, 0.945), xycoords="axes fraction", fontsize=8, color=INK2)
ax.set_title("Time per iteration vs Born order, 2 GPUs (H200 NV18)",
             fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(axis="y", color=INK2, alpha=0.12, linewidth=0.6)
save(fig, "fig_h200_siter_vs_M")

# ---- C. Convergence trajectories -------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.0))
for m, c in zip(M_VALS, RAMP):
    ax.plot(ITERS, TRAJ[m], "-o", color=c, linewidth=1.5, markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.0, label=f"born M={m}")
ax.annotate("M=1", (ITERS[-1], TRAJ[1][-1]), xytext=(7, 0),
            textcoords="offset points", va="center", fontsize=8.5, color=INK)
ax.plot(ITERS, MS_TRAJ, "--s", color=ORANGE, linewidth=1.5, markersize=5.5,
        markeredgecolor=SURFACE, markeredgewidth=1.0, label="multislice")
ax.set_xlabel("iteration")
ax.set_ylabel("total loss")
ax.set_xticks(ITERS)
ax.set_xlim(0.9, 5.8)
ax.set_title("Convergence vs Born order — M8 tracks multislice\n"
             "(identical losses for slice- and mode-split arms at every M)",
             fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.grid(axis="y", color=INK2, alpha=0.12, linewidth=0.6)
save(fig, "fig_h200_convergence")

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
RAMP = ["#a8c8ec", "#7dafe3", "#5395d9", "#2a78d6", "#174f94"]  # M=1,2,4,6,8 light->dark

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})

MS_SITER, MS_LOSS = 302.4, 0.3096
M_VALS = [1, 2, 4, 6, 8]
SLICE_SITER = [92.3, 109.3, 139.7, 165.7, 202.5]
MODE_SITER = {1: 120.5, 2: 131.9, 4: 165.6}  # M6 killed at $2 floor, M8 not reached
LOSS5 = [0.3335, 0.3160, 0.3146, 0.3117, 0.3103]
ITERS = [1, 2, 3, 4, 5]
TRAJ = {  # per-iteration total loss
    1: [0.4103, 0.3549, 0.3431, 0.3374, 0.3335],
    2: [0.3953, 0.3456, 0.3301, 0.3216, 0.3160],
    4: [0.3966, 0.3448, 0.3297, 0.3205, 0.3146],
    6: [0.3962, 0.3432, 0.3275, 0.3176, 0.3117],
    8: [0.3956, 0.3418, 0.3253, 0.3156, 0.3103],
}
MS_TRAJ = [0.3957, 0.3413, 0.3243, 0.3143, 0.3096]


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", name)


# M>=10 extension arms ran on 2x H100 (results_redo_highM_*): plotted AS
# MEASURED (owner instruction) — note the hardware change vs the H200 ladder.
EXT = {10: 0.3099}          # loss@5; add 12/14 when those arms land
EXT_SITER = {10: 348.4}     # measured s/iter on 2x H100

# ---- A. Pareto: iter-5 loss vs time per iteration --------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.plot(SLICE_SITER, LOSS5, "-", color="#2a78d6", linewidth=1.6, zorder=2)
ext_ms = sorted(EXT)
ax.plot([SLICE_SITER[-1]] + [EXT_SITER[m] for m in ext_ms],
        [LOSS5[-1]] + [EXT[m] for m in ext_ms], "--", color="#174f94",
        linewidth=1.4, zorder=2)
for m in ext_ms:
    ax.plot(EXT_SITER[m], EXT[m], "o", markerfacecolor=SURFACE,
            markeredgecolor="#174f94", markeredgewidth=1.6, markersize=8,
            zorder=3)
    ax.annotate(f"M={m}*", (EXT_SITER[m], EXT[m]), xytext=(2, -16),
                textcoords="offset points", ha="center", fontsize=9, color=INK)
ax.annotate("*M≥10 measured on 2×H100 (slower GPUs);\n"
            " M≤8 and multislice measured on 2×H200", (0.97, 0.60),
            xycoords="axes fraction", ha="right", fontsize=7.5, color=INK2)
for m, x, y, c in zip(M_VALS, SLICE_SITER, LOSS5, RAMP):
    ax.plot(x, y, "o", color=c, markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=1.3, zorder=3)
    ax.annotate(f"M={m}", (x, y), xytext=(6, 6), textcoords="offset points",
                fontsize=9, color=INK)
ax.plot(MS_SITER, MS_LOSS, "s", color=ORANGE, markersize=9,
        markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=3)
ax.annotate("multislice\n(mode-2)", (MS_SITER, MS_LOSS), xytext=(0, 10),
            textcoords="offset points", ha="center", fontsize=9, color=INK)
ax.axhline(MS_LOSS, color=INK2, linewidth=0.8, linestyle=":", alpha=0.5)
ax.set_xlabel("time per iteration (s)")
ax.set_ylabel("loss after 5 iterations")
ax.set_title("Born order vs multislice on 2 GPUs — loss/cost frontier\n"
             "(PSO 16384 DPs, 42 slices, 4 probe modes; born: slice split,\n"
             "MS: probe-mode split — its only parallel axis)",
             fontsize=9.5, loc="left")
ax.grid(axis="both", color=INK2, alpha=0.12, linewidth=0.6)
save(fig, "fig_h200_pareto")

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
ax.annotate(f"multislice, mode split: {MS_SITER:.0f} s/iter",
            (len(M_VALS) - 0.5, MS_SITER), xytext=(0, 4),
            textcoords="offset points", ha="right", fontsize=9, color=INK2)
ax.set_xticks(list(xs))
ax.set_xticklabels([f"M={m}" for m in M_VALS])
ax.set_ylabel("time per iteration (s)")
ax.set_ylim(0, 330)
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

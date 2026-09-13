"""
Convergence comparison on tBL_WSe2 (16384 views, dp128, 12 slices, batch 32):
standard first Born (Adam), exact-line-search first Born, multislice (Adam).
Parses the per-iteration `Iter: N, Total Loss: X, ..., in T` lines that
loss_logger writes, so all three series use the identical loss metric.

Outputs (next to demo/): convergence_linesearch.png + convergence_data.csv.
Run from anywhere; paths are absolute.
"""

import csv
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEMO = "/home/dnz75396/ptyrad/demo"
T100 = f"{DEMO}/output/test_100"

RUNS = [
    # (label, palette slot color, log path)
    (
        "First Born (Adam)",
        "#2a78d6",
        f"{T100}/tBL_WSe2_born/20260731_full_N16384_dp128_flipT100_random32_p6_1obj_"
        "12slice_dz1_plr1e-4_oalr5e-4_oplr5e-4_orblur0.5_ozblur1.0_mamp0.03_4.0_"
        "oathr0.98_oposc_sng1.0_spr0.1/20260731_120522_ptyrad_log_born.txt",
    ),
    (
        "First Born + line search",
        "#eb6834",
        "LATEST_LINESEARCH",  # resolved below: newest linesearch log in the random32 folder
    ),
    (
        "Multislice (Adam)",
        "#1baf7a",
        None,  # resolved below: the single log in the multislice folder
    ),
]

ITER_RE = re.compile(
    r"Iter: (\d+), Total Loss: ([0-9.]+),.*, in (?:(\d+) hr )?(?:(\d+) min )?([0-9.]+) sec"
)


def parse_log(path):
    iters, losses, secs = [], [], []
    with open(path) as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                n, loss, hr, mn, s = m.groups()
                iters.append(int(n))
                losses.append(float(loss))
                secs.append(3600 * int(hr or 0) + 60 * int(mn or 0) + float(s))
    if not iters:
        raise ValueError(f"no loss lines parsed from {path}")
    cum_min = []
    t = 0.0
    for s in secs:
        t += s
        cum_min.append(t / 60.0)
    return iters, losses, cum_min


def resolve_paths():
    import glob

    resolved = []
    for label, color, path in RUNS:
        if path == "LATEST_LINESEARCH":
            cands = sorted(
                glob.glob(f"{T100}/tBL_WSe2_born/20260913_*random32*/"
                          "*_ptyrad_log_linesearch_born_tBL_WSe2.txt")
            )
            path = cands[-1]
        elif path is None:
            cands = sorted(glob.glob(f"{T100}/tBL_WSe2_multislice/*/[0-9]*_ptyrad_log*.txt"))
            path = cands[-1]
        resolved.append((label, color, path))
    return resolved


def main():
    series = []
    for label, color, path in resolve_paths():
        iters, losses, cum_min = parse_log(path)
        series.append((label, color, iters, losses, cum_min))
        print(f"{label}: {len(iters)} iters, loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
              f"total {cum_min[-1]:.1f} min  ({path.split('/')[-1]})")

    # table view alongside the figure
    with open(f"{DEMO}/convergence_data.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "iteration", "total_loss", "cumulative_minutes"])
        for label, _, iters, losses, cum_min in series:
            for n, ls_, t in zip(iters, losses, cum_min, strict=True):
                wr.writerow([label, n, f"{ls_:.4f}", f"{t:.3f}"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=160, sharey=True)
    ink, muted = "#1a1a19", "#6b6a60"
    Y_LO, Y_HI = 0.372, 0.412  # clip the iteration-1 spike; declared in the note below
    for ax, xkey, xlabel in [(axes[0], 3, "Iteration"), (axes[1], 4, "Wall time (min)")]:
        # dodge the end-value labels so near-identical finals stay readable
        finals = sorted(
            ((losses[-1], (iters if xkey == 3 else cum_min)[-1], label)
             for label, _, iters, losses, cum_min in series),
        )
        min_gap = (Y_HI - Y_LO) * 0.045
        ys = []
        for fy, _, _ in finals:
            y = fy if not ys else max(fy, ys[-1] + min_gap)
            ys.append(y)
        label_y = {lbl: y for (fy, fx, lbl), y in zip(finals, ys, strict=True)}

        for label, color, iters, losses, cum_min in series:
            x = iters if xkey == 3 else cum_min
            ax.plot(x, losses, color=color, lw=2, solid_capstyle="round", label=label)
            ax.annotate(
                f"{losses[-1]:.4f}", (x[-1], label_y[label]),
                xytext=(5, 0), textcoords="offset points",
                fontsize=8, color=ink, va="center",
            )
        ax.set_xlabel(xlabel, color=ink)
        ax.set_ylim(Y_LO, Y_HI)
        ax.grid(True, color="#e8e7de", lw=0.6)
        ax.tick_params(colors=muted, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.margins(x=0.09)
    axes[0].set_ylabel("Total loss (normalized amplitude RMSE + sparse)", color=ink)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right", labelcolor=ink)
    starts = ", ".join(f"{lbl.split(' (')[0]} {losses[0]:.3f}"
                       for lbl, _, _, losses, _ in series)
    fig.suptitle(
        "tBL_WSe2 convergence — batch 32, 16384 views, 12 slices (dz 1 Å), RTX A4000",
        fontsize=10, color=ink,
    )
    fig.text(0.5, 0.905, f"y-axis clipped at {Y_HI}; iteration-1 values: {starts}",
             ha="center", fontsize=8, color=muted)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = f"{DEMO}/convergence_linesearch.png"
    fig.savefig(out, facecolor="white")
    print(f"saved {out} and {DEMO}/convergence_data.csv")


if __name__ == "__main__":
    main()

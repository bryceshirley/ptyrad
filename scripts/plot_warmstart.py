"""PSO warm-start test: 3 iterations of line-search ISS as a
preconditioner for the tuned multislice (Adam) reconstruction, vs the
cold-start multislice reference. Honest accounting: the warm curve includes
its 3 seed iterations on the pass axis and the seed's wall time on the time
axis. Outputs demo/warmstart_pso.png + demo/warmstart_pso.csv."""

import csv
import glob
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEMO = "/home/dnz75396/ptyrad/demo"
ITER_RE = re.compile(
    r"Iter: (\d+), Total Loss: ([0-9.]+),.*, in (?:(\d+) hr )?(?:(\d+) min )?([0-9.]+) sec"
)


def parse(path):
    it, ls_, sec = [], [], []
    with open(path) as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                n, loss, hr, mn, s = m.groups()
                it.append(int(n))
                ls_.append(float(loss))
                sec.append(3600 * int(hr or 0) + 60 * int(mn or 0) + float(s))
    return it, ls_, sec


def cum_min(secs, offset=0.0):
    out, t = [], offset
    for s in secs:
        t += s
        out.append(t / 60.0)
    return out


def main():
    cold_log = glob.glob(
        f"{DEMO}/output/PSO_ms_paper/2026*/20260912_171554_ptyrad_log.txt")[0]
    seed_log = glob.glob(
        f"{DEMO}/output/PSO_warmstart/2026*/[0-9]*ls_warm3_PSO.txt")
    seed_log = seed_log[0] if seed_log else f"{DEMO}/run_ls_warm3_console.log"
    warm_log = f"{DEMO}/run_ms_warm_console.log"

    ci, cl, cs = parse(cold_log)
    si, sl, ss = parse(seed_log)
    wi, wl, ws = parse(warm_log)
    seed_t = sum(ss) / 60.0

    cold_t = cum_min(cs)
    seed_tm = cum_min(ss)
    warm_t = cum_min(ws, offset=seed_t)
    # pass axis: warm passes = 3 seed + multislice iterations
    warm_pass = [n + len(si) for n in wi]

    with open(f"{DEMO}/warmstart_pso.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["series", "data_pass", "total_loss", "cumulative_minutes"])
        for n, ls_, t in zip(ci, cl, cold_t, strict=True):
            wr.writerow(["multislice cold", n, ls_, f"{t:.2f}"])
        for n, ls_, t in zip(si, sl, seed_tm, strict=True):
            wr.writerow(["LS ISS seed", n, ls_, f"{t:.2f}"])
        for n, ls_, t in zip(warm_pass, wl, warm_t, strict=True):
            wr.writerow(["multislice warm", n, ls_, f"{t:.2f}"])

    cold_c, warm_c = "#1baf7a", "#eda100"  # palette slots 3 (entity: multislice), 4
    ink, muted = "#1a1a19", "#6b6a60"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=160, sharey=True)
    for ax, cx, wx, sx, xlabel in [
        (axes[0], ci, warm_pass, si, "Full data passes (seed included)"),
        (axes[1], cold_t, warm_t, seed_tm, "Wall time (min, seed included)"),
    ]:
        ax.plot(cx, cl, color=cold_c, lw=2, solid_capstyle="round",
                label="Multislice, cold start")
        ax.plot(sx, sl, color=warm_c, lw=2, ls=(0, (3, 2)),
                label="3-iter line-search ISS seed")
        ax.plot(wx, wl, color=warm_c, lw=2, solid_capstyle="round",
                label="Multislice, warm start")
        ax.annotate(f"{cl[-1]:.4f}", (cx[-1], cl[-1]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color=ink)
        ax.annotate(f"{wl[-1]:.4f}", (wx[-1], wl[-1]), xytext=(5, -9),
                    textcoords="offset points", fontsize=8, color=ink)
        ax.set_xlabel(xlabel, color=ink)
        ax.set_ylim(0.28, 0.46)
        ax.grid(True, color="#e8e7de", lw=0.6)
        ax.tick_params(colors=muted, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.margins(x=0.06)
    axes[0].set_ylabel("Total loss", color=ink)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right", labelcolor=ink)
    fig.suptitle(
        "PSO: 3 exact-line-search ISS iterations as a multislice preconditioner "
        "(batch 32, identical Adam/constraints)",
        fontsize=10, color=ink,
    )
    fig.text(0.5, 0.905,
             "y-axis clipped at 0.46; cold start begins at 0.691, seed at 0.429",
             ha="center", fontsize=8, color=muted)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f"{DEMO}/warmstart_pso.png", facecolor="white")
    print(f"cold: {cl[0]:.4f} -> {cl[-1]:.4f} in {cold_t[-1]:.1f} min | "
          f"warm: seed {sl[-1]:.4f} ({seed_t:.1f} min) -> {wl[-1]:.4f} "
          f"in {warm_t[-1]:.1f} min")


if __name__ == "__main__":
    main()

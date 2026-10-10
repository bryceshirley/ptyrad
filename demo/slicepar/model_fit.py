"""Step 6.2: latency model T(P=2) = W/2 + c_scheme overlaid on measurements.

alpha/beta per transport from micro_results.csv (linear fit ms vs bytes);
W from the P=1 eager forward; c_scheme per the task's table, applied per
order (M carries per forward pass).

Writes fig_model.png and appends the alpha/beta table to analysis.md.
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {"chain": "#2a78d6", "chain+pinned": "#4a3aa7", "transpose": "#e87ba4"}
INK, MUT = "#1a1a19", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e8e7e0",
    "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "figure.dpi": 150})

micro = {}
for r in csv.DictReader(open("micro_results.csv")):
    micro.setdefault(r["case"], []).append(
        (int(r["bytes"]), float(r["ms_median"])))


def fit_ab(case):
    pts = sorted(micro[case])
    x = np.array([b for b, _ in pts], float)
    y = np.array([t for _, t in pts], float)
    beta, alpha = np.polyfit(x, y, 1)
    return alpha, beta  # ms, ms/byte


grid = {}
for r in csv.DictReader(open("slicepar_results.csv")):
    if r["status"] != "ok":
        continue
    k = (r["mode"], r["P"], r["scheme"], r["transport"], r["skipz"],
         r["Nz"], r["n"], r["M"], r["pmode"])
    grid[k] = float(r["ms_med"])  # duplicates: last wins


def main():
    a_peer, b_peer = fit_ab("nonblocking_to")
    a_pin, b_pin = fit_ab("pinned_staged")
    a_a2a, b_a2a = fit_ab("a2a_vol_N32")
    with open("analysis.md", "a") as f:
        f.write("\n## Link model fits (ms, ms/MB)\n\n"
                "| transport | alpha (ms) | beta (ms/MB) |\n|---|---|---|\n"
                f"| peer .to (driver staged) | {a_peer:.3f} | {b_peer*1e6:.4f} |\n"
                f"| pinned bounce | {a_pin:.3f} | {b_pin*1e6:.4f} |\n"
                f"| all-to-all (volume) | {a_a2a:.3f} | {b_a2a*1e6:.4f} |\n")
    print(f"peer:   alpha={a_peer:.3f} ms beta={b_peer*1e6:.4f} ms/MB")
    print(f"pinned: alpha={a_pin:.3f} ms beta={b_pin*1e6:.4f} ms/MB")
    print(f"a2a:    alpha={a_a2a:.3f} ms beta={b_a2a*1e6:.4f} ms/MB")

    NZS = [4, 8, 16, 32, 64, 128, 256]
    M = 4
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, n in zip(axes, (256, 1024, 2048)):
        bcar = 8 * n * n  # carry plane bytes
        meas = {s: ([], []) for s in C}
        model = {s: ([], []) for s in C}
        for Nz in NZS:
            W = grid.get(("fwd", "1", "single", "-", "False",
                          str(Nz), str(n), str(M), "1"))
            if W is None:
                continue
            for s, tr in (("chain", "peer"), ("chain+pinned", "pinned"),
                          ("transpose", "peer")):
                sch = s.split("+")[0]
                mv = grid.get(("fwd", "2", sch, tr, "False",
                               str(Nz), str(n), str(M), "1"))
                if mv is not None:
                    meas[s][0].append(Nz)
                    meas[s][1].append(mv)
                al, be = (a_peer, b_peer) if tr == "peer" else (a_pin, b_pin)
                if s == "transpose":
                    # per order: 2 a2a of ~Nz/2 cross planes halved + Dn row
                    cross = 2 * (Nz / 2) * (bcar / 2)
                    c = M * (2 * a_a2a + 2 * cross * b_a2a)
                else:
                    c = M * (al + bcar * be)
                model[s][0].append(Nz)
                model[s][1].append(W / 2 + c)
        for s in C:
            if meas[s][0]:
                ax.plot(*meas[s], "o-", color=C[s], lw=2, ms=4, label=f"{s} meas")
            if model[s][0]:
                ax.plot(*model[s], "--", color=C[s], lw=1.2, label=f"{s} model")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(NZS, NZS)
        ax.set_xlabel("Nz")
        ax.set_title(f"n = {n}", loc="left")
    axes[0].set_ylabel("forward time (ms), M=4")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("T = W/2 + c_scheme model (dashed) vs measured (solid), P=2",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig("fig_model.png", bbox_inches="tight")
    print("wrote fig_model.png")


if __name__ == "__main__":
    main()

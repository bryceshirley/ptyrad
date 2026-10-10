"""Step 6 analysis: Gantt charts, crossover tables, latency model, best-scheme map.

Inputs:  timeline_*.json, micro_results.csv, slicepar_results.csv
Outputs: fig_timeline_current.png, fig_timeline_best.png, fig_best_scheme.png,
         fig_crossover.png, fig_model.png, analysis.md (tables)
"""

import csv
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# dataviz fixed categorical order (validated)
C = {"chain": "#2a78d6", "chain_reordered": "#eb6834", "direct": "#1baf7a",
     "logscan": "#eda100", "transpose": "#e87ba4", "chain+pinned": "#4a3aa7",
     "transpose+pinned": "#008300", "single": "#52514e"}
PHASE_C = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
INK, MUT = "#1a1a19", "#52514e"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.grid": True, "grid.color": "#e8e7e0", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "figure.dpi": 150})


def gantt(timeline_json, out_png, title):
    d = json.load(open(timeline_json))
    devs = sorted({s["dev"] for s in d["segments"]})
    fig, ax = plt.subplots(figsize=(9, 2.8))
    for s in d["segments"]:
        y = devs.index(s["dev"])
        ph = s["name"][0]
        dur = s["end"] - s["start"]
        ax.barh(y, dur, left=s["start"], height=0.52,
                color=PHASE_C.get(ph, MUT), edgecolor="#fcfcfb", linewidth=0.8)
        if dur > 0.25:
            ax.text(s["start"] + dur / 2, y, s["name"], ha="center",
                    va="center", fontsize=7, color="#ffffff")
    # carry arrows: B<n>.src end on dev0 -> B<n>.dst start on dev1
    seg = {(s["dev"], s["name"]): s for s in d["segments"]}
    for nm in sorted({s["name"][1] for s in d["segments"]
                      if s["name"].startswith("B")}):
        src = seg.get((devs[0], f"B{nm}.src"))
        dst = seg.get((devs[1], f"B{nm}.dst"))
        if src and dst:
            ax.annotate("", xy=(dst["start"], 1 - 0.3), xytext=(src["end"], 0.3),
                        arrowprops=dict(arrowstyle="->", color=MUT, lw=1.0))
    ax.set_yticks(range(len(devs)), devs)
    ax.set_xlabel("time (ms)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_C[k]) for k in "ABC"]
    ax.legend(handles, ["phase 1 (scatter+scan)", "carry", "phase 2 (psi)"],
              loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False,
              fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_png)


def load_grid(path="slicepar_results.csv"):
    rows = []
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok":
            rows.append(r)
            continue
        r["ms_med"] = float(r["ms_med"])
        r["ms_iqr"] = float(r["ms_iqr"])
        rows.append(r)
    return rows


def key(r):
    nm = r["scheme"]
    if r["transport"] == "pinned":
        nm += "+pinned"
    if r["skipz"] == "True":
        nm += "+skipz"
    return nm


def crossover_and_best(rows, out_md):
    """Speed-up vs P=1 eager; crossover depth N*; best-scheme map."""
    base = {}
    for r in rows:
        if r.get("status") != "ok" if "status" in r else False:
            continue
        if r["P"] == "1" and r["scheme"] == "single" and r["status"] == "ok":
            base[(r["mode"], r["n"], r["M"], r["pmode"], r["Nz"])] = r["ms_med"]
    lines = ["\n## Speed-up vs P=1 (eager) — mode=step, pmode=1\n",
             "| n | M | scheme | " + " | ".join(
                 f"Nz={z}" for z in (4, 8, 16, 32, 64, 128, 256)) + " |",
             "|" + "---|" * 9]
    best = defaultdict(dict)
    cross = []
    schemes = ["chain", "chain_reordered", "direct", "logscan", "transpose",
               "chain+pinned", "transpose+pinned"]
    for n in ("256", "1024", "2048"):
        for M in ("1", "4"):
            for sch in schemes:
                sp, nstar = [], None
                for Nz in ("4", "8", "16", "32", "64", "128", "256"):
                    b = base.get(("step", n, M, "1", Nz))
                    rr = [r for r in rows if r["status"] == "ok"
                          and r["mode"] == "step" and r["P"] == "2"
                          and key(r) == sch and r["n"] == n and r["M"] == M
                          and r["pmode"] == "1" and r["Nz"] == Nz]
                    if b and rr:
                        s = b / rr[0]["ms_med"]
                        sp.append(f"{s:.2f}")
                        if s > 1 and nstar is None:
                            nstar = int(Nz)
                        cur = best[(n, M)].get(Nz)
                        if cur is None or rr[0]["ms_med"] < cur[1]:
                            best[(n, M)][Nz] = (sch, rr[0]["ms_med"], b)
                    else:
                        sp.append("—")
                if any(x != "—" for x in sp):
                    lines.append(f"| {n} | {M} | {sch} | " + " | ".join(sp) + " |")
                    cross.append((n, M, sch, nstar))
    lines.append("\n## Crossover depth N* (first Nz with speed-up > 1, step mode)\n")
    lines.append("| n | M | scheme | N* |")
    lines.append("|---|---|---|---|")
    for n, M, sch, nstar in cross:
        lines.append(f"| {n} | {M} | {sch} | {nstar if nstar else '>256 (none)'} |")
    open(out_md, "a").write("\n".join(lines) + "\n")
    return best, base


def best_scheme_fig(best, base, rows):
    NZS = ["4", "8", "16", "32", "64", "128", "256"]
    ns = ["256", "1024", "2048"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, M in zip(axes, ("1", "4")):
        grid_names = np.empty((len(ns), len(NZS)), dtype=object)
        for i, n in enumerate(ns):
            for j, Nz in enumerate(NZS):
                e = best.get((n, M), {}).get(Nz)
                b = base.get(("step", n, M, "1", Nz))
                if e is None:
                    grid_names[i, j] = ""
                    continue
                sch, ms, bms = e
                # P=1 wins if faster than best P=2
                grid_names[i, j] = "single (P=1)" if (b and b <= ms) else sch
        for i in range(len(ns)):
            for j in range(len(NZS)):
                nm = grid_names[i, j]
                col = ("#c3c2b7" if nm == "" else
                       "#eeede6" if nm == "single (P=1)" else
                       C.get(nm.replace("+skipz", ""), MUT))
                ax.add_patch(plt.Rectangle((j, i), 0.94, 0.94, color=col))
                if nm:
                    ax.text(j + 0.47, i + 0.47,
                            nm.replace("single (P=1)", "P=1")
                              .replace("chain_reordered", "reord")
                              .replace("+pinned", "\n+pinned"),
                            ha="center", va="center", fontsize=6.5,
                            color=INK if nm == "single (P=1)" else "#ffffff")
        ax.set_xticks(np.arange(len(NZS)) + 0.47, NZS)
        ax.set_yticks(np.arange(len(ns)) + 0.47, ns)
        ax.set_xlabel("Nz (depth)")
        ax.set_title(f"M = {M}", loc="left")
        ax.set_xlim(0, len(NZS))
        ax.set_ylim(0, len(ns))
        ax.invert_yaxis()
        ax.grid(False)
    axes[0].set_ylabel("n (plane size)")
    fig.suptitle("Fastest arm per (n, Nz) — optimizer step, P∈{1,2}, PCIe no-P2P",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig("fig_best_scheme.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_best_scheme.png")


def crossover_fig(rows, base):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
    NZS = [4, 8, 16, 32, 64, 128, 256]
    show = ["chain", "transpose", "chain+pinned"]
    for ax, n in zip(axes, ("256", "1024", "2048")):
        for sch in show:
            xs, ys = [], []
            for Nz in NZS:
                b = base.get(("step", n, "4", "1", str(Nz)))
                rr = [r for r in rows if r["status"] == "ok" and
                      r["mode"] == "step" and r["P"] == "2" and key(r) == sch
                      and r["n"] == n and r["M"] == "4" and r["pmode"] == "1"
                      and r["Nz"] == str(Nz)]
                if b and rr:
                    xs.append(Nz)
                    ys.append(b / rr[0]["ms_med"])
            if xs:
                ax.plot(xs, ys, "-o", color=C[sch], lw=2, ms=4, label=sch)
                ax.annotate(sch, (xs[-1], ys[-1]), textcoords="offset points",
                            xytext=(4, 0), fontsize=8, color=C[sch])
        ax.axhline(1.0, color=MUT, lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(NZS, NZS)
        ax.set_xlabel("Nz")
        ax.set_title(f"n = {n}", loc="left")
    axes[0].set_ylabel("speed-up vs P=1 eager (step)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("P=2 speed-up vs depth, M=4", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig("fig_crossover.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_crossover.png")


# latency model fit lives in model_fit.py (needs micro_results.csv + grid)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "gantt"):
        gantt("timeline_chain_Nz128_M4.json", "fig_timeline_current.png",
              "chain (current), P=2, Nz=128, M=4, n=256 — forward")
    if which in ("all", "grid"):
        rows = load_grid()
        open("analysis.md", "w").write("# slicepar analysis tables\n")
        best, base = crossover_and_best(rows, "analysis.md")
        best_scheme_fig(best, base, rows)
        crossover_fig(rows, base)

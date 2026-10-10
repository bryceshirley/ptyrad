"""Gate 4: loss curves of the recon-gate arms vs the chain baseline."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"chain (peer)": "#2a78d6", "chain (pinned)": "#4a3aa7",
     "transpose (peer)": "#e87ba4"}
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e8e7e0",
    "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "figure.dpi": 150})

def curve(path):
    out = []
    for line in open(path, errors="ignore"):
        m = re.search(r"Iter: (\d+), Total Loss: ([0-9.]+)", line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out

fig, ax = plt.subplots(figsize=(6.5, 3.4))
base = dict(curve("gate_chain.log"))
rows = []
for lbl, f in (("chain (peer)", "gate_chain.log"),
               ("chain (pinned)", "gate_pinned.log"),
               ("transpose (peer)", "gate_transpose.log")):
    c = curve(f)
    if not c:
        continue
    xs, ys = zip(*c)
    ax.plot(xs, ys, "-o", ms=3, lw=1.8, color=C[lbl], label=lbl)
    worst = max(abs(y - base[x]) for x, y in c if x in base) if base else None
    rows.append((lbl, len(c), ys[-1], worst))
ax.set_xlabel("iteration"); ax.set_ylabel("total loss")
ax.set_title("20-iter slicedist gate, real PSO born6, seed 42", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("fig_gate_loss.png", bbox_inches="tight")
print("| arm | iters | final loss | max |Δloss| vs chain |")
print("|---|---|---|---|")
for lbl, n, fl, w in rows:
    print(f"| {lbl} | {n} | {fl:.4f} | {('%.2e' % w) if w is not None else '—'} |")

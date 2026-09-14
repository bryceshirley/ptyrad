"""PSO (PrScO3, 300 kV) Born-vs-multislice comparison at matched iteration 80.

Reads existing checkpoints and loss logs only (no reconstruction is run).
Outputs: electron_recon_pso.png (phase pair) and pso_loss_comparison.png.
"""
import glob, re
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "/home/dnz75396/ptypy/draft_paper"
BORN = glob.glob("/home/dnz75396/ptyrad/demo/output/PSO_born_paper/2026*/model_iter0080.hdf5")[0]
MS = glob.glob("/home/dnz75396/ptyrad/demo/output/PSO_ms_paper/2026*/model_iter0080.hdf5")[0]
LOGB = "/home/dnz75396/ptyrad/demo/pso_born_paper.log"
LOGM = "/home/dnz75396/ptyrad/demo/pso_ms_paper.log"
PAT = re.compile(r"Iter:\s*(\d+), Total Loss:\s*([\d.]+).*?in\s*([\d.]+)\s*sec")

def load_phase(path):
    with h5py.File(path, "r") as f:
        p = f["optimizable_tensors/objp"][0].sum(0)
        dx = float(f["model_attributes/dx"][()])
    return p, dx

def parse(log, nmax):
    it, ls, tt, tot = [], [], [], 0.0
    for line in open(log):
        m = PAT.search(line)
        if m and int(m.group(1)) <= nmax:
            it.append(int(m.group(1))); ls.append(float(m.group(2)))
            tot += float(m.group(3)); tt.append(tot)
    return it, ls, tt

pb, dx = load_phase(BORN)
pm, _ = load_phase(MS)
c = 96
pb, pm = pb[c:-c, c:-c], pm[c:-c, c:-c]
lo = min(np.percentile(pb,1), np.percentile(pm,1))
hi = max(np.percentile(pb,99), np.percentile(pm,99))
fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.6))
for a, im, t in ((ax[0], pb, "ISS"), (ax[1], pm, "multislice")):
    a.imshow(im, cmap="gray", vmin=lo, vmax=hi); a.set_axis_off()
    a.set_title(t, fontsize=11)
px = 10.0/dx; ny, nx = pb.shape
ax[0].plot([nx*0.06, nx*0.06+px], [ny*0.94]*2, "w-", lw=3)
ax[0].text(nx*0.06+px/2, ny*0.90, "1 nm", color="w", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUTDIR}/electron_recon_pso.png", dpi=220, bbox_inches="tight")
print("recon pair saved | clim %.2f..%.2f rad" % (lo, hi))

itb, lb, tb = parse(LOGB, 80); itm, lm, tm = parse(LOGM, 80)
print("iter80: born %.4f (%.1f s/it), ms %.4f (%.1f s/it)" % (lb[-1], tb[-1]/80, lm[-1], tm[-1]/80))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
for a, xb, xm in ((a1, itb, itm), (a2, tb, tm)):
    a.semilogy(xb, lb, color="tab:blue", lw=1.5, label="ISS")
    a.semilogy(xm, lm, color="tab:green", lw=1.5, label="multislice")
    a.grid(True, which="both", ls=":", alpha=0.5); a.legend()
    a.set_ylabel("total loss")
a1.set_xlabel("iteration"); a2.set_xlabel("cumulative wall time (s)")
fig.tight_layout(); fig.savefig(f"{OUTDIR}/pso_loss_comparison.png", dpi=200)
print("loss comparison saved")

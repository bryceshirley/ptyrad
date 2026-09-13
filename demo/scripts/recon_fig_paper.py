"""Real-space reconstruction figures for the paper.

Renders the recovered phase (summed over slices) from existing PtyRAD
checkpoints -- no reconstruction is run. Outputs go to draft_paper/.

  tBL-WSe2 : Born-fitted vs multislice-fitted, iter 0200, common colour scale
  PSO      : multislice-fitted, iter 0100 (no Born PSO run exists on disk)
"""

import glob

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "/home/dnz75396/ptypy/draft_paper"


def load_phase(path, per_slice=False):
    with h5py.File(path, "r") as f:
        objp = f["optimizable_tensors/objp"][0]      # (N, ny, nx)
        dx = float(f["model_attributes/dx"][()])     # Angstrom
    return (objp if per_slice else objp.sum(0)), dx


def scalebar(ax, dx, length_A, ny, nx, label):
    px = length_A / dx
    x0, y0 = nx * 0.06, ny * 0.94
    ax.plot([x0, x0 + px], [y0, y0], "w-", lw=3)
    ax.text(x0 + px / 2, y0 - ny * 0.03, label, color="w",
            ha="center", va="bottom", fontsize=9)


def tbl_figure():
    born = glob.glob("/home/dnz75396/ptyrad/demo/output/tBL_WSe2_born/2026*/"
                     "model_iter0200.hdf5")[0]
    ms = glob.glob("/home/dnz75396/ptyrad/demo/output/tBL_WSe2_multislice/"
                   "2026*/model_iter0200.hdf5")[0]
    pb, dx = load_phase(born, per_slice=True)     # (12, ny, nx)
    pm, _ = load_phase(ms, per_slice=True)
    # central crop away from the unscanned buffer; zoom so atoms are visible
    c = 176
    pb, pm = pb[:, c:-c, c:-c], pm[:, c:-c, c:-c]
    slices = (3, 11)
    lo = min(np.percentile(pb[list(slices)], 1),
             np.percentile(pm[list(slices)], 1))
    hi = max(np.percentile(pb[list(slices)], 99),
             np.percentile(pm[list(slices)], 99))
    ncol = len(slices)
    fig, axes = plt.subplots(2, ncol, figsize=(3.4 * ncol, 7.0))
    for r, (stack, rname) in enumerate(((pb, "first Born"),
                                        (pm, "multislice"))):
        for cix, j in enumerate(slices):
            ax = axes[r, cix]
            ax.imshow(stack[j], cmap="gray", vmin=lo, vmax=hi,
                      origin="upper")
            ax.set_axis_off()
            if r == 0:
                ax.set_title(r"$z = %d\,$Å" % j, fontsize=10)
        axes[r, 0].text(-0.08, 0.5, rname, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=11)
    scalebar(axes[1, 0], dx, 10.0, *pb[0].shape, "1 nm")
    fig.tight_layout()
    out = f"{OUTDIR}/electron_recon_tblwse2.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", out, "| dx", dx, "A | crop", pb.shape,
          "| clim %.3f..%.3f rad" % (lo, hi))


def pso_figure():
    ms = glob.glob("/home/dnz75396/ptyrad/demo/output/multislice_subslices/"
                   "20260827_full_N4096_dp256_sparse8_p4_1obj_21slice_dz10_*/"
                   "model_iter0100.hdf5")[0]
    pm, dx = load_phase(ms)
    c = 96
    pm = pm[c:-c, c:-c]
    lo, hi = np.percentile(pm, 1), np.percentile(pm, 99)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.imshow(pm, cmap="gray", vmin=lo, vmax=hi, origin="upper")
    ax.set_axis_off()
    scalebar(ax, dx, 10.0, *pm.shape, "1 nm")
    fig.tight_layout()
    out = f"{OUTDIR}/electron_recon_pso_multislice.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", out, "| dx", dx, "A | crop", pm.shape,
          "| clim %.3f..%.3f rad" % (lo, hi))


if __name__ == "__main__":
    tbl_figure()
    pso_figure()

"""Iteration-10 reconstruction comparison figure (tBL-WSe2, batch 32).

Same layout and conventions as recon_fig_paper.py's tbl_figure (the paper's
electron_recon_tblwse2.png): per-slice phase at z = 3 and 11 Å, central crop
c = 176, one common colour scale, 1 nm scale bar — but three rows (first
Born, first Born + line search, multislice), each at its iter-0010
checkpoint from the matched test_100 runs. No reconstruction is run.
"""

import glob

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T100 = "/home/dnz75396/ptyrad/demo/output/test_100"
OUT = "/home/dnz75396/ptyrad/demo/recon_iter10_comparison.png"

ROWS = [
    ("first Born", f"{T100}/tBL_WSe2_born/20260731_*/model_iter0010.hdf5"),
    ("first Born\n+ line search", f"{T100}/tBL_WSe2_born/20260913_*random32*/model_iter0010.hdf5"),
    ("multislice", f"{T100}/tBL_WSe2_multislice/20260731_*/model_iter0010.hdf5"),
]
SLICES = (3, 11)
CROP = 176


def load_phase(pattern):
    path = glob.glob(pattern)[0]
    with h5py.File(path, "r") as f:
        objp = f["optimizable_tensors/objp"][0]  # (N, ny, nx)
        dx = float(f["model_attributes/dx"][()])
    return objp[:, CROP:-CROP, CROP:-CROP], dx, path


def scalebar(ax, dx, length_A, ny, nx, label):
    px = length_A / dx
    x0, y0 = nx * 0.06, ny * 0.94
    ax.plot([x0, x0 + px], [y0, y0], "w-", lw=3)
    ax.text(x0 + px / 2, y0 - ny * 0.03, label, color="w",
            ha="center", va="bottom", fontsize=9)


def main():
    stacks = []
    for rname, pattern in ROWS:
        stack, dx, path = load_phase(pattern)
        stacks.append((rname, stack))
        print(f"{rname.replace(chr(10), ' ')}: {path.split('/')[-2][:30]}... "
              f"crop {stack.shape}, dx {dx:.4f} A")

    sel = list(SLICES)
    lo = min(np.percentile(s[sel], 1) for _, s in stacks)
    hi = max(np.percentile(s[sel], 99) for _, s in stacks)

    ncol = len(SLICES)
    fig, axes = plt.subplots(len(ROWS), ncol, figsize=(3.4 * ncol, 3.5 * len(ROWS)))
    for r, (rname, stack) in enumerate(stacks):
        for cix, j in enumerate(SLICES):
            ax = axes[r, cix]
            ax.imshow(stack[j], cmap="gray", vmin=lo, vmax=hi, origin="upper")
            ax.set_axis_off()
            if r == 0:
                ax.set_title(r"$z = %d\,$Å" % j, fontsize=10)
        axes[r, 0].text(-0.08, 0.5, rname, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=11)
    scalebar(axes[-1, 0], dx, 10.0, *stacks[0][1][0].shape, "1 nm")
    fig.tight_layout()
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT, "| clim %.3f..%.3f rad" % (lo, hi))


if __name__ == "__main__":
    main()

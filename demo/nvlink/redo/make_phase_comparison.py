"""Three-panel projected-object-phase comparison (grayscale), recreating the
old comparison.png with the 200-iter local runs on REAL 4096-DP data
(21 slices, dz=10 A, pmode 4): first Born (M=1) / eighth-order Born / MS.

Shared linear gray scale across panels (vmin=0, vmax=global max) so the
M=1 contrast deficit shows honestly. Crop derived from the checkpoint's
crop_pos (validated against the saved objp_zsum_crop tif).
"""

import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/dnz75396/ptyrad-nvlink/demo/output/PSO"
TAIL = ("_200it_full_N4096_dp256_random1_p4_1obj_21slice_dz10_plr1e-4_"
        "oalr5e-4_oplr5e-4_dpblur1.0_orblur0.4_ozblur1.0_mamp0.03_4.0_"
        "oathr0.96_oposc_sng1.0_spr0.1")
RUNS = [
    (f"{BASE}/20261013_local_m1{TAIL}", "First-order Born (M=1)\nProjected Object Phase"),
    (f"{BASE}/20261012_local_m8{TAIL}", "Eighth-order inter-slice Born (tuned $c_m$)\nProjected Object Phase"),
    (f"{BASE}/20261012_local_ms{TAIL}", "Multislice\nProjected Object Phase"),
]


def zsum_crop(run_dir):
    with h5py.File(f"{run_dir}/model_iter0200.hdf5") as f:
        objp = f["optimizable_tensors/objp"][0]          # (21, Ny, Nx)
        cp = f["model_attributes/crop_pos"][:]           # (4096, 2) patch corners
    z = objp.sum(axis=0)
    n = 256  # patch size; crop FOV spans the patch-centre extent of the scan
    lo = cp.min(axis=0) + n // 2
    hi = cp.max(axis=0) + n // 2
    return z[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1]


imgs = [zsum_crop(d) for d, _ in RUNS]
print("crop shapes:", [im.shape for im in imgs])
vmax = max(im.max() for im in imgs)
print("vmax:", vmax, "| per-panel max:", [f"{im.max():.2f}" for im in imgs])

fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
for ax, im, (_, title) in zip(axes, imgs, RUNS):
    h = ax.imshow(im, cmap="gray", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
cbar = fig.colorbar(h, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("phase [rad]", fontsize=9)
out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/comparison_200it_m1_m8_ms.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
print("wrote", out)

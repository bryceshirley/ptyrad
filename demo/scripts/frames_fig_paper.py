"""Electron detector-frame comparison for the paper.

For three scan positions of the tBL-WSe2 data: the measured frame, the two
forward models evaluated from the same (Born-fitted) reconstruction at the
same position, and |multislice - Born|, on one log scale per row -- the
electron analogue of the X-ray frames figure. Reuses the loaders and the
roots-of-unity order extraction from ptyrad_diagnostics.py.
"""

import glob
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/dnz75396/ptyrad/demo")
sys.path.insert(0, "/home/dnz75396/ptyrad/src")
from ptyrad_diagnostics import Geom, electron_wavelength_A, gauge_fix, \
    load_model, orders  # noqa: E402
from ptyrad.load import load_raw  # noqa: E402  (EMPAD gap-aware reader)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "/home/dnz75396/ptypy/draft_paper/electron_frames.png"
RAW = "/home/dnz75396/ptyrad/demo/data/tBL_WSe2/Panel_g-h_Themis/scan_x128_y128.raw"
PICK = (8200,)                      # one representative frame for the paper


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdt = torch.complex128

    m = load_model(glob.glob(
        "/home/dnz75396/ptyrad/demo/output/tBL_WSe2_born/2026*/"
        "model_iter0200.hdf5")[0])
    obj = m["obj"][0] if m["obj"].ndim == 4 else m["obj"]
    obj, scale = gauge_fix(obj)
    probe = m["probe"] * scale
    lam = electron_wavelength_A(m["kv"])
    N = obj.shape[0]
    ny, nx = probe.shape[-2:]
    geom = Geom((ny, nx), m["dx"], lam, m["dz"], N, dev, cdt, H0=m.get("H0"))

    pos = np.round(m["pos"]).astype(int)
    raw = load_raw(RAW, shape=(16384, 128, 128))   # gap=1024 EMPAD default

    ob = torch.as_tensor(obj, dtype=cdt, device=dev)
    pr = torch.as_tensor(np.ascontiguousarray(probe), dtype=cdt, device=dev)

    fig, axes = plt.subplots(len(PICK), 3, figsize=(8.4, 2.7 * len(PICK)))
    axes = np.atleast_2d(axes)
    titles = ("measured", "multislice", "first Born")
    for r, idx in enumerate(PICK):
        p = pos[idx]
        rows = torch.as_tensor(p[0] + np.arange(ny), dtype=torch.long,
                               device=dev)
        cols = torch.as_tensor(p[1] + np.arange(nx), dtype=torch.long,
                               device=dev)
        O = torch.stack([ob[j][rows[:, None], cols[None, :]]
                         for j in range(N)])[:, None]      # (N,1,ny,nx)
        psi_ms, ords = orders(O, pr, geom, 1)
        F_ms = torch.fft.fft2(psi_ms, norm="ortho")
        F_fb = torch.fft.fft2(ords[0] + ords[1], norm="ortho")
        # measured frames are centre-ordered; shift the corner-ordered model
        # spectra to match (verified: correlation 0.97 in this layout)
        I_ms = np.fft.fftshift((F_ms.abs() ** 2).sum(1)[0].cpu().numpy())
        I_fb = np.fft.fftshift((F_fb.abs() ** 2).sum(1)[0].cpu().numpy())
        # PtyRAD's meas_flipT (1,0,0): transpose, as the model was fitted;
        # clip negatives as PtyRAD's clip_neg does
        meas = np.clip(np.asarray(raw[idx], dtype=np.float64).T, 0.0, None)
        # scale the models to the measured total: compare patterns only
        I_ms *= meas.sum() / I_ms.sum()
        I_fb *= meas.sum() / I_fb.sum()
        res_ms = np.abs(meas - I_ms)
        res_fb = np.abs(meas - I_fb)
        floor = max(meas[meas > 0].min(), 1e-6)
        vmax = np.log10(meas.max())
        vmin = np.log10(floor)
        for c, im in enumerate((meas, I_ms, I_fb)):
            ax = axes[r, c]
            ax.imshow(np.log10(im + floor),
                      vmin=vmin, vmax=vmax, cmap="inferno")
            ax.set_axis_off()
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
        print("pos %d: rel |meas-MS| %.3e, rel |meas-FB| %.3e"
              % (idx, res_ms.sum() / meas.sum(), res_fb.sum() / meas.sum()))
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()

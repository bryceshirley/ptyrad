"""PSO (PrScO3): ISS/Born vs multislice reconstructions at iteration 200 —
phase reconstructions, model fit and data.

Loads the two paper checkpoints (PSO_born_paper and PSO_ms_paper,
model_iter0200.hdf5), and shows, per model (rows):

  * the z-summed object phase over the scanned region,
  * the model diffraction amplitude DP^0.5 at one scan position, computed
    with the model's OWN forward (ISS = chord series truncated at M = 1 for
    the Born run, which used born_iterations = 1; exact multislice for the
    reference run),
  * the measured DP^0.5 at the same position,
  * |model - data| of DP^0.5, annotated with the relative amplitude error
    || sqrt(I_model) - sqrt(I_data) || / || sqrt(I_data) ||.

Forward fields are evaluated in complex128 with the unimodular-renormalized
propagator, as in analysis/born_remainder_bounds_pso.py; the model DP uses
the forward-model normalization sum_modes |psihat|^2 occu / (Ny Nx).

Run (from the repo root):
  CUDA_VISIBLE_DEVICES=0 ~/ptyrad/.venv/bin/python analysis/pso_born_vs_ms_recon.py
"""

import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ptyrad.forward_models.born_helpers import (  # noqa: E402
    _born_advance,
    _born_scatter,
    born_multislice_target,
)
from ptyrad.load import load_params  # noqa: E402
from ptyrad.models import PtychoAD  # noqa: E402
from ptyrad.reconstruction import PtyRADSolver  # noqa: E402
from ptyrad.utils import fftshift2  # noqa: E402
from torch.fft import fft2, ifft2  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
PARAMS = os.path.join(REPO, "demo", "params", "pso_ms_b1_n100.yml")
RUN = (
    "20260912_full_N4096_dp256_sparse32_p4_1obj_21slice_dz10_plr1e-4_"
    "oalr5e-4_oplr5e-4_slr5e-4_dpblur1_orblur0.4_ozblur1_mamp0.03_4_"
    "oathr0.96_oposc_sng1.0_spr0.1"
)
OUT_BASE = os.path.join(os.path.expanduser("~"), "ptyrad", "demo", "output")
CKPT = {
    "born": os.path.join(OUT_BASE, "PSO_born_paper", RUN, "model_iter0200.hdf5"),
    "ms": os.path.join(OUT_BASE, "PSO_ms_paper", RUN, "model_iter0200.hdf5"),
}
OUT_PNG = os.path.join(REPO, "demo", "pso_born_vs_ms_recon.png")
IDX = 2064
N_LAYER, DZ = 21, 10.0

C_SURFACE = "#fcfcfb"
CMAP = "viridis"  # matches the repo's forward-pass summaries


def load_model(ckpt, device):
    params = load_params(PARAMS, validate=True)
    ip = params["init_params"]
    ip["obj_source"], ip["obj_params"] = "PtyRAD", ckpt
    ip["probe_source"], ip["probe_params"] = "PtyRAD", ckpt
    ip["pos_source"], ip["pos_params"] = "PtyRAD", ckpt
    ip["obj_Nlayer"], ip["obj_slice_thickness"] = N_LAYER, DZ
    params["recon_params"]["if_quiet"] = True
    solver = PtyRADSolver(params, device=device, seed=42)
    model = PtychoAD(
        solver.init.init_variables, params["model_params"], device=device,
        verbose=False,
    )
    meas = solver.init.init_variables["measurements"]
    return model, meas


def forward_fields(model, sl, kind):
    """Detector field(s) of the model's own forward at indices sl."""
    patches = model.get_obj_patches(sl).to(torch.float64)
    probes = model.get_probes(sl).to(torch.complex128)
    H1 = model.get_propagators(sl).to(torch.complex128)
    H1 = H1 / H1.abs()
    powers = [torch.ones_like(H1)]
    for _ in range(model.n_slice - 1):
        powers.append(powers[-1] * H1)
    H3 = torch.stack(powers, dim=1).unsqueeze(1).unsqueeze(1)
    if kind == "ms":
        return born_multislice_target(patches, probes, H3)
    # ISS: chord series truncated at M = 1 (born_iterations = 1, plain)
    B, omode, Nz, Ny, Nx, _ = patches.shape
    probe_k = fft2(probes).view(-1, probes.shape[1], 1, 1, Ny, Nx)
    Psi0 = ifft2(H3 * probe_k)
    D0 = probe_k.squeeze(3).expand(B, -1, omode, -1, -1)
    obj = (torch.polar(patches[..., 0], patches[..., 1]) - 1.0).unsqueeze(1)
    scat = _born_scatter(obj, Psi0, H3, 0)
    return D0 + scat.sum(dim=3)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.chdir(os.path.join(REPO, "demo"))

    rows = {}
    data_amp = None
    for kind in ("born", "ms"):
        model, meas = load_model(CKPT[kind], device)
        occu = model.omode_occu.to(torch.float64).view(1, 1, -1, 1, 1)
        sl = torch.tensor([IDX], device=device)
        with torch.no_grad():
            field = forward_fields(model, sl, kind)
            Ny, Nx = field.shape[-2:]
            dp = fftshift2(
                (field.abs().square() * occu).sum(dim=(1, 2)) / (Ny * Nx)
            )[0].cpu().numpy()
            # z-summed phase over the scanned region
            objp = model.opt_objp.detach()[0].sum(dim=0).cpu().numpy()
            cp = model.crop_pos.cpu().numpy()
            y0, x0 = cp.min(axis=0)
            y1, x1 = cp.max(axis=0) + model.opt_probe.shape[-1]
            phase = objp[y0:y1, x0:x1]
        if data_amp is None:
            data_amp = np.sqrt(np.clip(np.asarray(meas[IDX]), 0, None))
        # the measurements are native (unpadded); the model DP lives on the
        # on-the-fly padded 256 grid — compare on the measured central region
        md, nd = data_amp.shape
        c0, c1 = (dp.shape[0] - md) // 2, (dp.shape[1] - nd) // 2
        dp = dp[c0 : c0 + md, c1 : c1 + nd]
        rows[kind] = {"dp_amp": np.sqrt(np.clip(dp, 0, None)), "phase": phase}
        del model
        torch.cuda.empty_cache()

    for kind in rows:
        r = rows[kind]
        r["resid"] = np.abs(r["dp_amp"] - data_amp)
        r["err"] = np.linalg.norm(r["resid"]) / np.linalg.norm(data_amp)

    # shared color scales
    ph_hi = max(np.percentile(r["phase"], 99.9) for r in rows.values())
    dp_hi = max(
        np.percentile(data_amp, 99.9),
        *(np.percentile(r["dp_amp"], 99.9) for r in rows.values()),
    )
    rs_hi = max(np.percentile(r["resid"], 99.9) for r in rows.values())

    labels = {"born": "ISS (Born, $M=1$)", "ms": "multislice"}
    fig, axes = plt.subplots(
        2, 4, figsize=(14.6, 7.6), dpi=150, facecolor=C_SURFACE
    )
    for i, kind in enumerate(("born", "ms")):
        r = rows[kind]
        panels = [
            (r["phase"], dict(vmin=0, vmax=ph_hi),
             "object phase, z-sum [rad]"),
            (r["dp_amp"], dict(vmin=0, vmax=dp_hi),
             rf"model $\sqrt{{I_{{\mathrm{{model}}}}}}$ (idx {IDX})"),
            (data_amp, dict(vmin=0, vmax=dp_hi),
             rf"data $\sqrt{{I_{{\mathrm{{data}}}}}}$ (idx {IDX})"),
            (r["resid"], dict(vmin=0, vmax=rs_hi),
             rf"$|\sqrt{{I_{{\mathrm{{model}}}}}}-\sqrt{{I_{{\mathrm{{data}}}}}}|$"
             rf", rel. err {r['err']:.3f}"),
        ]
        for j, (img, kw, title) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(img, cmap=CMAP, **kw)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0 or j == 3:
                ax.set_title(title, fontsize=10)
            elif j in (1, 2):
                ax.set_title(title.split(" (")[0], fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        axes[i, 0].set_ylabel(labels[kind], fontsize=12)
    fig.suptitle(
        "PrScO$_3$ at iteration 200: ISS (Born, $M=1$) vs multislice — "
        "phase reconstruction, model fit and data",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, facecolor=C_SURFACE)
    print(f"wrote {OUT_PNG}")
    for kind in ("born", "ms"):
        print(f"{labels[kind]}: relative amplitude error at idx {IDX}: "
              f"{rows[kind]['err']:.4f}")


if __name__ == "__main__":
    main()

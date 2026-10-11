"""Term norms of the terminating Born series on the multislice reference.

Recreates demo/born_term_norms.png (original made 2026-09-29 by an unsaved
inline script during the L-curve/ridge benchmark; recreated 2026-09-30).

Replays the converged 21-slice PSO multislice benchmark checkpoint
(20260918_ms_b1_n100, iteration 100) on the 32 evenly-spaced calibration
views (the same torch.linspace selection the refit uses), builds the full
order-1..21 Born detector basis and, for every model order M = 1..21, the
TSVD-fitted coefficients c^(M) from born_qr_coeffs. Plots, vs scattering
order m:

  - measured term norms ||psi^(m)|| / ||psi^(0)|| (median over views, with
    the view-to-view min-max band), and
  - the tuned |c_m^(M)| * ||psi^(m)|| / ||psi^(0)|| profile for each M
    (viridis-coloured by M).

The nilpotent termination guarantees the series ends at m = Nz = 21; the
figure shows the tuned coefficients re-weighting the low orders to absorb
the truncation tail while the high orders decay ~9 decades.

Run (from the repo root; GPU strongly recommended for the basis recursion):
  CUDA_VISIBLE_DEVICES=<UUID> .venv/bin/python analysis/born_term_norms.py
"""

import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ptyrad.forward_models.born_helpers import (  # noqa: E402
    born_detector_basis,
    born_multislice_target,
    born_qr_coeffs,
)
from ptyrad.load import load_params  # noqa: E402
from ptyrad.models import PtychoAD  # noqa: E402
from ptyrad.reconstruction import PtyRADSolver  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
PARAMS = os.path.join(REPO, "demo", "params", "pso_ms_b1_n100.yml")
CKPT = os.path.join(
    REPO,
    "demo",
    "output",
    "PSO",
    "20260918_ms_b1_n100_full_N4096_dp256_random1_p4_1obj_21slice_dz10_"
    "plr1e-4_oalr5e-4_oplr5e-4_dpblur1.0_orblur0.4_ozblur1.0_mamp0.03_4.0_"
    "oathr0.96_oposc_sng1.0_spr0.1",
    "model_iter0100.hdf5",
)
OUT_PNG = os.path.join(REPO, "demo", "born_term_norms.png")
N_VIEWS = 32
M_COLOR_MAX = 21  # shared colour normalization across the term-norm figures
N_ORDER = 21


def build_model(device):
    os.chdir(os.path.join(REPO, "demo"))  # params reference data/ relative to demo/
    params = load_params(PARAMS, validate=True)
    ip = params["init_params"]
    ip["obj_source"], ip["obj_params"] = "PtyRAD", CKPT
    ip["probe_source"], ip["probe_params"] = "PtyRAD", CKPT
    ip["pos_source"], ip["pos_params"] = "PtyRAD", CKPT
    params["recon_params"]["if_quiet"] = True
    solver = PtyRADSolver(params, device=device, seed=42)
    return PtychoAD(
        solver.init.init_variables, params["model_params"], device=device, verbose=False
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device)
    n_pos = model.crop_pos.shape[0]
    idx = torch.linspace(0, n_pos - 1, N_VIEWS).long().to(device)
    w64 = model.omode_occu.to(torch.float64).view(1, 1, -1)

    D0_parts, D_parts, T_parts = [], [], []
    d0n2 = 0.0
    with torch.no_grad():
        for sl in idx.split(8):
            patches = model.get_obj_patches(sl)
            probes = model.get_probes(sl)
            H3 = model.get_propagators_3d(model.get_propagators(sl))
            D0c, Dc = born_detector_basis(patches, probes, H3, N_ORDER)
            T_parts.append(born_multislice_target(patches, probes, H3) - D0c)
            D_parts.append(Dc)
            D0_parts.append(D0c)
            d0n2 += (
                (D0c.abs().square().sum(dim=(-2, -1)).to(torch.float64) * w64).sum().item()
            )
        D0 = torch.cat(D0_parts, dim=0)
        D = torch.cat(D_parts, dim=1)
        T = torch.cat(T_parts, dim=0)

        # per-view occupancy-weighted field norms, orders 0..21
        wf = model.omode_occu.to(torch.float64).clamp(min=0).view(1, 1, -1, 1, 1)

        def vnorm(field):  # (V,) norm per view over pmode/omode/pixels
            return (field.abs().square().to(torch.float64) * wf).sum(dim=(1, 2, 3, 4)).sqrt()

        n0 = vnorm(D0)
        ratios = torch.stack([vnorm(D[m]) / n0 for m in range(N_ORDER)])  # (21, V)
        ratios = ratios.cpu().numpy()

        # tuned coefficients per model order M (production TSVD solve)
        coeffs = {}
        for M in range(1, N_ORDER + 1):
            c, _ = born_qr_coeffs(D[:M], T, d0n2, omode_occu=model.omode_occu)
            coeffs[M] = c.abs().cpu().numpy()
            print(f"M = {M:2d}: max|c| = {coeffs[M].max():.3f}")

    m_axis = np.arange(1, N_ORDER + 1)
    med = np.median(ratios, axis=1)
    lo, hi = ratios.min(axis=1), ratios.max(axis=1)

    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(8.53, 6.1), dpi=150)
    ax.fill_between(m_axis, lo, hi, color="tab:blue", alpha=0.25, label="view-to-view range")
    for M in range(1, N_ORDER + 1):
        ax.plot(
            m_axis[:M],
            coeffs[M] * med[:M],
            marker="D",
            ms=5,
            lw=1.2,
            color=cmap((M - 1) / (M_COLOR_MAX - 1)),
            zorder=2 + M / 10,
        )
    ax.plot(
        m_axis,
        med,
        marker="o",
        ms=6,
        lw=2.2,
        color="tab:blue",
        zorder=5,
        label=r"measured $\|\hat\psi^{(m)}\|/\|\hat\psi^{(0)}\|$",
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(1, N_ORDER + 1, 2))
    ax.set_xlabel("scattering order $m$")
    ax.set_ylabel("term norm, relative to the unscattered field")
    ax.set_title(
        f"Term norms of the terminating Born series on the multislice reference "
        f"({N_VIEWS} views)"
    )
    ax.grid(alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=M_COLOR_MAX))
    cb = fig.colorbar(sm, ax=ax, ticks=[1, 5, 9, 13, 17, 21])
    cb.set_label(
        r"tuned $|c_m^{(M)}|\;\|\hat\psi^{(m)}\|/\|\hat\psi^{(0)}\|$: model order $M$"
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()

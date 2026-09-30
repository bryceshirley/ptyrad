"""Term norms of the terminating Born series — tBL-WSe2 multislice reference.

Recreates demo/born_term_norms_wse2.png (original 2026-09-29, unsaved
inline script; recreated 2026-09-30): the weak-scattering counterpart of
analysis/born_term_norms.py. The reference is the converged 100-iteration
ML-MS multislice reconstruction of the tBL-WSe2 dataset (12 slices, 1 A);
term norms decay monotonically (no multiple-scattering hump) from ~2.5e-1
at m=1 to ~1e-13 at m=12. The tuned profiles sit on the measured curve at
the leading orders; at m >= 5 the TSVD solve lets |c_m| grow large in the
near-null directions (fitted field unchanged — the documented "don't
over-read individual c_m" effect), so the tuned bulge there is larger
than in the ridge-era original of this figure.

Run (from the repo root):
  CUDA_VISIBLE_DEVICES=<UUID> .venv/bin/python analysis/born_term_norms_wse2.py
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
PARAMS = os.path.join(REPO, "demo", "params", "tBL_WSe2_reconstruct.yml")
CKPT = os.path.join(
    REPO,
    "demo",
    "output",
    "test_100",
    "tBL_WSe2_multislice",
    "20260923_full_mlms_full_N16384_dp128_flipT100_random32_p6_1obj_12slice_"
    "dz1_plr1e-4_oalr5e-4_oplr5e-4_orblur0.5_ozblur1.0_mamp0.03_4.0_"
    "oathr0.98_oposc_sng1.0_spr0.1",
    "model_iter0100.hdf5",
)
OUT_PNG = os.path.join(REPO, "demo", "born_term_norms_wse2.png")
OUT_CSV = os.path.join(REPO, "demo", "born_term_norms_wse2.csv")
N_VIEWS = 32
M_COLOR_MAX = 21  # shared colour normalization across the term-norm figures
N_ORDER = 12


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.chdir(os.path.join(REPO, "demo"))  # params reference data/ relative to demo/
    params = load_params(PARAMS, validate=True)
    ip = params["init_params"]
    ip["obj_source"], ip["obj_params"] = "PtyRAD", CKPT
    ip["probe_source"], ip["probe_params"] = "PtyRAD", CKPT
    ip["pos_source"], ip["pos_params"] = "PtyRAD", CKPT
    ip["obj_Nlayer"], ip["obj_slice_thickness"] = 12, 1.0  # checkpoint geometry
    params["recon_params"]["if_quiet"] = True
    solver = PtyRADSolver(params, device=device, seed=42)
    model = PtychoAD(
        solver.init.init_variables, params["model_params"], device=device, verbose=False
    )

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

        wf = model.omode_occu.to(torch.float64).clamp(min=0).view(1, 1, -1, 1, 1)

        def vnorm(field):
            return (field.abs().square().to(torch.float64) * wf).sum(dim=(1, 2, 3, 4)).sqrt()

        n0 = vnorm(D0)
        ratios = torch.stack([vnorm(D[m]) / n0 for m in range(N_ORDER)]).cpu().numpy()

        coeffs = {}
        for M in range(1, N_ORDER + 1):
            c, _ = born_qr_coeffs(D[:M], T, d0n2, omode_occu=model.omode_occu)
            coeffs[M] = c.abs().cpu().numpy()
            print(f"M = {M:2d}: max|c| = {coeffs[M].max():.3f}")

    m_axis = np.arange(1, N_ORDER + 1)
    mean = ratios.mean(axis=1)
    lo, hi = ratios.min(axis=1), ratios.max(axis=1)

    with open(OUT_CSV, "w") as f:
        f.write("m,rel_norm_mean,rel_norm_min,rel_norm_max\n")
        for m in range(N_ORDER):
            f.write(f"{m + 1},{mean[m]:.8g},{lo[m]:.8g},{hi[m]:.8g}\n")

    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(8.53, 6.1), dpi=150)
    ax.fill_between(m_axis, lo, hi, color="tab:blue", alpha=0.2, label="view-to-view range")
    for M in range(1, N_ORDER + 1):
        ax.plot(
            m_axis[:M],
            coeffs[M] * mean[:M],
            marker="D",
            ms=5,
            lw=1.2,
            color=cmap((M - 1) / (M_COLOR_MAX - 1)),
            zorder=2 + M / 10,
        )
    ax.plot(
        m_axis,
        mean,
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
    ax.legend(loc="lower left", framealpha=0.9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=M_COLOR_MAX))
    cb = fig.colorbar(sm, ax=ax, ticks=[1, 5, 9, 13, 17, 21])
    cb.set_label(
        r"tuned $|c_m^{(M)}|\;\|\hat\psi^{(m)}\|/\|\hat\psi^{(0)}\|$: model order $M$"
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"wrote {OUT_PNG} and {OUT_CSV}")


if __name__ == "__main__":
    main()

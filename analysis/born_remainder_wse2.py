"""Relative remainder norm of the truncated inter-slice Born series — tBL-WSe2.

Computes the detector-plane remainder of paper-1.tex Eq. (remainder),
R_M = psihat_MS - psihat^{(<=M)}, as a relative norm ||R_M|| / ||psihat_MS||
for truncation orders M = 1..12 on the converged 12-slice Adam multislice
reconstruction of the tBL-WSe2 dataset. Two perturbations are compared:

  * chord  (non-linearised): Delta O_j = |O_j| e^{i phi_j} - 1   — the ISS
    family of the paper; nilpotent, so R_12 = 0 to machine precision.
  * tangent (linearised):    Delta O_j = i phi_j                 — the
    slice-wise first-Born linearisation of Balakrishnan/Yeo; its series
    converges to the product of (1 + i phi_j), not to multislice, so the
    relative remainder plateaus at the intra-slice linearisation error.

All fields are promoted to complex128 so the M = 12 chord floor shows the
nilpotent termination rather than float32 round-off. The detector gauge
factor is unimodular and fft2 is unitary up to a constant, so these ratios
equal the exit-plane ones.

Run (from the repo root):
  CUDA_VISIBLE_DEVICES=0 ~/ptyrad/.venv/bin/python analysis/born_remainder_wse2.py
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
from torch.fft import fft2, ifft2  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
PARAMS = os.path.join(REPO, "demo", "params", "tBL_WSe2_reconstruct.yml")
# the 20260923 adam_ms run referenced by born_term_norms_wse2.py was deleted;
# this is the surviving Adam multislice run with the same geometry/settings
CKPT = os.path.join(
    os.path.expanduser("~"),
    "ptyrad",
    "demo",
    "output",
    "tBL_WSe2_multislice",
    "20260723_full_N16384_dp128_flipT100_random32_p6_1obj_12slice_"
    "dz1_plr1e-4_oalr5e-4_oplr5e-4_orblur0.5_ozblur1.0_mamp0.03_4.0_"
    "oathr0.98_oposc_sng1.0_spr0.1",
    "model_iter0100.hdf5",
)
OUT_PNG = os.path.join(REPO, "demo", "born_remainder_wse2.png")
OUT_CSV = os.path.join(REPO, "demo", "born_remainder_wse2.csv")
N_VIEWS = 32
N_ORDER = 12


def detector_orders(obj, Psi_state, H, n_max):
    """Per-order detector fields D_1..D_n for an arbitrary perturbation obj,
    same recursion as born_detector_basis but with the perturbation injected
    directly (chord or tangent)."""
    orders = []
    for k in range(n_max):
        scat = _born_scatter(obj, Psi_state, H, k)
        if k == n_max - 1:
            orders.append(scat.sum(dim=3))
        else:
            Psi_state, D_k = _born_advance(scat, H, k)
            orders.append(D_k.clone())
    return torch.stack(orders)


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
    wf = model.omode_occu.to(torch.float64).clamp(min=0).view(1, 1, -1, 1, 1)

    def vnorm(field):  # per-view occupancy-weighted norm
        return (field.abs().square().to(torch.float64) * wf).sum(dim=(1, 2, 3, 4)).sqrt()

    rel = {"chord": [], "tangent": []}
    with torch.no_grad():
        for sl in idx.split(8):
            patches = model.get_obj_patches(sl).to(torch.float64)
            probes = model.get_probes(sl).to(torch.complex128)
            # build H^j by cumulative multiplication in complex128 so the
            # sequential multislice target and the parallel series use
            # algebraically identical propagator powers (model.get_propagators_3d
            # uses complex64 pow(), which floors the remainder at ~3e-7)
            H1 = model.get_propagators(sl).to(torch.complex128)  # (N|1, Ny, Nx)
            # the float32-built phasor has |H| = 1 + O(1e-7); the series
            # telescopes H^j conj(H^j) per site while sequential multislice
            # accumulates |H^(Nz-1)|^2, so a non-unimodular H floors the
            # remainder at float32 rounding — renormalize to |H| = 1 exactly
            H1 = H1 / H1.abs()
            powers = [torch.ones_like(H1)]
            for _ in range(model.n_slice - 1):
                powers.append(powers[-1] * H1)
            H3 = torch.stack(powers, dim=1).unsqueeze(1).unsqueeze(1)
            assert H3.shape[-3] == model.n_slice and H3.ndim == 6, H3.shape

            B, omode, Nz, Ny, Nx, _ = patches.shape
            probe_k = fft2(probes).view(-1, probes.shape[1], 1, 1, Ny, Nx)
            Psi0 = ifft2(H3 * probe_k)
            D0 = probe_k.squeeze(3).expand(B, -1, omode, -1, -1)

            obj_chord = (
                torch.polar(patches[..., 0], patches[..., 1]) - 1.0
            ).unsqueeze(1)
            obj_tan = (1j * patches[..., 1].to(torch.complex128)).unsqueeze(1)

            T = born_multislice_target(patches, probes, H3)  # full multislice field
            t_norm = vnorm(T)

            for name, obj in (("chord", obj_chord), ("tangent", obj_tan)):
                D = detector_orders(obj, Psi0, H3, N_ORDER)
                psi_leq = D0.clone()
                rel_v = []
                for m in range(N_ORDER):
                    psi_leq = psi_leq + D[m]
                    rel_v.append((vnorm(T - psi_leq) / t_norm).cpu())
                rel[name].append(torch.stack(rel_v))  # (N_ORDER, B)

    curves = {k: torch.cat(v, dim=1).numpy() for k, v in rel.items()}  # (12, 32)
    m_axis = np.arange(1, N_ORDER + 1)

    with open(OUT_CSV, "w") as f:
        f.write(
            "M,chord_mean,chord_min,chord_max,tangent_mean,tangent_min,tangent_max\n"
        )
        for m in range(N_ORDER):
            c, t = curves["chord"][m], curves["tangent"][m]
            f.write(
                f"{m + 1},{c.mean():.8g},{c.min():.8g},{c.max():.8g},"
                f"{t.mean():.8g},{t.min():.8g},{t.max():.8g}\n"
            )

    fig, ax = plt.subplots(figsize=(8.0, 5.8), dpi=150)
    styles = {
        "chord": ("tab:blue", "o", r"chord $\Delta O = O - 1$ (non-linearised, ISS family)"),
        "tangent": ("tab:red", "s", r"tangent $\Delta O = i\varphi$ (linearised)"),
    }
    for name, (color, marker, label) in styles.items():
        v = curves[name]
        ax.fill_between(m_axis, v.min(axis=1), v.max(axis=1), color=color, alpha=0.18)
        ax.plot(m_axis, v.mean(axis=1), marker=marker, ms=6, lw=2.0, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xticks(m_axis)
    ax.set_xlabel("truncation order $M$ (retained scattering orders)")
    ax.set_ylabel(r"$\|\hat R_M\| \,/\, \|\hat\psi_{\mathrm{MS}}\|$")
    ax.set_title(
        f"Relative detector-plane remainder vs truncation order\n"
        f"tBL-WSe$_2$ multislice reference, 12 slices, {N_VIEWS} views "
        f"(bands: view-to-view range)"
    )
    ax.grid(alpha=0.4, which="both")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"wrote {OUT_PNG} and {OUT_CSV}")
    for m in range(N_ORDER):
        print(
            f"M = {m + 1:2d}: chord {curves['chord'][m].mean():.3e}  "
            f"tangent {curves['tangent'][m].mean():.3e}"
        )


if __name__ == "__main__":
    main()

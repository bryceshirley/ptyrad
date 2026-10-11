"""Detector intensity error of the coefficient-fitted Born model vs order M.

Recreates demo/born_recon_error_vs_n_qr_all.png (original 2026-09-21,
unsaved inline script; recreated 2026-09-30). On the converged 21-slice
PSO multislice benchmark, for M in {1, 2, 4, 8, 16}:

  - fit c^(M) on the 32 evenly-spaced calibration views (production TSVD
    solve, born_qr_coeffs);
  - evaluate the relative detector INTENSITY error vs the exact multislice
    patterns, ||dp_M - dp_ms||_F / ||dp_ms||_F (detector blur applied to
    both, matching the loss), three ways: plain series c_m = 1 on the full
    4096-view scan, fitted c on the 32 calibration views, and fitted c on
    the full scan.

The plain truncated series diverges through the multiple-scattering hump
(error ~1e3-1e4 at M = 4-8) while the fitted coefficients keep the model
convergent at every order, and the 32-view fit transfers to the full scan
essentially unchanged — the justification for the cheap calibration-view
refit. Streams the full scan in 8-view chunks (nothing above ~2 GB GPU).

Run (from the repo root; ~10 min on an A100):
  CUDA_VISIBLE_DEVICES=<UUID> .venv/bin/python analysis/born_recon_error_vs_order.py
"""

import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur

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
OUT_PNG = os.path.join(REPO, "demo", "born_recon_error_vs_n_qr_all.png")
OUT_CSV = os.path.join(REPO, "demo", "born_recon_error_vs_n_qr_all.csv")
N_CALIB = 32
ORDERS = [1, 2, 4, 8, 16]
CHUNK = 8


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


def make_dp_fn(model, Ny, Nx):
    norm_weight = (model.omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)

    def _dp(field):  # matches the refit's diagnostic dp
        dp = torch.fft.fftshift(
            torch.sum(field.abs().square() * norm_weight, dim=(1, 2)), dim=(-2, -1)
        )
        if model.detector_blur_std:
            dp = gaussian_blur(dp, kernel_size=[5, 5], sigma=model.detector_blur_std)
        return dp

    return _dp


def basis_and_target(model, sl, n_max):
    patches = model.get_obj_patches(sl)
    probes = model.get_probes(sl)
    H3 = model.get_propagators_3d(model.get_propagators(sl))
    D0, D = born_detector_basis(patches, probes, H3, n_max)
    T = born_multislice_target(patches, probes, H3) - D0
    return D0, D, T


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device)
    n_pos = model.crop_pos.shape[0]
    n_max = max(ORDERS)
    w64 = model.omode_occu.to(torch.float64).view(1, 1, -1)

    # ---- fit c^(M) on the calibration views -----------------------------
    calib = torch.linspace(0, n_pos - 1, N_CALIB).long().to(device)
    coeffs = {}
    with torch.no_grad():
        D0_p, D_p, T_p, d0n2 = [], [], [], 0.0
        for sl in calib.split(CHUNK):
            D0c, Dc, Tc = basis_and_target(model, sl, n_max)
            D0_p.append(D0c)
            D_p.append(Dc)
            T_p.append(Tc)
            d0n2 += (
                (D0c.abs().square().sum(dim=(-2, -1)).to(torch.float64) * w64).sum().item()
            )
        D0 = torch.cat(D0_p, dim=0)
        D = torch.cat(D_p, dim=1)
        T = torch.cat(T_p, dim=0)
        for M in ORDERS:
            c, delta = born_qr_coeffs(D[:M], T, d0n2, omode_occu=model.omode_occu)
            coeffs[M] = c.to(D.dtype).to(device)
            print(f"M = {M:2d}: fit delta = {delta:.3e}, max|c| = {c.abs().max():.3f}")

        Ny, Nx = D0.shape[-2:]
        _dp = make_dp_fn(model, Ny, Nx)

        # eps on the calibration views (each model's own fit views)
        dp_ms = _dp(D0 + T)
        eps_calib = {}
        for M in ORDERS:
            F = D0 + (coeffs[M].view(-1, 1, 1, 1, 1, 1) * D[:M]).sum(dim=0)
            eps_calib[M] = (
                ((_dp(F) - dp_ms).norm() / dp_ms.norm()).item()
            )
        del D0, D, T, dp_ms

        # ---- stream the full scan -----------------------------------------
        num_fit = {M: 0.0 for M in ORDERS}
        num_plain = {M: 0.0 for M in ORDERS}
        den = 0.0
        all_idx = torch.arange(n_pos, device=device)
        for k, sl in enumerate(all_idx.split(CHUNK)):
            D0c, Dc, Tc = basis_and_target(model, sl, n_max)
            dp_ref = _dp(D0c + Tc)
            den += dp_ref.square().sum().item()
            for M in ORDERS:
                F_fit = D0c + (coeffs[M].view(-1, 1, 1, 1, 1, 1) * Dc[:M]).sum(dim=0)
                F_pln = D0c + Dc[:M].sum(dim=0)
                num_fit[M] += (_dp(F_fit) - dp_ref).square().sum().item()
                num_plain[M] += (_dp(F_pln) - dp_ref).square().sum().item()
            if k % 64 == 0:
                print(f"  full scan: {min((k + 1) * CHUNK, n_pos)}/{n_pos} views")

    eps_fit_full = {M: (num_fit[M] / den) ** 0.5 for M in ORDERS}
    eps_plain_full = {M: (num_plain[M] / den) ** 0.5 for M in ORDERS}

    with open(OUT_CSV, "w") as f:
        f.write("M,plain_eps_int_fullscan,fitted_eps_int_fullscan,fitted_eps_int_calib32\n")
        for M in ORDERS:
            f.write(
                f"{M},{eps_plain_full[M]:.10g},{eps_fit_full[M]:.10g},{eps_calib[M]:.10g}\n"
            )

    Ms = np.array(ORDERS)
    fig, ax = plt.subplots(figsize=(7.6, 5.7), dpi=150)
    ax.plot(
        Ms, [eps_plain_full[M] for M in ORDERS],
        "-o", color="#c0392b", lw=2.2, ms=7, label=r"$c_m = 1$ (full scan)",
    )
    ax.plot(
        Ms, [eps_calib[M] for M in ORDERS],
        "-s", color="#9dc3e6", lw=4.0, ms=11, label=r"fitted $c_m$ (32-view fit)",
    )
    ax.plot(
        Ms, [eps_fit_full[M] for M in ORDERS],
        "-s", color="#1f3864", lw=1.8, ms=5,
        label=r"32-view fitted $c_m$ on full scan (4096 views)",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ms)
    ax.set_xticklabels([str(M) for M in ORDERS])
    ax.set_xlabel("order $M$")
    ax.set_ylabel("relative intensity error vs multislice")
    ax.grid(alpha=0.4, which="both")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"wrote {OUT_PNG} and {OUT_CSV}")


if __name__ == "__main__":
    main()

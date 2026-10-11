"""Detector-pattern montages of the coefficient-fitted Born series (PSO).

Recreates four figures from the 2026-09-19/21 QR-fit study (originals made
by unsaved inline scripts; recreated 2026-09-30):

  demo/born_qr_term_buildup_M6.png   cumulative partial sums of the M = 6
                                     fit: Phi_0, + c1 Phi_1, ..., + c6 Phi_6
                                     vs multislice, with |diff| panels. The
                                     intermediate rel. errors blow up through
                                     the multiple-scattering hump and cancel
                                     only in the full sum.
  demo/born_qr_terms_M6.png          the individual terms c_m Phi_m with
                                     their field-norm ratios ||c_m Phi_m|| /
                                     ||psi_MS|| — the cancellation budget.
  demo/born_recon_detectors_optc.png fitted-c model vs multislice at
                                     n = 1, 2, 4, 8, 16.
  demo/born_recon_detectors_plain.png same with the plain series c = 1.

Everything is computed on the converged 21-slice PSO multislice benchmark
(the same replay as analysis/born_term_norms.py): one order-16 detector
basis + exact multislice target on the 32 evenly-spaced calibration views,
coefficients from the production TSVD solve. Displayed pattern = the middle
calibration view; the quoted rel. err is the global L2 intensity error over
all 32 views (detector blur applied to both sides, matching the loss).
Note: the originals predate the ridge -> TSVD change, so high-order fitted
rel. errors land lower here (e.g. n = 16: ~1e-5 vs the ridge-era 5.5e-4).

Run (from the repo root, ~2 min on an A100):
  CUDA_VISIBLE_DEVICES=<UUID> .venv/bin/python analysis/born_detector_montages.py
"""

import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import matplotlib.pyplot as plt
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
DEMO = os.path.join(REPO, "demo")
N_CALIB = 32
N_MAX = 16
BUILDUP_M = 6
ORDERS = [1, 2, 4, 8, 16]
GAMMA = 0.25  # display: dp**GAMMA, magma


def build_model(device):
    os.chdir(DEMO)  # params reference data/ relative to demo/
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


def show(ax, img, vmax, cmap="magma"):
    ax.imshow(img.clamp(min=0).pow(GAMMA).cpu(), cmap=cmap, vmin=0, vmax=vmax**GAMMA)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device)
    n_pos = model.crop_pos.shape[0]
    idx = torch.linspace(0, n_pos - 1, N_CALIB).long().to(device)
    w64 = model.omode_occu.to(torch.float64).view(1, 1, -1)

    with torch.no_grad():
        D0_p, D_p, T_p, d0n2 = [], [], [], 0.0
        for sl in idx.split(8):
            patches = model.get_obj_patches(sl)
            probes = model.get_probes(sl)
            H3 = model.get_propagators_3d(model.get_propagators(sl))
            D0c, Dc = born_detector_basis(patches, probes, H3, N_MAX)
            T_p.append(born_multislice_target(patches, probes, H3) - D0c)
            D_p.append(Dc)
            D0_p.append(D0c)
            d0n2 += (
                (D0c.abs().square().sum(dim=(-2, -1)).to(torch.float64) * w64).sum().item()
            )
        D0 = torch.cat(D0_p, dim=0)
        D = torch.cat(D_p, dim=1)
        T = torch.cat(T_p, dim=0)

        Ny, Nx = D0.shape[-2:]
        norm_weight = (model.omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)

        def _dp(field):
            dp = torch.fft.fftshift(
                torch.sum(field.abs().square() * norm_weight, dim=(1, 2)), dim=(-2, -1)
            )
            if model.detector_blur_std:
                dp = gaussian_blur(dp, kernel_size=[5, 5], sigma=model.detector_blur_std)
            return dp

        def rel_err(dp_model, dp_ref):
            return ((dp_model - dp_ref).norm() / dp_ref.norm()).item()

        wf = model.omode_occu.to(torch.float64).clamp(min=0).view(1, 1, -1, 1, 1)

        def fnorm(field):
            return (field.abs().square().to(torch.float64) * wf).sum().sqrt().item()

        psi_ms = D0 + T
        dp_ms = _dp(psi_ms)
        v = N_CALIB // 2  # displayed view
        ms_img = dp_ms[v]
        vmax = ms_img.max().item()
        ms_norm = fnorm(psi_ms)

        coeffs = {}
        for M in sorted(set(ORDERS + [BUILDUP_M])):
            c, _ = born_qr_coeffs(D[:M], T, d0n2, omode_occu=model.omode_occu)
            coeffs[M] = c.to(D.dtype)

        # ---- figure A: cumulative buildup at M = BUILDUP_M ----------------
        cM = coeffs[BUILDUP_M]
        rows = BUILDUP_M + 1
        fig, axes = plt.subplots(rows, 3, figsize=(9.6, 3.2 * rows), dpi=110)
        F = D0.clone()
        for m in range(rows):
            if m > 0:
                F = F + cM[m - 1] * D[m - 1]
            dp_m = _dp(F)
            err = rel_err(dp_m, dp_ms)
            title = r"$\Phi_0$" if m == 0 else rf"$+\;c_{{{m}}}\Phi_{{{m}}}$"
            show(axes[m, 0], dp_m[v], vmax)
            axes[m, 0].set_title(rf"{title}  ($M={BUILDUP_M}$ fit)", fontsize=10)
            show(axes[m, 1], ms_img, vmax)
            axes[m, 1].set_title("multislice", fontsize=10)
            show(axes[m, 2], (dp_m[v] - ms_img).abs(), vmax, cmap="magma")
            axes[m, 2].set_title(rf"|diff|  (rel. err {err:.1e})", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(DEMO, f"born_qr_term_buildup_M{BUILDUP_M}.png"))
        plt.close(fig)

        # ---- figure B: individual terms at M = BUILDUP_M ------------------
        fig, axes = plt.subplots(rows, 2, figsize=(6.6, 3.2 * rows), dpi=110)
        for m in range(rows):
            term = D0 if m == 0 else cM[m - 1] * D[m - 1]
            ratio = fnorm(term) / ms_norm
            label = (
                r"$\Phi_0$" if m == 0 else rf"$c_{{{m}}}\Phi_{{{m}}}$"
            )
            show(axes[m, 0], _dp(term)[v], vmax)
            axes[m, 0].set_title(
                rf"{label}  ($\|{label.strip('$')}\|/\|\hat\psi_{{\rm MS}}\|$"
                rf" = {ratio:.2g})",
                fontsize=9,
            )
            show(axes[m, 1], ms_img, vmax)
            axes[m, 1].set_title("multislice", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(DEMO, f"born_qr_terms_M{BUILDUP_M}.png"))
        plt.close(fig)

        # ---- figures C/D: fitted vs plain models at n in ORDERS -----------
        for tag, use_fit in (("optc", True), ("plain", False)):
            fig, axes = plt.subplots(
                len(ORDERS), 3, figsize=(9.6, 3.2 * len(ORDERS)), dpi=110
            )
            for r, M in enumerate(ORDERS):
                c = coeffs[M] if use_fit else torch.ones(M, dtype=D.dtype, device=D.device)
                Fm = D0 + (c.view(-1, 1, 1, 1, 1, 1) * D[:M]).sum(dim=0)
                dp_m = _dp(Fm)
                err = rel_err(dp_m, dp_ms)
                title = (
                    rf"Born $n={M}$, optimized $c$" if use_fit else rf"order $M={M}$, $c=1$"
                )
                show(axes[r, 0], dp_m[v], vmax)
                axes[r, 0].set_title(title, fontsize=10)
                show(axes[r, 1], ms_img, vmax)
                axes[r, 1].set_title("multislice", fontsize=10)
                show(axes[r, 2], (dp_m[v] - ms_img).abs(), vmax)
                axes[r, 2].set_title(rf"|diff|  (rel. err {err:.1e})", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(DEMO, f"born_recon_detectors_{tag}.png"))
            plt.close(fig)

    print("wrote born_qr_term_buildup, born_qr_terms, born_recon_detectors_{optc,plain}")


if __name__ == "__main__":
    main()

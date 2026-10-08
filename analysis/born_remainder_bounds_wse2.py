"""Relative remainder of the truncated Born series with bounds and tuned
coefficients — tBL-WSe2.

Extends born_remainder_wse2.py. For each truncation order M = 1..12 and both
perturbations (chord Delta O = O - 1, tangent Delta O = i phi), computes:

  * measured   ||T - psihat^{(<=M)}|| / ||T||            (plain unit weights)
  * tuned      ||T - psihat_c^{(<=M)}|| / ||T||          with least-squares
    coefficients c (the born_qr_coeffs fit, paper Eq. weighted-sum) fitted
    at batch size 1 — one independent fit per view against the true
    multislice target, so tuned <= plain holds view by view, not just in
    aggregate. Solved fully in complex128 on a per-view column-normalized
    basis (born_qr_coeffs returns complex64 c and its basis columns span
    ~14 decades at high M, which pushed the tuned curve above the plain
    one at the round-off floor)
  * bound A (a posteriori, triangle inequality on the computed term norms):
      chord:   sum_{m>M} ||D_m|| / ||T||
      tangent: sum_{m>M} ||D_m|| / ||T||  +  ||T - tangent full series|| / ||T||
    — computable without the multislice solve for the chord (run the series
    out to N and sum the neglected term norms)
  * bound B (a priori, sup-norm path counting): each order-m term is a sum of
    C(N,m) paths; unitary propagators and pointwise slice multiplications give
      ||psihat^{(m)}|| <= e_m(eps_1..eps_N) ||P||,
    with eps_j = max_x |Delta O_j(x)| over the view's patch and e_m the
    elementary symmetric polynomial, so
      ||R_M|| / ||T|| <= (||P|| / ||T||) sum_{m>M} e_m(eps).
    For the tangent the limit differs from multislice; the gap is bounded by
    the product-difference inequality
      ||MS - prod(1 + i phi)|| <= ||P|| sum_j h_j prod_{k != j} max(a_k, b_k),
    h_j = max|O_j - 1 - i phi_j|, a_k = max|O_k|, b_k = max|1 + i phi_k|.

Same complex128 + unimodular-H treatment as born_remainder_wse2.py.

Run (from the repo root):
  CUDA_VISIBLE_DEVICES=0 ~/ptyrad/.venv/bin/python analysis/born_remainder_bounds_wse2.py
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
OUT_PNG = os.path.join(REPO, "demo", "born_remainder_bounds_wse2.png")
OUT_CSV = os.path.join(REPO, "demo", "born_remainder_bounds_wse2.csv")
N_VIEWS = 32
N_ORDER = 12

# dataviz reference palette (light mode): series hues + neutral chart ink
C_PLAIN = "#2a78d6"  # categorical slot 1, blue  — measured plain series
C_TUNED = "#eb6834"  # categorical slot 2, orange — tuned coefficients
C_BOUND_A = "#52514e"  # secondary ink — a posteriori bound (reference line)
C_BOUND_B = "#898781"  # muted ink — a priori bound (reference line)
C_GRID = "#e1e0d9"
C_SURFACE = "#fcfcfb"


def detector_orders(obj, Psi_state, H, n_max):
    orders = []
    for k in range(n_max):
        scat = _born_scatter(obj, Psi_state, H, k)
        if k == n_max - 1:
            orders.append(scat.sum(dim=3))
        else:
            Psi_state, D_k = _born_advance(scat, H, k)
            orders.append(D_k.clone())
    return torch.stack(orders)


def esp_tails(eps):
    """eps: (B, Nz) numpy. Returns (B, Nz) where out[v, M-1] = sum_{m>M} e_m,
    with e_m the elementary symmetric polynomials of eps[v]."""
    B, Nz = eps.shape
    out = np.zeros((B, Nz))
    for v in range(B):
        poly = np.array([1.0])
        for e in eps[v]:
            poly = np.convolve(poly, [1.0, e])
        e_m = poly[1:]  # e_1..e_Nz (coefficient of t^m in prod(1 + eps t))
        tails = np.cumsum(e_m[::-1])[::-1]  # tails[m-1] = sum_{k>=m} e_k
        out[v, : Nz - 1] = tails[1:]
        out[v, Nz - 1] = 0.0
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.chdir(os.path.join(REPO, "demo"))
    params = load_params(PARAMS, validate=True)
    ip = params["init_params"]
    ip["obj_source"], ip["obj_params"] = "PtyRAD", CKPT
    ip["probe_source"], ip["probe_params"] = "PtyRAD", CKPT
    ip["pos_source"], ip["pos_params"] = "PtyRAD", CKPT
    ip["obj_Nlayer"], ip["obj_slice_thickness"] = 12, 1.0
    params["recon_params"]["if_quiet"] = True
    solver = PtyRADSolver(params, device=device, seed=42)
    model = PtychoAD(
        solver.init.init_variables, params["model_params"], device=device, verbose=False
    )

    n_pos = model.crop_pos.shape[0]
    idx = torch.linspace(0, n_pos - 1, N_VIEWS).long().to(device)
    wf = model.omode_occu.to(torch.float64).clamp(min=0).view(1, 1, -1, 1, 1)

    def vnorm(field):
        return (field.abs().square().to(torch.float64) * wf).sum(dim=(1, 2, 3, 4)).sqrt()

    D0_l, T_l = [], []
    D_l = {"chord": [], "tangent": []}
    eps_l = {"chord": [], "tangent": []}
    plateau_prior_l = []
    with torch.no_grad():
        for sl in idx.split(8):
            patches = model.get_obj_patches(sl).to(torch.float64)
            probes = model.get_probes(sl).to(torch.complex128)
            H1 = model.get_propagators(sl).to(torch.complex128)
            H1 = H1 / H1.abs()  # exact unimodularity (see born_remainder_wse2.py)
            powers = [torch.ones_like(H1)]
            for _ in range(model.n_slice - 1):
                powers.append(powers[-1] * H1)
            H3 = torch.stack(powers, dim=1).unsqueeze(1).unsqueeze(1)

            B, omode, Nz, Ny, Nx, _ = patches.shape
            probe_k = fft2(probes).view(-1, probes.shape[1], 1, 1, Ny, Nx)
            Psi0 = ifft2(H3 * probe_k)
            D0 = probe_k.squeeze(3).expand(B, -1, omode, -1, -1)

            amp, ph = patches[..., 0], patches[..., 1]
            O = torch.polar(amp, ph)
            obj_chord = (O - 1.0).unsqueeze(1)
            obj_tan = (1j * ph.to(torch.complex128)).unsqueeze(1)

            # per-view, per-slice sup-norm perturbation strengths (over omode too)
            eps_l["chord"].append(
                (O - 1.0).abs().amax(dim=(1, 3, 4)).cpu().numpy()
            )
            eps_l["tangent"].append(ph.abs().amax(dim=(1, 3, 4)).cpu().numpy())
            h = (O - 1.0 - 1j * ph).abs().amax(dim=(1, 3, 4)).cpu().numpy()  # (B, Nz)
            a = amp.amax(dim=(1, 3, 4)).cpu().numpy()
            b = np.sqrt(1.0 + eps_l["tangent"][-1] ** 2)
            g = np.maximum(a, b)
            # sum_j h_j prod_{k != j} g_k, per view
            prod_g = g.prod(axis=1, keepdims=True)
            plateau_prior_l.append((h * prod_g / g).sum(axis=1))

            T_l.append(born_multislice_target(patches, probes, H3))
            D0_l.append(D0)
            for name, obj in (("chord", obj_chord), ("tangent", obj_tan)):
                D_l[name].append(detector_orders(obj, Psi0, H3, N_ORDER))

        D0 = torch.cat(D0_l, dim=0)
        T = torch.cat(T_l, dim=0)
        D = {k: torch.cat(v, dim=1) for k, v in D_l.items()}
        eps = {k: np.concatenate(v, axis=0) for k, v in eps_l.items()}
        plateau_prior = np.concatenate(plateau_prior_l)

        n_T = vnorm(T)
        n_D0 = vnorm(D0)

        # sqrt occupancy weights for the fit norm (matches vnorm)
        w_fit = wf.sqrt().unsqueeze(0)  # (1, 1, 1, omode, 1, 1)

        B_all = T.shape[0]
        curves = {}  # (model, kind) -> (N_ORDER, N_VIEWS)
        for name in ("chord", "tangent"):
            term = torch.stack([vnorm(D[name][m]) for m in range(N_ORDER)])  # (12, B)
            # measured plain + tuned remainders. The tuned fit is the
            # born_qr_coeffs least squares at batch size 1: one independent
            # solve per view, fully in complex128 on a per-view
            # column-normalized basis, and the tuned remainder is evaluated
            # as a correction to the plain remainder field — c = 1 is
            # feasible per view, so tuned <= plain view by view beyond
            # float64 round-off.
            plain, tuned = [], []
            psi_leq = D0.clone()
            for M in range(1, N_ORDER + 1):
                psi_leq = psi_leq + D[name][M - 1]
                R_plain = T - psi_leq  # = T_tail of the order-M fit
                plain.append((vnorm(R_plain) / n_T).cpu())
                b = (R_plain * w_fit.squeeze(0)).reshape(B_all, -1, 1)
                s_v = term[:M].to(torch.complex128)  # per-view column norms (M, B)
                A = (
                    (D[name][:M] * w_fit).reshape(M, B_all, -1)
                    / s_v.unsqueeze(-1)
                ).permute(1, 2, 0)  # (B, L, M)
                y = torch.linalg.lstsq(
                    A.cpu(), b.cpu(), driver="gelsd"
                ).solution[..., 0].to(A.device)  # (B, M)
                x = (y.T / s_v).view(M, B_all, 1, 1, 1, 1)
                # nothing left to fit for a view: keep unit weights (c -> 1)
                fitted = (
                    b[..., 0].norm(dim=1) >= 1e-13 * n_T
                ).view(1, B_all, 1, 1, 1, 1)
                R_tuned = R_plain - (x * D[name][:M] * fitted).sum(dim=0)
                tuned.append((vnorm(R_tuned) / n_T).cpu())
            curves[name, "plain"] = torch.stack(plain).numpy()
            curves[name, "tuned"] = torch.stack(tuned).numpy()

            # bound A: triangle tail of computed term norms
            tail = torch.flip(torch.cumsum(torch.flip(term, [0]), 0), [0])  # sum_{m>=M}
            bound_a = torch.zeros_like(term)
            bound_a[:-1] = tail[1:]
            bound_a = (bound_a / n_T).cpu().numpy()
            if name == "tangent":
                bound_a = bound_a + curves["tangent", "plain"][-1][None, :]
            curves[name, "boundA"] = bound_a

            # bound B: a priori sup-norm path counting
            bound_b = esp_tails(eps[name]).T * (n_D0 / n_T).cpu().numpy()[None, :]
            if name == "tangent":
                bound_b = bound_b + (
                    plateau_prior * (n_D0 / n_T).cpu().numpy()
                )[None, :]
            curves[name, "boundB"] = bound_b

    m_axis = np.arange(1, N_ORDER + 1)
    kinds = ["plain", "tuned", "boundA", "boundB"]
    with open(OUT_CSV, "w") as f:
        cols = [f"{n}_{k}_{s}" for n in ("chord", "tangent") for k in kinds
                for s in ("mean", "min", "max")]
        f.write("M," + ",".join(cols) + "\n")
        for m in range(N_ORDER):
            row = [str(m + 1)]
            for n in ("chord", "tangent"):
                for k in kinds:
                    v = curves[n, k][m]
                    row += [f"{v.mean():.8g}", f"{v.min():.8g}", f"{v.max():.8g}"]
            f.write(",".join(row) + "\n")

    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 5.4), dpi=150, sharey=True, facecolor=C_SURFACE
    )
    panel = {
        "chord": r"chord $\Delta O = O - 1$ (non-linearised)",
        "tangent": r"tangent $\Delta O = i\varphi$ (linearised)",
    }
    for ax, name in zip(axes, panel):
        ax.set_facecolor(C_SURFACE)
        v = curves[name, "boundB"]
        ax.plot(m_axis, v.mean(axis=1), ls=":", lw=1.8, color=C_BOUND_B, zorder=2)
        ax.annotate("a priori bound", (m_axis[2], v.mean(axis=1)[2]),
                    textcoords="offset points", xytext=(6, 6),
                    color=C_BOUND_B, fontsize=9)
        v = curves[name, "boundA"]
        ax.plot(m_axis, v.mean(axis=1), ls="--", lw=1.8, color=C_BOUND_A, zorder=3)
        ax.annotate("term-norm tail bound", (m_axis[4], v.mean(axis=1)[4]),
                    textcoords="offset points", xytext=(6, 5),
                    color=C_BOUND_A, fontsize=9)
        for kind, color, marker, label in (
            ("plain", C_PLAIN, "o", "measured, unit weights"),
            ("tuned", C_TUNED, "s", "tuned $c_m$ (per-view fit, batch 1)"),
        ):
            v = curves[name, kind]
            ax.fill_between(m_axis, v.min(axis=1), v.max(axis=1),
                            color=color, alpha=0.15, lw=0)
            ax.plot(m_axis, v.mean(axis=1), marker=marker, ms=5.5, lw=2.0,
                    color=color, label=label, zorder=5)
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 3e2)
        ax.set_xticks(m_axis)
        ax.set_xlabel("truncation order $M$")
        ax.set_title(panel[name], fontsize=11)
        ax.grid(alpha=0.9, which="major", color=C_GRID, lw=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"$\|\hat R_M\| \,/\, \|\hat\psi_{\mathrm{MS}}\|$")
    axes[0].legend(loc="lower left", framealpha=0.9, fontsize=9)
    fig.suptitle(
        f"Relative detector-plane remainder vs truncation order — tBL-WSe$_2$, "
        f"12 slices, {N_VIEWS} views (bands: view range)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, facecolor=C_SURFACE)
    print(f"wrote {OUT_PNG} and {OUT_CSV}")
    for name in ("chord", "tangent"):
        print(f"--- {name}")
        for m in range(N_ORDER):
            print(
                f"M = {m + 1:2d}: plain {curves[name, 'plain'][m].mean():.3e}  "
                f"tuned {curves[name, 'tuned'][m].mean():.3e}  "
                f"tailA {curves[name, 'boundA'][m].mean():.3e}  "
                f"priorB {curves[name, 'boundB'][m].mean():.3e}"
            )


if __name__ == "__main__":
    main()

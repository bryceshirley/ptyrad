"""Relative remainder of the truncated Born series with bounds and tuned
coefficients — PSO (PrScO3).

The PSO counterpart of born_remainder_bounds_wse2.py (see that script for
the full method notes): detector-plane remainder ||R_M|| / ||psihat_MS||
for truncation orders M = 1..21 on the converged 21-slice (dz 10 A)
multislice reconstruction of the PrScO3 dataset, chord vs tangent, with

  * measured plain (unit weights) remainder
  * tuned coefficients at batch size 1 (one complex128 least-squares fit
    per view on a per-view column-normalized basis)
  * the a priori elementary-symmetric sup-norm bound

All fields complex128; the propagator is renormalized to exact unit
modulus so the chord series terminates to the multislice target at the
float64 floor.

Run (from the repo root):
  CUDA_VISIBLE_DEVICES=0 ~/ptyrad/.venv/bin/python analysis/born_remainder_bounds_pso.py
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
PARAMS = os.path.join(REPO, "demo", "params", "pso_ms_b1_n100.yml")
CKPT = os.path.join(
    os.path.expanduser("~"),
    "ptyrad",
    "demo",
    "output",
    "PSO_ms_paper",
    "20260912_full_N4096_dp256_sparse32_p4_1obj_21slice_dz10_plr1e-4_"
    "oalr5e-4_oplr5e-4_slr5e-4_dpblur1_orblur0.4_ozblur1_mamp0.03_4_"
    "oathr0.96_oposc_sng1.0_spr0.1",
    "model_iter0080.hdf5",
)
OUT_PNG = os.path.join(REPO, "demo", "born_remainder_bounds_pso.png")
OUT_CSV = os.path.join(REPO, "demo", "born_remainder_bounds_pso.csv")
N_VIEWS = 32
N_ORDER = 21
N_LAYER = 21
DZ = 10.0

C_PLAIN = "#2a78d6"
C_TUNED = "#eb6834"
C_BOUND_B = "#898781"
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
    """eps: (B, Nz) numpy. out[v, M-1] = sum_{m>M} e_m(eps[v])."""
    B, Nz = eps.shape
    out = np.zeros((B, Nz))
    for v in range(B):
        poly = np.array([1.0])
        for e in eps[v]:
            poly = np.convolve(poly, [1.0, e])
        e_m = poly[1:]
        tails = np.cumsum(e_m[::-1])[::-1]
        out[v, : Nz - 1] = tails[1:]
        out[v, Nz - 1] = 0.0
    return out


def cone_tails(w1, eps):
    """Cone-aware a priori tails (see born_remainder_bounds_wse2.py): the
    first scattering event of every path acts on the known freely propagated
    (defocused) probe, so its sup-norm factor eps_j ||P|| is replaced by the
    computed w1_j = ||Delta O_j P_j||, which sees the beam cone's footprint
    at slice j exactly; later events keep sup-norm factors.
    out[v, M-1] = sum_j w1[v, j] * sum_{k>=M} e_k(eps[v, j+1:])."""
    B, Nz = eps.shape
    out = np.zeros((B, Nz))
    for v in range(B):
        for j in range(Nz):
            poly = np.array([1.0])
            for e in eps[v, j + 1 :]:
                poly = np.convolve(poly, [1.0, e])
            tails = np.cumsum(poly[::-1])[::-1]
            for M in range(1, Nz + 1):
                if M < len(tails):
                    out[v, M - 1] += w1[v, j] * tails[M]
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.chdir(os.path.join(REPO, "demo"))
    params = load_params(PARAMS, validate=True)
    ip = params["init_params"]
    ip["obj_source"], ip["obj_params"] = "PtyRAD", CKPT
    ip["probe_source"], ip["probe_params"] = "PtyRAD", CKPT
    ip["pos_source"], ip["pos_params"] = "PtyRAD", CKPT
    ip["obj_Nlayer"], ip["obj_slice_thickness"] = N_LAYER, DZ
    params["recon_params"]["if_quiet"] = True
    solver = PtyRADSolver(params, device=device, seed=42)
    model = PtychoAD(
        solver.init.init_variables, params["model_params"], device=device, verbose=False
    )

    n_pos = model.crop_pos.shape[0]
    idx = torch.linspace(0, n_pos - 1, N_VIEWS).long().to(device)
    # fields are staged to CPU as they are computed (21 orders x 2 models of
    # 256px complex128 frames exceed the GPU); all curve arithmetic runs on CPU
    wf = model.omode_occu.to(torch.float64).clamp(min=0).view(1, 1, -1, 1, 1).cpu()

    def vnorm(field):
        return (field.abs().square().to(torch.float64) * wf).sum(dim=(1, 2, 3, 4)).sqrt()

    D0_l, T_l = [], []
    D_l = {"chord": [], "tangent": []}
    eps_l = {"chord": [], "tangent": []}
    w1_l = {"chord": [], "tangent": []}
    plateau_prior_l = []
    with torch.no_grad():
        for sl in idx.split(4):  # 256px frames, 21 slices: small chunks
            patches = model.get_obj_patches(sl).to(torch.float64)
            probes = model.get_probes(sl).to(torch.complex128)
            H1 = model.get_propagators(sl).to(torch.complex128)
            H1 = H1 / H1.abs()  # exact unimodularity
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

            eps_l["chord"].append((O - 1.0).abs().amax(dim=(1, 3, 4)).cpu().numpy())
            eps_l["tangent"].append(ph.abs().amax(dim=(1, 3, 4)).cpu().numpy())
            h = (O - 1.0 - 1j * ph).abs().amax(dim=(1, 3, 4)).cpu().numpy()
            a = amp.amax(dim=(1, 3, 4)).cpu().numpy()
            b = np.sqrt(1.0 + eps_l["tangent"][-1] ** 2)
            g = np.maximum(a, b)
            prod_g = g.prod(axis=1, keepdims=True)
            plateau_prior_l.append((h * prod_g / g).sum(axis=1))

            T_l.append(born_multislice_target(patches, probes, H3).cpu())
            D0_l.append(D0.cpu())
            occ6 = wf.view(1, 1, -1, 1, 1, 1)
            for name, obj in (("chord", obj_chord), ("tangent", obj_tan)):
                D_l[name].append(detector_orders(obj, Psi0, H3, N_ORDER).cpu())
                # first-event norms ||Delta O_j P_j|| for the cone-aware bound
                scat1 = _born_scatter(obj, Psi0, H3, 0)
                w1_l[name].append(
                    (scat1.abs().square().to(torch.float64) * occ6.to(scat1.device))
                    .sum(dim=(1, 2, 4, 5)).sqrt().cpu().numpy()
                )
                del scat1
            del Psi0, obj_chord, obj_tan, O
            torch.cuda.empty_cache()

        D0 = torch.cat(D0_l, dim=0)
        T = torch.cat(T_l, dim=0)
        D = {k: torch.cat(v, dim=1) for k, v in D_l.items()}
        D_l.clear()
        eps = {k: np.concatenate(v, axis=0) for k, v in eps_l.items()}
        w1 = {k: np.concatenate(v, axis=0) for k, v in w1_l.items()}
        plateau_prior = np.concatenate(plateau_prior_l)

        n_T = vnorm(T)
        n_D0 = vnorm(D0)
        w_fit = wf.sqrt().unsqueeze(0)

        def intensity(field):  # occupancy-weighted detector intensity map
            return (field.abs().square().to(torch.float64) * wf).sum(dim=(1, 2))

        I_ref = intensity(T)
        n_I = I_ref.flatten(1).norm(dim=1)  # per-view Frobenius norm

        B_all = T.shape[0]
        curves = {}
        for name in ("chord", "tangent"):
            term = torch.stack([vnorm(D[name][m]) for m in range(N_ORDER)])
            plain, tuned, plain_I, tuned_I = [], [], [], []
            psi_leq = D0.clone()
            for M in range(1, N_ORDER + 1):
                psi_leq = psi_leq + D[name][M - 1]
                R_plain = T - psi_leq
                plain.append((vnorm(R_plain) / n_T).cpu())
                plain_I.append(
                    ((intensity(psi_leq) - I_ref).flatten(1).norm(dim=1) / n_I).cpu()
                )
                b = (R_plain * w_fit.squeeze(0)).reshape(B_all, -1, 1)
                s_v = term[:M].to(torch.complex128)
                A = (
                    (D[name][:M] * w_fit).reshape(M, B_all, -1)
                    / s_v.unsqueeze(-1)
                ).permute(1, 2, 0)
                y = torch.linalg.lstsq(
                    A, b, driver="gelsd"
                ).solution[..., 0]
                x = (y.T / s_v).view(M, B_all, 1, 1, 1, 1)
                fitted = (
                    b[..., 0].norm(dim=1) >= 1e-13 * n_T
                ).view(1, B_all, 1, 1, 1, 1)
                R_tuned = R_plain - (x * D[name][:M] * fitted).sum(dim=0)
                tuned.append((vnorm(R_tuned) / n_T).cpu())
                tuned_I.append(
                    ((intensity(T - R_tuned) - I_ref).flatten(1).norm(dim=1) / n_I)
                    .cpu()
                )
            curves[name, "plain"] = torch.stack(plain).numpy()
            curves[name, "tuned"] = torch.stack(tuned).numpy()
            curves[name, "plainI"] = torch.stack(plain_I).numpy()
            curves[name, "tunedI"] = torch.stack(tuned_I).numpy()

            bound_b = esp_tails(eps[name]).T * (n_D0 / n_T).cpu().numpy()[None, :]
            if name == "tangent":
                bound_b = bound_b + (
                    plateau_prior * (n_D0 / n_T).cpu().numpy()
                )[None, :]
            curves[name, "boundB"] = bound_b

            # cone-aware a priori bound: first event weighted by the
            # defocused probe at its slice (see cone_tails)
            bound_c = cone_tails(w1[name], eps[name]).T / n_T.cpu().numpy()[None, :]
            if name == "tangent":
                bound_c = bound_c + (
                    plateau_prior * (n_D0 / n_T).cpu().numpy()
                )[None, :]
            curves[name, "boundC"] = bound_c

    m_axis = np.arange(1, N_ORDER + 1)
    kinds = ["plain", "tuned", "plainI", "tunedI", "boundB", "boundC"]
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

    # two panels: left = field remainder (with cone-aware a priori bounds),
    # right = detector intensity error. color = model family (chord blue,
    # tangent orange), line style = variant (solid measured, dashed tuned,
    # dotted bound)
    fig, (ax, axI) = plt.subplots(
        1, 2, figsize=(12.6, 5.8), dpi=150, facecolor=C_SURFACE
    )
    for a in (ax, axI):
        a.set_facecolor(C_SURFACE)
        a.set_yscale("log")
        a.set_xticks(np.arange(1, N_ORDER + 1, 2))
        a.set_xlabel("truncation order $M$")
        a.grid(alpha=0.9, which="major", color=C_GRID, lw=0.7)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    series = {
        "chord": (C_PLAIN, r"chord $\Delta O = O - 1$ (non-linearised)"),
        "tangent": (C_TUNED, r"tangent $\Delta O = i\varphi$ (linearised)"),
    }
    bound_labels = {
        "chord":
            r"chord a priori bound (cone) "
            r"$\sum_j \|\Delta O_j P_j\| \sum_{k\geq M} e_k(\varepsilon_{>j})$",
        "tangent":
            r"tangent a priori bound (cone) "
            r"$\sum_j \|\Delta O_j P_j\| \sum_{k\geq M} e_k(\varepsilon_{>j})"
            r" + \|P\|\sum_j h_j \prod_{k\neq j} g_k$",
    }
    for name, (color, mdl_label) in series.items():
        v = curves[name, "boundC"]
        ax.plot(m_axis, v.mean(axis=1), ls=":", lw=1.6, color=color,
                alpha=0.7, zorder=2, label=bound_labels[name])
        v = curves[name, "plain"]
        ax.fill_between(m_axis, v.min(axis=1), v.max(axis=1),
                        color=color, alpha=0.13, lw=0)
        ax.plot(m_axis, v.mean(axis=1), marker="o", ms=5, lw=2.0,
                color=color, zorder=5, label=f"{mdl_label}: measured")
        v = curves[name, "tuned"]
        ax.plot(m_axis, v.mean(axis=1), marker="s", ms=4.5, lw=1.8, ls="--",
                mfc="none", color=color, zorder=5,
                label=f"{mdl_label.split(' ')[0]}: tuned $c_m$ (per-view fit, batch 1)")
        v = curves[name, "plainI"]
        axI.fill_between(m_axis, v.min(axis=1), v.max(axis=1),
                         color=color, alpha=0.13, lw=0)
        axI.plot(m_axis, v.mean(axis=1), marker="o", ms=5, lw=2.0,
                 color=color, zorder=5, label=f"{mdl_label}: measured")
        v = curves[name, "tunedI"]
        axI.plot(m_axis, v.mean(axis=1), marker="s", ms=4.5, lw=1.8, ls="--",
                 mfc="none", color=color, zorder=5,
                 label=f"{mdl_label.split(' ')[0]}: tuned $c_m$")
    ax.set_ylim(bottom=1e-16)
    ax.set_ylabel(r"$\|\hat R_M\| \,/\, \|\hat\psi_{\mathrm{MS}}\|$")
    ax.set_title("field remainder", fontsize=11)
    # legend order: chord measured/tuned/bound, tangent measured/tuned/bound
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 0, 4, 5, 3]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower left", framealpha=0.9, fontsize=8)
    axI.set_ylabel(
        r"$\|\hat I_M - \hat I_{\mathrm{MS}}\|_2 \,/\, \|\hat I_{\mathrm{MS}}\|_2$"
    )
    axI.set_title("detector intensity error", fontsize=11)
    axI.legend(loc="lower left", framealpha=0.9, fontsize=8)
    fig.suptitle(
        f"Truncated Born series vs multislice — PrScO$_3$, {N_LAYER} slices "
        f"of {DZ:g} Å, {N_VIEWS} views (bands: view range)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PNG, facecolor=C_SURFACE)
    print(f"wrote {OUT_PNG} and {OUT_CSV}")
    for name in ("chord", "tangent"):
        print(f"--- {name}")
        for m in range(N_ORDER):
            print(
                f"M = {m + 1:2d}: plain {curves[name, 'plain'][m].mean():.3e}  "
                f"tuned {curves[name, 'tuned'][m].mean():.3e}  "
                f"priorB {curves[name, 'boundB'][m].mean():.3e}  "
                f"cone {curves[name, 'boundC'][m].mean():.3e}"
            )


if __name__ == "__main__":
    main()

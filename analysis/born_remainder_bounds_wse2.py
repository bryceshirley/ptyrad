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
  * a priori bound (sup-norm path counting): each order-m term is a sum of
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
OUT_ALT = os.path.join(REPO, "demo", "born_remainder_bounds_alt_wse2.png")
OUT_CSV = os.path.join(REPO, "demo", "born_remainder_bounds_wse2.csv")
N_VIEWS = 32
N_ORDER = 12

# dataviz reference palette (light mode): series hues + neutral chart ink
C_PLAIN = "#2a78d6"  # categorical slot 1, blue  — measured plain series
C_TUNED = "#eb6834"  # categorical slot 2, orange — tuned coefficients
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


def cone_tails(w1, eps):
    """Cone-aware a priori tails. w1, eps: (B, Nz) numpy.

    The first scattering event of every path acts on the KNOWN freely
    propagated (defocused) probe, so its sup-norm factor eps_j ||P|| is
    replaced by the computed w1_j = ||Delta O_j P_j|| — this term sees the
    beam cone's footprint, position and defocus spread at slice j exactly.
    Later events keep sup-norm factors: after a scattering event the field
    is no longer confined to the vacuum cone, so without an assumption on
    the object's maximum scattering angle no smaller support is rigorous.

    out[v, M-1] = sum_j w1[v, j] * sum_{k>=M} e_k(eps[v, j+1:]).
    """
    B, Nz = eps.shape
    out = np.zeros((B, Nz))
    for v in range(B):
        for j in range(Nz):
            poly = np.array([1.0])
            for e in eps[v, j + 1 :]:
                poly = np.convolve(poly, [1.0, e])
            tails = np.cumsum(poly[::-1])[::-1]  # tails[k] = sum_{i>=k} e_i
            for M in range(1, Nz + 1):
                if M < len(tails):
                    out[v, M - 1] += w1[v, j] * tails[M]
    return out


def pair_tails(w2, eps):
    """Pair (second-order cone) a priori tails. w2: (B, Nz, Nz) upper
    triangular, eps: (B, Nz) numpy.

    One step beyond cone_tails: for a path j1 < j2 < ..., the field arriving
    at the SECOND event is also fully known without any forward solve — it is
    the once-scattered, freely propagated probe — so both of the first two
    sup-norm factors are replaced by the computed pair weight
    w2[j1, j2] = ||Delta O_{j2} P_{z_{j2}-z_{j1}}[Delta O_{j1} P_{j1}]||.
    Events three onward keep sup-norm factors. Valid as a remainder bound
    for every M >= 1 (all neglected orders have m >= 2):

    out[v, M-1] = sum_{j1<j2} w2[v, j1, j2] * sum_{k>=M-1} e_k(eps[v, j2+1:]).
    """
    B, Nz = eps.shape
    out = np.zeros((B, Nz))
    for v in range(B):
        for j2 in range(1, Nz):
            colsum = w2[v, :j2, j2].sum()
            if colsum == 0.0:
                continue
            poly = np.array([1.0])
            for e in eps[v, j2 + 1 :]:
                poly = np.convolve(poly, [1.0, e])
            tails = np.cumsum(poly[::-1])[::-1]  # tails[k] = sum_{i>=k} e_i
            for M in range(1, Nz + 1):
                if M - 1 < len(tails):
                    out[v, M - 1] += colsum * tails[M - 1]
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
    w1_l = {"chord": [], "tangent": []}
    w2_l = {"chord": [], "tangent": []}
    h_l, a_l, u_l, plateau_prior_l = [], [], [], []
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
            h_l.append(h)
            a_l.append(a)
            # ||E_j P_{z_j} P|| with E_j = O_j - 1 - i phi_j: the computable
            # single-scattering part of the linearisation gap (cone gauge:
            # k-space norm, hence the sqrt(Ny*Nx) fft factor)
            occ6 = wf.view(1, 1, -1, 1, 1, 1)
            err = (O - 1.0 - 1j * ph).unsqueeze(1) * Psi0
            u_l.append(
                (err.abs().square().to(torch.float64) * occ6.to(err.device))
                .sum(dim=(1, 2, 4, 5)).sqrt().cpu().numpy() * np.sqrt(Ny * Nx)
            )
            del err

            T_l.append(born_multislice_target(patches, probes, H3))
            D0_l.append(D0)
            for name, obj in (("chord", obj_chord), ("tangent", obj_tan)):
                D_l[name].append(detector_orders(obj, Psi0, H3, N_ORDER))
                # first-event norms ||Delta O_j P_{z_j} P|| for the cone bound
                # (the H.conj() in _born_scatter is unimodular: norm unchanged)
                scat1 = _born_scatter(obj, Psi0, H3, 0)
                w1_l[name].append(
                    (scat1.abs().square().to(torch.float64) * occ6.to(scat1.device))
                    .sum(dim=(1, 2, 4, 5)).sqrt().cpu().numpy()
                )
                # pair weights ||Delta O_k P_{z_k-z_j}[Delta O_j P_{z_j} P]||,
                # the double-scattering kernels of Eq. (double-scattering):
                # scat1[j] carries conj(H^j), so scat1[j] * H^k propagates by
                # z_k - z_j (H is unimodular)
                w2v = torch.zeros(B, Nz, Nz, dtype=torch.float64)
                for j1 in range(Nz - 1):
                    prop = ifft2(
                        scat1[..., j1 : j1 + 1, :, :] * H3[..., j1 + 1 :, :, :]
                    )
                    pair = obj[..., j1 + 1 :, :, :] * prop
                    w2v[:, j1, j1 + 1 :] = (
                        (pair.abs().square().to(torch.float64) * occ6.to(pair.device))
                        .sum(dim=(1, 2, 4, 5)).sqrt().cpu()
                    ) * np.sqrt(Ny * Nx)
                w2_l[name].append(w2v.numpy())
                del scat1

        D0 = torch.cat(D0_l, dim=0)
        T = torch.cat(T_l, dim=0)
        D = {k: torch.cat(v, dim=1) for k, v in D_l.items()}
        eps = {k: np.concatenate(v, axis=0) for k, v in eps_l.items()}
        w1 = {k: np.concatenate(v, axis=0) for k, v in w1_l.items()}
        w2 = {k: np.concatenate(v, axis=0) for k, v in w2_l.items()}
        h_arr = np.concatenate(h_l)
        a_arr = np.concatenate(a_l)
        u_arr = np.concatenate(u_l)
        plateau_prior = np.concatenate(plateau_prior_l)

        n_T = vnorm(T)
        n_D0 = vnorm(D0)

        I_ref = (T.abs().square().to(torch.float64) * wf).sum(dim=(1, 2))
        n_I = I_ref.flatten(1).norm(dim=1)  # per-view Frobenius norm

        def intensity_err(R):
            """Relative detector intensity error of the field T - R.
            I(T - R) - I(T) = |R|^2 - 2 Re(conj(T) R), evaluated directly:
            subtracting two O(1) intensity maps loses the difference to
            round-off once it falls below ~1e-13, which showed up as kinks
            at the floor of the tuned curve."""
            dI = (
                (R.abs().square() - 2 * (T.conj() * R).real).to(torch.float64) * wf
            ).sum(dim=(1, 2))
            return (dI.flatten(1).norm(dim=1) / n_I).cpu()

        # sqrt occupancy weights for the fit norm (matches vnorm)
        w_fit = wf.sqrt().unsqueeze(0)  # (1, 1, 1, omode, 1, 1)

        # tangent linearisation gap, cone-gauged: the tangent partial product
        # through slices i < j is P_{z_j} P plus a scattered part of norm at
        # most up_j = sum_{i<j} eps_i prod_{i<k<j} b_k (telescoping,
        # b_k = max|1 + i phi_k|), so the insertion at j costs
        # ||E_j P_{z_j} P|| + h_j up_j ||P||, carried downstream by
        # prod_{k>j} a_k. Like the cone weights, u_j = ||E_j P_{z_j} P|| uses
        # only the beam intensity footprint — no scattering computation. The
        # WSe2 linearisation error has a nearly uniform amplitude component
        # where probe weighting gains nothing, so keep the better of the two
        # valid gap bounds (the min of two upper bounds is an upper bound).
        t_tan = eps["tangent"]
        B_v = t_tan.shape[0]
        ones = np.ones((B_v, 1))
        b_tan = np.sqrt(1.0 + t_tan**2)
        cb = np.cumprod(b_tan, axis=1)
        c_excl = np.concatenate([ones, cb[:, :-1]], axis=1)
        s = np.cumsum(t_tan / cb, axis=1)
        s_excl = np.concatenate([np.zeros((B_v, 1)), s[:, :-1]], axis=1)
        up = c_excl * s_excl
        sa = np.cumprod(a_arr[:, ::-1], axis=1)[:, ::-1]
        down = np.concatenate([sa[:, 1:], ones], axis=1)
        nD0np = n_D0.cpu().numpy()
        gap_cone = (down * (u_arr + h_arr * up * nD0np[:, None])).sum(axis=1)
        gap_tan = np.minimum(gap_cone, plateau_prior * nD0np)

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
            plain, tuned, plain_I, tuned_I = [], [], [], []
            psi_leq = D0.clone()
            for M in range(1, N_ORDER + 1):
                psi_leq = psi_leq + D[name][M - 1]
                R_plain = T - psi_leq  # = T_tail of the order-M fit
                plain.append((vnorm(R_plain) / n_T).cpu())
                plain_I.append(intensity_err(R_plain))
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
                tuned_I.append(intensity_err(R_tuned))
            curves[name, "plain"] = torch.stack(plain).numpy()
            curves[name, "tuned"] = torch.stack(tuned).numpy()
            curves[name, "plainI"] = torch.stack(plain_I).numpy()
            curves[name, "tunedI"] = torch.stack(tuned_I).numpy()

            # a priori bound: sup-norm path counting
            bound_b = esp_tails(eps[name]).T * (n_D0 / n_T).cpu().numpy()[None, :]
            if name == "tangent":
                bound_b = bound_b + (
                    plateau_prior * (n_D0 / n_T).cpu().numpy()
                )[None, :]
            curves[name, "boundB"] = bound_b

            # cone-aware a priori bound: first event weighted by the beam
            # intensity footprint at its slice (see cone_tails); the tangent
            # adds the cone-gauged linearisation gap, a priori by the same
            # standard
            bound_c = cone_tails(w1[name], eps[name]).T / n_T.cpu().numpy()[None, :]
            if name == "tangent":
                bound_c = bound_c + (gap_tan / n_T.cpu().numpy())[None, :]
            curves[name, "boundC"] = bound_c

            # pair bound: the first TWO events carry their exact norms (the
            # double-scattering kernels of Eq. (double-scattering)). NOT
            # fully a priori: it evaluates order-2 kernels, so it only
            # certifies truncations M >= 2 without computing neglected
            # orders (the order-M model evaluates its own order-M kernels
            # anyway); at M = 1 its leading term is the triangle inequality
            # on psihat^(2) — the paper's residual monitor turned into a
            # full-tail bound.
            bound_d = pair_tails(w2[name], eps[name]).T / n_T.cpu().numpy()[None, :]
            if name == "tangent":
                bound_d = bound_d + (gap_tan / n_T.cpu().numpy())[None, :]
            curves[name, "boundD"] = bound_d

    m_axis = np.arange(1, N_ORDER + 1)
    kinds = ["plain", "tuned", "plainI", "tunedI", "boundB", "boundC", "boundD"]
    for name in ("chord", "tangent"):  # a bound that dips below measured is a bug
        for kind in ("boundB", "boundC", "boundD"):
            ok = np.all(curves[name, kind] >= curves[name, "plain"] - 1e-12)
            print(f"{kind} >= measured for every view and order ({name}): {ok}")
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
        1, 2, figsize=(12.6, 6.8), dpi=150, facecolor=C_SURFACE
    )
    for a in (ax, axI):
        a.set_facecolor(C_SURFACE)
        a.set_yscale("log")
        a.set_xticks(m_axis)
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
            r"chord a priori cone bound $\sum_j \|\Delta O_j"
            r" \mathcal{P}_{z_j} P\| \sum_{n\geq M} e_n(\varepsilon_{>j})$",
        "tangent":
            r"tangent a priori cone bound $\sum_j \|\Delta O_j"
            r" \mathcal{P}_{z_j} P\| \sum_{n\geq M} e_n(\varepsilon_{>j})"
            r"\ +$ lin. gap",
    }
    for name, (color, mdl_label) in series.items():
        v = curves[name, "boundC"]
        ax.plot(m_axis, v.mean(axis=1), ls=":", lw=1.6, color=color,
                alpha=0.7, zorder=2, label=bound_labels[name])
        v = curves[name, "plain"]
        ax.fill_between(m_axis, v.min(axis=1), v.max(axis=1),
                        color=color, alpha=0.13, lw=0)
        ax.plot(m_axis, v.mean(axis=1), marker="o", ms=5.5, lw=2.0,
                color=color, zorder=5, label=f"{mdl_label}: measured")
        v = curves[name, "tuned"]
        ax.plot(m_axis, v.mean(axis=1), marker="s", ms=5, lw=1.8, ls="--",
                mfc="none", color=color, zorder=5,
                label=f"{mdl_label.split(' ')[0]}: tuned $c_m$ (per-view fit, batch 1)")
        v = curves[name, "plainI"]
        axI.fill_between(m_axis, v.min(axis=1), v.max(axis=1),
                         color=color, alpha=0.13, lw=0)
        axI.plot(m_axis, v.mean(axis=1), marker="o", ms=5.5, lw=2.0,
                 color=color, zorder=5)
        v = curves[name, "tunedI"]
        axI.plot(m_axis, v.mean(axis=1), marker="s", ms=5, lw=1.8, ls="--",
                 mfc="none", color=color, zorder=5)
    ax.set_ylim(1e-16, 3e2)
    ax.set_ylabel(r"$\|\hat R_M\| \,/\, \|\hat\psi_{\mathrm{MS}}\|$")
    ax.set_title("field remainder", fontsize=11)
    axI.set_ylabel(
        r"$\|\hat I_M - \hat I_{\mathrm{MS}}\|_2 \,/\, \|\hat I_{\mathrm{MS}}\|_2$"
    )
    axI.set_title("detector intensity error", fontsize=11)
    # one shared legend below both panels (the intensity panel reuses the
    # same styles), keeping the axes large
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 0, 4, 5, 3]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=2, framealpha=0.9, fontsize=9,
               columnspacing=1.4)
    fig.suptitle(
        f"Truncated Born series vs multislice — tBL-WSe$_2$, 12 slices, "
        f"{N_VIEWS} views (bands: view range)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(OUT_PNG, facecolor=C_SURFACE)

    # ------------------------------------------------------------------
    # alternative-bounds comparison: measured vs cone-aware vs sup-norm
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10.0, 7.4), dpi=150, facecolor=C_SURFACE)
    ax2.set_facecolor(C_SURFACE)
    alt_labels = {
        ("chord", "boundD"):
            r"chord pair (order-2 kernels, certifies $M{\geq}2$): "
            r"$\sum_{j<k} \|\Delta O_k \mathcal{P}_{z_k-z_j}"
            r"[\Delta O_j \mathcal{P}_{z_j} P]\| \sum_{n\geq M-1}"
            r" e_n(\varepsilon_{>k})$",
        ("chord", "boundC"):
            r"chord cone (object + beam footprint $|\mathcal{P}_{z_j}P|^2$): "
            r"$\sum_j \|\Delta O_j \mathcal{P}_{z_j} P\|"
            r" \sum_{n\geq M} e_n(\varepsilon_{>j})$",
        ("chord", "boundB"):
            r"chord sup-norm (object only): $\|P\|\sum_{m>M} e_m(\varepsilon)$, "
            r"$\varepsilon_j{=}\max_x|\Delta O_j|$",
        ("tangent", "boundD"):
            r"tangent pair (order-2 kernels) $+$ lin. gap",
        ("tangent", "boundC"):
            r"tangent cone (object + beam footprint) $+$ lin. gap",
        ("tangent", "boundB"):
            r"tangent sup-norm (object only) "
            r"$+\ \|P\|\sum_j h_j \prod_{k\neq j} g_k$",
    }
    for name, (color, mdl_label) in series.items():
        v = curves[name, "plain"]
        ax2.fill_between(m_axis, v.min(axis=1), v.max(axis=1),
                         color=color, alpha=0.13, lw=0)
        ax2.plot(m_axis, v.mean(axis=1), marker="o", ms=5.5, lw=2.0,
                 color=color, zorder=5, label=f"{mdl_label}: measured")
        v = curves[name, "boundD"]
        ax2.plot(m_axis, v.mean(axis=1), ls=(0, (5, 1)), lw=1.9, color=color,
                 zorder=4, label=alt_labels[name, "boundD"])
        v = curves[name, "boundC"]
        ax2.plot(m_axis, v.mean(axis=1), ls=":", lw=1.7, color=color,
                 alpha=0.7, zorder=3, label=alt_labels[name, "boundC"])
        v = curves[name, "boundB"]
        ax2.plot(m_axis, v.mean(axis=1), ls="-.", lw=1.3, color=color,
                 alpha=0.45, zorder=2, label=alt_labels[name, "boundB"])
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=1e-16)
    ax2.set_xticks(m_axis)
    ax2.set_xlabel("truncation order $M$")
    ax2.set_ylabel(r"$\|\hat R_M\| \,/\, \|\hat\psi_{\mathrm{MS}}\|$")
    ax2.grid(alpha=0.9, which="major", color=C_GRID, lw=0.7)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    fig2.legend(loc="lower center", ncol=2, framealpha=0.9, fontsize=8.5,
                columnspacing=1.2)
    ax2.set_title(
        f"A priori remainder bounds: sup-norm vs cone vs pair — tBL-WSe$_2$, "
        f"{N_VIEWS} views",
        fontsize=11,
    )
    fig2.tight_layout(rect=(0, 0.22, 1, 1))
    fig2.savefig(OUT_ALT, facecolor=C_SURFACE)
    print(f"wrote {OUT_ALT}")
    print(f"wrote {OUT_PNG} and {OUT_CSV}")
    for name in ("chord", "tangent"):
        print(f"--- {name}")
        for m in range(N_ORDER):
            print(
                f"M = {m + 1:2d}: plain {curves[name, 'plain'][m].mean():.3e}  "
                f"tuned {curves[name, 'tuned'][m].mean():.3e}  "
                f"priorB {curves[name, 'boundB'][m].mean():.3e}  "
                f"cone {curves[name, 'boundC'][m].mean():.3e}  "
                f"pair {curves[name, 'boundD'][m].mean():.3e}"
            )


if __name__ == "__main__":
    main()

"""
ML-MS: maximum likelihood on the MULTISLICE forward model, built to match
ML-ISS (ptyrad.mliss) as closely as the model allows — same Gaussian
likelihood L_G = sum w/(2 sigma^2) (I_model - I_data)^2 with
sigma^2 = I_data + 1/counts_per_unit (one-count floor, same count scaling),
same l1 handling (logged only, never searched), same alternating
object-then-probe scheme, same damping default (1.0), same fallbacks.

Model: psi_exit_m = O_{N-1} Prop ... O_1 Prop O_0 P_m per probe mode m,
detector field F_m = FFT[psi_exit_m] (unnormalised; the §4.2 unit map
occu/(Nx*Ny) of ptyrad.linesearch.dp_from_fields then reproduces
multislice_forward's norm='ortho' intensity exactly, incl. fftshift + eps).

Differences from ML-ISS, forced by the model:

- The exit field is a DEGREE-N polynomial in gamma along a joint slice
  direction d (not affine), so the exact object step carries N+1 coefficient
  fields through the stack: at slice j the polynomial is multiplied by
  (O_j + gamma d_j) (degree +1), then every coefficient is propagated. The
  zeroth coefficient chain equals the stored forward fields, so it is reused
  rather than recomputed — the sweep then costs exactly N^2 frame transforms
  per position per mode (Table 2 of the paper).
- The misfit along gamma is a degree-4N polynomial; its coefficients are NOT
  formed (degree 47 at N = 12, badly conditioned). L_G(gamma) is instead
  evaluated pixelwise from the detector coefficient fields by Horner's rule
  (double precision, no transforms) and minimised in 1D by bracketing +
  Brent (scipy). Evaluation counts are logged (state.ls_evals).
- The probe step IS the same exact cubic step as ML-ISS: multislice is
  linear in the probe, so along d_P the field is F + gamma a_P with a_P from
  ONE forward pass through the UPDATED slices; ml_line_search is reused.

Preconditioning (the ePIE/3PIE analogue of ML-ISS):
- slice j: divide the gradient by the spatial maximum of the
  batch-accumulated sum_{views, pmode} |psi_inc_j|^2, where psi_inc_j is the
  field INCIDENT on slice j (computed with the object, i.e. the 3PIE
  illumination), scatter-accumulated onto the object canvas across the
  batch, one scalar per slice ('max' recipe, matching ML-ISS's K_j which
  uses the vacuum illumination — vacuum and 3PIE coincide for ISS since its
  branches are illuminated by the unscattered probe).
- probe: identical to ML-ISS — divide by the spatial max of
  sum_{views, omode, slices} |O_j|^2 at the pre-step object
  (ptyrad.linesearch.probe_denominator recipe).

Transform accounting per position per probe mode (from this code, N slices):
  gradient forward 2N-1 (N-1 propagations x2 + detector FFT), autodiff
  backward mirrors it: 4N-2 total; polynomial sweep N^2 (coefficients
  1..k propagated at each of the N-1 interfaces = N(N-1), plus N detector
  FFTs for F_1..F_N; F_0 and its chain reused from the forward);
  probe response forward 2N-1. Total (N^2 + 6N - 3) per position — the
  Table 2 multislice line. Counted live in state.transforms.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.fft import fft2, ifft2

from ptyrad.linesearch import DN_EPS, dp_from_fields, response_terms
from ptyrad.mliss import ml_line_search, mliss_loss, mliss_sigma2

DP_EPS = 1e-10  # multislice_forward's intensity floor


# --------------------------------------------------------------------------- #
# Field-level multislice forward (same maths as multislice_forward)            #
# --------------------------------------------------------------------------- #


def ms_fields(O, probe, H2, keep_incident=False):
    """Detector-plane field (B, pmode, omode, Ny, Nx), UNNORMALISED k-space
    (dp_from_fields' occu/(Nx*Ny) weight reproduces norm='ortho').

    O: complex (B, omode, Nz, Ny, Nx); probe: (B|1, pmode, Ny, Nx);
    H2: (B|1, Ny, Nx) single-step propagator.
    keep_incident=True also returns [psi_inc_0 .. psi_inc_{N-1}] (detached),
    the fields incident on each slice — the 3PIE preconditioner input and
    the zeroth-coefficient chain of the polynomial sweep."""
    Nz = O.shape[2]
    psi = probe[:, :, None]  # (B, pmode, 1->omode, Ny, Nx)
    Hb = H2[:, None, None]
    incid = []
    for n in range(Nz):
        if keep_incident:
            incid.append(psi.detach())
        psi = psi * O[:, None, :, n]
        if n < Nz - 1:
            psi = ifft2(Hb * fft2(psi))
    F = fft2(psi)
    return (F, incid) if keep_incident else F


# --------------------------------------------------------------------------- #
# Polynomial coefficient sweep for the exact object step                       #
# --------------------------------------------------------------------------- #


def ms_poly_coeffs(O, d, probe, H2, psi_inc, F0, state=None):
    """Detector coefficient fields [F_0 .. F_N] of the exit field along
    O + gamma d: psi_exit(gamma) = sum_k gamma^k IFFT-free coefficients.

    At slice j the running polynomial (coefficients C_1..C_k; C_0 is the
    stored forward chain psi_inc[j]) is multiplied by (O_j + gamma d_j):
        C'_i = O_j C_i + d_j C_{i-1}   (C'_0 handled by the stored chain),
    then every C'_i is propagated. F_0 is the stored forward field F0.
    Costs N(N-1) + N = N^2 transforms per position (per mode)."""
    Nz = O.shape[2]
    Hb = H2[:, None, None]
    C = []  # coefficients of order 1..k at the current plane (order 0 = psi_inc)
    for n in range(Nz):
        Oj = O[:, None, :, n]
        dj = d[:, None, :, n]
        # multiply the polynomial [psi_inc[n], C_1..C_k] by (O_j + gamma d_j):
        #   new_i = O_j C_i + d_j C_{i-1},  C_0 = psi_inc[n], C_{k+1} = 0
        prev = [psi_inc[n]] + C
        newC = []
        for i in range(1, len(prev) + 1):
            t = dj * prev[i - 1]
            if i < len(prev):
                t = t + Oj * prev[i]
            newC.append(t)
        if n < Nz - 1:
            newC = [ifft2(Hb * fft2(c)) for c in newC]
            if state is not None:
                state.transforms["sweep"] += 2 * len(newC) * _frames(newC[0])
        C = newC
    Fk = [fft2(c) for c in C]
    if state is not None:
        state.transforms["sweep"] += len(Fk) * _frames(Fk[0])
    return [F0] + Fk


def _frames(x):
    return x.numel() // (x.shape[-2] * x.shape[-1])


# --------------------------------------------------------------------------- #
# Horner likelihood evaluation + bracket/Brent 1D search                       #
# --------------------------------------------------------------------------- #


def _lg_of_gamma_factory(Fk, I_dat_us, omega_us, occu, postmap=None):
    """Returns L_G(gamma) evaluated pixelwise by Horner on the detector
    coefficient fields (double precision, no transforms). Works in the
    UNSHIFTED detector frame: I_dat_us/omega_us are ifftshift'ed once by the
    caller; fftshift is a permutation, so sums are identical."""
    Fk64 = [f.double() if f.is_floating_point() else f.to(torch.complex128) for f in Fk]
    nw = (occu.double() / (Fk64[0].shape[-1] * Fk64[0].shape[-2])).view(1, 1, -1, 1, 1)

    def lg(gamma):
        g = float(gamma)
        acc = Fk64[-1]
        for f in reversed(Fk64[:-1]):
            acc = acc * g + f
        u = torch.sum(acc.abs().square() * nw, dim=(1, 2)) + DP_EPS
        if postmap is not None:
            u = postmap(u)
        r = omega_us * (u - I_dat_us).square()
        return 0.5 * float(r.sum())

    return lg


def brent_min(f, x0=0.0, step=0.1, max_expand=40, tol=1e-10, max_shrink=14):
    """Bracketing + Brent in 1D, double precision, host-side control.
    Returns (x_min, f_min, n_evals) or (None, f(0), n_evals) if no bracket
    with interior decrease is found. Uses scipy's Brent on a bracket built
    by geometric expansion from x0 toward the descent side."""
    from scipy.optimize import brent as _brent

    evals = [0]

    def fc(x):
        evals[0] += 1
        return f(x)

    f0 = fc(0.0)
    # pick the descent side, shrinking the probe step if both sides ascend
    # (the direction is a descent direction, so a small enough step descends
    # unless gamma = 0 is already the minimum)
    fs = None
    for _ in range(max_shrink):
        fs = fc(step)
        if fs < f0:
            break
        fs_neg = fc(-step)
        if fs_neg < f0:
            step, fs = -step, fs_neg
            break
        step /= 8.0
        fs = None
    if fs is None:
        return None, f0, evals[0]
    # expand until f turns up
    a, fa = 0.0, f0
    b, fb = step, fs
    c, fcv = 2 * step, fc(2 * step)
    n = 0
    while fcv < fb and n < max_expand:
        a, fa = b, fb
        b, fb = c, fcv
        c = 2 * c
        fcv = fc(c)
        n += 1
    if fcv < fb:
        return None, f0, evals[0]  # never turned up
    lo, hi = (a, c) if a < c else (c, a)
    xmin, fmin, _, nf = _brent(fc, brack=(lo, b, hi), tol=tol, full_output=True)
    return float(xmin), float(fmin), evals[0]


# --------------------------------------------------------------------------- #
# Config / state                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class MLMSConfig:
    """Mirrors MLISSConfig where the model allows: same likelihood knobs,
    same fallbacks, same damping default, same probe preconditioner."""

    counts_per_unit: float | None = None  # REQUIRED, never guessed
    alpha: float = 1.0  # object fallback step alpha/N
    beta: float = 1.0  # probe fallback step beta/N
    damp: float = 1.0  # full step by default, as ML-ISS
    max_step: float = 0.0
    probe_denom: str = "max"
    denom_reg: float = 0.01
    brent_step: float = 0.25  # initial bracketing step (in units of gamma)
    brent_tol: float = 1e-10


@dataclass
class MLMSState:
    steps_o: list = field(default_factory=list)
    steps_p: list = field(default_factory=list)
    fallback_o: int = 0
    fallback_p: int = 0
    ls_evals: list = field(default_factory=list)  # Horner evals per object step
    transforms: dict = field(
        default_factory=lambda: {"gradient": 0, "sweep": 0, "probe": 0}
    )


# --------------------------------------------------------------------------- #
# Model-layer update (canvas-batched; per-view = batch of one)                 #
# --------------------------------------------------------------------------- #


def mlms_model_update(model, index, config, state=None, update_probe=True, loss_fn=None):
    return mlms_model_update_batched(
        model, [int(index)], config, state=state, update_probe=update_probe, loss_fn=loss_fn
    )


def mlms_model_update_batched(model, indices, config, state=None, update_probe=True, loss_fn=None):
    """One ML-MS update over a batch of B views on a PtychoAD multislice
    model: autodiff gradient of L_G scatter-accumulated on the canvas, 3PIE
    'max' preconditioner per slice, exact object step by polynomial sweep +
    Horner/Brent, then the exact ML-ISS-style cubic probe step against the
    Horner-updated field with the response at the UPDATED slices.

    The l1 sparsity term is logged only (loss_fn diagnostics), never part of
    the direction or the step — identical to ML-ISS/LISS/BLISS."""
    cfg = config
    st = state if state is not None else MLMSState()
    if model.solver_type != "multislice":
        raise ValueError("ML-MS requires solver_type='multislice'")
    if model.obj_preblur_std not in (None, 0):
        raise NotImplementedError("obj_preblur not supported in ML-MS")
    if model.shift_probes:
        raise NotImplementedError("per-view probe shifts not wired for ML-MS")
    mliss_sigma2(torch.zeros(1), cfg.counts_per_unit)

    device = model.opt_obja.device
    idx = torch.as_tensor(np.asarray(indices).reshape(-1), device=device)
    B = idx.numel()
    N = model.opt_obja.shape[-3]

    H2 = model.get_propagators(idx).detach()  # (B|1, Ny, Nx)
    probe = model.get_probes(idx).detach()  # (1, pmode, Ny, Nx)
    I_dat = model.get_measurements(idx).detach()
    gy = model.rpy_grid[None] + model.crop_pos[idx, 0, None, None]
    gx = model.rpx_grid[None] + model.crop_pos[idx, 1, None, None]

    postmap = None
    if model.detector_blur_std is not None and model.detector_blur_std != 0:
        try:
            from torchvision.transforms.functional import gaussian_blur
        except ImportError:
            from ptyrad.utils import gaussian_blur_2d as gaussian_blur
        std = model.detector_blur_std

        def postmap(x):
            return gaussian_blur(x, kernel_size=[5, 5], sigma=std)

    sigma2 = mliss_sigma2(I_dat, cfg.counts_per_unit)
    omega = 1.0 / sigma2

    # ---- forward + L_G backward through a differentiable canvas gather -----
    O_canvas = torch.polar(model.opt_obja.data, model.opt_objp.data).requires_grad_(True)
    O_b = O_canvas[:, :, gy, gx].permute(2, 0, 1, 3, 4)  # (B, omode, Nz, Ny, Nx)
    P_leaf = probe.clone().requires_grad_(update_probe)
    F, psi_inc = ms_fields(O_b, P_leaf, H2, keep_incident=True)
    u_p = dp_from_fields(F, model.omode_occu)
    if postmap is not None:
        u_p = postmap(u_p)
    L = mliss_loss(u_p, I_dat, None, sigma2)
    L.backward()
    Nz_frames = _frames(F)
    st.transforms["gradient"] += 2 * (2 * N - 1) * Nz_frames  # fwd + autodiff bwd
    F = F.detach()
    u_p = u_p.detach()

    patches0 = None
    if loss_fn is not None:
        with torch.no_grad():
            patches0 = torch.stack(
                [
                    model.opt_obja.data[:, :, gy, gx].permute(2, 0, 1, 3, 4),
                    model.opt_objp.data[:, :, gy, gx].permute(2, 0, 1, 3, 4),
                ],
                dim=-1,
            )

    with torch.no_grad():
        # ---- 3PIE 'max' preconditioner: canvas-accumulated |psi_inc_j|^2 ----
        Nz, Hc, Wc = model.opt_obja.shape[-3:]
        lin = (gy.long() * Wc + gx.long()).reshape(-1)
        K_canvas = torch.zeros(Nz, Hc * Wc, device=device, dtype=torch.float32)
        for j in range(Nz):
            Kj = psi_inc[j].abs().square().sum(dim=(1, 2))  # (B, Ny, Nx)
            if Kj.shape[0] == 1:
                Kj = Kj.expand(B, *Kj.shape[1:])
            K_canvas[j].index_add_(0, lin, Kj.reshape(-1))
        peak = K_canvas.amax(dim=1).view(Nz, 1, 1).clamp_min(DN_EPS)  # per-slice max
        assert O_canvas.grad is not None
        d_canvas = (-O_canvas.grad) / peak

        # ---- exact object step: polynomial sweep + Horner/Brent -------------
        d_wins = d_canvas[:, :, gy, gx].permute(2, 0, 1, 3, 4)
        Fk = ms_poly_coeffs(O_b.detach(), d_wins, probe, H2, psi_inc, F, state=st)
        from torch.fft import ifftshift

        I_us = ifftshift(I_dat, dim=(-2, -1)).double()
        om_us = ifftshift(omega, dim=(-2, -1)).double()

        if postmap is None:
            lg = _lg_of_gamma_factory(Fk, I_us, om_us, model.omode_occu)
        else:
            # blur commutes with nothing here; evaluate in the SHIFTED frame
            I_s = I_dat.double()
            om_s = omega.double()

            def lg(gamma):
                g = float(gamma)
                acc = Fk[-1].to(torch.complex128)
                for f in reversed(Fk[:-1]):
                    acc = acc * g + f.to(torch.complex128)
                u = dp_from_fields(acc, model.omode_occu.double())
                u = postmap(u)
                return 0.5 * float((om_s * (u - I_s).square()).sum())

        gmin, fmin, nev = brent_min(lg, step=cfg.brent_step, tol=cfg.brent_tol)
        st.ls_evals.append(nev)
        f0 = lg(0.0)
        if gmin is None or fmin >= f0:
            a, fb = cfg.alpha / N, True
        else:
            a, fb = gmin, False
        a *= cfg.damp
        if cfg.max_step > 0:
            a = float(np.clip(a, -cfg.max_step, cfg.max_step))
        st.steps_o.append(a)
        st.fallback_o += int(fb)

        O_new = O_canvas.detach() + a * d_canvas
        moved = (d_canvas.real != 0) | (d_canvas.imag != 0)
        model.opt_obja.data.copy_(torch.where(moved, O_new.abs(), model.opt_obja.data))
        model.opt_objp.data.copy_(torch.where(moved, O_new.angle(), model.opt_objp.data))

        diagnostics = {
            "a": a, "b": None, "loss": L.detach(), "model_dp": u_p,
            "fb_o": fb, "fb_p": None, "ls_evals": nev,
        }

        # ---- exact probe step (multislice is linear in P) -------------------
        if update_probe:
            # field after the object step, exactly, by Horner in complex64
            F2 = Fk[-1].to(F.dtype)
            for f in reversed(Fk[:-1]):
                F2 = F2 * a + f.to(F.dtype)
            u2 = dp_from_fields(F2, model.omode_occu)
            if postmap is not None:
                u2 = postmap(u2)
            e2 = u2 - I_dat

            from ptyrad.linesearch import probe_denominator

            dn_p = probe_denominator(
                O_b.detach(), mode=cfg.probe_denom, denom_reg=cfg.denom_reg
            )
            q = (-P_leaf.grad) / dn_p
            g2 = O_b.detach() + a * d_wins
            D_P = ms_fields(g2, q, H2)  # ONE forward through the UPDATED slices
            st.transforms["probe"] += (2 * N - 1) * _frames(D_P)
            v_p, w_p = response_terms(F2, D_P, model.omode_occu)
            if postmap is not None:
                v_p, w_p = postmap(v_p), postmap(w_p)
            b, fbp = ml_line_search(
                e2, v_p, w_p, omega, fallback=cfg.beta / N, damp=cfg.damp,
                max_step=cfg.max_step,
            )
            st.steps_p.append(b)
            st.fallback_p += int(fbp)
            torch.view_as_complex(model.opt_probe.data).add_(b * q[0])
            diagnostics["b"] = b
            diagnostics["fb_p"] = fbp

        if loss_fn is not None:
            _, diagnostics["losses"] = loss_fn(u_p, I_dat, patches0, model.omode_occu)

        pint = torch.view_as_complex(model.opt_probe.data).abs().square()
        diagnostics["probe_peak"] = pint.max()
        diagnostics["probe_mean"] = pint.mean()
    return diagnostics

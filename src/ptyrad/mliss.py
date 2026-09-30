"""
ML-ISS: maximum-likelihood exact line search for the ISS engine.

Implements maximum likelihood in the sense of Thibault & Guizar-Sicairos,
New J. Phys. 14, 063004 (2012): the search direction and the exact line
search both come from the SAME Gaussian likelihood

    L_G = sum_{pix, pos, batch} ( w / (2 sigma^2) ) * (I_model - I_data)^2,

with w the detector mask and sigma^2 = I_data + s0 the Poisson variance
floored at ONE DETECTOR COUNT. The measurements PtyRAD optimises against are
normalised (default 'max_at_one'), so one count is s0 = 1/counts_per_unit in
normalised units; counts_per_unit is a required config value (see
MLISSConfig.counts_per_unit — it is never guessed). Working in counts
instead multiplies L_G by the constant counts_per_unit, which changes
neither the direction (up to a positive scale absorbed by the exact step)
nor the step, so the floored-sigma^2 form is used throughout.

Differences from the retired LISS/BLISS engines (formerly ptyrad.linesearch;
see "Recovering LISS/BLISS" below — both are members of this family):

- Direction objective is L_G itself (LISS uses the amplitude loss); the
  detector residual is (w / sigma^2) * (I_data - I_model) * psi_hat per mode,
  obtained by autodiff of L_G (torch Wirtinger convention, descent = -grad).
- The line search minimises the same L_G: the weighted quartic
  Q(a) = sum omega (e + 2 a v + a^2 w_r)^2 with omega = w / sigma^2 equals
  2 * L_G(a) exactly, so direction and step optimise one objective.
- No damping by default: `damp` is exposed with default 1.0 (LISS's 0.5
  reconciles its mixed objectives; ML-ISS has nothing to reconcile).
- Fallback a = alpha/N fires when the cubic degenerates or no root strictly
  lowers L_G, and is COUNTED explicitly (state.fallback_o / fallback_p), not
  inferred from step values.

Formerly shared with LISS/BLISS and now local to this module (moved verbatim
from linesearch.py when that engine was retired): the ISS field map, the §4.2
unit map, the response terms, the per-slice direction response, and the
ePIE-style preconditioners K_j (batch-accumulated |phi_j|^2, per-slice
spatial max) and K_P (batch/slice-accumulated |O|^2, spatial max).

Recovering LISS/BLISS from ML-ISS (verified in the reproduction tests):
every step rule in this family minimises sum omega * (I(gamma) - I_data)^2
with omega = 1/(I_data + s); the engines differ only in the variance floor s
and the direction objective. ML-ISS proper uses s = 1/counts_per_unit (one
detector count).

- BLISS = MLISSConfig(objective='amplitude', step_sigma_floor=1.0, damp=0.5)
  with mliss_model_update_batched: amplitude direction + damped intensity
  step with omega = 1/(I + 1), i.e. s = 1 (ptypy's Irenorm = 1 default).
  Verified on tBL-WSe2 by trajectory overlap with the retired engine over
  100 iterations (per-iteration loss differences 0.00000 at the logged
  precision; finals 0.37485009 vs 0.37485003).
- LISS = the same configuration driven per view through mliss_model_update
  (B = 1) instead of the batched entry point.
- The amplitude objective itself is the s -> 0 end of the family: the
  linearised amplitude weight is omega ~ 1/(4u).

The probe step follows the object step and is searched against the exactly
updated field F + a*D; its response D_P uses the UPDATED slices
(a_P = sum_j D_j[(O'_j + a d_j) * Prop_{z_j} d_P]), same as LISS/BLISS.

The l1 sparsity regulariser (loss_sparse) receives the same treatment as in
LISS/BLISS: it is NOT part of the direction and NOT part of the line search;
it is only evaluated for logging through the optional `loss_fn` diagnostics.

Batching: `mliss_model_update` is the per-view (B = 1) entry point,
`mliss_model_update_batched` the joint-batch one — mirroring LISS/BLISS,
with the batch size chosen by the caller (driver: recon_params.BATCH_SIZE).
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.fft import fft2, fftshift, ifft2

# --------------------------------------------------------------------------- #
# Shared line-search machinery (moved verbatim from the retired linesearch.py) #
# --------------------------------------------------------------------------- #

# §6: floors. DN_EPS is float32 eps, used on the preconditioner denominators.
DP_EPS = 1e-10  # matches forward_models/iss.py intensity floor
SQRT_FLOOR = 1e-12  # clamp inside any sqrt of a model intensity
DN_EPS = float(torch.finfo(torch.float32).eps)  # ≈ 1.19e-7


def _fftshift2(x):
    return fftshift(x, dim=(-2, -1))


# At B = 1 the per-view update is launch-overhead- and sync-bound, not
# FLOP-bound: the 128^2 kernels are tiny. torch.compile fuses the pointwise
# soup between FFTs; the host-sync reductions live in the line searches (single
# batched transfer for the coefficients, single vectorized transfer for Q at
# the candidate roots). All compiled functions fall back to eager when dynamo
# is disabled (the tests set TORCHDYNAMO_DISABLE=1), with identical results.
def _compiled(fn):
    import os

    mode = os.environ.get("PTYRAD_LS_COMPILE_MODE", "default")
    return torch.compile(fn, dynamic=False, mode=mode)


@_compiled
def _fields_from_complex(O, probe, H):
    """Detector-plane field from complex object O (B, omode, Nz, Ny, Nx).

    Same maths as forward_models/iss.py::iss_forward up to (and
    excluding) the intensity reduction; F is exactly affine in (O - 1) and
    exactly linear in the probe — the two facts the line search rests on
    (spec §1)."""
    Ny, Nx = O.shape[-2:]
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    psi = ifft2(H * probe_k)  # unscattered illumination phi_j
    g = (O - 1.0).unsqueeze(1)  # chord perturbation, (B, 1, omode, Nz, Ny, Nx)
    scattered = torch.sum(fft2(g * psi) * H.conj(), dim=3)
    return probe_k.squeeze(3) + scattered  # (B, pmode, omode, Ny, Nx)


def iss_fields(object_patches, probe, H):
    """Detector field from PtyRAD (amp, phase) patches. Dtype-preserving."""
    O = torch.polar(object_patches[..., 0], object_patches[..., 1])
    return _fields_from_complex(O, probe, H)


@_compiled
def _fields_from_complex_m2(O, probe, H):
    """Order-2 (double-scattering) detector field, the M = 2 member of the
    terminating hierarchy with plain c = 1:

        F = P_k + sum_j T_j + sum_j FF[g_j * IFFT(H_j S_j)] H_j^*,

    with T_j = FF[g_j phi_j] H_j^* the entrance-referred single-scattering
    terms of _fields_from_complex and S_j = sum_{k<j} T_k the EXCLUSIVE
    prefix over slices (the cumulative-sum construction): H_j S_j is the
    singly-scattered field arriving at slice j, g_j scatters it once more,
    H_j^* refers it back to the detector. Same global-H_N convention as the
    M = 1 field (a common unimodular factor, invisible to intensities).

    F is exactly QUADRATIC in g = O - 1 and exactly LINEAR in the probe:
    along an object step O + a*d the field is F + a*D1 + a^2*D2 (so the
    intensity is an exact quartic in a — the joint-mode polynomial), and
    the probe step machinery is unchanged."""
    Ny, Nx = O.shape[-2:]
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    psi = ifft2(H * probe_k)
    g = (O - 1.0).unsqueeze(1)
    t = fft2(g * psi) * H.conj()
    S = torch.cumsum(t, dim=3) - t  # exclusive prefix over the slice axis
    inner = ifft2(H * S)
    second = fft2(g * inner) * H.conj()
    return probe_k.squeeze(3) + (t + second).sum(dim=3)


def _m2_direction_fields(O, d, probe, H, F0):
    """Exact field responses of the M = 2 forward along O + a*d:
    F(a) = F0 + a*D1 + a^2*D2. Because the field is a polynomial of degree
    2 in the object, the symmetric differences are EXACT (no truncation):
    D1 = (F(O+d) - F(O-d))/2, D2 = (F(O+d) + F(O-d))/2 - F0. Two extra
    forward evaluations, no adjoint."""
    F_p = _fields_from_complex_m2(O + d, probe, H)
    F_m = _fields_from_complex_m2(O - d, probe, H)
    return 0.5 * (F_p - F_m), 0.5 * (F_p + F_m) - F0


def dp_from_fields(F, omode_occu, eps=DP_EPS):
    """§4.2 unit map: PtyRAD model DP from the per-mode detector field.

    dp = fftshift2( sum_{pmode,omode} |F|^2 * occu_o/(Nx*Ny) + eps ), matching
    iss_forward exactly (pinned by oracle test 3b). Note the per-omode
    occupancy sits INSIDE the mode sum — it is not a scalar when omode > 1."""
    Ny, Nx = F.shape[-2:]
    nw = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return _fftshift2(torch.sum(F.abs().square() * nw, dim=(1, 2)) + eps)


@_compiled
def response_terms(F, D, omode_occu):
    """Per-pixel linear (v) and quadratic (w) intensity coefficients, carrying
    the same §4.2 transformation as dp_from_fields but WITHOUT the +eps floor
    (a constant: it lives in u, hence in e = u - I_dat, and drops from v, w)."""
    Ny, Nx = F.shape[-2:]
    nw = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    v = _fftshift2(torch.sum((F.conj() * D).real * nw, dim=(1, 2)))
    w = _fftshift2(torch.sum(D.abs().square() * nw, dim=(1, 2)))
    return v, w


@_compiled
def direction_response(object_patches, d, probe, H, per_slice=False):
    """Object direction response D = F(g + d) - F(g), exact for any d because
    F is affine in g (spec §2 — one extra forward half-pass, no adjoint).

    Computed as the explicit per-slice sum D = sum_j FF[ d_j * phi_j ] * H_j^*
    (identical to the difference form; the identity path cancels), so that D_j
    stays addressable for the future N-D vector search (spec §9, oracle test 8).

    per_slice=False -> (B, pmode, omode, Ny, Nx)
    per_slice=True  -> (B, pmode, omode, Nz, Ny, Nx), .sum(dim=3) is the total.

    (`object_patches` is kept for API stability but only d and probe carry
    information — the identity path cancels in the response.)
    """
    Ny, Nx = d.shape[-2:]
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    psi = ifft2(H * probe_k)
    D_slices = fft2(d.unsqueeze(1) * psi) * H.conj()
    if per_slice:
        return D_slices
    return D_slices.sum(dim=3)


def unscattered_illumination(probe, H):
    """phi_j = IFFT[H_j FFT(P)] — depends only on the probe (spec §1).
    Shape (B, pmode, 1, Nz, Ny, Nx). Any probe change invalidates it."""
    Ny, Nx = probe.shape[-2:]
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    return ifft2(H * probe_k)


def object_denominator(phi, mode="max", denom_reg=0.01):
    """K_j = sum_{batch, pmode} |phi_j|^2, then per-slice denominator:
    'max' (winning recipe) -> per-slice spatial peak, which starves off-focus
    pixels; 'local' -> K_j + peak*denom_reg, degenerating to 'max' on a
    caustic slice. Returns (Nz, Ny, Nx)-broadcastable, floored at DN_EPS."""
    K = phi.abs().square().sum(dim=(0, 1))  # (1|omode, Nz, Ny, Nx) -> keep as is
    K = K.sum(dim=0) if K.dim() == 4 else K  # (Nz, Ny, Nx)
    peak = K.amax(dim=(-2, -1), keepdim=True)
    dn = peak if mode == "max" else K + peak * denom_reg
    return dn.clamp_min(DN_EPS)


def probe_denominator(obj_complex, mode="max", denom_reg=0.01):
    """K_P = sum_{batch, omode, Nz} |O|^2 at the PRE-STEP object — the correct
    probe diagonal (spec §3 step 7). 'local' spikes tight high-NA probes;
    'max' is the default. Returns (Ny, Nx)-broadcastable, floored."""
    K = obj_complex.abs().square()
    K = K.sum(dim=tuple(range(K.dim() - 2)))  # -> (Ny, Nx)
    peak = K.amax()
    dn = peak if mode == "max" else K + peak * denom_reg
    return dn.clamp_min(DN_EPS)


@_compiled
def _quartic_terms(e, v, w, omega):
    """All four coefficient sums as one (4,) float64 tensor — a single kernel
    group and, crucially, a single device-to-host transfer at the call site
    instead of four separate float() syncs."""
    e, v, w, omega = (t.double() for t in (e, v, w, omega))
    c0 = (omega * e * v).sum()
    c1 = (omega * (e * w + 2.0 * v * v)).sum()
    c2 = 3.0 * (omega * v * w).sum()
    c3 = (omega * w * w).sum()
    return torch.stack([c0, c1, c2, c3])


@_compiled
def _q_at(e, v, w, omega, a_vec):
    """Q(a) for a whole vector of candidate steps in one kernel group:
    (K,) float64 out, one device-to-host transfer at the call site."""
    e, v, w, omega = (t.double() for t in (e, v, w, omega))
    a = a_vec.view(-1, *([1] * e.dim()))
    r = e + (2.0 * a) * v + (a * a) * w
    return (omega * r * r).sum(dim=tuple(range(1, r.dim())))


def _real_cubic_roots(c0, c1, c2, c3):
    """Host-side §3 step-5 guards + np.roots. Returns the real candidate
    roots, or None when the cubic is degenerate (caller takes the fallback):
    non-finite coefficient, fewer than 2 coefficients after stripping
    |c| < 1e-300, or no root with |imag| <= 1e-8*(1+|real|)."""
    if not np.all(np.isfinite([c0, c1, c2, c3])):
        return None
    coeffs = [c3, c2, c1, c0]
    while coeffs and abs(coeffs[0]) < 1e-300:
        coeffs = coeffs[1:]
    if len(coeffs) < 2:
        return None
    roots = np.roots(coeffs)
    cands = [float(r.real) for r in roots if abs(r.imag) <= 1e-8 * (1.0 + abs(r.real))]
    return cands or None


def apply_object_step(obja, objp, a, d):
    """O <- O + a*d IN COMPLEX, written back into PtyRAD's float (amp, phase)
    storage in place. One scalar a applied jointly to all N slices — there is
    no per-slice loop of steps anywhere in the object update (oracle test 8).

    The quartic is exact only for the complex step; stepping in (amp, phase)
    coordinates makes the map non-affine and silently degrades the search
    (spec §4.1, oracle test 4 is the tripwire).

    Phase branch: angle() returns the principal value in (-pi, pi]. Measured
    stored per-slice phase on real runs stays far inside the branch and is
    one-sided under objp_postiv (PSO_born_paper iter 200: [0, 0.836] rad;
    tBL_WSe2_born: [0, 0.185] rad — see oracle test 9), so the round trip is
    lossless. If a future specimen approaches |phase| ~ pi per slice, track
    the branch explicitly instead of hoping."""
    O = torch.polar(obja, objp) + a * d
    obja.copy_(O.abs())
    objp.copy_(O.angle())


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


def _shift_views(x, shifts, grid):
    """Per-view sub-pixel Fourier shift: x (B|1, pmode, Ny, Nx) shifted by
    shifts (B, 2) [y, x] px -> (B, pmode, Ny, Nx). Same phasor convention as
    utils.imshift_batch (which broadcasts ONE image over the shifts and so
    cannot shift B distinct images by B distinct shifts). Unitary; the
    adjoint/inverse is the same call with -shifts."""
    ky, kx = grid[0], grid[1]
    phase = -2.0 * torch.pi * (shifts[:, 1, None, None] * kx + shifts[:, 0, None, None] * ky)
    w = torch.polar(torch.ones_like(phase), phase).unsqueeze(1)  # (B, 1, Ny, Nx)
    return ifft2(fft2(x) * w)

# --------------------------------------------------------------------------- #
# The Gaussian likelihood L_G                                                  #
# --------------------------------------------------------------------------- #


def mliss_sigma2(I_dat, counts_per_unit):
    """sigma^2 = I_data + 1/counts_per_unit: Poisson variance in the data's
    normalised units, floored at one detector count. counts_per_unit is the
    number of detector counts per normalised intensity unit (the measurement
    normalisation constant when the raw data are in counts)."""
    if counts_per_unit is None or counts_per_unit <= 0:
        raise ValueError(
            "ML-ISS needs counts_per_unit > 0 (detector counts per normalised "
            "intensity unit). It is not guessed: pass the measurement "
            "normalisation constant (raw mean-pattern max for 'max_at_one'), "
            "or 1.0 if the data are already in counts."
        )
    return I_dat + 1.0 / float(counts_per_unit)


def mliss_loss(u_p, I_dat, mask, sigma2):
    """L_G = sum w/(2 sigma^2) (u - I)^2 in PtyRAD units (u includes the
    +eps intensity floor of the forward, exactly as the data comparison in
    the line search does through e = u - I)."""
    r = (u_p - I_dat).square() / sigma2
    if mask is not None:
        r = mask * r
    return 0.5 * r.sum()


@_compiled
def _fwd_lg(O, P, H, I_dat, mask, sigma2, omode_occu):
    """Fused forward + L_G (one compiled region so the backward compiles too).
    Callers with an intensity postmap use the eager pieces instead."""
    F = _fields_from_complex(O, P, H)
    u_p = dp_from_fields(F, omode_occu)
    L = mliss_loss(u_p, I_dat, mask, sigma2)
    return L, F, u_p


def mliss_direction(obj_complex, probe, H, I_dat, mask, sigma2, omode_occu):
    """Descent directions of L_G w.r.t. complex O and complex P, BEFORE
    preconditioning (torch Wirtinger convention: grad = 2 dL/d(conj)).
    Same forward/backward ops as mliss_batch_update; exists for the gradient
    oracle test (parallel to linesearch.direction_gradient)."""
    O = obj_complex.detach().clone().requires_grad_(True)
    P = probe.detach().clone().requires_grad_(True)
    F = _fields_from_complex(O, P, H)
    L = mliss_loss(dp_from_fields(F, omode_occu), I_dat, mask, sigma2)
    L.backward()
    return -O.grad, -P.grad


# --------------------------------------------------------------------------- #
# Exact quartic line search on L_G with explicit fallback flag                 #
# --------------------------------------------------------------------------- #


def ml_line_search(e, v, w, omega, fallback, damp=1.0, max_step=0.0):
    """Exact step on Q(a) = sum omega (e + 2av + a^2 w)^2 = 2 L_G(a), with
    omega = mask/sigma^2 — the SAME weight as the direction, so both optimise
    one objective. Reuses the float64-before-products coefficient sums and
    the host-side cubic guards (both now local to this module) verbatim.

    Returns (a, used_fallback). The root must be STRICTLY below Q(0), else
    fallback (alpha/N or beta/N, supplied by the caller). `damp` multiplies
    every step (default 1.0 — ML-ISS takes the full step)."""
    c0, c1, c2, c3 = _quartic_terms(e, v, w, omega).cpu().tolist()
    cands = _real_cubic_roots(c0, c1, c2, c3)
    used_fallback = True
    a = fallback
    if cands is not None:
        a_vec = torch.tensor([0.0, *cands], dtype=torch.float64, device=e.device)
        qs = _q_at(e, v, w, omega, a_vec).cpu().tolist()
        best, bestq = None, qs[0]
        for cand, qa in zip(cands, qs[1:], strict=True):
            if qa < bestq:
                best, bestq = cand, qa
        if best is not None:
            a, used_fallback = best, False
    a *= damp
    if max_step > 0:
        a = float(np.clip(a, -max_step, max_step))
    return float(a), used_fallback


# --------------------------------------------------------------------------- #
# Joint probe–object step (Thibault & Guizar-Sicairos NJP 14, 063004, App. B)  #
# --------------------------------------------------------------------------- #
#
# With object direction d and probe direction d_P taken from the SAME residual,
# psi_hat(gamma) = psi_hat + gamma a + gamma^2 b exactly (the field is affine
# in the object and linear in the probe, so the only nonlinearity is the
# bilinear d * d_P term):
#   a = F(d_P; O) + R(d; P)      (R = object direction response)
#   b = R(d; d_P)                (response of d under illumination by d_P)
# In PtyRAD units the intensity is I(gamma) = u + sum_{n>=1} A_n gamma^n with
#   A1 = 2 v_a, A2 = 2 v_b + w_a, A3 = 2 <a, b>, A4 = w_b,
# where (v_x, w_x) are the response terms of x against F and <a, b> is the
# same weighted Re(conj(a) b) map. L_G(gamma) is then (1/2) of the degree-8
# polynomial P(gamma) = sum omega (A0 + A1 g + ... + A4 g^4)^2, A0 = e.


@_compiled
def _octic_terms(e, va, wa, vb, ab, wb, omega):
    """The nine float64 coefficient sums C_k of P(gamma), k = 0..8, as one
    (9,) tensor (single device-to-host transfer at the call site). All inputs
    are CAST TO FLOAT64 BEFORE THE PRODUCTS — w_b^2 overflows float32 exactly
    like the quartic's c3 (spec §4.3)."""
    e, va, wa, vb, ab, wb, omega = (t.double() for t in (e, va, wa, vb, ab, wb, omega))
    A = (e, 2.0 * va, 2.0 * vb + wa, 2.0 * ab, wb)
    out = []
    for k in range(9):
        s = None
        for i in range(max(0, k - 4), min(k, 4) + 1):
            term = A[i] * A[k - i]
            s = term if s is None else s + term
        out.append((omega * s).sum())
    return torch.stack(out)


@_compiled
def _p_at(e, va, wa, vb, ab, wb, omega, g_vec):
    """P(gamma) for a vector of candidate steps, float64, one transfer."""
    e, va, wa, vb, ab, wb, omega = (t.double() for t in (e, va, wa, vb, ab, wb, omega))
    g = g_vec.view(-1, *([1] * e.dim()))
    r = e + (2.0 * va) * g + (2.0 * vb + wa) * g * g + (2.0 * ab) * g**3 + wb * g**4
    return (omega * r * r).sum(dim=tuple(range(1, r.dim())))


def _real_poly_roots(coeffs_desc):
    """Real roots of a polynomial (coefficients highest-order first) with the
    same guards as the cubic solve: non-finite -> None; strip |c| < 1e-300;
    fewer than 2 coefficients -> None; companion-matrix roots via np.roots;
    keep |imag| <= 1e-8 (1 + |real|); none real -> None."""
    if not np.all(np.isfinite(coeffs_desc)):
        return None
    coeffs = list(coeffs_desc)
    while coeffs and abs(coeffs[0]) < 1e-300:
        coeffs = coeffs[1:]
    if len(coeffs) < 2:
        return None
    roots = np.roots(coeffs)
    cands = [float(r.real) for r in roots if abs(r.imag) <= 1e-8 * (1.0 + abs(r.real))]
    return cands or None


def ml_joint_line_search(e, va, wa, vb, ab, wb, omega, fallback, damp=1.0, max_step=0.0):
    """Exact joint step: minimise the degree-8 polynomial 2 L_G(gamma) along
    (O + gamma d, P + gamma d_P). Differentiates to degree 7, takes the real
    roots (companion matrix), keeps the root with the lowest L_G STRICTLY
    below L_G(0), else the fallback. Returns (gamma, used_fallback)."""
    C = _octic_terms(e, va, wa, vb, ab, wb, omega).cpu().tolist()
    dcoeffs = [k * C[k] for k in range(8, 0, -1)]  # P'(g), highest first
    cands = _real_poly_roots(dcoeffs)
    used_fallback = True
    g = fallback
    if cands is not None:
        g_vec = torch.tensor([0.0, *cands], dtype=torch.float64, device=e.device)
        ps = _p_at(e, va, wa, vb, ab, wb, omega, g_vec).cpu().tolist()
        best, bestp = None, ps[0]
        for cand, pv in zip(cands, ps[1:], strict=True):
            if pv < bestp:
                best, bestp = cand, pv
        if best is not None:
            g, used_fallback = best, False
    g *= damp
    if max_step > 0:
        g = float(np.clip(g, -max_step, max_step))
    return float(g), used_fallback


def ml_vector_line_search(e, F, D_slices, omega, omode_occu, fallback, damp=1.0, postmap=None):
    """N-dimensional exact object step: minimise Q(gamma_1..gamma_N) =
    sum omega (e + 2 sum_j gamma_j v_j + sum_jk gamma_j gamma_k G_jk)^2 — the
    exact multivariate quartic of the per-slice steps (the field is affine in
    every slice) — by Newton on the N x N Gram, warm-started from the scalar
    step. Costs NO additional transforms: the D_j are already computed; only
    pixelwise products and reductions are added.

    Returns (gamma_vec float64 tensor (N,), used_fallback, extras dict).
    Fallback ladder: vector accepted only if strictly below both L(0) and the
    scalar step; else the scalar root; else fallback/N * ones. `damp`
    multiplies the returned steps."""
    Nz = D_slices.shape[3]
    dev = e.device
    V = []
    G = torch.zeros(Nz, Nz, *e.shape, dtype=torch.float64, device=dev)
    for j in range(Nz):
        vj, wj = response_terms(F, D_slices[:, :, :, j], omode_occu)
        if postmap is not None:
            vj, wj = postmap(vj), postmap(wj)
        V.append(vj.double())
        G[j, j] = wj.double()
        for k in range(j):
            gjk = response_terms(D_slices[:, :, :, k], D_slices[:, :, :, j], omode_occu)[0]
            if postmap is not None:
                gjk = postmap(gjk)
            G[j, k] = G[k, j] = gjk.double()
    V = torch.stack(V)  # (N, B, Ny, Nx) float64
    e64 = e.double()
    om64 = omega.double() if torch.is_tensor(omega) else torch.as_tensor(omega).double()

    # scalar warm start (identical to the scalar path: v = sum_j V_j, w = sum G)
    v_s = V.sum(dim=0)
    w_s = G.sum(dim=(0, 1))
    a_s, fb_s = ml_line_search(e64, v_s, w_s, om64, fallback=fallback, damp=1.0)

    def resid(g):
        return e64 + 2.0 * torch.einsum("n,n...->...", g, V) + torch.einsum(
            "m,n,mn...->...", g, g, G
        )

    def L(g):
        r = resid(g)
        return float((om64 * r * r).sum())

    g = torch.full((Nz,), float(a_s), dtype=torch.float64, device=dev)
    L_scalar = L(g)
    L0 = L(torch.zeros_like(g))
    cur = L_scalar
    for _ in range(12):
        r = resid(g)
        t = 2.0 * V + 2.0 * torch.einsum("m,mn...->n...", g, G)
        grad = 2.0 * torch.einsum("...,n...->n", om64 * r, t)
        Hm = 2.0 * (
            torch.einsum("n...,m...->nm", om64 * t, t)
            + 2.0 * torch.einsum("...,nm...->nm", om64 * r, G)
        )
        lam = 1e-10 * torch.diagonal(Hm).abs().max().clamp_min(1e-300)
        try:
            step = torch.linalg.solve(Hm + lam * torch.eye(Nz, dtype=torch.float64, device=dev), -grad)
        except Exception:
            break
        ok = False
        for _bt in range(6):
            Lnew = L(g + step)
            if Lnew < cur:
                g = g + step
                cur = Lnew
                ok = True
                break
            step = 0.5 * step
        if not ok or float(step.abs().max()) < 1e-10 * (1.0 + float(g.abs().max())):
            break

    extras = {"L0": L0, "L_scalar": L_scalar, "L_vector": cur, "a_scalar": float(a_s)}
    if cur < L_scalar and cur < L0 and torch.isfinite(g).all():
        return damp * g.cpu(), False, extras
    if not fb_s and L_scalar < L0:
        return damp * torch.full((Nz,), float(a_s), dtype=torch.float64), False, extras
    return damp * torch.full((Nz,), float(fallback), dtype=torch.float64), True, extras


# --------------------------------------------------------------------------- #
# Config / state                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class MLISSConfig:
    """ML-ISS knobs. Preconditioner knobs deliberately mirror LineSearchConfig
    (same K_j / K_P recipes); the differences are the objective (L_G), the
    default damping (1.0) and the required counts_per_unit."""

    counts_per_unit: float | None = None  # REQUIRED: detector counts per unit
    alpha: float = 1.0  # object fallback step alpha/N (cubic degeneracy only)
    beta: float = 1.0  # probe fallback step beta/N
    damp: float = 1.0  # multiplier on every step; ML-ISS default is FULL step
    max_step: float = 0.0  # symmetric clip on the damped step; 0 = off
    object_denom: str = "max"  # 'max' | 'local' (same K_j as LISS/BLISS)
    probe_denom: str = "max"  # same K_P as LISS/BLISS
    denom_reg: float = 0.01
    # 'alternating': exact object step, then exact probe step against the
    # updated field (the original ML-ISS behaviour). 'joint': ONE gamma moves
    # object and probe together along directions from the same residual
    # (NJP 14, 063004 App. B); exact degree-8 line search, fallback alpha/N.
    step_mode: str = "alternating"
    # Joint mode only: scalar weight on the probe direction d_P. One gamma
    # moves both, so the relative scale matters; exposed, not tuned.
    probe_dir_weight: float = 1.0
    # Objective switch (alternating mode only). 'gaussian' is the ML-ISS
    # definition (L_G). 'amplitude': direction AND exact 1D step both on the
    # amplitude loss L_A = sum mask (sqrt I_model - sqrt I_data)^2 — undamped
    # amplitude maximum likelihood. 'poisson': direction and exact 1D step on
    # the Poisson NLL in counts, L_P = sum mask (c u_c - c I_c log(c u_c))
    # with c = counts_per_unit and the model intensity floored. The 1D steps
    # use I(gamma) = u + 2 gamma v + gamma^2 w pixelwise (float64) with
    # bracketing + Brent; inner evaluation counts land in state.ls_evals.
    objective: str = "gaussian"
    # Poisson objective only: model-intensity floor in COUNTS (bounds the
    # Poisson weight 1 - I/u; 0.01 count is negligible bias, finite gradient)
    poisson_floor: float = 0.01
    # TEST knob (step-rule override): when set, BOTH line searches use the
    # Gaussian intensity quartic with sigma^2 = I_data + step_sigma_floor
    # (NORMALISED units) regardless of the direction objective. With
    # objective='amplitude', step_sigma_floor=1.0, damp=0.5 this reconstructs
    # BLISS's algorithm exactly (amplitude direction + omega=1/(I+1) damped
    # intensity step). None (default) leaves every objective's own step rule.
    step_sigma_floor: float | None = None
    # 'vector' replaces the SCALAR object step with the N-dimensional per-slice
    # step (batched alternating path): the likelihood restricted to
    # psi + sum_j gamma_j D_j is an exact multivariate quartic; its stationary
    # point is found by Newton on the N x N Gram (no additional transforms —
    # the per-slice responses D_j are already computed). Requires a Gaussian
    # step weight (objective='gaussian' or step_sigma_floor set).
    slice_step: str = "scalar"


@dataclass
class MLISSState:
    """Cross-batch state: step logs plus explicit fallback counters (the
    fallback fraction per iteration is n_fallback/n_steps over the sweep).
    ls_evals is filled only by the 'amplitude' objective's 1D searches."""

    steps_o: list = field(default_factory=list)
    steps_p: list = field(default_factory=list)
    fallback_o: int = 0
    fallback_p: int = 0
    ls_evals: list = field(default_factory=list)
    steps_o_vec: list = field(default_factory=list)  # per-slice steps ('vector' mode)


def _la_loss(u_p, I_dat, mask):
    """Amplitude loss L_A = sum mask (sqrt u - sqrt I)^2 (sqrt floored)."""
    r = (u_p.clamp_min(1e-12).sqrt() - I_dat.clamp_min(0).sqrt()).square()
    if mask is not None:
        r = mask * r
    return r.sum()


def _lp_loss(u_p, I_dat, mask, counts_per_unit, floor_counts=0.01):
    """Poisson NLL in counts: L_P = sum mask [c u - c I log(c u)], with the
    model intensity floored at `floor_counts` COUNTS (constant log Gamma
    terms dropped). The floor bounds the weight (1 - I/u) — without it a
    vacuum start (dark-field u ~ 0) produces unbounded gradients.
    c = counts_per_unit converts normalised units to counts."""
    c = float(counts_per_unit)
    cu = (c * u_p).clamp_min(float(floor_counts))
    r = cu - (c * I_dat.clamp_min(0)) * torch.log(cu)
    if mask is not None:
        r = mask * r
    return r.sum()


def lp_line_search(
    u, v, w, I_dat, mask, counts_per_unit, fallback, damp=1.0, max_step=0.0, floor_counts=0.01
):
    """Exact 1D step on the POISSON NLL along I(gamma) = u + 2 gamma v +
    gamma^2 w: bracketing + Brent in float64, intensity floored at the SAME
    floor_counts as the direction loss. Returns (gamma, used_fallback,
    n_evals). A non-finite L_P(0) returns a ZERO step (freeze, flagged as
    fallback) instead of stepping alpha/N along a possibly huge direction."""

    c = float(counts_per_unit)
    u64, v64, w64 = u.double(), v.double(), w.double()
    cI = (c * I_dat.clamp_min(0)).double()
    m64 = None if mask is None else mask.double()

    def lp(gamma):
        g = float(gamma)
        cu = (c * (u64 + (2.0 * g) * v64 + (g * g) * w64)).clamp_min(float(floor_counts))
        r = cu - cI * torch.log(cu)
        if m64 is not None:
            r = m64 * r
        return float(r.sum())

    f0 = lp(0.0)
    if not np.isfinite(f0):
        return 0.0, True, 1  # frozen step: never move along a non-finite objective
    gmin, fmin, nev = brent_min(lp, step=0.25)
    used_fallback = gmin is None or fmin >= f0
    a = fallback if used_fallback else gmin
    a *= damp
    if max_step > 0:
        a = float(np.clip(a, -max_step, max_step))
    return float(a), used_fallback, nev


def la_line_search(u, v, w, I_dat, mask, fallback, damp=1.0, max_step=0.0):
    """Exact 1D step on the AMPLITUDE loss along I(gamma) = u + 2 gamma v +
    gamma^2 w: bracketing + Brent in float64, I(gamma) floored inside the
    sqrt. Returns (gamma, used_fallback, n_evals)."""

    u64, v64, w64 = u.double(), v.double(), w.double()
    sI = I_dat.double().clamp_min(0).sqrt()
    m64 = None if mask is None else mask.double()

    def la(gamma):
        g = float(gamma)
        r = ((u64 + (2.0 * g) * v64 + (g * g) * w64).clamp_min(1e-12).sqrt() - sI).square()
        if m64 is not None:
            r = m64 * r
        return float(r.sum())

    gmin, fmin, nev = brent_min(la, step=0.25)
    used_fallback = gmin is None or fmin >= la(0.0)
    a = fallback if used_fallback else gmin
    a *= damp
    if max_step > 0:
        a = float(np.clip(a, -max_step, max_step))
    return float(a), used_fallback, nev


# --------------------------------------------------------------------------- #
# Quartic-intensity 1D searches (M = 2 object step: I(a) is degree 4 in a)     #
# --------------------------------------------------------------------------- #
#
# Along the M = 2 object step the field is F + a*D1 + a^2*D2, so the pixelwise
# intensity is the SAME quartic as the joint mode's:
#   I(a) = u + 2 va a + (2 vb + wa) a^2 + 2 ab a^3 + wb a^4,
# with (va, wa) = response_terms(F, D1), (vb, wb) = response_terms(F, D2) and
# ab = response_terms(D1, D2)[0]. The Gaussian objective therefore reuses
# ml_joint_line_search verbatim; amplitude and Poisson bracket+Brent on the
# quartic exactly as their quadratic (M = 1) counterparts do.


def _quartic_I(u, va, wa, vb, ab, wb):
    """Float64 pixelwise coefficient maps (A1..A4) of I(a) above."""
    va, wa, vb, ab, wb = (t.double() for t in (va, wa, vb, ab, wb))
    return 2.0 * va, 2.0 * vb + wa, 2.0 * ab, wb


def la_line_search_quartic(u, va, wa, vb, ab, wb, I_dat, mask, fallback, damp=1.0, max_step=0.0):
    """Exact 1D AMPLITUDE step along the quartic intensity (M = 2 object
    step): bracketing + Brent in float64, I(a) floored inside the sqrt.
    Returns (a, used_fallback, n_evals). Reduces to la_line_search when
    D2 = 0 (vb = ab = wb = 0)."""
    u64 = u.double()
    A1, A2, A3, A4 = _quartic_I(u, va, wa, vb, ab, wb)
    sI = I_dat.double().clamp_min(0).sqrt()
    m64 = None if mask is None else mask.double()

    def la(a):
        g = float(a)
        Ia = u64 + g * (A1 + g * (A2 + g * (A3 + g * A4)))
        r = (Ia.clamp_min(1e-12).sqrt() - sI).square()
        if m64 is not None:
            r = m64 * r
        return float(r.sum())

    gmin, fmin, nev = brent_min(la, step=0.25)
    used_fallback = gmin is None or fmin >= la(0.0)
    a = fallback if used_fallback else gmin
    a *= damp
    if max_step > 0:
        a = float(np.clip(a, -max_step, max_step))
    return float(a), used_fallback, nev


def lp_line_search_quartic(
    u, va, wa, vb, ab, wb, I_dat, mask, counts_per_unit, fallback,
    damp=1.0, max_step=0.0, floor_counts=0.01,
):
    """Exact 1D POISSON step along the quartic intensity (M = 2 object step),
    same floors and freeze-on-non-finite behaviour as lp_line_search."""
    c = float(counts_per_unit)
    u64 = u.double()
    A1, A2, A3, A4 = _quartic_I(u, va, wa, vb, ab, wb)
    cI = (c * I_dat.clamp_min(0)).double()
    m64 = None if mask is None else mask.double()

    def lp(a):
        g = float(a)
        Ia = u64 + g * (A1 + g * (A2 + g * (A3 + g * A4)))
        cu = (c * Ia).clamp_min(float(floor_counts))
        r = cu - cI * torch.log(cu)
        if m64 is not None:
            r = m64 * r
        return float(r.sum())

    f0 = lp(0.0)
    if not np.isfinite(f0):
        return 0.0, True, 1
    gmin, fmin, nev = brent_min(lp, step=0.25)
    used_fallback = gmin is None or fmin >= f0
    a = fallback if used_fallback else gmin
    a *= damp
    if max_step > 0:
        a = float(np.clip(a, -max_step, max_step))
    return float(a), used_fallback, nev


# --------------------------------------------------------------------------- #
# One batch update on full-frame tensors (alternating: object, then probe)     #
# --------------------------------------------------------------------------- #


def mliss_batch_update(
    obja,
    objp,
    probe,
    H,
    I_dat,
    mask,
    omode_occu,
    config,
    state=None,
    update_probe=True,
    intensity_postmap=None,
    order=1,
):
    """One ML-ISS update on full-frame tensors: exact object step on L_G,
    then exact probe step on L_G against the exactly updated field.
    Structure mirrors linesearch_batch_update (which is not modified);
    shapes and in-place semantics are identical to it.

    order=2 (ML-ISS2): the M = 2 member with plain c = 1. The field is
    quadratic in the object, so the object step is the exact QUARTIC
    intensity search (joint-mode polynomial machinery for the Gaussian
    weight, quartic Brent for amplitude/Poisson); the field stays linear
    in the probe, so the probe step is unchanged. Alternating scalar only.

    Returns (probe, diagnostics); diagnostics["loss"] is L_G of the batch at
    the pre-step state (0-dim tensor, call .item() at logging cadence)."""
    cfg = config
    st = state if state is not None else MLISSState()
    if probe.shape[0] != 1:
        raise ValueError("mliss_batch_update expects a shared probe (B = 1 design point)")
    if order not in (1, 2):
        raise ValueError(f"mliss_batch_update: order must be 1 or 2, got {order}")
    if order == 2 and cfg.step_mode != "alternating":
        raise NotImplementedError("order=2 supports step_mode='alternating' only")
    N = obja.shape[-3]
    fwd = _fields_from_complex if order == 1 else _fields_from_complex_m2

    sigma2 = mliss_sigma2(I_dat, cfg.counts_per_unit)
    omega = (1.0 / sigma2) if mask is None else mask / sigma2

    # ---- forward with leaves for both gradients (one backward) -------------
    O_leaf = torch.polar(obja, objp).unsqueeze(0).detach().requires_grad_(True)
    P_leaf = probe.detach().clone().requires_grad_(update_probe)
    if getattr(cfg, "objective", "gaussian") != "gaussian":
        F = fwd(O_leaf, P_leaf, H)
        u_p = dp_from_fields(F, omode_occu)
        if intensity_postmap is not None:
            u_p = intensity_postmap(u_p)
        L = (
            _la_loss(u_p, I_dat, mask)
            if cfg.objective == "amplitude"
            else _lp_loss(u_p, I_dat, mask, cfg.counts_per_unit, cfg.poisson_floor)
        )
    elif intensity_postmap is None and order == 1:
        L, F, u_p = _fwd_lg(O_leaf, P_leaf, H, I_dat, mask, sigma2, omode_occu)
    else:
        F = fwd(O_leaf, P_leaf, H)
        u_p = dp_from_fields(F, omode_occu)
        if intensity_postmap is not None:
            u_p = intensity_postmap(u_p)
        L = mliss_loss(u_p, I_dat, mask, sigma2)
    L.backward()
    F = F.detach()
    u_p = u_p.detach()
    e = u_p - I_dat

    # ---- preconditioned descent direction of L_G ---------------------------
    phi = unscattered_illumination(P_leaf.detach(), H)
    dn_o = object_denominator(phi, mode=cfg.object_denom, denom_reg=cfg.denom_reg)
    assert O_leaf.grad is not None
    d = (-O_leaf.grad) / dn_o  # descent = -O.grad (torch Wirtinger convention)

    # ---- JOINT step (one gamma along d and d_P, NJP 14 App. B) --------------
    if cfg.step_mode == "joint" and update_probe:
        dn_p = probe_denominator(O_leaf.detach(), mode=cfg.probe_denom, denom_reg=cfg.denom_reg)
        q = cfg.probe_dir_weight * ((-P_leaf.grad) / dn_p)
        D_d = direction_response(None, d, P_leaf.detach(), H)
        # a = F(q; O) + R(d; P): Prop_z d_P is built once inside the forward
        # (q is the shared B=1 probe direction, broadcast over the batch)
        a_f = _fields_from_complex(O_leaf.detach(), q, H) + D_d
        b_f = direction_response(None, d, q, H)
        va, wa = _ml_response_terms(F, a_f, omode_occu, intensity_postmap)
        vb, wb = _ml_response_terms(F, b_f, omode_occu, intensity_postmap)
        ab = _ml_response_terms(a_f, b_f, omode_occu, intensity_postmap)[0]
        g, fb = ml_joint_line_search(
            e, va, wa, vb, ab, wb, omega,
            fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step,
        )
        st.steps_o.append(g)
        st.fallback_o += int(fb)
        apply_object_step(obja, objp, g, d[0])
        probe = probe + g * q
        diagnostics = {
            "a": g, "b": g, "loss": L.detach(), "model_dp": u_p, "fb_o": fb, "fb_p": None,
        }
        pint = probe.abs().square()
        diagnostics["probe_peak"] = pint.max()
        diagnostics["probe_mean"] = pint.mean()
        return probe, diagnostics

    # ---- M = 2: exact quartic object step, then probe against F(a) ---------
    if order == 2:
        D1, D2 = _m2_direction_fields(O_leaf.detach(), d, P_leaf.detach(), H, F)
        va, wa = _ml_response_terms(F, D1, omode_occu, intensity_postmap)
        vb, wb = _ml_response_terms(F, D2, omode_occu, intensity_postmap)
        ab = _ml_response_terms(D1, D2, omode_occu, intensity_postmap)[0]
        if cfg.step_sigma_floor is not None:
            om_s = 1.0 / (I_dat + cfg.step_sigma_floor)
            if mask is not None:
                om_s = mask * om_s
            a, fb = ml_joint_line_search(
                e, va, wa, vb, ab, wb, om_s,
                fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step,
            )
        elif cfg.objective == "amplitude":
            a, fb, nev = la_line_search_quartic(
                u_p, va, wa, vb, ab, wb, I_dat, mask,
                fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step,
            )
            st.ls_evals.append(nev)
        elif cfg.objective == "poisson":
            a, fb, nev = lp_line_search_quartic(
                u_p, va, wa, vb, ab, wb, I_dat, mask, cfg.counts_per_unit,
                fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step,
                floor_counts=cfg.poisson_floor,
            )
            st.ls_evals.append(nev)
        else:
            a, fb = ml_joint_line_search(
                e, va, wa, vb, ab, wb, omega,
                fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step,
            )
        st.steps_o.append(a)
        st.fallback_o += int(fb)
        apply_object_step(obja, objp, a, d[0])
        diagnostics = {
            "a": a, "b": None, "loss": L.detach(), "model_dp": u_p, "fb_o": fb, "fb_p": None,
        }
        if update_probe:
            # exact post-step field and intensity: F is quadratic in the object
            F = F + a * D1 + (a * a) * D2
            A1, A2, A3, A4 = _quartic_I(u_p, va, wa, vb, ab, wb)
            u_p2 = (u_p.double() + a * (A1 + a * (A2 + a * (A3 + a * A4)))).to(u_p.dtype)
            e2 = u_p2 - I_dat
            dn_p = probe_denominator(O_leaf.detach(), mode=cfg.probe_denom, denom_reg=cfg.denom_reg)
            q = (-P_leaf.grad) / dn_p
            # D_P: order-2 response at the UPDATED slices (field linear in probe)
            D_P = _fields_from_complex_m2(torch.polar(obja, objp).unsqueeze(0), q, H)
            v_p, w_p = _ml_response_terms(F, D_P, omode_occu, intensity_postmap)
            if cfg.step_sigma_floor is not None:
                om_s = 1.0 / (I_dat + cfg.step_sigma_floor)
                if mask is not None:
                    om_s = mask * om_s
                b, fbp = ml_line_search(
                    e2, v_p, w_p, om_s, fallback=cfg.beta / N, damp=cfg.damp,
                    max_step=cfg.max_step,
                )
            elif cfg.objective == "amplitude":
                b, fbp, nev_p = la_line_search(
                    u_p2, v_p, w_p, I_dat, mask, fallback=cfg.beta / N, damp=cfg.damp,
                    max_step=cfg.max_step,
                )
                st.ls_evals.append(nev_p)
            elif cfg.objective == "poisson":
                b, fbp, nev_p = lp_line_search(
                    u_p2, v_p, w_p, I_dat, mask, cfg.counts_per_unit,
                    fallback=cfg.beta / N, floor_counts=cfg.poisson_floor,
                    damp=cfg.damp, max_step=cfg.max_step,
                )
                st.ls_evals.append(nev_p)
            else:
                b, fbp = ml_line_search(
                    e2, v_p, w_p, omega, fallback=cfg.beta / N, damp=cfg.damp,
                    max_step=cfg.max_step,
                )
            st.steps_p.append(b)
            st.fallback_p += int(fbp)
            probe = probe + b * q
            diagnostics["b"] = b
            diagnostics["fb_p"] = fbp
        pint = probe.abs().square()
        diagnostics["probe_peak"] = pint.max()
        diagnostics["probe_mean"] = pint.mean()
        return probe, diagnostics

    # ---- response (per-slice container kept, like LISS), quartic, solve ----
    D_slices = direction_response(None, d, P_leaf.detach(), H, per_slice=True)
    D = D_slices.sum(dim=3)
    v, w = _ml_response_terms(F, D, omode_occu, intensity_postmap)
    if cfg.step_sigma_floor is not None:
        om_s = 1.0 / (I_dat + cfg.step_sigma_floor)
        if mask is not None:
            om_s = mask * om_s
        a, fb = ml_line_search(
            e, v, w, om_s, fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step
        )
    elif cfg.objective == "amplitude":
        a, fb, nev = la_line_search(
            u_p, v, w, I_dat, mask, fallback=cfg.alpha / N, damp=cfg.damp,
            max_step=cfg.max_step,
        )
        st.ls_evals.append(nev)
    elif cfg.objective == "poisson":
        a, fb, nev = lp_line_search(
            u_p, v, w, I_dat, mask, cfg.counts_per_unit, fallback=cfg.alpha / N, floor_counts=cfg.poisson_floor,
            damp=cfg.damp, max_step=cfg.max_step,
        )
        st.ls_evals.append(nev)
    else:
        a, fb = ml_line_search(
            e, v, w, omega, fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step
        )
    st.steps_o.append(a)
    st.fallback_o += int(fb)

    # ---- joint complex object step -----------------------------------------
    apply_object_step(obja, objp, a, d[0])

    diagnostics = {"a": a, "b": None, "loss": L.detach(), "model_dp": u_p, "fb_o": fb, "fb_p": None}

    # ---- probe step against the exactly-updated field ----------------------
    if update_probe:
        F = F + a * D  # exact — handles the bilinear cross term
        u_p2 = u_p + (2.0 * a) * v + (a * a) * w  # exact, no re-forward
        e2 = u_p2 - I_dat

        dn_p = probe_denominator(O_leaf.detach(), mode=cfg.probe_denom, denom_reg=cfg.denom_reg)
        q = (-P_leaf.grad) / dn_p

        # D_P = F(q; g2): response at the UPDATED slices (phi rebuilt from q)
        patches2 = torch.stack([obja, objp], dim=-1).unsqueeze(0)
        D_P = iss_fields(patches2, q, H)
        v_p, w_p = _ml_response_terms(F, D_P, omode_occu, intensity_postmap)
        if cfg.step_sigma_floor is not None:
            om_s = 1.0 / (I_dat + cfg.step_sigma_floor)
            if mask is not None:
                om_s = mask * om_s
            b, fbp = ml_line_search(
                e2, v_p, w_p, om_s, fallback=cfg.beta / N, damp=cfg.damp,
                max_step=cfg.max_step,
            )
        elif cfg.objective == "amplitude":
            b, fbp, nev_p = la_line_search(
                u_p2, v_p, w_p, I_dat, mask, fallback=cfg.beta / N, damp=cfg.damp,
                max_step=cfg.max_step,
            )
            st.ls_evals.append(nev_p)
        elif cfg.objective == "poisson":
            b, fbp, nev_p = lp_line_search(
                u_p2, v_p, w_p, I_dat, mask, cfg.counts_per_unit, fallback=cfg.beta / N, floor_counts=cfg.poisson_floor,
                damp=cfg.damp, max_step=cfg.max_step,
            )
            st.ls_evals.append(nev_p)
        else:
            b, fbp = ml_line_search(
                e2, v_p, w_p, omega, fallback=cfg.beta / N, damp=cfg.damp,
                max_step=cfg.max_step,
            )
        st.steps_p.append(b)
        st.fallback_p += int(fbp)
        probe = probe + b * q
        diagnostics["b"] = b
        diagnostics["fb_p"] = fbp

    pint = probe.abs().square()
    diagnostics["probe_peak"] = pint.max()
    diagnostics["probe_mean"] = pint.mean()
    return probe, diagnostics


def _ml_response_terms(F, D, omode_occu, intensity_postmap):
    v, w = response_terms(F, D, omode_occu)
    if intensity_postmap is not None:
        v, w = intensity_postmap(v), intensity_postmap(w)
    return v, w


# --------------------------------------------------------------------------- #
# Model-layer wiring: per-view (B = 1) update on a PtychoAD ISS model          #
# --------------------------------------------------------------------------- #


def _model_guard(model, cfg):
    if model.solver_type != "born" or model.born_iterations not in (1, 2):
        raise ValueError(
            "ML-ISS requires solver_type='born' with born_iterations 1 (ISS: "
            "field affine in the object) or 2 (ML-ISS2: field quadratic, "
            "exact quartic object step)."
        )
    if model.born_iterations == 2:
        if cfg.step_mode != "alternating" or getattr(cfg, "slice_step", "scalar") != "scalar":
            raise NotImplementedError(
                "ML-ISS2 (born_iterations=2) supports alternating scalar steps only"
            )
        if getattr(model, "use_born_coeffs", False):
            raise NotImplementedError(
                "ML-ISS2 assumes the plain series (c = 1): disable born_coeffs "
                "init/refit/tuning."
            )
    if model.obj_preblur_std not in (None, 0):
        raise NotImplementedError(
            "obj_preblur changes the parameter-to-field map; thread it through "
            "the direction response before enabling it with ML-ISS."
        )
    if cfg.step_mode not in ("alternating", "joint"):
        raise ValueError(f"Unknown ML-ISS step_mode: {cfg.step_mode!r}")
    if cfg.objective not in ("gaussian", "amplitude", "poisson"):
        raise ValueError(f"Unknown ML-ISS objective: {cfg.objective!r}")
    if getattr(cfg, "slice_step", "scalar") not in ("scalar", "vector"):
        raise ValueError(f"Unknown ML-ISS slice_step: {cfg.slice_step!r}")
    if (
        getattr(cfg, "slice_step", "scalar") == "vector"
        and cfg.objective != "gaussian"
        and cfg.step_sigma_floor is None
    ):
        raise NotImplementedError("slice_step='vector' needs a Gaussian step weight")
    if cfg.objective != "gaussian" and cfg.step_mode == "joint":
        raise NotImplementedError("amplitude/poisson objectives are alternating-only")
    mliss_sigma2(torch.zeros(1), cfg.counts_per_unit)  # validate counts_per_unit


def _detector_postmap(model):
    if model.detector_blur_std is None or model.detector_blur_std == 0:
        return None
    try:
        from torchvision.transforms.functional import gaussian_blur
    except ImportError:
        from ptyrad.utils import gaussian_blur_2d as gaussian_blur
    std = model.detector_blur_std

    def postmap(x):
        return gaussian_blur(x, kernel_size=[5, 5], sigma=std)

    return postmap


def mliss_model_update(model, index, config, state=None, update_probe=True, loss_fn=None, H=None):
    """One per-view ML-ISS update on a PtychoAD model (solver_type='born',
    born_iterations=1). Same gather/scatter windows, probe-shift handling and
    detector-blur postmap as linesearch_model_update; the update itself is
    mliss_batch_update. Constraints remain the caller's per-iteration
    responsibility. When loss_fn is given, PtyRAD's standard losses (incl.
    the l1 sparsity term — logged only, never searched) are evaluated on the
    pre-step model DP as diagnostics["losses"]."""
    cfg = config
    st = state if state is not None else MLISSState()
    _model_guard(model, cfg)

    device = model.opt_obja.device
    idx = torch.as_tensor([index], device=device)

    gy = model.rpy_grid + model.crop_pos[index, 0]
    gx = model.rpx_grid + model.crop_pos[index, 1]
    obja_win = model.opt_obja.data[:, :, gy, gx]
    objp_win = model.opt_objp.data[:, :, gy, gx]
    if H is None:
        H = model.get_propagators_3d(model.get_propagators(idx)).detach()
    probe = model.get_probes(idx).detach()
    I_dat = model.get_measurements(idx).detach()
    postmap = _detector_postmap(model)

    patches0 = None
    if loss_fn is not None:
        patches0 = torch.stack([obja_win, objp_win], dim=-1).unsqueeze(0)

    probe_new, diag = mliss_batch_update(
        obja_win,
        objp_win,
        probe,
        H,
        I_dat,
        None,
        model.omode_occu,
        config=cfg,
        state=st,
        update_probe=update_probe,
        intensity_postmap=postmap,
        order=model.born_iterations,
    )

    if loss_fn is not None:
        with torch.no_grad():
            _, diag["losses"] = loss_fn(diag["model_dp"], I_dat, patches0, model.omode_occu)

    with torch.no_grad():
        model.opt_obja.data[:, :, gy, gx] = obja_win
        model.opt_objp.data[:, :, gy, gx] = objp_win
        if update_probe:
            dP = (probe_new - probe)[0]
            if model.shift_probes:
                from ptyrad.utils import imshift_batch

                dP = imshift_batch(
                    dP,
                    shifts=-model.opt_probe_pos_shifts[idx].detach(),
                    grid=model.shift_probes_grid,
                )[0]
            torch.view_as_complex(model.opt_probe.data).add_(dP)
    return diag


# --------------------------------------------------------------------------- #
# Model-layer wiring: joint batch update (B >= 1)                              #
# --------------------------------------------------------------------------- #


def mliss_model_update_batched(
    model, indices, config, state=None, update_probe=True, loss_fn=None, H=None,
):
    """Joint ML-ISS update over a batch of B views: object gradient of L_G
    scatter-accumulated on the canvas, canvas-accumulated K_j, ONE scalar
    step for the whole batch (exact — the field is affine in the object for
    every view), then the batch-averaged probe step against the exactly
    updated fields. Same batching contract as the per-view entry point,
    with the L_G objective, omega = mask/sigma^2, damp default 1.0 and
    explicit fallback counting. Directions are the preconditioned gradients
    only — no conjugate-gradient recursion."""
    cfg = config
    st = state if state is not None else MLISSState()
    _model_guard(model, cfg)
    order = model.born_iterations
    fwd = _fields_from_complex if order == 1 else _fields_from_complex_m2

    device = model.opt_obja.device
    idx = torch.as_tensor(np.asarray(indices).reshape(-1), device=device)
    B = idx.numel()
    N = model.opt_obja.shape[-3]

    if H is None:
        H = model.get_propagators_3d(model.get_propagators(idx)).detach()
    probe = model.get_probes(idx).detach()
    I_dat = model.get_measurements(idx).detach()
    gy = model.rpy_grid[None] + model.crop_pos[idx, 0, None, None]
    gx = model.rpx_grid[None] + model.crop_pos[idx, 1, None, None]
    postmap = _detector_postmap(model)

    sigma2 = mliss_sigma2(I_dat, cfg.counts_per_unit)
    omega = 1.0 / sigma2

    # ---- forward through a differentiable canvas gather --------------------
    O_canvas = torch.polar(model.opt_obja.data, model.opt_objp.data).requires_grad_(True)
    O_b = O_canvas[:, :, gy, gx].permute(2, 0, 1, 3, 4)
    P_leaf = probe.clone().requires_grad_(update_probe)
    if cfg.objective != "gaussian":
        F = fwd(O_b, P_leaf, H)
        u_p = dp_from_fields(F, model.omode_occu)
        if postmap is not None:
            u_p = postmap(u_p)
        L = (
            _la_loss(u_p, I_dat, None)
            if cfg.objective == "amplitude"
            else _lp_loss(u_p, I_dat, None, cfg.counts_per_unit, cfg.poisson_floor)
        )
    elif postmap is None and order == 1:
        L, F, u_p = _fwd_lg(O_b, P_leaf, H, I_dat, None, sigma2, model.omode_occu)
    else:
        F = fwd(O_b, P_leaf, H)
        u_p = dp_from_fields(F, model.omode_occu)
        if postmap is not None:
            u_p = postmap(u_p)
        L = mliss_loss(u_p, I_dat, None, sigma2)
    L.backward()
    F = F.detach()
    u_p = u_p.detach()
    e = u_p - I_dat

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
        # ---- canvas preconditioner K_j (scatter-accumulated over the batch) --
        phi = unscattered_illumination(P_leaf.detach(), H)
        K_win = phi.abs().square().sum(dim=1).squeeze(1)
        if K_win.shape[0] == 1:
            K_win = K_win.expand(B, *K_win.shape[1:])
        Nz, Hc, Wc = model.opt_obja.shape[-3:]
        K_canvas = torch.zeros(Nz, Hc * Wc, device=device, dtype=K_win.dtype)
        lin = (gy.long() * Wc + gx.long()).reshape(-1)
        K_canvas.index_add_(1, lin, K_win.permute(1, 0, 2, 3).reshape(Nz, -1))
        K_canvas = K_canvas.view(Nz, Hc, Wc)
        peak = K_canvas.amax(dim=(-2, -1), keepdim=True)
        dn = peak if cfg.object_denom == "max" else K_canvas + peak * cfg.denom_reg
        assert O_canvas.grad is not None
        d_canvas = (-O_canvas.grad) / dn.clamp_min(DN_EPS)

        # ---- JOINT step (one gamma along d and d_P, NJP 14 App. B) -----------
        if cfg.step_mode == "joint" and update_probe:
            d_wins = d_canvas[:, :, gy, gx].permute(2, 0, 1, 3, 4)
            K_P = O_b.detach().abs().square().sum(dim=(0, 1, 2))
            peak_P = K_P.amax()
            dn_p = peak_P if cfg.probe_denom == "max" else K_P + peak_P * cfg.denom_reg
            grad_P = P_leaf.grad
            if model.shift_probes:
                shifts = model.opt_probe_pos_shifts[idx].detach()
                grid = model.shift_probes_grid
                grad_P = _shift_views(grad_P, -shifts, grid).sum(dim=0, keepdim=True)
            q = cfg.probe_dir_weight * ((-grad_P) / dn_p.clamp_min(DN_EPS))
            q_views = _shift_views(q, shifts, grid) if model.shift_probes else q
            D_d = direction_response(None, d_wins, probe, H)
            a_f = _fields_from_complex(O_b.detach(), q_views, H) + D_d
            b_f = direction_response(None, d_wins, q_views, H)
            va, wa = _ml_response_terms(F, a_f, model.omode_occu, postmap)
            vb, wb = _ml_response_terms(F, b_f, model.omode_occu, postmap)
            ab = _ml_response_terms(a_f, b_f, model.omode_occu, postmap)[0]
            g, fb = ml_joint_line_search(
                e, va, wa, vb, ab, wb, omega,
                fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step,
            )
            st.steps_o.append(g)
            st.fallback_o += int(fb)
            O_new = O_canvas.detach() + g * d_canvas
            moved = (d_canvas.real != 0) | (d_canvas.imag != 0)
            model.opt_obja.data.copy_(torch.where(moved, O_new.abs(), model.opt_obja.data))
            model.opt_objp.data.copy_(torch.where(moved, O_new.angle(), model.opt_objp.data))
            torch.view_as_complex(model.opt_probe.data).add_(g * q[0])
            diagnostics = {
                "a": g, "b": g, "loss": L.detach(), "model_dp": u_p, "fb_o": fb, "fb_p": None,
            }
            if loss_fn is not None:
                _, diagnostics["losses"] = loss_fn(u_p, I_dat, patches0, model.omode_occu)
            pint = torch.view_as_complex(model.opt_probe.data).abs().square()
            diagnostics["probe_peak"] = pint.max()
            diagnostics["probe_mean"] = pint.mean()
            return diagnostics

        # ---- response, quartic on L_G, solve ---------------------------------
        d_wins = d_canvas[:, :, gy, gx].permute(2, 0, 1, 3, 4)

        # ---- M = 2: exact quartic object step, then probe against F(a) -------
        if order == 2:
            D1, D2 = _m2_direction_fields(O_b.detach(), d_wins, probe, H, F)
            va, wa = _ml_response_terms(F, D1, model.omode_occu, postmap)
            vb, wb = _ml_response_terms(F, D2, model.omode_occu, postmap)
            ab = _ml_response_terms(D1, D2, model.omode_occu, postmap)[0]
            if cfg.step_sigma_floor is not None:
                om_s = 1.0 / (I_dat + cfg.step_sigma_floor)
                a, fb = ml_joint_line_search(
                    e, va, wa, vb, ab, wb, om_s,
                    fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step,
                )
            elif cfg.objective == "amplitude":
                a, fb, nev = la_line_search_quartic(
                    u_p, va, wa, vb, ab, wb, I_dat, None,
                    fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step,
                )
                st.ls_evals.append(nev)
            elif cfg.objective == "poisson":
                a, fb, nev = lp_line_search_quartic(
                    u_p, va, wa, vb, ab, wb, I_dat, None, cfg.counts_per_unit,
                    fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step,
                    floor_counts=cfg.poisson_floor,
                )
                st.ls_evals.append(nev)
            else:
                a, fb = ml_joint_line_search(
                    e, va, wa, vb, ab, wb, omega,
                    fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step,
                )
            st.steps_o.append(a)
            st.fallback_o += int(fb)

            O_new = O_canvas.detach() + a * d_canvas
            moved = (d_canvas.real != 0) | (d_canvas.imag != 0)
            model.opt_obja.data.copy_(torch.where(moved, O_new.abs(), model.opt_obja.data))
            model.opt_objp.data.copy_(torch.where(moved, O_new.angle(), model.opt_objp.data))

            diagnostics = {
                "a": a, "b": None, "loss": L.detach(), "model_dp": u_p,
                "fb_o": fb, "fb_p": None,
            }

            if update_probe:
                # exact post-step field and intensity (field quadratic in object)
                F = F + a * D1 + (a * a) * D2
                A1, A2, A3, A4 = _quartic_I(u_p, va, wa, vb, ab, wb)
                u2 = (u_p.double() + a * (A1 + a * (A2 + a * (A3 + a * A4)))).to(u_p.dtype)
                e2 = u2 - I_dat
                K_P = O_b.detach().abs().square().sum(dim=(0, 1, 2))
                peak_P = K_P.amax()
                dn_p = peak_P if cfg.probe_denom == "max" else K_P + peak_P * cfg.denom_reg
                grad_P = P_leaf.grad
                if model.shift_probes:
                    shifts = model.opt_probe_pos_shifts[idx].detach()
                    grid = model.shift_probes_grid
                    grad_P = _shift_views(grad_P, -shifts, grid).sum(dim=0, keepdim=True)
                q = (-grad_P) / dn_p.clamp_min(DN_EPS)
                g2 = O_b.detach() + a * d_wins
                q_views = _shift_views(q, shifts, grid) if model.shift_probes else q
                D_P = _fields_from_complex_m2(g2, q_views, H)
                v_p, w_p = _ml_response_terms(F, D_P, model.omode_occu, postmap)
                if cfg.step_sigma_floor is not None:
                    om_s = 1.0 / (I_dat + cfg.step_sigma_floor)
                    b, fbp = ml_line_search(
                        e2, v_p, w_p, om_s, fallback=cfg.beta / N, damp=cfg.damp,
                        max_step=cfg.max_step,
                    )
                elif cfg.objective == "amplitude":
                    b, fbp, nev_p = la_line_search(
                        u2, v_p, w_p, I_dat, None, fallback=cfg.beta / N,
                        damp=cfg.damp, max_step=cfg.max_step,
                    )
                    st.ls_evals.append(nev_p)
                elif cfg.objective == "poisson":
                    b, fbp, nev_p = lp_line_search(
                        u2, v_p, w_p, I_dat, None, cfg.counts_per_unit,
                        fallback=cfg.beta / N, floor_counts=cfg.poisson_floor,
                        damp=cfg.damp, max_step=cfg.max_step,
                    )
                    st.ls_evals.append(nev_p)
                else:
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

        D_slices = direction_response(None, d_wins, probe, H, per_slice=True)
        D = D_slices.sum(dim=3)
        v, w = _ml_response_terms(F, D, model.omode_occu, postmap)
        if getattr(cfg, "slice_step", "scalar") == "vector":
            om_v = (
                1.0 / (I_dat + cfg.step_sigma_floor)
                if cfg.step_sigma_floor is not None
                else omega
            )
            a_vec, fb, vx = ml_vector_line_search(
                e, F, D_slices, om_v, model.omode_occu, fallback=cfg.alpha / N,
                damp=cfg.damp, postmap=postmap,
            )
            a_t = a_vec.to(device=device, dtype=d_canvas.real.dtype)
            st.steps_o.append(float(a_vec.mean()))
            st.steps_o_vec.append([float(x) for x in a_vec])
            st.fallback_o += int(fb)
            O_new = O_canvas.detach() + d_canvas * a_t.view(1, -1, 1, 1)
            moved = (d_canvas.real != 0) | (d_canvas.imag != 0)
            model.opt_obja.data.copy_(torch.where(moved, O_new.abs(), model.opt_obja.data))
            model.opt_objp.data.copy_(torch.where(moved, O_new.angle(), model.opt_objp.data))
            diagnostics = {
                "a": float(a_vec.mean()), "b": None, "loss": L.detach(), "model_dp": u_p,
                "fb_o": fb, "fb_p": None, "vec_extras": vx,
            }
            if update_probe:
                # exact field after the per-slice step: F + sum_j a_j D_j
                F = F + (D_slices * a_t.view(1, 1, 1, -1, 1, 1).to(D_slices.dtype)).sum(dim=3)
                u2 = dp_from_fields(F, model.omode_occu)
                if postmap is not None:
                    u2 = postmap(u2)
                e2 = u2 - I_dat
                K_P = O_b.detach().abs().square().sum(dim=(0, 1, 2))
                peak_P = K_P.amax()
                dn_p = peak_P if cfg.probe_denom == "max" else K_P + peak_P * cfg.denom_reg
                grad_P = P_leaf.grad
                if model.shift_probes:
                    shifts = model.opt_probe_pos_shifts[idx].detach()
                    grid = model.shift_probes_grid
                    grad_P = _shift_views(grad_P, -shifts, grid).sum(dim=0, keepdim=True)
                q = (-grad_P) / dn_p.clamp_min(DN_EPS)
                g2 = O_b.detach() + d_wins * a_t.view(1, 1, -1, 1, 1).to(d_wins.dtype)
                q_views = _shift_views(q, shifts, grid) if model.shift_probes else q
                D_P = _fields_from_complex(g2, q_views, H)
                v_p, w_p = _ml_response_terms(F, D_P, model.omode_occu, postmap)
                if cfg.step_sigma_floor is not None:
                    b, fbp = ml_line_search(
                        e2, v_p, w_p, 1.0 / (I_dat + cfg.step_sigma_floor),
                        fallback=cfg.beta / N, damp=cfg.damp, max_step=cfg.max_step,
                    )
                else:
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
        if cfg.step_sigma_floor is not None:
            om_s = 1.0 / (I_dat + cfg.step_sigma_floor)
            a, fb = ml_line_search(
                e, v, w, om_s, fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step
            )
        elif cfg.objective == "amplitude":
            a, fb, nev = la_line_search(
                u_p, v, w, I_dat, None, fallback=cfg.alpha / N, damp=cfg.damp,
                max_step=cfg.max_step,
            )
            st.ls_evals.append(nev)
        elif cfg.objective == "poisson":
            a, fb, nev = lp_line_search(
                u_p, v, w, I_dat, None, cfg.counts_per_unit, fallback=cfg.alpha / N, floor_counts=cfg.poisson_floor,
                damp=cfg.damp, max_step=cfg.max_step,
            )
            st.ls_evals.append(nev)
        else:
            a, fb = ml_line_search(
                e, v, w, omega, fallback=cfg.alpha / N, damp=cfg.damp, max_step=cfg.max_step
            )
        st.steps_o.append(a)
        st.fallback_o += int(fb)

        # ---- one joint complex step on the canvas ----------------------------
        O_new = O_canvas.detach() + a * d_canvas
        moved = (d_canvas.real != 0) | (d_canvas.imag != 0)
        model.opt_obja.data.copy_(torch.where(moved, O_new.abs(), model.opt_obja.data))
        model.opt_objp.data.copy_(torch.where(moved, O_new.angle(), model.opt_objp.data))

        diagnostics = {
            "a": a,
            "b": None,
            "loss": L.detach(),
            "model_dp": u_p,
            "fb_o": fb,
            "fb_p": None,
        }

        # ---- probe step against the exactly-updated field --------------------
        if update_probe:
            F = F + a * D
            u2 = u_p + (2.0 * a) * v + (a * a) * w
            e2 = u2 - I_dat
            K_P = O_b.detach().abs().square().sum(dim=(0, 1, 2))
            peak_P = K_P.amax()
            dn_p = peak_P if cfg.probe_denom == "max" else K_P + peak_P * cfg.denom_reg
            grad_P = P_leaf.grad
            if model.shift_probes:
                shifts = model.opt_probe_pos_shifts[idx].detach()
                grid = model.shift_probes_grid
                grad_P = _shift_views(grad_P, -shifts, grid).sum(dim=0, keepdim=True)
            q = (-grad_P) / dn_p.clamp_min(DN_EPS)
            g2 = O_b.detach() + a * d_wins
            q_views = _shift_views(q, shifts, grid) if model.shift_probes else q
            D_P = _fields_from_complex(g2, q_views, H)
            v_p, w_p = _ml_response_terms(F, D_P, model.omode_occu, postmap)
            if cfg.step_sigma_floor is not None:
                om_s = 1.0 / (I_dat + cfg.step_sigma_floor)
                b, fbp = ml_line_search(
                    e2, v_p, w_p, om_s, fallback=cfg.beta / N, damp=cfg.damp,
                    max_step=cfg.max_step,
                )
            elif cfg.objective == "amplitude":
                b, fbp, nev_p = la_line_search(
                    u2, v_p, w_p, I_dat, None, fallback=cfg.beta / N, damp=cfg.damp,
                    max_step=cfg.max_step,
                )
                st.ls_evals.append(nev_p)
            elif cfg.objective == "poisson":
                b, fbp, nev_p = lp_line_search(
                    u2, v_p, w_p, I_dat, None, cfg.counts_per_unit, fallback=cfg.beta / N, floor_counts=cfg.poisson_floor,
                    damp=cfg.damp, max_step=cfg.max_step,
                )
                st.ls_evals.append(nev_p)
            else:
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

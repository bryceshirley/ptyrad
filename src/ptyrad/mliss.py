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

Differences from LISS/BLISS (ptyrad.linesearch, which is NOT modified):

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

Shared with LISS/BLISS (imported, unmodified): the ISS field map, the §4.2
unit map, the response terms, the per-slice direction response, and the
ePIE-style preconditioners K_j (batch-accumulated |phi_j|^2, per-slice
spatial max) and K_P (batch/slice-accumulated |O|^2, spatial max).

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

from ptyrad.linesearch import (
    DN_EPS,
    _compiled,
    _fields_from_complex,
    _q_at,
    _quartic_terms,
    _real_cubic_roots,
    apply_object_step,
    direction_response,
    dp_from_fields,
    firstborn_fields as iss_fields,
    object_denominator,
    probe_denominator,
    unscattered_illumination,
)

def _shift_views(x, shifts, grid):
    """Per-view sub-pixel Fourier shift (private copy: aa212b7's linesearch
    predates this helper). Same phasor convention as utils.imshift_batch."""
    from torch.fft import fft2, ifft2

    ky, kx = grid[0], grid[1]
    phase = -2.0 * torch.pi * (shifts[:, 1, None, None] * kx + shifts[:, 0, None, None] * ky)
    w = torch.polar(torch.ones_like(phase), phase).unsqueeze(1)
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
    the host-side cubic guards of ptyrad.linesearch verbatim.

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
    from ptyrad.mlms import brent_min  # lazy: mlms imports mliss at module load

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
    from ptyrad.mlms import brent_min  # lazy: mlms imports mliss at module load

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
):
    """One ML-ISS update on full-frame tensors: exact object step on L_G,
    then exact probe step on L_G against the exactly updated field.
    Structure mirrors linesearch_batch_update (which is not modified);
    shapes and in-place semantics are identical to it.

    Returns (probe, diagnostics); diagnostics["loss"] is L_G of the batch at
    the pre-step state (0-dim tensor, call .item() at logging cadence)."""
    cfg = config
    st = state if state is not None else MLISSState()
    if probe.shape[0] != 1:
        raise ValueError("mliss_batch_update expects a shared probe (B = 1 design point)")
    N = obja.shape[-3]

    sigma2 = mliss_sigma2(I_dat, cfg.counts_per_unit)
    omega = (1.0 / sigma2) if mask is None else mask / sigma2

    # ---- forward with leaves for both gradients (one backward) -------------
    O_leaf = torch.polar(obja, objp).unsqueeze(0).detach().requires_grad_(True)
    P_leaf = probe.detach().clone().requires_grad_(update_probe)
    if getattr(cfg, "objective", "gaussian") != "gaussian":
        F = _fields_from_complex(O_leaf, P_leaf, H)
        u_p = dp_from_fields(F, omode_occu)
        if intensity_postmap is not None:
            u_p = intensity_postmap(u_p)
        L = (
            _la_loss(u_p, I_dat, mask)
            if cfg.objective == "amplitude"
            else _lp_loss(u_p, I_dat, mask, cfg.counts_per_unit, cfg.poisson_floor)
        )
    elif intensity_postmap is None:
        L, F, u_p = _fwd_lg(O_leaf, P_leaf, H, I_dat, mask, sigma2, omode_occu)
    else:
        F = _fields_from_complex(O_leaf, P_leaf, H)
        u_p = intensity_postmap(dp_from_fields(F, omode_occu))
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

    # ---- response (per-slice container kept, like LISS), quartic, solve ----
    D_slices = direction_response(None, d, P_leaf.detach(), H, per_slice=True)
    D = D_slices.sum(dim=3)
    v, w = _ml_response_terms(F, D, omode_occu, intensity_postmap)
    if cfg.objective == "amplitude":
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
        if cfg.objective == "amplitude":
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
    from ptyrad.linesearch import response_terms

    v, w = response_terms(F, D, omode_occu)
    if intensity_postmap is not None:
        v, w = intensity_postmap(v), intensity_postmap(w)
    return v, w


# --------------------------------------------------------------------------- #
# Model-layer wiring: per-view (B = 1) update on a PtychoAD ISS model          #
# --------------------------------------------------------------------------- #


def _model_guard(model, cfg):
    if model.solver_type != "born" or model.born_iterations != 1:
        raise ValueError(
            "ML-ISS requires solver_type='born' with born_iterations=1: "
            "the field is affine in the object only for single scattering."
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
    model, indices, config, state=None, update_probe=True, loss_fn=None, H=None
):
    """Joint ML-ISS update over a batch of B views: object gradient of L_G
    scatter-accumulated on the canvas, canvas-accumulated K_j, ONE scalar
    step for the whole batch (exact — the field is affine in the object for
    every view), then the batch-averaged probe step against the exactly
    updated fields. Mirrors linesearch_model_update_batched (unmodified),
    with the L_G objective, omega = mask/sigma^2, damp default 1.0 and
    explicit fallback counting."""
    cfg = config
    st = state if state is not None else MLISSState()
    _model_guard(model, cfg)

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
        F = _fields_from_complex(O_b, P_leaf, H)
        u_p = dp_from_fields(F, model.omode_occu)
        if postmap is not None:
            u_p = postmap(u_p)
        L = (
            _la_loss(u_p, I_dat, None)
            if cfg.objective == "amplitude"
            else _lp_loss(u_p, I_dat, None, cfg.counts_per_unit, cfg.poisson_floor)
        )
    elif postmap is None:
        L, F, u_p = _fwd_lg(O_b, P_leaf, H, I_dat, None, sigma2, model.omode_occu)
    else:
        F = _fields_from_complex(O_b, P_leaf, H)
        u_p = postmap(dp_from_fields(F, model.omode_occu))
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
        D_slices = direction_response(None, d_wins, probe, H, per_slice=True)
        D = D_slices.sum(dim=3)
        v, w = _ml_response_terms(F, D, model.omode_occu, postmap)
        if cfg.objective == "amplitude":
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
            if cfg.objective == "amplitude":
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

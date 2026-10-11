"""
maximum-likelihood exact line search for the ISS engine
(paper-1 Sec. "Line search", sec:ml-solver).

Implements maximum likelihood in the sense of Thibault & Guizar-Sicairos,
New J. Phys. 14, 063004 (2012): the search direction and the line search
both come from the SAME likelihood. The ISS field is affine in the object
and linear in the probe, so along either search line the model intensity is
the exact quadratic I(g) = u + 2 g v + g^2 w. Three data fidelities are
supported, matching the paper:

- 'gaussian' (default): L_G = sum w/(2 sigma^2) (I_model - I_data)^2 with
  sigma^2 = I_data + s0 the Poisson variance floored at ONE DETECTOR COUNT.
  The measurements PtyRAD optimises against are normalised (default
  'max_at_one'), so one count is s0 = 1/counts_per_unit in normalised
  units; counts_per_unit is a required config value and is never guessed.
  L_G(g) is then an exact quartic and the step is the real root of its
  cubic derivative with the lowest L_G — an exact line search.
- 'amplitude' / 'poisson': the likelihood is not polynomial in g, but
  L'(g) and L''(g) are closed-form pixelwise sums through I(g), so the
  step is the Newton step g = -L'(0)/L''(0) (the paper's Eq. newton-step;
  ptypy's ML rule). cfg.newton_iters > 1 refines it with re-evaluated
  Newton iterations at no additional transforms; cfg.newton_safeguard adds
  the accept-only-if-decreasing rule with the 1/N fallback.

Updates alternate object then probe (block coordinate descent). The probe
step is searched against the EXACTLY updated field F + a*D (closed form, no
re-forward), with the probe response built at the UPDATED slices; the probe
DIRECTION is the pre-step gradient — one backward per batch, as in the
paper. It is then no longer exactly the gradient at the updated object, but
the search still finds the best (possibly negative) step along it.
Preconditioners are the ePIE-style diagonals K_j (batch-accumulated
|phi_j|^2, per-slice spatial max) and K_P (batch/slice-accumulated |O|^2,
spatial max); the line search absorbs any common scale, so they only set
the relative weighting.

Fallback a = 1/N fires when the solve degenerates or (gaussian /
safeguarded Newton) no step strictly lowers the loss, and is COUNTED
explicitly (state.fallback_o / fallback_p), not inferred from step values.

The l1 sparsity regulariser (loss_sparse) is NOT part of the direction and
NOT part of the line search; it is only evaluated for logging through the
optional `loss_fn` diagnostics.

Batching: `mliss_model_update_batched` is the model-layer entry point (batch
size from the caller; driver: recon_params.BATCH_SIZE), and
`mliss_model_update` is its per-view (B = 1) shorthand. `mliss_batch_update`
is the tensor-level core the oracle tests drive directly.

Removed variants live in git history: Brent/'all' step rules, the joint
object+probe degree-8 search (the paper searches object and probe
separately on purpose), the per-slice vector step, the M = 2
quadratic-field engine, the post-step probe gradient, the depth-coupled
object preconditioner, and the LISS/BLISS reproduction knobs.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.fft import fft2, ifft2

from ptyrad.forward_models.born_helpers import _born_scatter, unscattered_illumination
from ptyrad.utils import fftshift2

# §6: floors. DN_EPS is float32 eps, used on the preconditioner denominators.
DP_EPS = 1e-10  # matches forward_models/born.py intensity floor
DN_EPS = float(torch.finfo(torch.float32).eps)  # ≈ 1.19e-7


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

    The scatter is born_helpers._born_scatter, so the recursion physics lives
    in one place: this is born_forward at n_max = 1 up to (and excluding) the
    intensity reduction. F is exactly affine in (O - 1) and exactly linear in
    the probe — the two facts the line search rests on (spec §1)."""
    Ny, Nx = O.shape[-2:]
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    psi = ifft2(H * probe_k)  # unscattered illumination phi_j
    g = (O - 1.0).unsqueeze(1)  # chord perturbation, (B, 1, omode, Nz, Ny, Nx)
    return probe_k.squeeze(3) + _born_scatter(g, psi, H, 0).sum(dim=3)


def dp_from_fields(F, omode_occu, eps=DP_EPS):
    """§4.2 unit map: PtyRAD model DP from the per-mode detector field.

    dp = fftshift2( sum_{pmode,omode} |F|^2 * occu_o/(Nx*Ny) + eps ), matching
    born_forward exactly (pinned by oracle test 3b). Note the per-omode
    occupancy sits INSIDE the mode sum — it is not a scalar when omode > 1."""
    Ny, Nx = F.shape[-2:]
    nw = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(F.abs().square() * nw, dim=(1, 2)) + eps)


@_compiled
def response_terms(F, D, omode_occu):
    """Per-pixel linear (v) and quadratic (w) intensity coefficients, carrying
    the same §4.2 transformation as dp_from_fields but WITHOUT the +eps floor
    (a constant: it lives in u, hence in e = u - I_dat, and drops from v, w)."""
    Ny, Nx = F.shape[-2:]
    nw = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    v = fftshift2(torch.sum((F.conj() * D).real * nw, dim=(1, 2)))
    w = fftshift2(torch.sum(D.abs().square() * nw, dim=(1, 2)))
    return v, w


@_compiled
def direction_response(object_patches, d, probe, H):
    """Object direction response D = F(g + d) - F(g), exact for any d because
    F is affine in g (one extra forward half-pass, no adjoint) — the paper's
    detector update direction sum_j D_j[P_j d_j] (Eq. slice-step-field),
    i.e. one born_helpers scatter of d under the unscattered illumination
    (the identity path cancels in the difference).

    (`object_patches` is kept for API stability but only d and probe carry
    information.)
    """
    psi = unscattered_illumination(probe, H)
    return _born_scatter(d.unsqueeze(1), psi, H, 0).sum(dim=3)


def object_denominator(phi):
    """K_j denominator: per-slice spatial peak of the batch/pmode-summed
    illumination intensity |phi_j|^2 (the 'max' recipe — starves off-focus
    pixels). phi from unscattered_illumination, (B, pmode, 1, Nz, Ny, Nx).
    Returns (Nz, 1, 1)-broadcastable, floored at DN_EPS."""
    K = phi.abs().square().sum(dim=(0, 1, 2))  # (Nz, Ny, Nx)
    return K.amax(dim=(-2, -1), keepdim=True).clamp_min(DN_EPS)


def probe_denominator(obj_complex):
    """K_P denominator: spatial peak of |O|^2 summed over batch/omode/slices,
    at the PRE-STEP object — the probe diagonal (spec §3 step 7). Floored."""
    K = obj_complex.abs().square()
    K = K.sum(dim=tuple(range(K.dim() - 2)))  # -> (Ny, Nx)
    return K.amax().clamp_min(DN_EPS)


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
    no per-slice loop of steps anywhere in the object update.

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
# The likelihoods                                                              #
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


@_compiled
def _fwd_lg(O, P, H, I_dat, mask, sigma2, omode_occu):
    """Fused forward + L_G (one compiled region so the backward compiles too).
    Callers with an intensity postmap use the eager pieces instead."""
    F = _fields_from_complex(O, P, H)
    u_p = dp_from_fields(F, omode_occu)
    L = mliss_loss(u_p, I_dat, mask, sigma2)
    return L, F, u_p


def _forward_loss(cfg, O, P, H, I_dat, mask, sigma2, omode_occu, postmap):
    """Forward + configured loss, with O and P as autograd leaves — the one
    forward/backward both entry points run. The default gaussian/no-postmap
    case takes the fused compiled region. Returns (L, F, u_p)."""
    if cfg.objective == "gaussian" and postmap is None:
        return _fwd_lg(O, P, H, I_dat, mask, sigma2, omode_occu)
    F = _fields_from_complex(O, P, H)
    u_p = dp_from_fields(F, omode_occu)
    if postmap is not None:
        u_p = postmap(u_p)
    if cfg.objective == "amplitude":
        L = _la_loss(u_p, I_dat, mask)
    elif cfg.objective == "poisson":
        L = _lp_loss(u_p, I_dat, mask, cfg.counts_per_unit, cfg.poisson_floor)
    else:
        L = mliss_loss(u_p, I_dat, mask, sigma2)
    return L, F, u_p


# --------------------------------------------------------------------------- #
# Gaussian: exact quartic line search with explicit fallback flag              #
# --------------------------------------------------------------------------- #


def ml_line_search(e, v, w, omega, fallback, max_step=0.0):
    """Exact step on Q(a) = sum omega (e + 2av + a^2 w)^2 = 2 L_G(a), with
    omega = mask/sigma^2 — the SAME weight as the direction, so both optimise
    one objective (paper Eq. cubic-poly: the real cubic roots in closed form,
    keep the one with the lowest L_G). Float64-before-products coefficient
    sums; host-side cubic guards. ML-ISS takes the full step.

    Returns (a, used_fallback). The root must be STRICTLY below Q(0), else
    fallback (1/N, supplied by the caller). Candidates are compared through
    Q(a) - Q(0) = 4 (c0 a + c1 a^2/2 + c2 a^3/3 + c3 a^4/4) — exact host-side
    integration of the computed derivative coefficients, so no second device
    reduction is needed."""
    c0, c1, c2, c3 = _quartic_terms(e, v, w, omega).cpu().tolist()
    cands = _real_cubic_roots(c0, c1, c2, c3)
    used_fallback = True
    a = fallback
    if cands is not None:
        best, bestq = None, 0.0  # Q(cand) - Q(0): accept strict decrease only
        for cand in cands:
            dq = 4.0 * cand * (c0 + cand * (c1 / 2.0 + cand * (c2 / 3.0 + cand * c3 / 4.0)))
            if dq < bestq:
                best, bestq = cand, dq
        if best is not None:
            a, used_fallback = best, False
    if max_step > 0:
        a = float(np.clip(a, -max_step, max_step))
    return float(a), used_fallback


# --------------------------------------------------------------------------- #
# Amplitude / Poisson: Newton steps on the closed-form intensity path          #
# --------------------------------------------------------------------------- #
#
# Along I(g) = u + 2 g v + g^2 w the ptypy ML engine (Thibault &
# Guizar-Sicairos, NJP 14, 063004) takes the single unsafeguarded Newton step
# g = -L'(0)/L''(0) — the paper's Eq. newton-step. These functions implement
# that step and up to K re-evaluated iterations (closed-form I(g), no
# additional transforms) for the amplitude and Poisson objectives.
#
# Floor treatment: la clamps I(g) at 1e-12 inside the sqrt; lp clamps c*I(g)
# at floor_counts. The clamped objective is locally constant in g on a floored
# pixel, so such pixels contribute ZERO to L' and L'' (this is the derivative
# of the actual objective the search minimises).

_NEWTON_L1_RTOL = 1e-14  # stop iterating when |L'| is at round-off of L


def _amp_derivs(u64, v64, w64, sI, m64, g):
    """(L, L', L'') of the amplitude loss at step g, one (3,) float64 tensor.
    L'  = sum m (1 - sqrt(I_dat/I(g))) I',       I'  = 2v + 2gw
    L'' = sum m [ sqrt(I_dat)/(2 I(g)^{3/2}) I'^2 + (1 - sqrt(I_dat/I(g))) I'' ]
    with floored pixels (I(g) <= 1e-12) contributing 0 to L' and L''."""
    g = float(g)
    Ig_raw = u64 + (2.0 * g) * v64 + (g * g) * w64
    Ig = Ig_raw.clamp_min(1e-12)
    sIg = Ig.sqrt()
    L = (sIg - sI).square()
    act = (Ig_raw > 1e-12).to(u64.dtype)
    Ip = 2.0 * v64 + (2.0 * g) * w64
    t = 1.0 - sI / sIg
    L1 = act * (t * Ip)
    L2 = act * (0.5 * sI * Ip * Ip / (sIg * Ig) + t * (2.0 * w64))
    if m64 is not None:
        L, L1, L2 = m64 * L, m64 * L1, m64 * L2
    return torch.stack([L.sum(), L1.sum(), L2.sum()])


def _poi_derivs(u64, v64, w64, cI, m64, c, floor, g):
    """(L, L', L'') of the Poisson NLL in counts at step g, (3,) float64.
    cu = clamp(c I(g), floor_counts);  L = sum m (cu - cI log cu)
    L'  = sum m (1 - cI/cu) c I'
    L'' = sum m [ (cI/cu^2)(c I')^2 + (1 - cI/cu) c I'' ]
    with floored pixels (c I(g) <= floor_counts) contributing 0."""
    g = float(g)
    cu_raw = c * (u64 + (2.0 * g) * v64 + (g * g) * w64)
    cu = cu_raw.clamp_min(floor)
    L = cu - cI * torch.log(cu)
    act = (cu_raw > floor).to(u64.dtype)
    cIp = c * (2.0 * v64 + (2.0 * g) * w64)
    t = 1.0 - cI / cu
    L1 = act * (t * cIp)
    L2 = act * ((cI / cu.square()) * cIp.square() + t * (c * 2.0 * w64))
    if m64 is not None:
        L, L1, L2 = m64 * L, m64 * L1, m64 * L2
    return torch.stack([L.sum(), L1.sum(), L2.sum()])


def _newton_1d(derivs_at, fallback, max_step=0.0, iters=1, safeguard=False):
    """Shared Newton driver. derivs_at(g) -> (L, L', L'') floats. Each
    iteration takes g <- g - L'(g)/L''(g) with PLAIN division (no curvature
    guard: L'' <= 0 steps uphill/away and L'' == 0 gives +-inf, exactly the
    unsafeguarded ptypy behaviour). Iteration stops early when g goes
    non-finite or |L'| reaches round-off of |L|.

    safeguard=True applies the acceptance rule: one extra L evaluation,
    accept only if L(g) < L(0) and g is finite, else `fallback`.
    Returns (a, used_fallback)."""
    g = 0.0
    L0 = None
    for _ in range(iters):
        L, L1, L2 = (float(x) for x in derivs_at(g))
        if L0 is None:
            L0 = L
        if not np.isfinite(L1) or not np.isfinite(L2):
            break
        if abs(L1) <= _NEWTON_L1_RTOL * (abs(L) + 1.0):
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            g = g - float(np.divide(L1, L2))
        if not np.isfinite(g):
            break
    used_fallback = False
    a = g
    if safeguard:
        ok = np.isfinite(g) and g != 0.0
        if ok:
            Lg = float(derivs_at(g)[0])
            ok = np.isfinite(Lg) and Lg < L0
        used_fallback = not ok
        if used_fallback:
            a = fallback
    if max_step > 0 and np.isfinite(a):
        a = float(np.clip(a, -max_step, max_step))
    return float(a), used_fallback


def _step_1d(cfg, u, e, v, w, I_dat, mask, omega, fallback):
    """THE 1D step along I(g) = u + 2gv + g^2 w, on the configured objective:
    exact cubic-root quartic search for 'gaussian', Newton step(s) for
    'amplitude'/'poisson'. One function so the object and probe steps of both
    entry points dispatch identically. Returns (step, used_fallback).

    Poisson freeze rule: a non-finite L(0) returns a ZERO step (flagged as
    fallback) instead of stepping 1/N along a possibly huge direction."""
    if cfg.objective == "gaussian":
        return ml_line_search(e, v, w, omega, fallback, max_step=cfg.max_step)
    u64, v64, w64 = u.double(), v.double(), w.double()
    m64 = None if mask is None else mask.double()
    if cfg.objective == "amplitude":
        sI = I_dat.double().clamp_min(0).sqrt()

        def derivs(g):
            return _amp_derivs(u64, v64, w64, sI, m64, g).cpu()
    else:
        c = float(cfg.counts_per_unit)
        cI = (c * I_dat.clamp_min(0)).double()

        def derivs(g):
            return _poi_derivs(u64, v64, w64, cI, m64, c, float(cfg.poisson_floor), g).cpu()

        if not np.isfinite(float(derivs(0.0)[0])):
            return 0.0, True
    return _newton_1d(
        derivs, fallback, max_step=cfg.max_step,
        iters=cfg.newton_iters, safeguard=cfg.newton_safeguard,
    )


# --------------------------------------------------------------------------- #
# Config / state                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class MLISSConfig:
    """ML-ISS knobs (paper-1 sec:ml-solver)."""

    counts_per_unit: float | None = None  # REQUIRED: detector counts per unit
    max_step: float = 0.0  # symmetric clip on the step; 0 = off
    # 'gaussian': exact quartic line search on L_G (the ML-ISS definition).
    # 'amplitude' / 'poisson': direction AND Newton step on the amplitude
    # loss L_A = sum mask (sqrt I_model - sqrt I_data)^2, respectively the
    # Poisson NLL in counts (model intensity floored at poisson_floor).
    objective: str = "gaussian"
    # Poisson objective only: model-intensity floor in COUNTS (bounds the
    # Poisson weight 1 - I/u; 0.01 count is negligible bias, finite gradient)
    poisson_floor: float = 0.01
    # Amplitude/Poisson only: number of Newton iterations on the 1D step
    # (1 = ptypy's single unsafeguarded step; more re-evaluate L', L'' at the
    # current g — closed-form I(g), no additional transforms).
    newton_iters: int = 1
    # Amplitude/Poisson only: accept the Newton step only if L strictly
    # decreases, else the 1/N fallback, counted in state.fallback_o/p.
    # False = unsafeguarded, exactly as ptypy.
    newton_safeguard: bool = False


@dataclass
class MLISSState:
    """Cross-batch state: step logs plus explicit fallback counters (the
    fallback fraction per iteration is n_fallback/n_steps over the sweep)."""

    steps_o: list = field(default_factory=list)
    steps_p: list = field(default_factory=list)
    fallback_o: int = 0
    fallback_p: int = 0


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
    """One ML-ISS update on full-frame tensors: object step, then probe step
    against the exactly updated field F + a*D (closed form, no re-forward).
    The probe direction is the pre-step gradient from the SAME backward as
    the object direction (paper: one backward per update).

    Returns (probe, diagnostics); diagnostics["loss"] is the batch loss at
    the pre-step state (0-dim tensor, call .item() at logging cadence)."""
    cfg = config
    st = state if state is not None else MLISSState()
    if probe.shape[0] != 1:
        raise ValueError("mliss_batch_update expects a shared probe (B = 1 design point)")
    N = obja.shape[-3]

    sigma2 = mliss_sigma2(I_dat, cfg.counts_per_unit)
    omega = (1.0 / sigma2) if mask is None else mask / sigma2

    # ---- one forward/backward for both directions --------------------------
    O_leaf = torch.polar(obja, objp).unsqueeze(0).detach().requires_grad_(True)
    P_leaf = probe.detach().clone().requires_grad_(update_probe)
    L, F, u_p = _forward_loss(
        cfg, O_leaf, P_leaf, H, I_dat, mask, sigma2, omode_occu, intensity_postmap
    )
    L.backward()
    F = F.detach()
    u_p = u_p.detach()
    e = u_p - I_dat

    # ---- preconditioned descent direction, response, 1D step ---------------
    phi = unscattered_illumination(P_leaf.detach(), H)
    dn_o = object_denominator(phi)
    assert O_leaf.grad is not None
    d = (-O_leaf.grad) / dn_o  # descent = -O.grad (torch Wirtinger convention)
    D = direction_response(None, d, P_leaf.detach(), H)
    v, w = _ml_response_terms(F, D, omode_occu, intensity_postmap)
    a, fb = _step_1d(cfg, u_p, e, v, w, I_dat, mask, omega, 1.0 / N)
    st.steps_o.append(a)
    st.fallback_o += int(fb)

    # ---- joint complex object step -----------------------------------------
    apply_object_step(obja, objp, a, d[0])

    diagnostics = {"a": a, "b": None, "loss": L.detach(), "fb_o": fb, "fb_p": None}

    # ---- probe step against the exactly-updated field ----------------------
    if update_probe:
        F = F + a * D  # exact — handles the bilinear cross term
        u2 = u_p + (2.0 * a) * v + (a * a) * w  # exact, no re-forward
        e2 = u2 - I_dat

        O2 = torch.polar(obja, objp).unsqueeze(0)  # updated (stored) object
        dn_p = probe_denominator(O_leaf.detach())
        q = (-P_leaf.grad) / dn_p  # pre-step probe gradient: no second backward

        # D_P = F(q; g2): response at the UPDATED slices (phi rebuilt from q)
        D_P = _fields_from_complex(O2, q, H)
        v_p, w_p = _ml_response_terms(F, D_P, omode_occu, intensity_postmap)
        b, fbp = _step_1d(cfg, u2, e2, v_p, w_p, I_dat, mask, omega, 1.0 / N)
        st.steps_p.append(b)
        st.fallback_p += int(fbp)
        probe = probe + b * q
        diagnostics["b"] = b
        diagnostics["fb_p"] = fbp

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
    if model.solver_type != "born" or model.born_iterations != 1:
        raise ValueError(
            "ML-ISS requires solver_type='born' with born_iterations=1 "
            "(ISS: field affine in the object)."
        )
    if model.obj_preblur_std not in (None, 0):
        raise NotImplementedError(
            "obj_preblur changes the parameter-to-field map; thread it through "
            "the direction response before enabling it with ML-ISS."
        )
    if cfg.objective not in ("gaussian", "amplitude", "poisson"):
        raise ValueError(f"Unknown ML-ISS objective: {cfg.objective!r}")
    if int(cfg.newton_iters) < 1:
        raise ValueError(f"ML-ISS newton_iters must be >= 1, got {cfg.newton_iters!r}")
    mliss_sigma2(torch.zeros(1), cfg.counts_per_unit)  # validate counts_per_unit


def _detector_postmap(model):
    """Detector-blur postmap, the SAME gaussian_blur (and torchvision
    fallback) the forward model applies in models.PtychoAD."""
    if model.detector_blur_std is None or model.detector_blur_std == 0:
        return None
    from ptyrad.models import gaussian_blur

    std = model.detector_blur_std

    def postmap(x):
        return gaussian_blur(x, kernel_size=[5, 5], sigma=std)

    return postmap


def mliss_model_update(model, index, config, state=None, update_probe=True, loss_fn=None, H=None):
    """One per-view ML-ISS update on a PtychoAD model: the batched update on
    a single index (at B = 1 the canvas K_j peak, the probe-shift algebra and
    the searches coincide with a dedicated per-view path)."""
    return mliss_model_update_batched(
        model, [index], config, state=state, update_probe=update_probe, loss_fn=loss_fn, H=H
    )


# --------------------------------------------------------------------------- #
# Model-layer wiring: joint batch update (B >= 1)                              #
# --------------------------------------------------------------------------- #


def mliss_model_update_batched(
    model, indices, config, state=None, update_probe=True, loss_fn=None, H=None,
):
    """Joint ML-ISS update over a batch of B views: object gradient
    scatter-accumulated on the canvas, canvas-accumulated K_j, ONE scalar
    step for the whole batch (exact — the field is affine in the object for
    every view), then the batch-averaged probe step against the exactly
    updated fields. Same batching contract as the per-view entry point.
    Directions are the preconditioned gradients only — no conjugate-gradient
    recursion."""
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
    L, F, u_p = _forward_loss(
        cfg, O_b, P_leaf, H, I_dat, None, sigma2, model.omode_occu, postmap
    )
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
        assert O_canvas.grad is not None
        # ---- canvas preconditioner K_j (scatter-accumulated over the batch)
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
        d_canvas = (-O_canvas.grad) / peak.clamp_min(DN_EPS)

        # ---- response, 1D step, one joint complex step on the canvas --------
        d_wins = d_canvas[:, :, gy, gx].permute(2, 0, 1, 3, 4)
        D = direction_response(None, d_wins, probe, H)
        v, w = _ml_response_terms(F, D, model.omode_occu, postmap)
        a, fb = _step_1d(cfg, u_p, e, v, w, I_dat, None, omega, 1.0 / N)
        st.steps_o.append(a)
        st.fallback_o += int(fb)

        O_new = O_canvas.detach() + a * d_canvas
        moved = (d_canvas.real != 0) | (d_canvas.imag != 0)
        model.opt_obja.data.copy_(torch.where(moved, O_new.abs(), model.opt_obja.data))
        model.opt_objp.data.copy_(torch.where(moved, O_new.angle(), model.opt_objp.data))

        diagnostics = {"a": a, "b": None, "loss": L.detach(), "fb_o": fb, "fb_p": None}

        # ---- probe step against the exactly-updated field --------------------
        if update_probe:
            F = F + a * D
            u2 = u_p + (2.0 * a) * v + (a * a) * w
            e2 = u2 - I_dat
            g2 = O_b.detach() + a * d_wins
            dn_p = probe_denominator(O_b.detach())
            grad_P = P_leaf.grad  # pre-step probe gradient: no second backward
            if model.shift_probes:
                shifts = model.opt_probe_pos_shifts[idx].detach()
                grid = model.shift_probes_grid
                grad_P = _shift_views(grad_P, -shifts, grid).sum(dim=0, keepdim=True)
            q = (-grad_P) / dn_p
            q_views = _shift_views(q, shifts, grid) if model.shift_probes else q
            D_P = _fields_from_complex(g2, q_views, H)
            v_p, w_p = _ml_response_terms(F, D_P, model.omode_occu, postmap)
            b, fbp = _step_1d(cfg, u2, e2, v_p, w_p, I_dat, None, omega, 1.0 / N)
            st.steps_p.append(b)
            st.fallback_p += int(fbp)
            torch.view_as_complex(model.opt_probe.data).add_(b * q[0])
            diagnostics["b"] = b
            diagnostics["fb_p"] = fbp

        if loss_fn is not None:
            _, diagnostics["losses"] = loss_fn(u_p, I_dat, patches0, model.omode_occu)
    return diagnostics

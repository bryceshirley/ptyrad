"""
Exact quartic line search for the first-Born (single-scattering) engine.

Implements §3 of LINESEARCH_BORN_SPEC.md: preconditioned autograd direction,
direction response from one extra forward evaluation (no hand-derived adjoint
— see spec §0), exact quartic coefficients with float64 accumulation, a
host-side cubic solve, the joint complex object step, and the probe step taken
against the exactly-updated field.

The function contract here is pinned by test/test_linesearch_oracle.py; the
docstring of that file is the authoritative API description. Shapes follow
forward_models/born.py:

    object_patches : (B, omode, Nz, Ny, Nx, 2)  float, [..., 0]=amp, [..., 1]=phase
    probe          : (B|1, pmode, Ny, Nx)       complex
    H              : (1, 1, 1, Nz, Ny, Nx)      complex, H_j = H^j (entrance at j=0)
    fields F       : (B, pmode, omode, Ny, Nx)  complex, unshifted k-space

Design notes.
- Everything is dtype-preserving so the float64 oracle tests stay exact; the
  production path is float32/complex64 except the quartic sums (§4.3/§6).
- `direction_response` keeps D addressable per slice (spec §9). The N-D vector
  line search (per-slice steps, the N×N Gram) is deliberately NOT implemented;
  `linesearch_batch_update` accumulates D from the per-slice container so the
  seam costs nothing today.
- Batch design point is B = 1 (spec §3): probe updated every view. The batch
  update here operates on full-frame tensors; per-view crop windows
  (gather/scatter), the phi cache, and the reconstruction-loop switch are the
  model-layer integration seam, not part of this module.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.fft import fft2, fftshift, ifft2

# §6: floors. DN_EPS is float32 eps, used on the preconditioner denominators.
DP_EPS = 1e-10  # matches forward_models/born.py intensity floor
SQRT_FLOOR = 1e-12  # clamp inside any sqrt of a model intensity
DN_EPS = float(torch.finfo(torch.float32).eps)  # ≈ 1.19e-7


def _fftshift2(x):
    return fftshift(x, dim=(-2, -1))


# At B = 1 the per-view update is launch-overhead- and sync-bound, not
# FLOP-bound: the 128^2 kernels are tiny. torch.compile fuses the pointwise
# soup between FFTs; the host-sync reductions live in line_search (single
# batched transfer for the coefficients, single vectorized transfer for Q at
# the candidate roots). All compiled functions fall back to eager when dynamo
# is disabled (the tests set TORCHDYNAMO_DISABLE=1), with identical results.
def _compiled(fn):
    import os

    mode = os.environ.get("PTYRAD_LS_COMPILE_MODE", "default")
    return torch.compile(fn, dynamic=False, mode=mode)


# --------------------------------------------------------------------------- #
# Forward field and the §4.2 unit map                                          #
# --------------------------------------------------------------------------- #


@_compiled
def _fields_from_complex(O, probe, H):
    """Detector-plane field from complex object O (B, omode, Nz, Ny, Nx).

    Same maths as forward_models/born.py::firstborn_forward up to (and
    excluding) the intensity reduction; F is exactly affine in (O - 1) and
    exactly linear in the probe — the two facts the line search rests on
    (spec §1)."""
    Ny, Nx = O.shape[-2:]
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    psi = ifft2(H * probe_k)  # unscattered illumination phi_j
    g = (O - 1.0).unsqueeze(1)  # chord perturbation, (B, 1, omode, Nz, Ny, Nx)
    scattered = torch.sum(fft2(g * psi) * H.conj(), dim=3)
    return probe_k.squeeze(3) + scattered  # (B, pmode, omode, Ny, Nx)


def firstborn_fields(object_patches, probe, H):
    """Detector field from PtyRAD (amp, phase) patches. Dtype-preserving."""
    O = torch.polar(object_patches[..., 0], object_patches[..., 1])
    return _fields_from_complex(O, probe, H)


def dp_from_fields(F, omode_occu, eps=DP_EPS):
    """§4.2 unit map: PtyRAD model DP from the per-mode detector field.

    dp = fftshift2( sum_{pmode,omode} |F|^2 * occu_o/(Nx*Ny) + eps ), matching
    firstborn_forward exactly (pinned by oracle test 3b). Note the per-omode
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


# --------------------------------------------------------------------------- #
# Direction response (spec §2, §9)                                             #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Direction objective and its autograd gradient (spec §5, option (a) default)  #
# --------------------------------------------------------------------------- #


def _direction_loss(u_p, I_dat, mask, objective):
    """The §5 direction objective, in PtyRAD units.

    'amplitude' (default, §5 option (a) — matches ptypy, ls_damp = 0.5 is
    load-bearing): E_amp = sum m (sqrt(u) - sqrt(I_dat))^2.
    'intensity' (§5 option (b)): the Gaussian quartic Q itself.
    A callable(u_p, I_dat, mask) -> scalar covers §5 option (c)."""
    if callable(objective):
        return objective(u_p, I_dat, mask)
    if objective == "amplitude":
        resid = (u_p.clamp_min(SQRT_FLOOR).sqrt() - I_dat.clamp_min(0).sqrt()).square()
    elif objective == "intensity":
        omega = 1.0 / (I_dat + 1.0)
        resid = omega * (u_p - I_dat).square()
    else:
        raise ValueError(f"Unknown direction objective: {objective!r}")
    return (resid if mask is None else mask * resid).sum()


@_compiled
def _fwd_loss(O, P, H, I_dat, mask, omode_occu, objective):
    """Fused forward + direction objective (one compiled region, so the
    AOTAutograd backward is compiled too). Only for the built-in string
    objectives and no intensity postmap — callers fall back to the eager
    pieces otherwise."""
    F = _fields_from_complex(O, P, H)
    u_p = dp_from_fields(F, omode_occu)
    L = _direction_loss(u_p, I_dat, mask, objective)
    return L, F, u_p


def direction_gradient(obj_complex, probe, H, I_dat, mask, omode_occu, objective="amplitude"):
    """Descent direction of the direction objective w.r.t. complex O.

    Torch Wirtinger convention: O.grad = 2*dL/d(conj(O)); the returned descent
    direction is -O.grad (oracle test 1). NOTE: ptypy's acc_grad equals
    -dL/d(conj(O)) — a factor 2 smaller. The exact step absorbs any positive
    scaling of the direction; only the FALLBACK step magnitude feels it, so
    fallback parity with ptypy at matched alpha needs alpha halved here."""
    O = obj_complex.detach().clone().requires_grad_(True)
    F = _fields_from_complex(O, probe, H)
    L = _direction_loss(dp_from_fields(F, omode_occu), I_dat, mask, objective)
    L.backward()
    return -O.grad


# --------------------------------------------------------------------------- #
# Preconditioners (spec §3 step 2 and step 7)                                  #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Quartic coefficients and the host-side cubic solve (spec §3 steps 4-5, §6)   #
# --------------------------------------------------------------------------- #


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


def quartic_coeffs(e, v, w, omega):
    """The four coefficients of (1/4) dQ/da for Q(a) = sum omega (e + 2av +
    a^2 w)^2. All four sums are accumulated in float64 with the inputs CAST
    BEFORE THE PRODUCTS — casting after .sum() does not prevent the float32
    overflow of c3 = sum(omega w^2) that silently pins the run to the fixed
    fallback step (spec §4.3, oracle test 7)."""
    c0, c1, c2, c3 = _quartic_terms(e, v, w, omega).cpu().tolist()
    return c0, c1, c2, c3


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


def solve_step_cubic(c0, c1, c2, c3, Qfun, fallback):
    """Minimise the quartic along the line: dQ/da = 4(c0 + c1 a + c2 a^2 +
    c3 a^3) = 0, host-side via np.roots (spec §3 step 5). Fallback on any
    non-finite coefficient (guard kept despite the float64 promotion, §6),
    a degenerate cubic, no real root, or no root strictly below Q(0)."""
    cands = _real_cubic_roots(c0, c1, c2, c3)
    if cands is None:
        return fallback
    best, bestq = None, Qfun(0.0)
    for a in cands:
        qa = Qfun(a)
        if qa < bestq:
            best, bestq = a, qa
    return fallback if best is None else best


def line_search(e, v, w, omega, fallback, ls_damp=0.5, max_step=0.0, step_log=None):
    """Exact damped step: coefficients + cubic solve, then a *= ls_damp (the
    fallback is damped too — degenerate inputs return exactly ls_damp*fallback,
    oracle test 6), optional symmetric clip at max_step.

    ls_damp = 0.5 is load-bearing under the default amplitude direction (§5):
    it reconciles the amplitude-cost direction with the intensity-quartic step.

    Every accepted step should be logged (pass step_log): a healthy run shows
    live spread in a; every value clustered at exactly ls_damp*fallback is the
    only cheap symptom of the silent-fallback failure (spec §4.3, §6).

    Semantics identical to solve_step_cubic + damping, but with exactly two
    device-to-host syncs: one batched transfer for the four coefficients, one
    vectorized transfer for Q at {0} ∪ candidate roots."""
    c0, c1, c2, c3 = _quartic_terms(e, v, w, omega).cpu().tolist()
    cands = _real_cubic_roots(c0, c1, c2, c3)
    if cands is None:
        a = fallback
    else:
        a_vec = torch.tensor([0.0, *cands], dtype=torch.float64, device=e.device)
        qs = _q_at(e, v, w, omega, a_vec).cpu().tolist()
        best, bestq = None, qs[0]  # root must be STRICTLY below Q(0)
        for cand, qa in zip(cands, qs[1:], strict=True):
            if qa < bestq:
                best, bestq = cand, qa
        a = fallback if best is None else best
    a *= ls_damp
    if max_step > 0:
        a = float(np.clip(a, -max_step, max_step))
    a = float(a)
    if step_log is not None:
        step_log.append(a)
    return a


# --------------------------------------------------------------------------- #
# The complex object step (spec §3 step 6, §4.1)                               #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# The batch update (spec §3 steps 1-7)                                         #
# --------------------------------------------------------------------------- #


@dataclass
class LineSearchConfig:
    """Knobs (spec §8)."""

    alpha: float = 1.0  # object fallback step alpha/N, on cubic degeneracy only
    beta: float = 1.0  # probe fallback step beta/N, same condition
    ls_damp: float = 0.5  # multiplier on every step, exact and fallback (§5)
    max_step: float = 0.0  # symmetric clip on the damped step; 0 = off
    object_denom: str = "max"  # 'max' | 'local'
    probe_denom: str = "max"  # decoupled from object_denom
    denom_reg: float = 0.01  # floor as fraction of peak K for 'local'
    momentum: float = 0.0  # heavy ball, mixed in BEFORE the search
    direction_objective: object = "amplitude"  # 'amplitude' | 'intensity' | callable


@dataclass
class LineSearchState:
    """Cross-batch state: heavy-ball displacement and the step logs (§6 —
    the logs are the only cheap detector of a run stuck in fallback)."""

    disp_prev: torch.Tensor | None = None
    steps_o: list = field(default_factory=list)
    steps_p: list = field(default_factory=list)


def linesearch_batch_update(
    obja,
    objp,
    probe,
    H,
    I_dat,
    mask,
    omode_occu,
    config=None,
    state=None,
    update_probe=True,
    intensity_postmap=None,
):
    """One §3 batch update on full-frame tensors. Steps the object storage
    (obja, objp) in place and returns (probe, diagnostics) — the probe is
    returned rather than mutated so the caller controls its buffer and can
    invalidate any phi cache (spec §3 step 7).

    obja, objp : (omode, Nz, Ny, Nx) float storage
    probe      : (1, pmode, Ny, Nx) complex (B = 1 design point)
    H          : (1, 1, 1, Nz, Ny, Nx) complex
    I_dat      : (1, Ny, Nx) measured DP in PtyRAD units
    mask       : (1, Ny, Nx) or None
    intensity_postmap : optional LINEAR map applied identically to u, v and w
        (e.g. the detector blur that get_forward_meas applies after the §4.2
        expression). Linearity keeps the quartic exact; anything nonlinear
        voids the search.

    Model-layer integration (per-view crop windows, probe shifts, constraint
    call sites — constraints fire per iteration, after the batch loop, so
    F <- F + a*D stays valid within a batch, spec §4.4) lives outside this
    module."""
    cfg = config or LineSearchConfig()
    st = state if state is not None else LineSearchState()
    if probe.shape[0] != 1:
        raise ValueError("linesearch_batch_update expects a shared probe (B = 1 design point)")
    N = obja.shape[-3]

    # ---- step 1: forward, with leaves for both gradients (one backward) ----
    O_leaf = torch.polar(obja, objp).unsqueeze(0).detach().requires_grad_(True)
    P_leaf = probe.detach().clone().requires_grad_(update_probe)
    if intensity_postmap is None and isinstance(cfg.direction_objective, str):
        L, F, u_p = _fwd_loss(
            O_leaf, P_leaf, H, I_dat, mask, omode_occu, cfg.direction_objective
        )
    else:
        F = _fields_from_complex(O_leaf, P_leaf, H)
        u_p = dp_from_fields(F, omode_occu)
        if intensity_postmap is not None:
            u_p = intensity_postmap(u_p)
        L = _direction_loss(u_p, I_dat, mask, cfg.direction_objective)
    L.backward()
    F = F.detach()
    u_p = u_p.detach()

    omega = (1.0 / (I_dat + 1.0)) if mask is None else mask / (I_dat + 1.0)
    e = u_p - I_dat

    # ---- step 2: preconditioned descent direction (+ heavy ball) -----------
    phi = unscattered_illumination(P_leaf.detach(), H)
    dn_o = object_denominator(phi, mode=cfg.object_denom, denom_reg=cfg.denom_reg)
    d = (-O_leaf.grad) / dn_o  # descent = -O.grad (torch Wirtinger convention)
    if cfg.momentum > 0 and st.disp_prev is not None:
        d = d + cfg.momentum * st.disp_prev

    # ---- step 3: response, accumulated per slice (§9 seam) -----------------
    patches = torch.stack([obja, objp], dim=-1).unsqueeze(0)
    D_slices = direction_response(patches, d, P_leaf.detach(), H, per_slice=True)
    D = D_slices.sum(dim=3)

    # ---- steps 4-5: coefficients and cubic solve ---------------------------
    v, w = response_terms(F, D, omode_occu)
    if intensity_postmap is not None:
        v, w = intensity_postmap(v), intensity_postmap(w)
    a = line_search(
        e, v, w, omega,
        fallback=cfg.alpha / N, ls_damp=cfg.ls_damp, max_step=cfg.max_step,
        step_log=st.steps_o,
    )

    # ---- step 6: joint complex object step ---------------------------------
    apply_object_step(obja, objp, a, d[0])
    st.disp_prev = a * d

    # pre-step model DP (PtyRAD units, §4.2, incl. any postmap) — lets callers
    # evaluate PtyRAD's own losses without a second forward. "loss" and the
    # probe watchdog fields are 0-dim tensors, NOT floats: converting here
    # would force a device sync per view and stall the async pipeline. Call
    # .item() at whatever cadence you actually log.
    diagnostics = {"a": a, "b": None, "loss": L.detach(), "model_dp": u_p}

    # ---- step 7: probe step against the exactly-updated field --------------
    if update_probe:
        F = F + a * D  # exact — this handles the bilinear cross term
        u_p = u_p + (2.0 * a) * v + (a * a) * w  # exact, no re-forward
        e = u_p - I_dat

        # q from the gradient at the PRE-step object, K_P from pre-step O
        dn_p = probe_denominator(
            O_leaf.detach(), mode=cfg.probe_denom, denom_reg=cfg.denom_reg
        )
        q = (-P_leaf.grad) / dn_p

        # D_P = F(q; g2): F is linear in P, so one ordinary forward with q in
        # place of P at the UPDATED object — phi is rebuilt from q inside.
        patches2 = torch.stack([obja, objp], dim=-1).unsqueeze(0)
        D_P = firstborn_fields(patches2, q, H)
        v_p, w_p = response_terms(F, D_P, omode_occu)
        if intensity_postmap is not None:
            v_p, w_p = intensity_postmap(v_p), intensity_postmap(w_p)
        b = line_search(
            e, v_p, w_p, omega,
            fallback=cfg.beta / N, ls_damp=cfg.ls_damp, max_step=cfg.max_step,
            step_log=st.steps_p,
        )
        probe = probe + b * q
        diagnostics["b"] = b

    # §6 watchdog inputs: probe peak/mean intensity — a runaway shows a spike
    pint = probe.abs().square()
    diagnostics["probe_peak"] = pint.max()
    diagnostics["probe_mean"] = pint.mean()
    return probe, diagnostics


# --------------------------------------------------------------------------- #
# Model-layer wiring: one view update on a PtychoAD first-Born model           #
# --------------------------------------------------------------------------- #


def linesearch_model_update(
    model, index, config=None, state=None, update_probe=True, loss_fn=None, H=None
):
    """One line-search view update (B = 1, the §3 design point) on a PtychoAD
    model with solver_type='born' and born_iterations=1.

    Gathers the object window at scan position `index` (same crop grids as
    model.get_obj_ROI), runs linesearch_batch_update on it, scatters the
    stepped window back into the (omode, Nz, H, W) canvases, and adds the
    probe increment b*q into opt_probe — unshifted first when per-view
    sub-pixel probe shifts are active, so the shared probe is updated in its
    own frame. Detector blur, when configured, is threaded through the linear
    intensity_postmap so the quartic stays exact.

    Constraints are the caller's responsibility and fire per iteration after
    the view sweep (reconstruction.py convention, spec §4.4), so the in-batch
    exact field update is never invalidated here.

    When `loss_fn` (a CombinedLoss) is given, PtyRAD's standard losses are
    evaluated on the pre-step model DP and pre-step patches — the same
    quantities recon_step logs — and returned as diagnostics["losses"], so a
    line-search run can be compared against an optimizer run line for line.

    Returns the diagnostics dict of linesearch_batch_update.
    """
    cfg = config or LineSearchConfig()
    st = state if state is not None else LineSearchState()
    if model.solver_type != "born" or model.born_iterations != 1:
        raise ValueError(
            "The exact line search requires solver_type='born' with born_iterations=1: "
            "F is affine in the object only for single scattering (spec §1)."
        )
    if model.obj_preblur_std not in (None, 0):
        raise NotImplementedError(
            "obj_preblur changes the parameter-to-field map; thread it through the "
            "direction response before enabling it with the line search."
        )
    if cfg.momentum > 0:
        raise NotImplementedError(
            "Heavy ball at the model layer needs a global displacement canvas "
            "(per-view windows overlap); use momentum only with the tensor-level updater."
        )

    device = model.opt_obja.device
    idx = torch.as_tensor([index], device=device)

    # gather: window grids identical to get_obj_ROI, propagator stack H^j
    gy = model.rpy_grid + model.crop_pos[index, 0]
    gx = model.rpx_grid + model.crop_pos[index, 1]
    obja_win = model.opt_obja.data[:, :, gy, gx]  # advanced indexing -> copy
    objp_win = model.opt_objp.data[:, :, gy, gx]
    if H is None:  # callers may hoist the (static) 3D propagator out of the loop
        H = model.get_propagators_3d(model.get_propagators(idx)).detach()
    probe = model.get_probes(idx).detach()  # (1, pmode, Ny, Nx)
    I_dat = model.get_measurements(idx).detach()

    postmap = None
    if model.detector_blur_std is not None and model.detector_blur_std != 0:
        try:
            from torchvision.transforms.functional import gaussian_blur
        except ImportError:
            from ptyrad.utils import gaussian_blur_2d as gaussian_blur
        std = model.detector_blur_std

        def postmap(x):
            return gaussian_blur(x, kernel_size=5, sigma=std)

    # snapshot pre-step patches for loss evaluation (stack copies the data,
    # so the in-place window step below cannot alias it)
    patches0 = None
    if loss_fn is not None:
        patches0 = torch.stack([obja_win, objp_win], dim=-1).unsqueeze(0)

    probe_new, diag = linesearch_batch_update(
        obja_win, objp_win, probe, H, I_dat, None, model.omode_occu,
        config=cfg, state=st, update_probe=update_probe, intensity_postmap=postmap,
    )

    if loss_fn is not None:
        with torch.no_grad():
            _, diag["losses"] = loss_fn(
                diag["model_dp"], I_dat, patches0, model.omode_occu
            )

    with torch.no_grad():
        model.opt_obja.data[:, :, gy, gx] = obja_win
        model.opt_objp.data[:, :, gy, gx] = objp_win
        if update_probe:
            dP = (probe_new - probe)[0]  # (pmode, Ny, Nx)
            if model.shift_probes:
                from ptyrad.utils import imshift_batch

                dP = imshift_batch(
                    dP,
                    shifts=-model.opt_probe_pos_shifts[idx].detach(),
                    grid=model.shift_probes_grid,
                )[0]
            torch.view_as_complex(model.opt_probe.data).add_(dP)
    return diag


def _shift_views(x, shifts, grid):
    """Per-view sub-pixel Fourier shift: x (B|1, pmode, Ny, Nx) shifted by
    shifts (B, 2) [y, x] px -> (B, pmode, Ny, Nx). Same phasor convention as
    utils.imshift_batch (which broadcasts ONE image over the shifts and so
    cannot shift B distinct images by B distinct shifts). Unitary; the
    adjoint/inverse is the same call with -shifts."""
    ky, kx = grid[0], grid[1]
    phase = -2.0 * torch.pi * (
        shifts[:, 1, None, None] * kx + shifts[:, 0, None, None] * ky
    )
    w = torch.polar(torch.ones_like(phase), phase).unsqueeze(1)  # (B, 1, Ny, Nx)
    return ifft2(fft2(x) * w)


def linesearch_model_update_batched(
    model, indices, config=None, state=None, update_probe=True, loss_fn=None, H=None
):
    """Joint §3 update over a batch of B views on a PtychoAD first-Born model.

    The ptypy-batched structure: the object gradient scatter-accumulates into
    the full canvas (autograd does this for free through a differentiable
    window gather), the preconditioner K_j scatter-accumulates the batch
    illumination onto the canvas, and ONE scalar `a` steps the whole canvas
    jointly in complex — the quartic stays exact for the joint step because F
    is affine in g for every view. The probe step is averaged over the batch
    (spec §3 warns B > 1 averages the probe update; B = 1 is the design
    point — use this entry point when you deliberately want batch parity with
    an optimizer run).

    Momentum is supported here (the displacement lives on the canvas, so
    overlapping windows compose correctly). Per-view sub-pixel probe shifts
    are supported exactly: shifts are unitary and every view's field is
    linear in the SHARED probe, so the shared-frame gradient is the sum of
    the per-view gradients unshifted by -s_b, and the response of a shared
    direction q is one forward with shift_b(q) per view. The shift values
    themselves stay frozen (position refinement is not part of the port).
    """
    cfg = config or LineSearchConfig()
    st = state if state is not None else LineSearchState()
    if model.solver_type != "born" or model.born_iterations != 1:
        raise ValueError(
            "The exact line search requires solver_type='born' with born_iterations=1: "
            "F is affine in the object only for single scattering (spec §1)."
        )
    if model.obj_preblur_std not in (None, 0):
        raise NotImplementedError(
            "obj_preblur changes the parameter-to-field map; thread it through the "
            "direction response before enabling it with the line search."
        )
    device = model.opt_obja.device
    idx = torch.as_tensor(np.asarray(indices).reshape(-1), device=device)
    B = idx.numel()
    N = model.opt_obja.shape[-3]

    if H is None:
        H = model.get_propagators_3d(model.get_propagators(idx)).detach()
    probe = model.get_probes(idx).detach()  # (1,pmode,Ny,Nx) shared, or (B,...) shifted
    I_dat = model.get_measurements(idx).detach()  # (B, Ny, Nx)
    gy = model.rpy_grid[None] + model.crop_pos[idx, 0, None, None]  # (B, Ny, Nx)
    gx = model.rpx_grid[None] + model.crop_pos[idx, 1, None, None]

    postmap = None
    if model.detector_blur_std is not None and model.detector_blur_std != 0:
        try:
            from torchvision.transforms.functional import gaussian_blur
        except ImportError:
            from ptyrad.utils import gaussian_blur_2d as gaussian_blur
        std = model.detector_blur_std

        def postmap(x):
            return gaussian_blur(x, kernel_size=5, sigma=std)

    # ---- step 1: forward through a differentiable canvas gather ------------
    O_canvas = torch.polar(model.opt_obja.data, model.opt_objp.data).requires_grad_(True)
    O_b = O_canvas[:, :, gy, gx].permute(2, 0, 1, 3, 4)  # (B, omode, Nz, Ny, Nx)
    P_leaf = probe.clone().requires_grad_(update_probe)
    if postmap is None and isinstance(cfg.direction_objective, str):
        L, F, u_p = _fwd_loss(
            O_b, P_leaf, H, I_dat, None, model.omode_occu, cfg.direction_objective
        )
    else:
        F = _fields_from_complex(O_b, P_leaf, H)
        u_p = dp_from_fields(F, model.omode_occu)
        if postmap is not None:
            u_p = postmap(u_p)
        L = _direction_loss(u_p, I_dat, None, cfg.direction_objective)
    L.backward()  # O_canvas.grad scatter-adds over the B overlapping windows
    F = F.detach()
    u_p = u_p.detach()

    omega = 1.0 / (I_dat + 1.0)
    e = u_p - I_dat

    # pre-step patches snapshot for loss parity with recon_step (which logs
    # losses on the pre-update state)
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

    # ---- step 2: canvas preconditioner (scatter-accumulated over the batch) --
    with torch.no_grad():
        phi = unscattered_illumination(P_leaf.detach(), H)  # (B|1, pmode, 1, Nz, Ny, Nx)
        K_win = phi.abs().square().sum(dim=1).squeeze(1)  # (B|1, Nz, Ny, Nx), per view
        if K_win.shape[0] == 1:
            K_win = K_win.expand(B, *K_win.shape[1:])
        Nz, Hc, Wc = model.opt_obja.shape[-3:]
        Ny, Nx = K_win.shape[-2:]
        K_canvas = torch.zeros(Nz, Hc * Wc, device=device, dtype=K_win.dtype)
        lin = (gy.long() * Wc + gx.long()).reshape(-1)  # (B*Ny*Nx,)
        K_canvas.index_add_(1, lin, K_win.permute(1, 0, 2, 3).reshape(Nz, -1))
        K_canvas = K_canvas.view(Nz, Hc, Wc)
        peak = K_canvas.amax(dim=(-2, -1), keepdim=True)
        if cfg.object_denom == "max":
            dn = peak
        else:
            dn = K_canvas + peak * cfg.denom_reg
        d_canvas = (-O_canvas.grad) / dn.clamp_min(DN_EPS)
        if cfg.momentum > 0 and st.disp_prev is not None and st.disp_prev.shape == d_canvas.shape:
            d_canvas = d_canvas + cfg.momentum * st.disp_prev

        # ---- step 3-5: response (per-slice container, §9), quartic, solve ----
        d_wins = d_canvas[:, :, gy, gx].permute(2, 0, 1, 3, 4)
        D_slices = direction_response(None, d_wins, probe, H, per_slice=True)
        D = D_slices.sum(dim=3)
        v, w = response_terms(F, D, model.omode_occu)
        if postmap is not None:
            v, w = postmap(v), postmap(w)
        a = line_search(
            e, v, w, omega,
            fallback=cfg.alpha / N, ls_damp=cfg.ls_damp, max_step=cfg.max_step,
            step_log=st.steps_o,
        )

        # ---- step 6: one joint complex step on the canvas -------------------
        O_new = O_canvas.detach() + a * d_canvas
        moved = (d_canvas.real != 0) | (d_canvas.imag != 0)
        model.opt_obja.data.copy_(torch.where(moved, O_new.abs(), model.opt_obja.data))
        model.opt_objp.data.copy_(torch.where(moved, O_new.angle(), model.opt_objp.data))
        st.disp_prev = a * d_canvas

        diagnostics = {"a": a, "b": None, "loss": L.detach(), "model_dp": u_p}

        # ---- step 7: probe step against the exactly-updated field -----------
        if update_probe:
            F = F + a * D
            u2 = u_p + (2.0 * a) * v + (a * a) * w
            e2 = u2 - I_dat
            K_P = O_b.detach().abs().square().sum(dim=(0, 1, 2))  # pre-step, (Ny, Nx)
            peak_P = K_P.amax()
            dn_p = peak_P if cfg.probe_denom == "max" else K_P + peak_P * cfg.denom_reg
            # shared-frame probe gradient: per-view grads unshifted by -s_b and
            # summed (shifts are unitary; F_b is linear in the shared probe)
            grad_P = P_leaf.grad
            if model.shift_probes:
                shifts = model.opt_probe_pos_shifts[idx].detach()
                grid = model.shift_probes_grid
                grad_P = _shift_views(grad_P, -shifts, grid).sum(dim=0, keepdim=True)
            q = (-grad_P) / dn_p.clamp_min(DN_EPS)  # (1, pmode, Ny, Nx), shared frame
            g2 = O_b.detach() + a * d_wins
            # response of the shared direction: view b sees shift_b(q) — exact
            q_views = _shift_views(q, shifts, grid) if model.shift_probes else q
            D_P = _fields_from_complex(g2, q_views, H)
            v_p, w_p = response_terms(F, D_P, model.omode_occu)
            if postmap is not None:
                v_p, w_p = postmap(v_p), postmap(w_p)
            b = line_search(
                e2, v_p, w_p, omega,
                fallback=cfg.beta / N, ls_damp=cfg.ls_damp, max_step=cfg.max_step,
                step_log=st.steps_p,
            )
            torch.view_as_complex(model.opt_probe.data).add_(b * q[0])
            diagnostics["b"] = b

        if loss_fn is not None:
            _, diagnostics["losses"] = loss_fn(u_p, I_dat, patches0, model.omode_occu)

        pint = torch.view_as_complex(model.opt_probe.data).abs().square()
        diagnostics["probe_peak"] = pint.max()
        diagnostics["probe_mean"] = pint.mean()
    return diagnostics

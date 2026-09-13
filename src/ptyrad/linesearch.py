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


# --------------------------------------------------------------------------- #
# Forward field and the §4.2 unit map                                          #
# --------------------------------------------------------------------------- #


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


def direction_response(object_patches, d, probe, H, per_slice=False):
    """Object direction response D = F(g + d) - F(g), exact for any d because
    F is affine in g (spec §2 — one extra forward half-pass, no adjoint).

    Computed as the explicit per-slice sum D = sum_j FF[ d_j * phi_j ] * H_j^*
    (identical to the difference form; the identity path cancels), so that D_j
    stays addressable for the future N-D vector search (spec §9, oracle test 8).

    per_slice=False -> (B, pmode, omode, Ny, Nx)
    per_slice=True  -> (B, pmode, omode, Nz, Ny, Nx), .sum(dim=3) is the total.
    """
    Ny, Nx = object_patches.shape[-3:-1]
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


def quartic_coeffs(e, v, w, omega):
    """The four coefficients of (1/4) dQ/da for Q(a) = sum omega (e + 2av +
    a^2 w)^2. All four sums are accumulated in float64 with the inputs CAST
    BEFORE THE PRODUCTS — casting after .sum() does not prevent the float32
    overflow of c3 = sum(omega w^2) that silently pins the run to the fixed
    fallback step (spec §4.3, oracle test 7)."""
    e, v, w, omega = (t.double() for t in (e, v, w, omega))
    c0 = float((omega * e * v).sum())
    c1 = float((omega * (e * w + 2.0 * v * v)).sum())
    c2 = 3.0 * float((omega * v * w).sum())
    c3 = float((omega * w * w).sum())
    return c0, c1, c2, c3


def solve_step_cubic(c0, c1, c2, c3, Qfun, fallback):
    """Minimise the quartic along the line: dQ/da = 4(c0 + c1 a + c2 a^2 +
    c3 a^3) = 0, host-side via np.roots (spec §3 step 5). Fallback on any
    non-finite coefficient (guard kept despite the float64 promotion, §6),
    a degenerate cubic, no real root, or no root strictly below Q(0)."""
    if not np.all(np.isfinite([c0, c1, c2, c3])):
        return fallback
    coeffs = [c3, c2, c1, c0]
    while coeffs and abs(coeffs[0]) < 1e-300:
        coeffs = coeffs[1:]
    if len(coeffs) < 2:
        return fallback
    roots = np.roots(coeffs)
    cands = [float(r.real) for r in roots if abs(r.imag) <= 1e-8 * (1.0 + abs(r.real))]
    if not cands:
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
    only cheap symptom of the silent-fallback failure (spec §4.3, §6)."""
    c0, c1, c2, c3 = quartic_coeffs(e, v, w, omega)
    e64, v64, w64, o64 = (t.double() for t in (e, v, w, omega))

    def Qfun(a):
        r = e64 + (2.0 * a) * v64 + (a * a) * w64
        return float((o64 * r * r).sum())

    a = solve_step_cubic(c0, c1, c2, c3, Qfun, fallback)
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

    diagnostics = {"a": a, "b": None, "loss": float(L.detach())}

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
    diagnostics["probe_peak"] = float(pint.max())
    diagnostics["probe_mean"] = float(pint.mean())
    return probe, diagnostics

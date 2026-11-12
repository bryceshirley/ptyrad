"""Born-coefficient fitting: the shared least-squares kernel and both fits
built on it.

The coefficients c_m of the tuned series (parallel_in_depth_rewrite-3.pdf,
eq. 3.18) reweight the retained scattering orders of the Fourier-view Born
series. Two fits produce them, and both minimise

    min_c || sum_m c_m A_m - b ||

over the same kind of operand — a handful of columns A_m (one per retained
order, M at most a few tens) against a right-hand side with millions of
rows. Only the rows differ:

  REMAINDER FIT (born_remainder_coeffs, born_volume_remainder_rows, and
    the multislice targets). b is the exact multislice field of the
    current object. Needs a sequential multislice sweep per view, so it
    runs on a small calibration SUBSAMPLE and yields ONE shared
    coefficient vector per iteration, driven by
    reconstruction._refit_born_coeffs_fourier.

  RESIDUAL FIT (born_residual_coeffs). b is the fixed-point residual of
    the Lippmann-Schwinger system, so no multislice target is needed at
    all — one extra Born order suffices. Cheap enough to run at EVERY scan
    position, which is why it carries a batch axis: one Gram per position.
    This is the GMRES minimiser over the Born-order Krylov space.

Both are data-free (the measured intensities never enter, except under the
opt-in 'hybrid_amp' target), so the coefficients are a deterministic
function of the current object operator and cannot absorb object error.

Because M is tiny and the rows are many, the economical route is the
normal equations, formed once and solved on the (M, M) moments — the tall
factor is never built, never moved off the device and never decomposed.
What makes that safe is the pair of measured conditioning constraints
below. The kernel lives in one place so its callers cannot drift apart
(they previously did, as three copies: one reported a ridge-inflated
residual, and another blocked the reduction differently).

Two conditioning constraints, both measured (GT-patch detector error 0.33
when violated vs 0.037 when respected):

  * the moment reductions MUST run in complex128. The equilibrated Gram
    reaches cond ~1e8 and a complex64 dot over ~1e7 elements carries only
    ~1e-4.
  * the solve must run in column-equilibrated coordinates, with the ridge
    acting on the unit-diagonal Gram. A trace-relative ridge on the raw
    Gram over-regularises, because the Born-order column norms are
    strongly disparate at high scattering strength (the orders decay
    geometrically: s_M/s_1 ~ 1e-8 at M = 8, ~1e-13 at M = Nz = 21).

Forming the Gram squares the condition number, so directions below
~sqrt(eps64) of the leading one are not resolvable from the moments; the
ridge is what keeps them bounded rather than letting them blow up. The
solve has a ZERO prior, so a direction it cannot resolve sends its
coefficient to 0 (the order is dropped) rather than to 1 (the plain
series). Callers that want a c = 1 fallback on degenerate input must
guard for it themselves — the remainder fit does, the residual fit
deliberately does not (a vacuum patch there degrades to the free wave).

The kernel is shape-generic in a leading batch: pass columns as
(M, *batch, rows) and right-hand sides as (nrhs, *batch, rows), and the
moments come back as (*batch, M, M) and (*batch, nrhs, M). The
subsample-shared fits use no batch axis and accumulate their moments over
view chunks; the per-position fit uses batch = (B,).
"""

import torch
from torch.fft import fft2, ifft2

from .born import _born_apply_G, _born_apply_V, _born_gauges, _born_init
from .multislice import multislice_exit_wave


def born_mode_weights(
    omode_occu: torch.Tensor | None,
    omode: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """sqrt(omode_occu) row weights, so the fit norm matches the forward
    model's mode-weighted intensity. None means uniform 1/omode, which is
    a single global factor on every row and therefore cannot change the
    minimiser — it is returned anyway so callers agree on the convention.
    Returns a (omode,) tensor to be broadcast against the mode axis."""
    w = (
        torch.full((omode,), 1.0 / omode, dtype=dtype, device=device)
        if omode_occu is None
        else omode_occu.to(dtype)
    )
    return w.clamp(min=0).sqrt()


def born_gram_moments(
    C: torch.Tensor,
    R: torch.Tensor,
    k_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split-k complex128 moments (G, g) = (C C^H, C R^H) of the
    least-squares problem whose columns are the rows of C.

    C : (M, *batch, rows) columns, one per retained order
    R : (nrhs, *batch, rows) right-hand sides
    k_block : rows per reduction block — pass Ny*Nx, the natural frame
        size. A single batched zgemm with m = n = M and k in the millions
        occupies only a handful of thread blocks (profiled 131 ms/call vs
        15.5 split); carving k into frame-sized blocks restores occupancy
        while the block partials stay complex128, so nothing is lost
        numerically. The reduction is exact either way — k_block changes
        only the summation order.

    Returns (G, g) of shape (*batch, M, M) and (*batch, nrhs, M), in
    complex128. Sum them over chunks to accumulate a fit across view
    chunks; the leading (n, n) block of G with the leading n columns of g
    is exactly the fit at the lower order n, which is what makes adaptive
    order growth free of a second pass."""
    C = C.to(torch.complex128)
    R = R.to(torch.complex128)
    M, rows = C.shape[0], C.shape[-1]
    nrhs = R.shape[0]
    batch = C.shape[1:-1]
    Kb = (rows + k_block - 1) // k_block
    if Kb * k_block != rows:
        pad = Kb * k_block - rows
        C = torch.nn.functional.pad(C, (0, pad))
        R = torch.nn.functional.pad(R, (0, pad))
    Ck = C.view(M, *batch, Kb, k_block)
    Rk = R.view(nrhs, *batch, Kb, k_block)
    # k stays in the output so the contraction is a Kb-batched GEMM over z;
    # the blocks are then summed in complex128
    G = torch.einsum("m...kz,l...kz->...kml", Ck.conj(), Ck).sum(dim=-3)
    g = torch.einsum("m...kz,r...kz->...krm", Ck.conj(), Rk).sum(dim=-3)
    return G, g


def born_gram_solve(
    G: torch.Tensor,
    g: torch.Tensor,
    b2: torch.Tensor,
    lam_rel: float = 1e-12,
    breakdown_eps: float = 1e-20,
    offset: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Column-equilibrated ridge solve of the moments from
    born_gram_moments.

    offset=True (default) solves in x = c - 1 coordinates, so the ridge
    and any unresolved direction shrink toward c = 1, the PLAIN series.
    That is the right prior once the series is centred: centring makes
    the untuned series already near-optimal (measured at M = 3, strong
    scattering: 0.102 centred-plain vs 4.68 uncentred-plain, with the
    fit gaining only 0.2% on the centred one), so c = 1 is close to the
    answer and c = 0 — dropping the order — is not.

    Measured on PSO, centred basis, field error vs multislice: offset and
    zero-prior are IDENTICAL to four significant figures for M <= 12, and
    offset is strictly better beyond — 6.49e-6 vs 8.63e-6 at M = 16, and
    at M = 21 it keeps |c| ~ 1 where the zero prior lets it reach 4e+04.
    Four different solvers (this one either way, iterative refinement, and
    a float64 lstsq on the tall factor) agree below M = 12, so in the
    useful range the fit is limited by the span of the Born orders, not by
    the projection onto it.

    Only the FIRST right-hand side takes the prior. A second rhs (the
    hybrid target's c_MS recovery) is a difference of targets, whose
    natural prior is zero, and c_MS = c + x_extra stays correct.

    G  : (*batch, M, M) Gram
    g  : (*batch, nrhs, M) moments of the right-hand sides
    b2 : (*batch,) squared norms of the FIRST right-hand side
    lam_rel : ridge on the unit-diagonal Gram (see the module docstring —
        it must not act on the raw Gram)

    Returns (sol, rel_res): sol (*batch, nrhs, M) complex128, the solution
    for each right-hand side, and rel_res (*batch,) the relative residual
    ||sum_m c_m A_m - b|| / ||b|| of the FIRST one, evaluated from the
    moments.

    The residual is taken against the RAW Gram, not the ridged one: the
    ridge is a regulariser, not part of the objective, and letting it into
    the reported number inflates it (at lam_rel = 1e-2 the inflated value
    reads 0.149 where the true residual is 0.079). It is cancellation-
    limited to ~1e-11 relative once the fit residual is orders below
    ||b||, which is the price of not forming the tall product; it is a
    logged diagnostic, never a solve input. breakdown_eps makes a dead
    system (b = 0, c = 0) report rel_res = 1 rather than 0/0."""
    M = G.shape[-1]
    s = torch.sqrt(torch.diagonal(G, dim1=-2, dim2=-1).real.clamp(min=0) + breakdown_eps)
    Gs = G / (s[..., :, None] * s[..., None, :])
    Gs = Gs + lam_rel * torch.eye(M, dtype=G.dtype, device=G.device)
    if offset:  # solve for x = c - 1 on the primary rhs only
        ones = torch.ones(M, dtype=G.dtype, device=G.device)
        g = torch.cat([(g[..., :1, :] - (G @ ones).unsqueeze(-2)), g[..., 1:, :]],
                      dim=-2)
    # lu_factor_ex + lu_solve rather than torch.linalg.solve: solve raises
    # on a singular matrix, which means reading LAPACK's info back on the
    # host — a sync in the middle of the forward. The _ex variant returns
    # info instead of checking it, and the ridge on the unit-diagonal Gram
    # already guarantees a non-singular system.
    lu, piv, _info = torch.linalg.lu_factor_ex(Gs)
    sol = torch.linalg.lu_solve(lu, piv, (g / s[..., None, :]).mT).mT / s[..., None, :]
    if offset:  # back to c coordinates; the residual below uses the full c
        sol = torch.cat([sol[..., :1, :] + 1.0, sol[..., 1:, :]], dim=-2)
    c = sol[..., 0, :]
    quad = torch.einsum("...m,...mn,...n->...", c.conj(), G, c).real
    g0 = g[..., 0, :] + ((G @ torch.ones(M, dtype=G.dtype, device=G.device))
                         if offset else 0)  # undo the shift: residual vs the true b
    cross = torch.einsum("...m,...m->...", c.conj(), g0).real
    res2 = (b2 - 2.0 * cross + quad).clamp(min=0.0)
    rel_res = torch.sqrt(res2 + breakdown_eps) / torch.sqrt(b2 + breakdown_eps)
    return sol, rel_res

# ------------------------------------------------------------------------
# REMAINDER FIT: against the exact multislice target
# ------------------------------------------------------------------------


def born_multislice_target(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
) -> torch.Tensor:
    """Exact multislice detector field of the current object, entrance-plane
    gauge, by one sequential sweep with a single rolling wavefield: 2*Nz
    slice-FFT passes per view and O(1) transient frames, independent of the
    model order. Same gauge as the Born detector orders (the unimodular
    vacuum power is divided out), so target - D0 is directly comparable to
    sum_m D_m — equal to it to float precision (nilpotent termination).
    Returns (B, pmode, omode, Ny, Nx) complex. Eager; call under
    torch.no_grad()."""
    B, omode, Nz, Ny, Nx, _ = object_patches.shape
    O = torch.polar(object_patches[..., 0], object_patches[..., 1])  # (B,omode,Nz,Ny,Nx)
    H1 = H[:, :, :, min(1, Nz - 1)]  # single-step propagator (ones when Nz == 1)
    # probe[:, :, None]: (B|1, pmode, 1, Ny, Nx) -> broadcast omode
    psi = multislice_exit_wave(O, probe[:, :, None], H1)
    F = fft2(psi)
    return (F * H[:, :, :, Nz - 1].conj()).expand(B, -1, omode, -1, -1)


def born_multislice_volume_target(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
) -> torch.Tensor:
    """Per-plane multislice wavefield of the current object IN THE FOURIER
    VIEW: Psi_hat_MS, with block j equal to U_j psi_j = H^{-z_j} F psi_j for
    the field psi_j incident on slice j, j = 0..Nz-1. One sequential sweep
    stores every plane and refers each to the entrance plane as it is
    produced.

    The referral is FREE. The multislice step already forms
    S_j = F[psi_j dO_j] and sets psi_{j+1} = F^{-1}[H^{dz} S_j], so
    F psi_{j+1} is exactly H^{dz} S_j — the argument of that inverse
    transform — and the hat block is (H^{dz} S_j) H^{-z_{j+1}} with no
    transform of its own. Taking fft2(psi_{j+1}) instead would re-do the
    round trip: 3 transforms per slice rather than 2, measured 1.75x the
    detector target. Reusing it also avoids the round-trip rounding, so
    the blocks are marginally MORE accurate than a re-transform.

    Same plane convention and normalisation as the per-order stacks of
    born_volume_basis, so sum_m V[m] equals this target to float precision
    (nilpotent termination); the exit/detector entry is NOT included — pair
    with born_multislice_target for the detector row block.
    Returns (B, pmode, omode, Nz, Ny, Nx) complex. Cost is now the same
    2*Nz slice-FFTs as born_multislice_target; what remains is memory, one
    full per-slice stack per view, so still call on small view chunks.
    Eager; call under torch.no_grad()."""
    B, omode, Nz, Ny, Nx, _ = object_patches.shape
    O = torch.polar(object_patches[..., 0], object_patches[..., 1])
    H1 = H[:, :, :, min(1, Nz - 1)]  # single-step propagator (ones when Nz == 1)
    psi = (probe[:, :, None] * torch.ones_like(O[:, None, :, 0])).to(O.dtype)
    planes = [fft2(psi) * H[:, :, :, 0].conj()]
    for j in range(Nz - 1):
        HS = H1 * fft2(psi * O[:, None, :, j])  # = F psi_{j+1}, propagated
        psi = ifft2(HS)
        planes.append(HS * H[:, :, :, j + 1].conj())
    return torch.stack(planes, dim=3).expand(B, -1, omode, -1, -1, -1)


def born_remainder_coeffs(
    D: torch.Tensor,
    T: torch.Tensor,
    d0_norm2: float,
    omode_occu: torch.Tensor | None = None,
    extra_rhs: torch.Tensor | None = None,
    lam_rel: float = 1e-12,
    breakdown_eps: float = 1e-20,
) -> tuple[torch.Tensor, float] | tuple[torch.Tensor, float, torch.Tensor]:
    """THE Born-coefficient solve: least squares of the exact multislice
    field onto the retained orders, solved for c DIRECTLY,

        min_c || sum_m c_m D_m - T ||^2,

    by the same solver born_residual uses — split-k complex128 Gram
    moments, column-equilibrated, with a tiny relative Tikhonov ridge. The
    whole solve stays on the device: the tall (rows, M) factor is never
    formed, never transferred, and never decomposed; only its (M, M)
    moments are. Measured 120-400x faster than the previous CPU 'gelsd'
    TSVD factorisation on the tall matrix (identical coefficients to
    machine precision wherever the basis is numerically full rank), which
    is why the fit moved here.

    Zero prior, like born_residual: under the ridge (and under any rank
    deficiency) the solution shrinks toward c = 0 — dropping an order it
    cannot resolve — not toward the plain series c = 1. The previous
    offset parametrisation x = c - 1, whose unresolved directions stayed
    at c = 1, is gone.

    Two conditioning constraints carried over from born_residual: the
    moment reductions MUST run in complex128 (a complex64 dot over ~1e7
    elements carries only ~1e-4), and the ridge must act on the
    unit-diagonal Gram — a trace-relative ridge on the raw Gram
    over-regularises, because the Born-order column norms are strongly
    disparate at high scattering strength (measured here: the orders decay
    geometrically, s_M/s_1 ~ 1e-8 at M = 8 and ~1e-13 at M = Nz = 21).
    Forming the Gram squares the condition number, so directions below
    ~sqrt(eps64) of the leading one are not resolvable from the moments;
    the ridge is what keeps them bounded.

    Data-free: T is the current object's own exact multislice field
    (born_multislice_target) — never measured data — so the coefficients
    are a deterministic function of the operator and cannot absorb object
    error.

    D : (M, B, pmode, omode, Ny, Nx) basis from born_detector_basis
        (concatenate view chunks along dim 1 before calling)
    T : (B, pmode, omode, Ny, Nx) exact scattered target
        (born_multislice_target - D0, same view concatenation)
    d0_norm2 : ||D0||^2 in the same (occupancy-weighted) norm — used only
        as the degenerate-input guard (a dead reference field returns
        c = 1 immediately)
    omode_occu : optional object-mode occupancies, applied as sqrt-weights
        on the field rows so the fit norm matches the forward model's
        mode-weighted intensity
    extra_rhs : optional additional target field, same shape as T, solved
        as a second right-hand side against the SAME Gram factorisation.
        Used for targets linear in an external knob: fitting against
        T + alpha * extra_rhs gives coefficients c + alpha * x_extra.

    Returns (c, delta): c complex64 of shape (M,), and
    delta = ||sum_m c_m D_m - T|| / ||T|| — the direct, calibration-free
    detector-error estimate, evaluated from the (M, M) moments rather than
    from a tall product, which keeps it free but makes it
    cancellation-limited to ~1e-11 relative once the fit residual is
    orders below ||T|| (it is a logged diagnostic, not a solve input).
    With extra_rhs, returns (c, delta, x_extra) where x_extra is the
    complex64 (M,) solution for extra_rhs. Eager; call under
    torch.no_grad().
    """
    M = D.shape[0]
    Ny, Nx = D.shape[-2], D.shape[-1]
    w = born_mode_weights(omode_occu, D.shape[3], D.real.dtype, D.device)
    w = w.view(1, 1, -1, 1, 1)
    D, T = D * w, T * w
    if extra_rhs is not None:
        extra_rhs = extra_rhs * w
    C = D.reshape(M, -1)  # (M, rows): one ROW per order
    t = T.reshape(-1)
    t_norm = torch.linalg.vector_norm(t.to(torch.complex128))
    if not (torch.isfinite(t_norm) and t_norm > 0 and d0_norm2 > 0):
        # degenerate input: fall back on the PLAIN SERIES, not on the
        # kernel's zero prior (born_residual makes the opposite choice)
        ones = torch.ones(M, dtype=torch.complex64, device=D.device)
        if extra_rhs is None:
            return ones, 0.0
        return ones, 0.0, torch.zeros(M, dtype=torch.complex64, device=D.device)
    R = t[None] if extra_rhs is None else torch.stack([t, extra_rhs.reshape(-1)])

    G, g = born_gram_moments(C, R, k_block=Ny * Nx)
    sol, rel_res = born_gram_solve(
        G, g, t_norm.square(), lam_rel=lam_rel, breakdown_eps=breakdown_eps
    )
    c, delta = sol[0], rel_res.item()
    if extra_rhs is None:
        return c.to(torch.complex64), delta
    return c.to(torch.complex64), delta, sol[1].to(torch.complex64)


def born_volume_remainder_rows(
    V: torch.Tensor,
    D: torch.Tensor,
    T_vol: torch.Tensor,
    T_det: torch.Tensor,
    D0: torch.Tensor,
    M: int,
    omode_occu: torch.Tensor | None = None,
    extra_det: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stacked rows of the VOLUME-remainder least-squares problem for one
    view chunk: columns Psi^(m) (m = 1..M) over every plane plus the
    detector entry, right-hand side the multislice field with order 0
    removed (T_vol - V[0] on the planes, T_det - D0 at the detector).
    Minimising ||A c - b|| for c DIRECTLY downstream
    (reconstruction._refit_born_coeffs_fourier) gives the remainder fit of
    born_remainder_coeffs extended from the detector rows to the full
    wavefield — the target is the stored multislice solution on every
    plane, not the fixed-point residual.

    Same normalisation conventions as born_volume_residual_rows: every row
    is a Fourier-view block, so plane and exit rows already carry the same
    fft2 factor and enter at unit weight; optional sqrt(omode_occu) row
    weights; plane-0 rows dropped (every order >= 1 and the tail are zero
    there — psi_hat_0 is F P for any object).

    V, D from born_volume_basis(..., n_max=M) — the remainder fit needs no
    order beyond those retained, unlike the residual rows. T_vol from
    born_multislice_volume_target, T_det from born_multislice_target, D0
    from the basis call.

    T_det carries whichever DETECTOR target is being fitted, so passing the
    hybrid field Upsilon * psi_MS in place of the multislice one switches
    the exit rows to the hybrid target while the plane rows stay data-free.
    extra_det, if given, is a second detector-only right-hand side (zero on
    every plane row, extra_det at the exit) returned as a third output —
    used to recover the pure-multislice coefficients from the same solve.

    Returns (A, b) or (A, b, b_extra): A (rows, M) and b (rows,) in V's
    dtype — cast to complex128 downstream. Eager; call under
    torch.no_grad()."""
    w = born_mode_weights(omode_occu, V.shape[3], V.real.dtype, V.device)
    wv, wd = w.view(1, 1, -1, 1, 1, 1), w.view(1, 1, -1, 1, 1)

    def _rows(vol_block, det_block):
        return torch.cat(
            [(vol_block * wv)[..., 1:, :, :].reshape(-1), (det_block * wd).reshape(-1)]
        )

    cols = [_rows(V[m].expand_as(T_vol), D[m - 1]) for m in range(1, M + 1)]
    b = _rows(T_vol - V[0].expand_as(T_vol), T_det - D0)
    A = torch.stack(cols, dim=1)
    if extra_det is None:
        return A, b
    return A, b, _rows(torch.zeros_like(T_vol), extra_det)

# ------------------------------------------------------------------------
# RESIDUAL FIT: target-free, per scan position
# ------------------------------------------------------------------------


def born_residual_order_gram(V_win, dets, n_ord, omode_occu):
    """The (n_ord, n_ord) Gram of the Born ORDERS, P_kl = <u_k, u_l> with
    u_k = (Psi_hat^(k) on the planes, D_{k-1} at the exit).

    This is the accumulable part of the residual fit: P adds over scan
    positions and over view chunks, so a fit shared across a calibration
    subsample is just the summed P (born_residual_solve_gram). Each order
    is weighted, cast and flattened EXACTLY ONCE into the rows of U and
    the Gram is one batched GEMM — doing the pairwise dots directly would
    rebuild every order once per partner, O(n^2) casts for O(n) data.

    V_win[k] holds order k on its own window, planes k..Nz-1 (it vanishes
    above that by nilpotency); dets[k-1] holds its exit block. Both must
    already be in the lab frame: the plane and exit gauges of _born_gauges
    differ, so they cannot be restored afterwards.

    Returns P of shape (B, n_ord, n_ord), complex128."""
    B = dets[0].shape[0]
    Ny, Nx = dets[0].shape[-2:]
    omode = dets[0].shape[2]
    w = born_mode_weights(omode_occu, omode, torch.float32, dets[0].device)
    wv, wd = w.view(1, 1, -1, 1, 1, 1), w.view(1, 1, -1, 1, 1)
    U = torch.stack([
        torch.cat([
            torch.nn.functional.pad(V_win[k] * wv, (0, 0, 0, 0, k, 0)).reshape(B, -1),
            (dets[k - 1] * wd).reshape(B, -1)], dim=1)
        for k in range(1, n_ord + 1)])                   # (n_ord, B, rows)
    P, _ = born_gram_moments(U, U[:1], k_block=Ny * Nx)
    return P


def born_residual_solve_gram(P, M, lam_rel=1e-12, breakdown_eps=1e-20):
    """Solve the residual fit from the order Gram of
    born_residual_order_gram (summed over whatever set of positions the
    fit is shared across).

    The residual columns are differences of consecutive orders,
    C_m = u_m - u_{m+1}, with u_k = 0 for k > n_ord and right-hand side
    r = u_1, so everything follows from P by
        G_ml = P_ml - P_{m,l+1} - P_{m+1,l} + P_{m+1,l+1},
        g_m  = P_m1 - P_{m+1,1},      ||r||^2 = P_11,
    assembled by SLICING a zero-padded P rather than looping over its
    M^2 entries. Returns (c, beta, rel_res)."""
    Pp = torch.nn.functional.pad(P, (0, 1, 0, 1))  # zero rows/cols for k > n_ord
    G = (Pp[..., :M, :M] - Pp[..., :M, 1:M + 1]
         - Pp[..., 1:M + 1, :M] + Pp[..., 1:M + 1, 1:M + 1])
    g = Pp[..., :M, 0] - Pp[..., 1:M + 1, 0]
    rhs2 = Pp[..., 0, 0].real.clamp(min=0)
    sol, rel_res = born_gram_solve(G, g[..., None, :], rhs2, lam_rel=lam_rel,
                                   breakdown_eps=breakdown_eps)
    return sol[..., 0, :], torch.sqrt(rhs2 + breakdown_eps), rel_res


def born_residual_from_orders(V_win, dets, M, n_ord, omode_occu,
                              lam_rel=1e-12, breakdown_eps=1e-20):
    """Per-position residual fit from orders someone has already built:
    born_residual_order_gram followed by born_residual_solve_gram."""
    P = born_residual_order_gram(V_win, dets, n_ord, omode_occu)
    return born_residual_solve_gram(P, M, lam_rel, breakdown_eps)


def _born_residual_orders(object_patches, probe, H, n_ord, centred=True):
    """The Born orders the residual fit needs, each on its OWN window.

    born_volume_basis zero-pads every order to the full Nz plane stack,
    which costs an allocation, a zero-fill and a copy per order — and
    then the Gram reduces over those zeros. Order k vanishes on planes
    < k (nilpotency), so here order k is returned on planes k..Nz-1 only.
    Order 0 is not built at all: the residual fit never references it
    (its columns start at u_1).

    Returns (V_win, dets): V_win[k] of shape
    (B, pmode, omode, Nz-k, Ny, Nx) for k = 1..n_ord, and dets[k] the
    exit block of order k+1, both in the lab frame (the per-plane and
    exit gauges of _born_gauges are applied here, and they differ)."""
    obj, _probe_k, Psi_hat, a_g = _born_init(
        object_patches, probe, H, centred=centred)
    gauge, gauge_planes = _born_gauges(a_g)
    Nz = object_patches.shape[2]
    V_win, dets = {}, []
    for k in range(n_ord):
        b_hat = _born_apply_V(obj, Psi_hat, H, k)
        if k == Nz - 1:  # no plane lies below the last slice
            dets.append(b_hat.sum(dim=3))
        else:
            Psi_hat, D_k = _born_apply_G(b_hat)
            dets.append(D_k.clone())
            # order k+1 lives on planes k+1..Nz-1, exactly Psi_hat's window
            V_win[k + 1] = (Psi_hat if gauge_planes is None
                            else gauge_planes[..., k + 1:, :, :] * Psi_hat)
    if gauge is not None:
        dets = [gauge * d for d in dets]
    if n_ord not in V_win:  # n_ord == Nz: that order has no plane block
        V_win[n_ord] = V_win[n_ord - 1].new_zeros(
            *V_win[n_ord - 1].shape[:3], 0, *V_win[n_ord - 1].shape[4:])
    return V_win, dets


@torch.no_grad()
def born_residual_coeffs(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    n_max: int,
    omode_occu: torch.Tensor | None = None,
    lam_rel: float = 1e-12,
    breakdown_eps: float = 1e-20,
    centred: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-position Born-residual coefficients (module docstring), entirely
    under torch.no_grad(), as a (B, n_max, 2) pseudo-complex float tensor
    for born.born_forward. Vectorised over the batch dim (one Gram per
    position), so a whole-scan coefficient table can be refreshed in view
    chunks (PtychoAD.refresh_born_residual_table).

    object_patches (B, omode, Nz, Ny, Nx, 2) pseudo-complex amp/phase,
    probe (B|1, pmode, Ny, Nx) complex64, H the propagator power stack of
    PtychoAD.get_H_3d, omode_occu (omode,) or None for uniform; mode rows
    carry sqrt(omode_occu) weights so the residual norm matches the
    mode-weighted intensity.

    Returns (coeffs, beta, rel_res): coeffs (B, n_max, 2) float32 (orders
    beyond min(n_max, Nz) padded with (0, 0) — those detector orders are
    identically zero by nilpotency, so the value is irrelevant); beta (B,)
    the initial residual norm ||Psi_hat^(1)|| (the first-Born field, ~0 on
    vacuum); rel_res (B,) the relative residual of the solve (1 on the
    vacuum fallback, where the solution stays at the free wave)."""
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    H = H.contiguous()
    B, omode, Nz, Ny, Nx, _ = object_patches.shape
    M = min(n_max, Nz)
    n_ord = min(M + 1, Nz)  # highest order computed (order M+1 unless M = Nz)

    V_win, dets = _born_residual_orders(object_patches, probe, H, n_ord,
                                        centred=centred)
    c, beta, rel_res = born_residual_from_orders(
        V_win, dets, M, n_ord, omode_occu, lam_rel=lam_rel,
        breakdown_eps=breakdown_eps)

    c64 = c.to(torch.complex64)
    coeffs = torch.zeros(B, n_max, 2, dtype=torch.float32, device=object_patches.device)
    coeffs[:, :M, 0], coeffs[:, :M, 1] = c64.real, c64.imag
    return coeffs, beta.to(torch.float32), rel_res.to(torch.float32)

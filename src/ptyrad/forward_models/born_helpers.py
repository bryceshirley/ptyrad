"""Born-series helpers: the shared recursion primitives, the per-order
detector fields, and the coefficient fits.

The recursion physics (_born_init / _born_scatter / _born_advance) is defined
once here and consumed by born.born_forward and the fit machinery. The
production coefficient fit is born_detector_basis + born_multislice_target +
born_qr_coeffs: least squares of the truncation tail onto the retained
orders, one rank-aware torch.linalg.lstsq solve (TSVD at the numerical
rank). It never sees measured data:
coefficients are a deterministic function of the current object operator, so
they cannot absorb object error (the data-fit gauge leak is closed by
construction). Removed predecessors (Krylov-Gram/GMRES fit, detector-Gram
normal equations, sequential-target Gram statistics) live in git history
(commit 18a2d35) if ever needed.

Alongside it live the target-free residual fits (born_exit_residual_coeffs
and the born_volume_basis / born_volume_residual_rows / born_tsqr_update /
born_residual_solve_from_r pipeline): they minimise the residual of the
multislice fixed-point equation (I - E) Psi = Psi^(0) at the trial field
sum_m c_m Psi^(m), which needs one Born order beyond those retained and NO
multislice solution. For a trial with c_0 = 1 the residual is exactly

    r(c) = Psi^(1) - sum_{m=1..M} c_m (Psi^(m) - Psi^(m+1)),

linear in c, so the fit is a direct least squares with columns
Psi^(m) - Psi^(m+1) and right-hand side Psi^(1); with every c_m = 1 it
telescopes to Psi^(M+1). These solve for c directly (zero prior), unlike
born_qr_coeffs which solves for offsets from c = 1 — under rank truncation
the former shrinks toward c = 0, the latter toward c = 1. Also data-free.
"""

import math

import torch
from torch.fft import fft2, ifft2

from ptyrad.forward_models.multislice import multislice_exit_wave


def unscattered_illumination(probe, H):
    """phi_j = ifft2(H^j FFT(P)) — the order-0 per-slice wave of the Born
    recursion, shape (B|1, pmode, 1, Nz, Ny, Nx). Depends only on the probe;
    any probe change invalidates it. Also consumed by ptyrad.mliss, whose
    hot call sites are torch.compile'd regions (this inlines there)."""
    Ny, Nx = probe.shape[-2:]
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    return ifft2(H * probe_k)


def _born_init(object_patches, probe, H):
    """Shared entry of the Born recursion: the scattering potential (O - 1),
    the k-space probe, and the unscattered per-slice wave v_0 = ifft2(H^j psi_0).
    Used by born_forward and the fit machinery so the recursion physics
    lives in one place."""
    Ny, Nx = object_patches.shape[-3], object_patches.shape[-2]
    O = torch.polar(object_patches[..., 0], object_patches[..., 1])
    obj = (O - 1.0).unsqueeze(1)
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    return obj, probe_k, ifft2(H * probe_k)


def _born_scatter(obj, Psi_state, H, n):
    """One bounce over sites n..Nz-1: transmit the incoming per-slice wave and
    refer it to the detector via the conjugate of the (unimodular) propagator
    powers. Scattering order n+1 needs n+1 distinct sites in increasing slice
    order, so its first allowed site is n — hence the window n: (nilpotency,
    applied order by order)."""
    return fft2(obj[..., n:, :, :] * Psi_state) * H[..., n:, :, :].conj()


def _born_advance(scat, H, n):
    """The propagation half of E, applied to this bounce's per-site scattered
    fields: E is strictly lower triangular in the slice index, and the prefix
    sums ARE that triangular structure — the next bounce at site k = n+1+i
    receives everything scattered at sites <= k-1, propagated forward to k.
    The entry including the last slice is dropped (nothing lies downstream of
    it). Returns (next_state, bounce_sum): the last prefix entry is the
    completed bounce's detector contribution, a free byproduct — no separate
    sum over sites is needed."""
    cs = torch.cumsum(scat, dim=3)
    return ifft2(cs[..., :-1, :, :] * H[..., n + 1 :, :, :]), cs[..., -1, :, :]


def born_detector_basis(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    n_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated per-order detector basis for the coefficient fit.

    Returns (D0, D): D0 the unscattered detector field of shape
    (B, pmode, omode, Ny, Nx), and D the order-1..min(n_max, Nz) detector
    fields stacked as (n, B, pmode, omode, Ny, Nx), such that born_forward's
    detector field is exactly D0 + sum_k c_k D[k-1] (unshifted k-space).
    Only the truncated recursion runs (~2*n*(Nz - n/2) slice-FFTs per view);
    the last basis order skips its advance (detector entry only, no next
    state). Dtype-preserving. Eager; call under torch.no_grad()."""
    B, omode, Nz, Ny, Nx, _ = object_patches.shape
    n = min(n_max, Nz)
    obj, probe_k, Psi_state = _born_init(object_patches, probe, H)
    D0 = probe_k.squeeze(3).expand(B, -1, omode, -1, -1)
    orders = []
    for k in range(n):
        scat = _born_scatter(obj, Psi_state, H, k)
        if k == n - 1:  # last basis order: detector entry only
            orders.append(scat.sum(dim=3))
        else:
            Psi_state, D_k = _born_advance(scat, H, k)
            # clone: D_k is a view into the advance's prefix sums
            orders.append(D_k.clone())
    return D0, torch.stack(orders)


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


def born_qr_coeffs(
    D: torch.Tensor,
    T: torch.Tensor,
    d0_norm2: float,
    omode_occu: torch.Tensor | None = None,
    extra_rhs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float] | tuple[torch.Tensor, float, torch.Tensor]:
    """THE Born-coefficient solve: truncated-SVD least squares of the
    truncation tail onto the retained orders.

    In x = c - 1 coordinates the fit is plain least squares of the tail
    T_tail = T - sum_m D_m (the part of the exact field the plain series
    misses) onto the basis columns:

        min_x || sum_m x_m D_m - T_tail ||^2,    c = 1 + x,

    solved by torch.linalg.lstsq with the rank-aware LAPACK driver 'gelsd'
    on the CPU (CUDA's only lstsq driver, 'gels', assumes full rank and
    breaks above ~2^23 rows). The ONLY regularisation is gelsd's TSVD
    cutoff at the numerical rank: a well-conditioned basis gets the exact
    minimum-norm projection and a degenerate one cannot blow up the
    coefficients. Data-free: T is the current object's own exact scattered
    field (born_multislice_target minus D0) — never measured data — so the
    coefficients are a deterministic function of the operator and cannot
    absorb object error. Limits: T_tail -> 0 (weak object, or M = Nz)
    => x -> 0, c -> 1.

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
        with ZERO prior as a second right-hand-side column of the SAME
        lstsq call. Used for targets linear in an external knob: fitting
        against T + alpha * extra_rhs gives coefficients c + alpha * x_extra.

    Returns (c, delta): c complex64 of shape (M,), and
    delta = ||sum_m c_m D_m - T|| / ||T|| — the direct, calibration-free
    detector-error estimate. With extra_rhs, returns (c, delta, x_extra)
    where x_extra is the complex64 (M,) zero-prior solution for extra_rhs.
    Eager; call under torch.no_grad().
    """
    M = D.shape[0]
    if omode_occu is not None:
        w = omode_occu.to(D.real.dtype).clamp(min=0).sqrt().view(1, 1, -1, 1, 1)
        D = D * w
        T = T * w
        if extra_rhs is not None:
            extra_rhs = extra_rhs * w
    A = D.reshape(M, -1).T.to(torch.complex128)  # tall factor: one column per order
    T_high_pres = T.reshape(-1).to(torch.complex128)
    t_norm = torch.linalg.vector_norm(T_high_pres)
    if not (torch.isfinite(t_norm) and t_norm > 0 and d0_norm2 > 0):
        ones = torch.ones(M, dtype=torch.complex64, device=D.device)
        if extra_rhs is None:
            return ones, 0.0
        return ones, 0.0, torch.zeros(M, dtype=torch.complex64, device=D.device)
    b = T_high_pres - A.sum(dim=1)  # T_tail: what the plain series misses
    rhs = b[:, None]
    if extra_rhs is not None:
        b_e = extra_rhs.reshape(-1).to(torch.complex128)
        rhs = torch.cat([rhs, b_e[:, None]], dim=1)
    sol = torch.linalg.lstsq(A.cpu(), rhs.cpu(), driver="gelsd").solution.to(A.device)
    x = sol[:, 0]
    c = 1.0 + x
    delta = (torch.linalg.vector_norm(A @ x - b) / t_norm).item()
    if extra_rhs is None:
        return c.to(torch.complex64), delta
    return c.to(torch.complex64), delta, sol[:, 1].to(torch.complex64)


def born_volume_basis(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    n_max: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-order PER-SLICE fields plus the detector entries, for the
    volume-residual fit and its checks.

    Returns (V, D0, D): V of shape (n+1, B, pmode, omode, Nz, Ny, Nx) with
    V[m, ..., j, :, :] the order-m real-space field at plane j (zero for
    j < m: order m needs m prior sites), D0 and D exactly as from
    born_detector_basis, n = min(n_max, Nz). Unlike born_detector_basis the
    last order's advance runs too, so V[n] is populated (identically zero
    when n == Nz — no plane lies at depth >= Nz, only the detector entry
    D[n-1] survives). Memory is (n+1) full per-slice stacks per view: call
    on small view chunks. Eager; call under torch.no_grad()."""
    B, omode, Nz, Ny, Nx, _ = object_patches.shape
    n = min(n_max, Nz)
    pmode = probe.shape[1]
    obj, probe_k, Psi_state = _born_init(object_patches, probe, H)
    D0 = probe_k.squeeze(3).expand(B, -1, omode, -1, -1)
    full = (B, pmode, omode, Nz, Ny, Nx)
    V = [Psi_state.expand(full).clone()]
    orders = []
    for k in range(n):
        scat = _born_scatter(obj, Psi_state, H, k)
        Vk = torch.zeros(full, dtype=D0.dtype, device=D0.device)
        if k == Nz - 1:  # order Nz: detector entry only, no plane holds it
            orders.append(scat.sum(dim=3))
        else:
            Psi_state, D_k = _born_advance(scat, H, k)
            # clone: D_k is a view into the advance's prefix sums
            orders.append(D_k.clone())
            Vk[..., k + 1 :, :, :] = Psi_state
        V.append(Vk)
    return torch.stack(V), D0, torch.stack(orders)


def born_exit_residual_coeffs(
    D: torch.Tensor,
    M: int,
    omode_occu: torch.Tensor | None = None,
    offset_from_one: bool = False,
) -> tuple[torch.Tensor, float, int, torch.Tensor]:
    """Exit-residual fit: least squares on the DETECTOR entries of the
    multislice fixed-point residual — columns D_m - D_{m+1} (m = 1..M),
    right-hand side D_1. Needs the detector basis run to order M+1, i.e.
    D from born_detector_basis(..., n_max=M+1); when M == Nz the basis
    tops out at order Nz and D_{Nz+1} = 0 is implied (nilpotency). No
    multislice target is ever formed, and no measured data enters.

    Solved in complex128 by the rank-aware 'gelsd' driver on the CPU, same
    as born_qr_coeffs, but for c DIRECTLY (zero prior): under rank
    truncation the solution shrinks toward c = 0, unlike born_qr_coeffs
    whose offset coordinates shrink toward c = 1. offset_from_one=True
    solves the same problem in x = c - 1 coordinates instead (rhs becomes
    D_1 - sum_m (D_m - D_{m+1}) = D_{M+1}) for the rank-deficient
    comparison.

    Returns (c, rel_res, rank, svals): c complex64 (M,),
    rel_res = ||A c - D_1|| / ||D_1|| in the (occupancy-weighted) row norm,
    the numerical rank reported by gelsd, and the singular values of the
    column matrix. Eager; call under torch.no_grad()."""
    n_det = D.shape[0]
    if n_det < M:
        raise ValueError(f"need detector orders 1..{M}, got {n_det}")
    Dm = D[:M]
    if n_det > M:
        Dnext = D[1 : M + 1]
    else:  # M == Nz: D_{Nz+1} = 0 by nilpotency
        Dnext = torch.cat([D[1:M], torch.zeros_like(D[:1])])
    A = Dm - Dnext
    b = D[0]
    if omode_occu is not None:
        w = omode_occu.to(A.real.dtype).clamp(min=0).sqrt().view(1, 1, -1, 1, 1)
        A = A * w
        b = b * w
    A = A.reshape(M, -1).T.to(torch.complex128)
    b = b.reshape(-1).to(torch.complex128)
    rhs = b - A.sum(dim=1) if offset_from_one else b
    out = torch.linalg.lstsq(A.cpu(), rhs.cpu()[:, None], driver="gelsd")
    x = out.solution[:, 0].to(A.device)
    c = 1.0 + x if offset_from_one else x
    b_norm = torch.linalg.vector_norm(b)
    rel_res = (torch.linalg.vector_norm(A @ c - b) / b_norm).item() if b_norm > 0 else 0.0
    return c.to(torch.complex64), rel_res, int(out.rank), out.singular_values


def born_volume_residual_rows(
    V: torch.Tensor,
    D: torch.Tensor,
    M: int,
    omode_occu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stacked rows of the VOLUME-residual least-squares problem for one
    view chunk: columns Psi^(m) - Psi^(m+1) (m = 1..M) over every plane
    plus the detector entry, right-hand side Psi^(1).

    Normalisation convention: unitary. The per-slice fields are real-space
    and kept as-is; the detector entries come from an unnormalised fft2,
    which inflates their norm by sqrt(Ny*Nx), so the detector block is
    divided by sqrt(Ny*Nx) before stacking. Object-mode rows are weighted
    by sqrt(omode_occu), matching born_qr_coeffs. Plane-0 rows are dropped
    (every order >= 1 and the rhs are zero there). Orders beyond those
    present in V / D are treated as zero (nilpotency at M = Nz).

    V, D from born_volume_basis(..., n_max=M+1) (n_max capped at Nz).
    Returns (A, b): A (rows, M) and b (rows,) in V's dtype — cast to
    complex128 downstream (born_tsqr_update does). Eager; call under
    torch.no_grad()."""
    n_vol = V.shape[0] - 1  # highest per-slice order present
    n_det = D.shape[0]
    Ny, Nx = V.shape[-2], V.shape[-1]
    det_scale = 1.0 / math.sqrt(Ny * Nx)
    if omode_occu is not None:
        w = omode_occu.to(V.real.dtype).clamp(min=0).sqrt()
        wv = w.view(1, 1, -1, 1, 1, 1)
        wd = w.view(1, 1, -1, 1, 1)
    else:
        wv = wd = None

    def _vol(m):
        return V[m] if m <= n_vol else torch.zeros_like(V[0])

    def _det(m):  # D[m-1] is order m; orders past the stack are zero
        return D[m - 1] if 1 <= m <= n_det else torch.zeros_like(D[0])

    def _rows(vol_block, det_block):
        if wv is not None:
            vol_block = vol_block * wv
            det_block = det_block * wd
        return torch.cat(
            [vol_block[..., 1:, :, :].reshape(-1), (det_block * det_scale).reshape(-1)]
        )

    cols = [_rows(_vol(m) - _vol(m + 1), _det(m) - _det(m + 1)) for m in range(1, M + 1)]
    b = _rows(_vol(1), _det(1))
    return torch.stack(cols, dim=1), b


def born_tsqr_update(R: torch.Tensor | None, block: torch.Tensor) -> torch.Tensor:
    """One accumulation step of the chunked (tall-skinny) QR: returns the
    R factor of [R_prev; block] in complex128. Feed row blocks of the
    least-squares matrix WITH the right-hand side appended as the last
    column; after the last block, R holds everything the solve needs
    (born_residual_solve_from_r). No Gram matrix is ever formed, and the
    accumulation is exact in the usual QR backward-stable sense."""
    block = block.to(torch.complex128)
    stacked = block if R is None else torch.cat([R, block])
    return torch.linalg.qr(stacked, mode="r").R


def born_residual_solve_from_r(
    R: torch.Tensor,
    M: int,
    n_rows: int,
    offset_from_one: bool = False,
) -> tuple[torch.Tensor, float, int, torch.Tensor, float]:
    """Solve the residual least squares from the accumulated R factor of
    [A_1 .. A_K, b] (born_tsqr_update), for the leading M <= K columns.

    Columns are scaled to unit norm (column norms are preserved by QR, so
    they are read off R itself — identical to scaling the full columns
    first, since diagonal scaling commutes with QR), then a truncated SVD
    at the gelsd-style numerical rank: s > s_max * eps64 * max(n_rows, M),
    with n_rows the TOTAL row count that went through the accumulation.
    Solves for c directly (zero prior; truncation shrinks toward c = 0) or,
    with offset_from_one=True, for x = c - 1 (truncation shrinks toward
    c = 1).

    Returns (c, rel_res, rank, svals, cond): c complex64 (M,),
    rel_res = ||A c - b|| / ||b|| (exact — both live in the accumulated Q
    basis), the truncation rank, the singular values of the unit-column
    matrix, and their condition number."""
    RA = R[:, :M]
    rb = R[:, -1]
    s_col = torch.linalg.vector_norm(RA, dim=0)
    s_safe = torch.where(s_col > 0, s_col, torch.ones_like(s_col))
    U, S, Vh = torch.linalg.svd(RA / s_safe, full_matrices=False)
    cutoff = S.max() * torch.finfo(torch.float64).eps * max(n_rows, M)
    rank = int((S > cutoff).sum())
    rhs = rb - RA.sum(dim=1) if offset_from_one else rb
    y = Vh[:rank].mH @ ((U[:, :rank].mH @ rhs) / S[:rank])
    x = y / s_safe
    c = 1.0 + x if offset_from_one else x
    b_norm = torch.linalg.vector_norm(rb)
    rel_res = (torch.linalg.vector_norm(RA @ c - rb) / b_norm).item() if b_norm > 0 else 0.0
    cond = (S.max() / S.min()).item() if S.min() > 0 else float("inf")
    return c.to(torch.complex64), rel_res, rank, S, cond

"""Born-series helpers: the shared recursion primitives, the per-order
detector fields, and the detector-space coefficient fit.

The recursion physics (_born_init / _born_scatter / _born_advance) is defined
once here and consumed by born.born_forward and the fit machinery. THE
coefficient fit is born_detector_basis + born_multislice_target +
born_qr_coeffs — there is no other coefficient-fitting code: least squares
of the truncation tail onto the retained orders, one rank-aware
torch.linalg.lstsq solve (TSVD at the numerical rank). It never sees
measured data:
coefficients are a deterministic function of the current object operator, so
they cannot absorb object error (the data-fit gauge leak is closed by
construction). Removed predecessors (Krylov-Gram/GMRES fit, detector-Gram
normal equations, sequential-target Gram statistics) live in git history
(commit 18a2d35) if ever needed.
"""

import torch
from torch.fft import fft2, ifft2


def _born_init(object_patches, probe, H):
    """Shared entry of the Born recursion: the scattering potential (O - 1),
    the k-space probe, and the unscattered per-slice wave v_0 = ifft2(H^j psi_0).
    Used by born_forward and the fit machinery so the recursion physics
    lives in one place."""
    Ny, Nx = object_patches.shape[-3], object_patches.shape[-2]
    obj = (torch.polar(object_patches[..., 0], object_patches[..., 1]) - 1.0).unsqueeze(1)
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
    psi = probe[:, :, None]  # (B|1, pmode, 1, Ny, Nx) -> broadcast omode
    for j in range(Nz - 1):
        psi = ifft2(H1 * fft2(psi * O[:, None, :, j]))
    F = fft2(psi * O[:, None, :, Nz - 1])
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

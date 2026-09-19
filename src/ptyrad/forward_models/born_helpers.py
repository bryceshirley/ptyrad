"""Born-series helpers: the shared recursion primitives, the per-order
detector fields, and the residual-minimizing (GMRES) coefficient machinery.

The recursion physics (_born_init / _born_scatter / _born_advance) is defined
once here and consumed by born.born_forward and born_krylov_gram.
The GMRES coefficient fit (born_krylov_gram +
born_gmres_coeffs) never sees measured data: coefficients are a deterministic
function of the current object operator, so they cannot absorb object error
(the data-fit gauge leak is closed by construction).
"""

import torch
from torch.fft import fft2, ifft2


def _born_init(object_patches, probe, H):
    """Shared entry of the Born recursion: the scattering potential (O - 1),
    the k-space probe, and the unscattered per-slice wave v_0 = ifft2(H^j psi_0).
    Used by born_forward and born_krylov_gram so the recursion physics
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


def born_fields_from_complex(
    O: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    n_max: int,
    coeffs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Detector-plane FIELD of the coefficient Born model from a complex object.

    Same recursion as born.born_forward, stopped before the |.|^2 intensity
    reduction: returns Psi(c) = D0 + sum_n c_n D_n of shape
    (B, pmode, omode, Ny, Nx), unshifted k-space — directly comparable to
    linesearch.iss_fields (equal to it at n_max=1, coeffs=None). Eager,
    dtype-preserving (complex128 in -> complex128 out), autograd-friendly:
    the exact line search differentiates through it for the direction
    gradient at order M > 1.

    O      : (B, omode, Nz, Ny, Nx) complex
    coeffs : optional (n, 2) pseudo-complex float or (n,) complex tensor,
             per-order detector weights (orders beyond n keep weight 1 is NOT
             assumed — n must cover n_max); None = plain series.
    """
    B, omode, Nz, Ny, Nx = O.shape
    obj = (O - 1.0).unsqueeze(1)  # (B, 1, omode, Nz, Ny, Nx)
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    Psi_state = ifft2(H * probe_k)
    F = probe_k.squeeze(3).expand(B, -1, omode, -1, -1)
    c = None
    if coeffs is not None:
        c = coeffs if torch.is_complex(coeffs) else torch.complex(coeffs[..., 0], coeffs[..., 1])
        c = c.to(F.dtype)
    n_orders = min(n_max, Nz)  # nilpotent termination
    for n in range(n_orders):
        scat = _born_scatter(obj, Psi_state, H, n)
        if n < n_orders - 1:
            Psi_state, D_n = _born_advance(scat, H, n)
        else:
            D_n = torch.sum(scat, dim=3)  # final order: no next bounce
        F = F + (D_n if c is None else c[n] * D_n)
    return F


def born_fields(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    n_max: int,
    coeffs: torch.Tensor | None = None,
) -> torch.Tensor:
    """born_fields_from_complex from PtyRAD (amp, phase) patches
    (B, omode, Nz, Ny, Nx, 2). Consistent with born.born_forward:
    fftshift2(sum_modes |born_fields|^2 * occu/(Ny*Nx)) + eps reproduces it."""
    O = torch.polar(object_patches[..., 0], object_patches[..., 1])
    return born_fields_from_complex(O, probe, H, n_max, coeffs)


def born_krylov_gram(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    n_max: int,
    omode_occu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-order detector fields plus the Gram matrix of the internal Krylov fields.

    Returns (D0, D, G): D0 the unscattered detector field of shape
    (B, pmode, omode, Ny, Nx); D the order-1..n detector fields stacked as
    (n, B, pmode, omode, Ny, Nx), such that born_forward's detector field is
    exactly D0 + sum_k c_k D[k-1]; and G of
    shape (n_max + 2, n_max + 2) complex128 with G[i, j] = <v_i, v_j> over the
    stacked internal wavefields v_k = E^k psi_0 for orders 0 .. n_max + 1
    (inner products weighted by omode_occu, summed over batch, probe modes,
    slices, and pixels; order k lives on slices k..Nz-1, so pairs are aligned
    on their overlap). Orders at or beyond the nilpotent cutoff (k >= Nz) are
    exactly zero and leave zero rows/columns.

    G is additive over views: accumulate over view chunks to bound memory —
    this function holds all retained order fields for the views it is given
    (roughly (n_max + 2) x the transient state of one Born pass).
    Feeds the residual-minimizing (GMRES) coefficient fit
    (born_gmres_coeffs), which never sees measured data. Eager; call under
    torch.no_grad().
    """
    # ---------------------------------------------------------------
    # 1. Setup: potential, 0th order fields, mode weights
    # ---------------------------------------------------------------
    B, omode, Nz, Ny, Nx, _ = object_patches.shape
    n_max = min(n_max, Nz)
    obj, probe_k, Psi_state = _born_init(object_patches, probe, H)
    D0 = probe_k.squeeze(3).expand(B, -1, omode, -1, -1).clone()
    if omode_occu is None:
        omode_occu = torch.ones(omode, device=object_patches.device) / omode
    w = omode_occu.to(torch.float64).view(1, 1, -1)

    # ---------------------------------------------------------------
    # 2. Born recursion: retain internal states and detector orders
    # ---------------------------------------------------------------
    states = [Psi_state]  # states[k] = v_k on slices k..Nz-1
    orders = []
    n_hi = min(n_max + 1, Nz)  # highest order with a nonzero internal field
    for n in range(n_hi):  # n = order - 1 = first allowed scattering site
        scat = _born_scatter(obj, Psi_state, H, n)
        if n < Nz - 1:
            Psi_state, D_n = _born_advance(scat, H, n)
            if n < n_max:
                # clone: D_n is a view into the advance's prefix sums
                orders.append(D_n.clone())
            states.append(Psi_state)
        elif n < n_max:  # order == Nz: no next state, sum the 1-slice window
            orders.append(scat.sum(dim=3))

    # ---------------------------------------------------------------
    # 3. Gram assembly: pairwise overlaps, float64 accumulation
    # ---------------------------------------------------------------
    m = n_max + 2
    G = torch.zeros(m, m, dtype=torch.complex128, device=object_patches.device)
    for i in range(len(states)):
        for j in range(i, len(states)):
            vi = states[i][..., j - i :, :, :] if j > i else states[i]
            prod = vi.conj() * states[j]
            # v_0 broadcasts over batch/omode when probe and H are shared;
            # restore the per-view multiplicity so G stays additive over chunks
            bfac = B / prod.shape[0]
            s = prod.sum(dim=(-2, -1)).to(torch.complex128).sum(dim=-1)
            g = (s * w).sum() * bfac
            G[i, j] = g
            if i != j:
                G[j, i] = g.conj()
    return D0, torch.stack(orders), G


def born_detector_gram(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    omode_occu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Full-depth detector Gram for the detector-space coefficient fit.

    One Born pass to the nilpotent cutoff (all Nz orders) yields every
    per-order detector field D_1..D_Nz, and D0 + sum_m D_m IS the exact
    multislice detector field of the current object — so the detector error
    of any truncated coefficient model is computable from their Gram alone,
    with no reconstruction oracle and no measured data.

    Returns (D0, D, Dg, d0_norm2): D0 the unscattered detector field of
    shape (B, pmode, omode, Ny, Nx); D the order-1..Nz detector fields
    stacked as (Nz, B, pmode, omode, Ny, Nx) (born_forward's detector field
    at order n with coefficients c is exactly D0 + sum_{m<=n} c_m D[m-1]);
    Dg of shape (Nz, Nz) complex128 with Dg[i, j] = <D_{i+1}, D_{j+1}>
    (summed over views, probe modes, and pixels; object modes
    occupancy-weighted); and d0_norm2 = ||D0||^2 under the same weighting
    (the ridge scale). Dg and d0_norm2 are additive over view chunks.
    Unlike born_krylov_gram this retains only one detector field per order
    (no internal states), so memory is ~Nz detector frames per view.
    Eager; call under torch.no_grad().
    """
    B, omode, Nz, Ny, Nx, _ = object_patches.shape
    obj, probe_k, Psi_state = _born_init(object_patches, probe, H)
    D0 = probe_k.squeeze(3).expand(B, -1, omode, -1, -1).clone()
    if omode_occu is None:
        omode_occu = torch.ones(omode, device=object_patches.device) / omode
    w = omode_occu.to(torch.float64).view(1, 1, -1)

    orders = []
    for n in range(Nz):
        scat = _born_scatter(obj, Psi_state, H, n)
        if n < Nz - 1:
            Psi_state, D_n = _born_advance(scat, H, n)
            # clone: D_n is a view into the advance's prefix sums
            orders.append(D_n.clone())
        else:
            orders.append(scat.sum(dim=3))

    Dg = torch.zeros(Nz, Nz, dtype=torch.complex128, device=object_patches.device)
    for i in range(Nz):
        for j in range(i, Nz):
            s = (orders[i].conj() * orders[j]).sum(dim=(-2, -1)).to(torch.complex128)
            g = (s * w).sum()
            Dg[i, j] = g
            if i != j:
                Dg[j, i] = g.conj()
    d0_norm2 = ((D0.abs().square().sum(dim=(-2, -1)).to(torch.float64)) * w).sum().item()
    return D0, torch.stack(orders), Dg, d0_norm2


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


def born_seq_detector_stats(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    n_max: int,
    omode_occu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Sequential-target statistics for the detector-space coefficient fit.

    Same objective as born_detector_gram + born_detector_coeffs, but the
    exact target T = Psi_MS - D0 comes from one sequential multislice sweep
    (born_multislice_target) instead of summing all Nz Born orders, and only
    the truncated basis D_1..D_n is built — so the cost is
    ~2*n*(Nz - n/2) + 2*Nz slice-FFTs per view (linear in depth) and the
    memory is n detector frames plus one rolling state, vs the full-depth
    pass's ~Nz^2 FFTs and Nz frames. The trade: only orders m <= n_max can
    be priced (carry one extra basis order as lookahead for adaptive
    growth); rhs_i = <D_i, T> is independent of the truncation, so leading
    blocks of (A, rhs) price every m <= n_max.

    Returns (D0, D, A, rhs, t, d0_norm2): D0 and the order-1..n_max detector
    fields D as in born_detector_gram; A of shape (n, n) complex128 with
    A[i, j] = <D_{i+1}, D_{j+1}>; rhs of shape (n,) complex128 with
    rhs[i] = <D_{i+1}, T>; t = ||T||^2; d0_norm2 = ||D0||^2 (all inner
    products view-summed, mode-occupancy weighted). A, rhs, t, d0_norm2 are
    additive over view chunks. Eager; call under torch.no_grad().
    """
    B, omode, Nz, Ny, Nx, _ = object_patches.shape
    n = min(n_max, Nz)
    obj, probe_k, Psi_state = _born_init(object_patches, probe, H)
    D0 = probe_k.squeeze(3).expand(B, -1, omode, -1, -1).clone()
    if omode_occu is None:
        omode_occu = torch.ones(omode, device=object_patches.device) / omode
    w = omode_occu.to(torch.float64).view(1, 1, -1)

    orders = []
    for k in range(n):
        scat = _born_scatter(obj, Psi_state, H, k)
        if k == n - 1:  # last basis order: detector entry only, no next state
            orders.append(scat.sum(dim=3))
        else:
            Psi_state, D_k = _born_advance(scat, H, k)
            # clone: D_k is a view into the advance's prefix sums
            orders.append(D_k.clone())
    T = born_multislice_target(object_patches, probe, H) - D0

    def _dot(x, y):
        return ((x.conj() * y).sum(dim=(-2, -1)).to(torch.complex128) * w).sum()

    A = torch.zeros(n, n, dtype=torch.complex128, device=object_patches.device)
    rhs = torch.zeros(n, dtype=torch.complex128, device=object_patches.device)
    for i in range(n):
        rhs[i] = _dot(orders[i], T)
        for j in range(i, n):
            g = _dot(orders[i], orders[j])
            A[i, j] = g
            if i != j:
                A[j, i] = g.conj()
    t = _dot(T, T).real.clamp(min=0.0).item()
    d0_norm2 = ((D0.abs().square().sum(dim=(-2, -1)).to(torch.float64)) * w).sum().item()
    return D0, torch.stack(orders), A, rhs, t, d0_norm2


def _detector_solve(
    A: torch.Tensor, rhs: torch.Tensor, t: float, lam: float
) -> tuple[torch.Tensor, float]:
    """Shared core of the detector-space least squares: solve
    (A + lam I) c = rhs + lam 1 and return (c, rel_res) with
    rel_res = sqrt(max(c^H A c - 2 Re(c^H rhs) + t, 0) / t)."""
    M = A.shape[0]
    eye = torch.eye(M, dtype=A.dtype, device=A.device)
    c = torch.linalg.solve(A + lam * eye, rhs + lam * torch.ones(M, dtype=A.dtype, device=A.device))
    res2 = ((c.conj() @ (A @ c)).real - 2.0 * (c.conj() @ rhs).real + t).clamp(min=0.0)
    rel_res = torch.sqrt(res2 / t).item() if t > 0 else 0.0
    return c, rel_res


def born_seq_detector_coeffs(
    A: torch.Tensor,
    rhs: torch.Tensor,
    t: float,
    n_max: int,
    ridge: float = 1e-3,
    d0_norm2: float | None = None,
) -> tuple[torch.Tensor, float]:
    """Detector-space Born coefficients from sequential-target statistics
    (born_seq_detector_stats). Identical objective, ridge scaling, and
    limits as born_detector_coeffs — see there; the basis may carry more
    orders than n_max (lookahead), priced via leading blocks.

    Returns (c, rel_res): c complex64 of shape (n_max,) for orders
    1..n_max (orders beyond the basis get coefficient 1), and
    rel_res = ||residual field|| / ||T||.
    """
    K = A.shape[0]
    M = min(n_max, K)
    diag = A.diagonal().real
    scale = float(d0_norm2) if d0_norm2 is not None else diag.mean().item()
    if not torch.isfinite(diag.sum()) or scale <= 0 or diag.max() <= 0:
        return torch.ones(n_max, dtype=torch.complex64, device=A.device), 0.0
    lam = max(float(ridge), 1e-12) * scale
    c, rel_res = _detector_solve(A[:M, :M], rhs[:M], float(t), lam)
    if n_max > M:
        c = torch.cat([c, torch.ones(n_max - M, dtype=c.dtype, device=c.device)])
    return c.to(torch.complex64), rel_res


def born_detector_coeffs(
    Dg: torch.Tensor,
    n_max: int,
    ridge: float = 1e-3,
    d0_norm2: float | None = None,
) -> tuple[torch.Tensor, float]:
    """Detector-space Born coefficients from the full-depth detector Gram.

    Minimizes the exact detector field error of the order-M model against
    the current object's own multislice field (T = sum_{m=1}^{Nz} D_m, the
    nilpotent-terminated series):

        min_c || sum_{m<=M} c_m D_m - T ||^2 + lam ||c - 1||^2,
        lam = ridge * d0_norm2,

    a convex least-squares over the detector Gram — closed form,
    deterministic, and data-free (the target is the current operator's exact
    solve, not a reconstruction oracle and not measured intensities, so the
    gauge-leak protection of the GMRES fit is preserved). The ridge pulls
    toward the plain series c = 1, scaled by the unscattered detector norm
    (the same role G[0, 0] plays in born_gmres_coeffs); at a weak object the
    scattered Gram is tiny vs d0_norm2, so c -> 1, and at M = Nz the target
    is in the span and c = 1 is the exact minimizer.

    Returns (c, rel_res): c complex64 of shape (n_max,) for orders
    1..n_max (orders beyond Nz get coefficient 1; their fields are zero),
    and rel_res = ||residual field|| / ||T||.
    """
    Nz = Dg.shape[0]
    M = min(n_max, Nz)
    diag = Dg.diagonal().real
    scale = float(d0_norm2) if d0_norm2 is not None else diag.mean().item()
    if not torch.isfinite(diag.sum()) or scale <= 0 or diag.max() <= 0:
        return torch.ones(n_max, dtype=torch.complex64, device=Dg.device), 0.0
    ones_full = torch.ones(Nz, dtype=Dg.dtype, device=Dg.device)
    rhs = Dg[:M] @ ones_full  # <D_m, T>
    t = (ones_full @ (Dg @ ones_full)).real.clamp(min=0.0).item()  # ||T||^2
    # lam floor keeps the solve invertible when high orders are numerically zero
    lam = max(float(ridge), 1e-12) * scale
    c, rel_res = _detector_solve(Dg[:M, :M], rhs, t, lam)
    if n_max > M:
        c = torch.cat([c, torch.ones(n_max - M, dtype=Dg.dtype, device=Dg.device)])
    return c.to(torch.complex64), rel_res


def born_gmres_coeffs(
    G: torch.Tensor,
    n_max: int,
    pin_first: bool = True,
    ridge: float = 1e-3,
) -> tuple[torch.Tensor, float]:
    """Residual-minimizing (GMRES) Born coefficients from the Krylov Gram matrix.

    With psi(c) = sum_{k=0}^{n} c_k E^k psi_0 and c_0 = 1 fixed, the
    linear-system residual psi_0 - (I - E) psi(c) telescopes to
    sum_{k=1}^{n+1} b_k v_k with b_k = c_{k-1} - c_k (c_{n+1} := 0), so
    ||r||^2 = b^H Gs b with Gs = G[1:, 1:], subject to sum(b) = 1. Minimized
    in closed form via KKT, Tikhonov-regularized toward the plain series
    (b = e_top, i.e. c = 1) with lam = ridge * G[0, 0] (the psi_0 norm — the
    same scale the relative residual is measured against, so ridge r means
    "pay up to ~sqrt(r) relative equation residual to stay near c = 1"; at a
    weak object the Krylov norms are tiny vs G[0, 0], so the pull to the plain
    series wins, while at strong scattering the exploding mid-order Krylov
    norms dwarf G[0, 0] and the ridge is negligible. Scaling by mean(diag Gs)
    instead lets those exploding norms inflate lam by orders of magnitude and
    over-damps the fit exactly at the divergence hump — measured 4.8e0 vs
    5.9e-2 detector error at ridge 1e-3 on the converged PSO object):
    min b^H (Gs + lam I) b - 2 lam Re(e_top^H b). The measured data never
    enters — the coefficients depend only on the current object operator, so
    they cannot absorb object error (the data-fit gauge leak is closed by
    construction). pin_first fixes c_1 = 1 by dropping b_1; with GMRES this
    is optional, not a leak guard. Weak object => c -> 1; n_max = Nz =>
    c -> 1 (exact termination).

    Returns (c, rel_res): c complex64 of shape (n_max,) for orders 1..n_max,
    and the relative equation residual sqrt(b^H Gs b / G[0, 0]).
    """
    Gs = G[1:, 1:]
    n_b = n_max + 1
    diag = Gs.diagonal().real
    scale = G[0, 0].real
    if not torch.isfinite(diag.sum()) or not torch.isfinite(scale) or scale <= 0 or diag.max() <= 0:
        return torch.ones(n_max, dtype=torch.complex64, device=G.device), 0.0
    keep = list(range(1, n_b)) if pin_first else list(range(n_b))
    A = Gs[keep][:, keep]
    k = len(keep)
    # lam floor keeps A invertible when trailing Krylov vectors are exactly
    # zero (nilpotent cutoff) or numerically dependent (near-converged series)
    lam = max(float(ridge), 1e-12) * scale.item()
    A = A + lam * torch.eye(k, dtype=A.dtype, device=A.device)
    ones = torch.ones(k, dtype=A.dtype, device=A.device)
    e_top = torch.zeros(k, dtype=A.dtype, device=A.device)
    e_top[-1] = 1.0
    X = torch.linalg.solve(A, torch.stack([ones, lam * e_top], dim=1))
    x1, x2 = X[:, 0], X[:, 1]
    nu = (1.0 - x2.sum()) / x1.sum()
    b = torch.zeros(n_b, dtype=G.dtype, device=G.device)
    b[keep] = x2 + nu * x1
    c = 1.0 - torch.cumsum(b, 0)[:n_max]
    res2 = (b.conj() @ (Gs @ b)).real.clamp(min=0.0)
    rel_res = torch.sqrt(res2 / G[0, 0].real).item()
    return c.to(torch.complex64), rel_res

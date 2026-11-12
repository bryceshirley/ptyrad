"""Parallel inter-slice Born-series forward model for multislice
ptychography, in the FOURIER VIEW (parallel_in_depth_rewrite-3.pdf,
Secs. 3.2-3.4).

The all-at-once multislice system, premultiplied by the free-space Green's
function and changed to the Fourier view (hat quantities referred to the
entrance plane in k-space), is the Lippmann-Schwinger form

    (I - G_hat V_hat) Psi_hat = Psi_hat^(0),   Psi_hat^(0) = 1 (F P),

and the inter-slice Born series is its Neumann series,
Psi_hat^(m) = (G_hat V_hat)^m Psi_hat^(0) (eq. 3.13). Because V_hat is
block diagonal, one order is the slice-wise V_hat_j — fully parallel across
depth, one inverse and one forward transform each — followed by G_hat,
which is a single cumulative sum over planes and the only operation that
couples the slices. The sequential propagation loop of multislice is
therefore replaced by a parallel prefix sum: depth M instead of Nz.
G_hat V_hat is strictly lower triangular, so the series is nilpotent and
terminates exactly at the number of slices; higher `n_max` adds nothing.

The exit block of Psi_hat IS the far-field detector wave, up to the
unimodular phase the detector cannot record (eq. 3.15), so the model never
propagates to the detector as a separate step: every order's contribution
falls out of the same scan that advances the field.

This module owns the recursion primitives of the factorisation
(_born_incident / _born_scatter / _born_apply_V / _born_apply_G, plus the
shared entry _born_init and the position-view bridge _born_advance) and
the per-order basis builders (born_detector_basis / born_volume_basis)
that every Born path consumes — the forward below, born_dist, the
per-position residual solver born_residual and the remainder fits of
born_remainder. Normalisation: torch's fft2 is unnormalised, so every hat
block carries the same factor sqrt(Ny*Nx) relative to the unitary
convention — plane and exit blocks are on the same footing in any
hat-view norm.

The coefficients passed into the forward are the c_m of eq. (3.18) and
only reweight the per-order sum at the exit block; they come from the
remainder fits (born_remainder, shared over a calibration subsample) or
the per-position residual solve (born_residual).
"""

import torch
from torch.fft import fft2, ifft2

from ptyrad.utils import fftshift2


def _born_incident(Psi_hat, H, n=0):
    """U_j^* = F^{-1} H^{z_j} on planes n..Nz-1: the first half of V_hat_j,
    taking the Fourier-view blocks back to psi_j, the wave INCIDENT on slice
    j. Psi_hat may have a singleton plane axis, in which case it broadcasts —
    that is exactly the source Psi_hat^(0) = 1 (F P), one block repeated."""
    return ifft2(H[..., n:, :, :] * Psi_hat)


def _born_scatter(obj, psi, H, n=0):
    """dO_j followed by U_j = H^{-z_j} F on planes n..Nz-1: the second half of
    V_hat_j. Takes the position-view incident wave psi_j and returns the
    Fourier-view scattered block, already referred to the entrance plane —
    which for the exit block means already at the detector (eq. 3.15), so
    nothing downstream ever propagates again."""
    return fft2(obj[..., n:, :, :] * psi) * H[..., n:, :, :].conj()


def _born_apply_V(obj, Psi_hat, H, n=0):
    """V_hat on planes n..Nz-1: V_hat_j = H^{-z_j} F dO_j F^{-1} H^{z_j}
    (eq. 3.11), block diagonal, so every plane in the window is independent —
    this is the whole parallel part of one Born order, and its only cost is
    the one inverse and one forward transform per plane."""
    return _born_scatter(obj, _born_incident(Psi_hat, H, n), H, n)


def _born_apply_G(b_hat):
    """G_hat on planes n..Nz-1: the Green's function of free space in the
    Fourier view is a copy of the exit block plus a cumulative sum over
    planes (eq. 3.8), and the source block of the Born recursion is zero at
    the exit, so G_hat here is the EXCLUSIVE prefix sum — plane k receives
    everything scattered strictly above it.

    Returns (Psi_hat_next, det): the window's exclusive prefix sums, which
    are the next order on planes n+1..Nz-1 (the order vanishes on plane n and
    above, by nilpotency), and the inclusive total, which is the exit block
    1^T b_hat of this order — a free byproduct of the same scan, since the
    exit block of a cumulative sum is the sum of all the blocks. No
    transform: G_hat is the ONLY operation that couples the depth axis, and
    it is a scan."""
    cs = torch.cumsum(b_hat, dim=3)
    return cs[..., :-1, :, :], cs[..., -1, :, :]


def unscattered_illumination(probe, H):
    """phi_j = ifft2(H^j FFT(P)) — the wave incident on slice j with the
    object absent, shape (B|1, pmode, 1, Nz, Ny, Nx). It is the position view
    of the Born source, U^* Psi_hat^(0): the same block F P on every plane,
    carried back through U_j^*. Depends only on the probe; any probe change
    invalidates it. Also consumed by ptyrad.mliss, whose hot call sites are
    torch.compile'd regions (this inlines there)."""
    Ny, Nx = probe.shape[-2:]
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    return _born_incident(probe_k, H)


def _born_init(object_patches, probe, H, centred=True):
    """Shared entry of the Born recursion: the scattering potential, the
    k-space probe F P, the Fourier-view source Psi_hat^(0) = 1 (F P) of
    eq. 3.12, and the background gauge.

    CENTRED BY DEFAULT (eq. 3.3). The background is shifted from 1 to the
    per-slice, per-window mean a_j = mean(O_j), so the perturbation the
    series expands in is

        eps_j = O_j / a_j - 1,    with  A = prod_j a_j

    restored on the detector field. The factorisation O_j = a_j(1 + eps_j)
    is an identity, so the model is unchanged at M = Nz; what changes is
    the truncation. Centring removes the DC part of the perturbation, and
    on real PSO patches it shrinks the overshoot peak ~3.5x (order norms
    29 vs 100 x ||T|| at m = 8), improves the conditioning of the basis
    5-8x, and lowers the detector-field error at EVERY order (measured
    0.91-0.98 of the uncentred error for M <= 12, 0.2-0.35 beyond).

    a_j is a function of the object, so autograd flows through the gauge:
    this is the exact gradient of the model actually evaluated, and it
    costs only a broadcast per slice. The probe-intensity weighting the
    paper also suggests for a_j is NOT used — measured indistinguishable
    on a homogeneous specimen (gauges differ 0.4%, field errors in the
    4th digit) and it couples the probe into the object gradient.

    centred=False restores the plain chord series dO_j = O_j - 1 and
    returns gauge=None.

    The source is returned with a SINGLETON plane axis, shape
    (B|1, pmode, 1, 1, Ny, Nx): in the Fourier view order zero is literally
    the same block on every plane, so it is a broadcast rather than a stack,
    and it is simultaneously the order-0 detector wave (squeeze the plane
    axis). Used by born_forward, born_forward_dist, born_residual and
    the fit machinery so the recursion physics lives in one place.

    Returns (obj, probe_k, Psi_hat, a_g): a_g holds the per-slice means
    with a plane axis, (B, 1, omode, Nz, 1, 1), or None when
    centred=False. Pass it to _born_gauges to get the exit and per-plane
    gauges — they are NOT the same factor (see there)."""
    Ny, Nx = object_patches.shape[-3], object_patches.shape[-2]
    O = torch.polar(object_patches[..., 0], object_patches[..., 1])
    if centred:
        a = O.mean(dim=(-2, -1), keepdim=True)  # (B, omode, Nz, 1, 1)
        obj = (O / a - 1.0).unsqueeze(1)
        a_g = a.unsqueeze(1)  # (B, 1, omode, Nz, 1, 1)
    else:
        obj = (O - 1.0).unsqueeze(1)
        a_g = None
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    return obj, probe_k, probe_k, a_g


def _born_gauges(a_g):
    """The two gauges of the centred series, from the per-slice means a_g
    of _born_init.

    They differ, and conflating them is a silent error. The wave INCIDENT
    on plane j has only passed slices k < j, so it carries the running
    product A_j = prod_{k<j} a_k — an EXCLUSIVE cumulative product, equal
    to 1 on plane 0. Only the exit field has passed every slice, so it
    carries the full A = prod_k a_k. Applying the exit gauge to the plane
    blocks overstates their attenuation by prod_{k>=j} a_k and breaks the
    identity sum_m Psi^(m) = Psi_MS that the volume fits are built on.

    Returns (A_exit, A_planes) shaped (B, 1, omode, 1, 1) and
    (B, 1, omode, Nz, 1, 1), or (None, None)."""
    if a_g is None:
        return None, None
    A_exit = a_g.prod(dim=3)  # (B, 1, omode, 1, 1)
    inclusive = torch.cumprod(a_g, dim=3)
    A_planes = torch.cat(
        [torch.ones_like(a_g[..., :1, :, :]), inclusive[..., :-1, :, :]], dim=3)
    return A_exit, A_planes


def _born_advance(scat, H, n):
    """Position-view bridge: G_hat on the window followed by U^* back to the
    waves incident on planes n+1..Nz-1, i.e. _born_apply_G composed with
    _born_incident. Kept for the error-analysis scripts, which walk the
    orders in the position view. The production recursion stays in the
    Fourier view and never converts back — _born_apply_V absorbs the U^*."""
    Psi_hat, det = _born_apply_G(scat)
    return _born_incident(Psi_hat, H, n + 1), det


def born_forward_eager(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    omode_occu: torch.Tensor,
    eps: float = 1e-10,
    n_max: int = 1,
    coeffs: torch.Tensor | None = None,
    centred: bool = True,
) -> torch.Tensor:
    """
    Parallel Born Series Forward Model for Multislice Ptychography.

    This function implements a parallelized formulation of the multislice algorithm.
    By expanding the multislice operator into a Born series in the Fourier view and
    factoring out the vacuum transmission, the traditional O(N_z) sequential
    propagation loop is replaced with parallel prefix sums (`cumsum`): the state
    carried from order to order is the Fourier-view wavefield Psi_hat on all N_z + 1
    planes, each block referred to the entrance plane, so the per-plane work
    V_hat_j = H^{-z_j} F dO_j F^{-1} H^{z_j} is independent across depth and the only
    cross-plane operation is G_hat, the cumulative sum.

    Scattering Regimes (`n_max`):
    -----------------------------
    * `n_max = 1` (First-Order Born Approximation):
        Models a single scattering event. The wave scatters at each slice and
        reaches the exit block without further object interactions. This
        completely bypasses the 3D prefix sum, utilizing a single, parallel
        reduction directly to the 2D exit plane.

    * `1 < n_max < N_z` (Truncated Born Approximation):
        Captures higher-order multiple scattering events up to `n_max` interactions.
        Provides a highly accurate approximate solution that models the dominant
        dynamical scattering effects.

    * `n_max == N_z` (Full Multislice Solution):
        Because the spatial scattering operator is nilpotent across the slices,
        the series naturally terminates at `N_z`. At this limit, the model is
        mathematically exact and produces identically equivalent results to the
        standard sequential multislice formulation.

    Args:
        object_patches (torch.Tensor): Tensor of shape (N, omode, Nz, Ny, Nx, 2), representing
            pseudo-complex object patches with float32 amplitude and phase components.
        probe (torch.Tensor): Tensor of shape (N, pmode, Ny, Nx) or (1, pmode, Ny, Nx) with complex64 values,
            representing the probe(s). N is the number of samples in the batch, pmode is the
            number of probe modes. By default, N is 1, assuming the same probe for all samples.
        H (torch.Tensor): Tensor of shape (N|1, 1, 1, Nz, Ny, Nx) with complex64 values —
            the stack of Fresnel propagator powers H^j = H^{z_j} (entrance to slice j),
            as built by PtychoAD.get_H_3d. Together with F it is the change of view
            U_j = H^{-z_j} F of eq. (3.6); the referral back to the entrance plane uses
            its conjugate (the vacuum propagator is unimodular, Lemma 3.1).
        omode_occu (torch.Tensor): Tensor of shape (omode,) with float32 values.
        eps (float, optional): A small value added for numerical stability. Defaults to 1e-10.
        n_max (int): Maximum order of the Born series iterations (orders of scattering).
        coeffs (torch.Tensor, optional): Tensor of shape (n_max, 2) with float32 values,
            a pseudo-complex (real, imag) coefficient per scattering order — the c_m of
            the tuned series (3.18). The detector
            field becomes Psi_0 + sum_n c_n D_n, so coeffs full of (1, 0) reproduces the
            plain series exactly. The coefficients reweight only the exit-block sum — the
            internal recursion stays the plain series, so each D_n remains the true
            n-th order field. Register as an nn.Parameter to tune the coefficients by
            autograd during data fitting alongside the object and probe. For a coefficient
            update that stays identifiable against the object scale, freeze c_1 at (1, 0)
            and optimize only the top order(s). Defaults to None (plain series, and no
            per-order detector fields are retained for a coefficient gradient).

    Returns:
        torch.Tensor: Tensor of shape (N, Ny, Nx) with float32 positive values, representing the
        forward diffraction pattern for each sample in the batch.
    """
    # ---------------------------------------------------------------
    # 1. Setup: contiguity and mode weights
    # ---------------------------------------------------------------
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    kernel_fwd = H.contiguous()

    _, omode, Nz, Ny, Nx, _ = object_patches.shape

    if omode_occu is None:
        omode_occu = (
            torch.ones(omode, dtype=object_patches.dtype, device=object_patches.device) / omode
        )
    norm_weight = omode_occu / (Nx * Ny)

    # ---------------------------------------------------------------
    # 2. Scattering potential, Fourier-view source and its exit block
    # ---------------------------------------------------------------
    # Psi_hat^(0) = 1 (F P): the same block on every plane, so the source is
    # a broadcast and its exit block is F P, the 0th-order detector wave.
    obj, probe_k, Psi_hat, a_g = _born_init(
        object_patches, probe, kernel_fwd, centred=centred)
    gauge, gauge_planes = _born_gauges(a_g)  # exit vs per-plane: see _born_gauges
    Psi_M_hat = probe_k.squeeze(3)

    # ---------------------------------------------------------------
    # 3. Born series loop: V_hat (parallel) -> G_hat (scan), tune the orders
    # ---------------------------------------------------------------
    n_orders = min(n_max, Nz)  # nilpotent termination
    if gauge is not None:
        Psi_M_hat = gauge * Psi_M_hat  # background A = prod_j a_j
    for n in range(n_orders):  # n = order - 1 = first allowed scattering site
        b_hat = _born_apply_V(obj, Psi_hat, kernel_fwd, n)
        if n < n_orders - 1:
            # the scan's running total IS the exit block of this order
            Psi_hat, D_n = _born_apply_G(b_hat)
        else:
            # final order: only the exit block is still needed, 1^T V_hat
            D_n = torch.sum(b_hat, dim=3)
        if gauge is not None:
            D_n = gauge * D_n
        # accumulate immediately: holding the orders in a list keeps them
        # and their producing intermediates alive for the whole recursion,
        # which cost ~1.5x peak memory (measured 4.89 vs 3.20 GiB at
        # batch 8, M = 8, 256^2) for no gain now that nothing downstream
        # needs the per-order fields
        if coeffs is None:
            Psi_M_hat = Psi_M_hat + D_n
        else:
            Psi_M_hat = Psi_M_hat + torch.complex(coeffs[n, 0], coeffs[n, 1]) * D_n

    # ---------------------------------------------------------------
    # 4. Detector measurement (incoherent sum over modes)
    # ---------------------------------------------------------------
    return fftshift2(
        torch.sum(Psi_M_hat.abs().square() * norm_weight.view(1, 1, -1, 1, 1), dim=(1, 2)) + eps
    )


def born_detector_basis(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    n_max: int,
    centred: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated per-order detector basis for the coefficient fit.

    Returns (D0, D): D0 the unscattered detector field of shape
    (B, pmode, omode, Ny, Nx), and D the order-1..min(n_max, Nz) detector
    fields stacked as (n, B, pmode, omode, Ny, Nx), such that born_forward's
    detector field is exactly D0 + sum_k c_k D[k-1] (unshifted k-space).
    The detector entries are exit blocks of the Fourier view, e_N^T
    Psi_hat^(m) (eq. 3.15), so D0 = F P needs no referral.
    Only the truncated recursion runs (~2*n*(Nz - n/2) slice-FFTs per view);
    the last basis order skips its cumulative sum (exit block only, no next
    state). Dtype-preserving. Eager; call under torch.no_grad().

    Centred by default (_born_init): the orders are built in eps_j and the
    gauge A is applied to D0 and to every order BEFORE returning, so the
    basis is always in the lab frame. Downstream fits therefore need no
    gauge bookkeeping — a common factor on all columns and on D0 leaves
    the least-squares minimiser unchanged in any case."""
    B, omode, Nz, Ny, Nx, _ = object_patches.shape
    n = min(n_max, Nz)
    obj, probe_k, Psi_hat, a_g = _born_init(
        object_patches, probe, H, centred=centred)
    gauge, _ = _born_gauges(a_g)  # exit blocks only
    D0 = probe_k.squeeze(3).expand(B, -1, omode, -1, -1)
    orders = []
    for k in range(n):
        b_hat = _born_apply_V(obj, Psi_hat, H, k)
        if k == n - 1:  # last basis order: exit block only
            orders.append(b_hat.sum(dim=3))
        else:
            Psi_hat, D_k = _born_apply_G(b_hat)
            # clone: D_k is a view into the scan's prefix sums
            orders.append(D_k.clone())
    D = torch.stack(orders)
    if gauge is not None:
        D0 = gauge * D0
        D = gauge * D
    return D0, D


def born_volume_basis(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    n_max: int,
    centred: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-order FOURIER-VIEW plane blocks plus the exit blocks, for the
    volume-residual fit and its checks.

    Returns (V, D0, D): V of shape (n+1, B, pmode, omode, Nz, Ny, Nx) with
    V[m, ..., j, :, :] the order-m hat block Psi_hat^(m)_j on plane j (zero
    for j < m: order m needs m prior sites), D0 and D exactly as from
    born_detector_basis, n = min(n_max, Nz). V[0] is the source
    Psi_hat^(0) = 1 (F P), materialised here as the stack the fit rows
    index. Unlike born_detector_basis the last order's cumulative sum runs
    too, so V[n] is populated (identically zero when n == Nz — no plane lies
    at depth >= Nz, only the exit block D[n-1] survives). Memory is (n+1)
    full per-slice stacks per view: call on small view chunks. Eager; call
    under torch.no_grad().

    Cost: this is where the Fourier view is strictly cheaper than the
    position view. The last order is CONSUMED as a hat block — it is a fit
    column, never scattered again — so G_hat (a cumsum) finishes it and no
    transform maps it back to position space. The position-view recursion
    had to pay one ifft over the last window to produce psi^(n), so this
    builder costs exactly Nz - n fewer plane-transforms (measured: 21/13
    -> 391 vs 399, 12/4 -> 85 vs 93). The saving is zero at n == Nz only
    because that window is empty. Do not reintroduce a trailing
    _born_incident here."""
    B, omode, Nz, Ny, Nx, _ = object_patches.shape
    n = min(n_max, Nz)
    pmode = probe.shape[1]
    obj, probe_k, Psi_hat, a_g = _born_init(
        object_patches, probe, H, centred=centred)
    gauge, gauge_planes = _born_gauges(a_g)
    D0 = probe_k.squeeze(3).expand(B, -1, omode, -1, -1)
    full = (B, pmode, omode, Nz, Ny, Nx)
    V = [probe_k.expand(full).clone()]
    orders = []
    for k in range(n):
        b_hat = _born_apply_V(obj, Psi_hat, H, k)
        Vk = torch.zeros(full, dtype=D0.dtype, device=D0.device)
        if k == Nz - 1:  # order Nz: exit block only, no plane holds it
            orders.append(b_hat.sum(dim=3))
        else:
            Psi_hat, D_k = _born_apply_G(b_hat)
            # clone: D_k is a view into the scan's prefix sums
            orders.append(D_k.clone())
            Vk[..., k + 1 :, :, :] = Psi_hat
        V.append(Vk)
    Vs, D = torch.stack(V), torch.stack(orders)
    if gauge is not None:  # lab frame; planes and exit carry DIFFERENT gauges
        Vs, D0, D = gauge_planes * Vs, gauge * D0, gauge * D
    return Vs, D0, D


# The compiled entry point. born_forward_eager stays exported for callers
# that need the undecorated function (tests comparing against an
# uncompiled run, and profiling).
#
# A per-position residual fit fused into this forward was tried and
# REMOVED (2026-11-12): at batch 1 it cost ~8.3 ms/position on top of the
# recursion and nothing brought that down — not row subsampling (the fit
# is insensitive to dropping 7/8 of the rows, yet the time plateaued), not
# vectorising the Gram assembly, and not CUDA graphs (forward+backward
# cannot be captured at all, and the no_grad fit alone replays at 7.76 ms
# vs 7.95 eager, so there is no launch overhead to remove). The residual
# fit is now subsample-shared like the remainder fits, in
# born_coeff_fit — 32 calibration views once per iteration instead of
# 4096 positions.
born_forward = torch.compile(mode="max-autotune")(born_forward_eager)

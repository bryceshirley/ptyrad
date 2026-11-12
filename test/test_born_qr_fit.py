"""The Born-coefficient remainder fit: born_detector_basis +
born_multislice_target + born_remainder_coeffs.

The identities the algorithm rests on: the truncated basis plus the
sequential multislice sweep reproduce the exact detector field at full order
(nilpotent termination); the equilibrated Gram solve equals the explicit
least-squares solution for c in float64; and the fit satisfies its limits:
c -> 1 for a weak object, c = 1 (zero residual) at M = Nz, and detector
field error at fixed truncation no worse than the plain series (nothing in
the span beats the projection).
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

torch._dynamo.config.disable = True

from torch.fft import fft2, ifft2

from ptyrad.forward_models.born import born_detector_basis
from ptyrad.forward_models.born_coeff_fit import (
    born_multislice_target,
    born_remainder_coeffs,
)


def _setup(B, omode, Nz, Ny, Nx, pmode, Bp, phase_scale=0.4, seed=0):
    g = torch.Generator().manual_seed(seed)
    patches = torch.rand(B, omode, Nz, Ny, Nx, 2, generator=g, dtype=torch.float64)
    patches[..., 0] = 1.0 + 0.2 * (patches[..., 0] - 0.5)
    patches[..., 1] *= phase_scale
    probe = torch.randn(Bp, pmode, Ny, Nx, generator=g, dtype=torch.float64) + 1j * torch.randn(
        Bp, pmode, Ny, Nx, generator=g, dtype=torch.float64
    )
    ky = torch.fft.fftfreq(Ny, dtype=torch.float64)
    kx = torch.fft.fftfreq(Nx, dtype=torch.float64)
    H1 = torch.exp(-1j * 0.3 * (ky[:, None] ** 2 + kx[None, :] ** 2) * Ny * Nx)
    H = (H1 ** torch.arange(Nz, dtype=torch.float64).view(Nz, 1, 1)).view(1, 1, 1, Nz, Ny, Nx)
    occu = torch.linspace(1.0, 0.5, omode, dtype=torch.float64)
    occu = occu / occu.sum()
    return patches, probe, H, occu


def _multislice_detector_field(patches, probe, H):
    """Independent sequential multislice, referred to the entrance plane in
    k-space (conj of the unimodular vacuum power) — the same gauge as the
    Born detector orders."""
    B, omode, Nz, Ny, Nx, _ = patches.shape
    O = torch.polar(patches[..., 0], patches[..., 1])  # (B, omode, Nz, Ny, Nx)
    H1 = H[0, 0, 0, 1]
    psi = probe[:, :, None]  # (B|1, pmode, 1, Ny, Nx) -> broadcast omode
    for j in range(Nz - 1):
        psi = ifft2(H1 * fft2(psi * O[:, None, :, j]))
    F = fft2(psi * O[:, None, :, Nz - 1])
    return F * (H1.conj() ** (Nz - 1))  # (B, pmode, omode, Ny, Nx)


def _field_err(c, D0, D, F_ref):
    model = D0 + (c.view(-1, 1, 1, 1, 1, 1) * D).sum(dim=0)
    return ((model - F_ref).norm() / F_ref.norm()).item()


def _d0_norm2(D0, occu):
    w = occu.view(1, 1, -1)
    return ((D0.abs().square().sum(dim=(-2, -1))) * w).sum().item()


def test_full_order_basis_sum_equals_multislice():
    # the identity the whole method rests on: nilpotent termination
    patches, probe, H, occu = _setup(2, 1, 5, 16, 16, 2, 1, phase_scale=1.0)
    D0, D = born_detector_basis(patches, probe, H, 5)
    F_ms = _multislice_detector_field(patches, probe, H)
    assert torch.allclose(D0 + D.sum(dim=0), F_ms, rtol=1e-9, atol=1e-11)


def test_multislice_target_matches_independent_sweep():
    # the target builder equals the independent reference sweep, same gauge
    patches, probe, H, occu = _setup(2, 2, 5, 16, 16, 2, 1, phase_scale=1.1)
    F = born_multislice_target(patches, probe, H)
    F_ms = _multislice_detector_field(patches, probe, H)
    assert torch.allclose(F, F_ms, rtol=1e-9, atol=1e-11)


def test_basis_truncation_is_prefix():
    # the truncated basis is exactly the leading orders of the full basis
    patches, probe, H, occu = _setup(2, 1, 6, 16, 16, 2, 1, phase_scale=0.9)
    _, D_full = born_detector_basis(patches, probe, H, 6)
    _, D_tr = born_detector_basis(patches, probe, H, 3)
    assert D_tr.shape[0] == 3
    assert torch.allclose(D_tr, D_full[:3], rtol=1e-11, atol=1e-13)


def test_gram_solve_matches_explicit_least_squares():
    # the equilibrated Gram solve equals the explicit least-squares
    # solution for c (float64, zero prior — NOT offset coordinates)
    patches, probe, H, occu = _setup(2, 2, 6, 16, 16, 2, 1, phase_scale=1.2)
    M = 4
    D0, D = born_detector_basis(patches, probe, H, M)
    T = born_multislice_target(patches, probe, H) - D0
    d0n2 = _d0_norm2(D0, occu)
    c_qr, delta = born_remainder_coeffs(D, T, d0n2, omode_occu=occu)

    w = occu.sqrt().view(1, 1, -1, 1, 1)
    A = (D * w).reshape(M, -1).T.to(torch.complex128)
    b = (T * w).reshape(-1).to(torch.complex128)
    c_ref = torch.linalg.lstsq(A, b[:, None]).solution[:, 0]
    # c is returned in complex64 (production dtype): compare at its precision
    assert torch.allclose(c_qr.to(torch.complex128), c_ref, rtol=1e-5, atol=1e-6)
    delta_ref = ((A @ c_ref - b).norm() / b.norm()).item()
    assert abs(delta - delta_ref) < 1e-9 * max(delta_ref, 1e-12)


def test_weak_object_coeffs_near_one():
    patches, probe, H, occu = _setup(2, 1, 6, 16, 16, 1, 1, phase_scale=1e-3)
    patches[..., 0] = 1.0  # pure weak-phase object
    D0, D = born_detector_basis(patches, probe, H, 3)
    T = born_multislice_target(patches, probe, H) - D0
    c, _ = born_remainder_coeffs(D, T, _d0_norm2(D0, occu), omode_occu=occu)
    assert torch.allclose(c, torch.ones_like(c), atol=1e-2)


def test_full_order_fit_is_exact():
    patches, probe, H, occu = _setup(2, 1, 4, 16, 16, 2, 1, phase_scale=1.2)
    D0, D = born_detector_basis(patches, probe, H, 4)
    T = born_multislice_target(patches, probe, H) - D0
    c, delta = born_remainder_coeffs(D, T, _d0_norm2(D0, occu), omode_occu=occu)
    assert torch.allclose(c, torch.ones_like(c), atol=1e-6)
    assert delta < 1e-6


def test_fit_beats_plain_in_detector_norm():
    # nothing in the span beats the projection — in EITHER gauge
    patches, probe, H, occu = _setup(2, 1, 6, 16, 16, 1, 1, phase_scale=1.5)
    n = 3
    F_ms = _multislice_detector_field(patches, probe, H)
    err = {}
    for centred in (False, True):
        D0, D = born_detector_basis(patches, probe, H, n, centred=centred)
        T = born_multislice_target(patches, probe, H) - D0
        c, _ = born_remainder_coeffs(D, T, _d0_norm2(D0, occu), omode_occu=occu)
        fit = _field_err(c.to(torch.complex128), D0, D, F_ms)
        plain = _field_err(torch.ones(n, dtype=torch.complex128), D0, D, F_ms)
        assert fit <= plain * (1 + 1e-6)  # the projection cannot be worse
        err[centred] = (fit, plain)

    # uncentred at strong scattering: the plain series OVERSHOOTS badly
    # (the C(N,m) path count outgrows the per-path decay, Sec. 3.4) and the
    # coefficients are what rescue it
    fit_u, plain_u = err[False]
    assert plain_u > 1.0            # >100% error: the overshoot is real
    assert fit_u < plain_u * 0.1    # tuning rescues it by >10x

    # centred: the background is already right, so the plain series is
    # close and the fit has little left to do
    fit_c, plain_c = err[True]
    assert plain_c < 0.2
    assert fit_c > plain_c * 0.9    # the fit gains only a few percent
    # and the UNTUNED centred series beats the TUNED uncentred one
    assert plain_c < fit_u


def test_delta_is_the_detector_field_error():
    # delta returned by the fit equals the directly-evaluated relative
    # detector field error of the fitted model (omode=1: unweighted norm)
    patches, probe, H, occu = _setup(2, 1, 6, 16, 16, 2, 1, phase_scale=1.3)
    n = 3
    D0, D = born_detector_basis(patches, probe, H, n)
    T = born_multislice_target(patches, probe, H) - D0
    c, delta = born_remainder_coeffs(D, T, _d0_norm2(D0, occu), omode_occu=occu)
    model = (c.to(torch.complex128).view(-1, 1, 1, 1, 1, 1) * D).sum(dim=0)
    delta_direct = ((model - T).norm() / T.norm()).item()
    assert abs(delta - delta_direct) < 1e-9 * max(delta_direct, 1e-12)


def test_extra_rhs_two_return_backward_compat():
    # without extra_rhs the return is the exact 2-tuple as before
    patches, probe, H, occu = _setup(2, 1, 6, 16, 16, 1, 1, phase_scale=1.2)
    D0, D = born_detector_basis(patches, probe, H, 3)
    T = born_multislice_target(patches, probe, H) - D0
    out = born_remainder_coeffs(D, T, _d0_norm2(D0, occu), omode_occu=occu)
    assert len(out) == 2


def test_extra_rhs_linearity_matches_shifted_target():
    # fitting against T + alpha * Delta (with the c-1 prior on the T part
    # only) must equal c_QR + alpha * c_Delta from the single factorization
    patches, probe, H, occu = _setup(2, 2, 6, 16, 16, 2, 1, phase_scale=1.2)
    M, alpha = 4, 0.7
    D0, D = born_detector_basis(patches, probe, H, M)
    T = born_multislice_target(patches, probe, H) - D0
    g = torch.Generator().manual_seed(7)
    delta_rhs = torch.randn(T.shape, generator=g, dtype=torch.float64) + 1j * torch.randn(
        T.shape, generator=g, dtype=torch.float64
    )
    d0n2 = _d0_norm2(D0, occu)
    c_qr, det_res, c_del = born_remainder_coeffs(
        D, T, d0n2, omode_occu=occu, extra_rhs=delta_rhs
    )
    # base solution and residual are untouched by the extra column (both
    # right-hand sides go through the same Gram factorisation)
    c_base, det_base = born_remainder_coeffs(D, T, d0n2, omode_occu=occu)
    assert torch.allclose(c_qr, c_base, rtol=0, atol=1e-12)
    # delta is evaluated from the (M, M) moments rather than from a tall
    # product, so it is cancellation-limited to ~1e-11 relative when the
    # fit residual is orders below ||T||; the solution itself is exact to
    # the ulp above.
    assert abs(det_res - det_base) < 1e-8 * max(det_base, 1e-12)
    # reference: explicit least squares for the combined target. The solve
    # is linear in the right-hand side and both parts are zero-prior, so
    # fitting T + alpha*Delta must give c_T + alpha*c_Delta exactly.
    w = occu.sqrt().view(1, 1, -1, 1, 1)
    A = (D * w).reshape(M, -1).T.to(torch.complex128)
    b = ((T + alpha * delta_rhs) * w).reshape(-1).to(torch.complex128)
    c_ref = torch.linalg.lstsq(A, b[:, None]).solution[:, 0]
    c_alpha = (c_qr + alpha * c_del).to(torch.complex128)
    assert torch.allclose(c_alpha, c_ref, rtol=1e-5, atol=1e-6)

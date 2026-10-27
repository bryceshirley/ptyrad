"""
Chebyshev exact-generator multislice forward model.

Each slab is propagated by U = exp(i k0 dz (sqrt(1 + B) - 1)), B = P (mu + eps) P,
evaluated to a set tolerance by a Chebyshev polynomial in B, so thick slabs
carry no splitting error. P is the band limit k < band_frac * K_Nyq,
mu = -k^2 / k0^2, and eps = n^2 - 1 with n - 1 = (phase - i log amp) / (k0 dz)
(phase and log amp summed over the slab), so a laterally uniform slab
reproduces amp * exp(i phase) exactly. Depth structure inside a slab is
averaged out.

    kernel = chebyshev_setup(shape, dx, lambd, dz_slab, phase_cap, amp_cap)
    dp = multislice_forward_chebyshev(object_patches, probe, *kernel, groups)

The recurrence runs in Fourier space, one FFT pair per degree:
B T = mu T + P fft2(eps ifft2(T)), masked every step so the field stays in the
band. It is built on B rather than 1 + B, with f(x) = exp(i k0 dz x / (1 + sqrt(1 + x)))
to avoid cancellation in float32.

phase_cap and amp_cap are the slab phase and amplitude ranges the
coefficients are built for. They fix a rectangle containing the field of
values of B; the coefficients are computed once, at the smallest degree whose
error on that rectangle is <= tol, which bounds the operator error per slab by
tol (phase-only, B Hermitian) or (1 + sqrt 2) tol (with absorption, by
Crouzeix's theorem). Outside the caps the error grows quickly, so make them
generous; the potential adds about (phase_cap[1] - phase_cap[0]) / 4 terms.

The degree is about k0 dz (sqrt(1 + hi) - sqrt(1 + lo)) / 2 plus ~10 terms.
Under torch.compile the slab and degree loops unroll; groups, lo, hi,
k0_dz_slab and len(coeffs) are compile-time constants.
"""

import numpy as np
import torch
from numpy.polynomial import chebyshev as npcheb
from torch.fft import fft2, ifft2

from ptyrad.forward_models.multislice import default_omode_occu
from ptyrad.utils import fftshift2, vprint


def chebyshev_setup(
    shape, dx, lambd, dz_slab, phase_cap, amp_cap=(0.8, 1.2), tol=1e-4, degree_cap=512,
    band_frac=2.0 / 3.0, device="cpu", dtype=torch.complex64, verbose=True,
):
    """
    Fixed propagator state for slabs of thickness dz_slab.

    Args:
        shape (tuple): (Ny, Nx) of the probe/patch grid.
        dx, lambd, dz_slab (float): pixel size, wavelength, slab thickness (same units).
        phase_cap (tuple): (lo, hi) slab phase in rad, e.g. a little below the
            minimum to ~2x the maximum slab phase.
        amp_cap (tuple): (lo, hi) slab amplitude; (1, 1) for a phase-only object.
        tol (float): truncation error per slab.
        degree_cap (int): interpolation degree and maximum polynomial degree.
        band_frac (float): band limit as a fraction of K_Nyquist.

    Returns:
        tuple: (mu_mul, band_mask, coeffs, lo, hi, k0_dz_slab).
    """
    k0 = 2.0 * np.pi / lambd
    theta = k0 * dz_slab
    mu_min = -((band_frac * np.pi / dx) ** 2) / k0**2

    def _f(z):
        """exp(i theta (sqrt(1 + z) - 1)) without the subtraction."""
        return np.exp(1j * theta * z / (1.0 + np.sqrt(1.0 + z)))

    # band mask and kinetic multiplier
    ky, kx = (2.0 * np.pi * torch.fft.fftfreq(n, d=dx, dtype=torch.float64, device=device) for n in shape)
    mu = -(ky[:, None] ** 2 + kx**2) / k0**2
    band_mask = (mu > mu_min).to(dtype.to_real())
    mu_mul = (mu * band_mask).to(dtype.to_real())

    # rectangle containing the field of values: eps at the cap extremes (Re eps is
    # monotone in phase and |log amp|, Im eps bilinear), plus the band edge
    y = -np.log(amp_cap[::-1]) / theta
    nm1 = np.add.outer(np.asarray(phase_cap) / theta, 1j * np.array([y[0], np.clip(0.0, *y), y[1]]))
    eps = nm1 * (nm1 + 2.0)
    re_lo, re_hi, im_lo, im_hi = mu_min + eps.real.min(), eps.real.max(), eps.imag.min(), eps.imag.max()
    ext = max(-im_lo, im_hi, 0.0)  # keep the rectangle's corners off the interval ends
    lo, hi = re_lo - ext, re_hi + ext
    if 1.0 + lo <= 0.0:
        raise ValueError(f"interval reaches the sqrt branch point (1 + lo = {1.0 + lo:g}); reduce band_frac")

    # interpolate with a growing degree until the coefficient tail is far below
    # tol, then drop the ~1e-15 noise tail (it would be amplified off the axis).
    # Interpolating once at a large degree_cap instead pads c with ~1e-13 noise
    # the cut cannot remove, which the off-axis error scan (|T_k| ~ rho^k) blows
    # up to overflow, clamping the selected degree to degree_cap.
    n = min(64, degree_cap)
    c = npcheb.Chebyshev.interpolate(_f, n, domain=[lo, hi]).coef
    while n < degree_cap and np.max(np.abs(c[-8:])) >= 0.01 * tol:
        n = min(2 * n, degree_cap)
        c = npcheb.Chebyshev.interpolate(_f, n, domain=[lo, hi]).coef
    c = c[: max(np.flatnonzero(np.abs(c) >= 1e-13 * np.abs(c).max())[-1] + 1, 2)]

    # smallest degree whose error on the rectangle's boundary stays <= tol
    s = 0.5 * (npcheb.chebpts1(2 * degree_cap) + 1.0)
    xs, ys = re_lo + (re_hi - re_lo) * s, im_lo + (im_hi - im_lo) * s
    z = np.concatenate([xs + 1j * im_lo, xs + 1j * im_hi, re_lo + 1j * ys, re_hi + 1j * ys])
    zeta = (2.0 * z - (hi + lo)) / (hi - lo)
    err = np.array([np.abs(npcheb.chebval(zeta, c[: m + 1]) - _f(z)).max() for m in range(c.size)])
    ok = np.maximum.accumulate(err[::-1])[::-1] <= tol
    M = max(int(np.argmax(ok)) if ok.any() else c.size - 1, 1)

    vprint(f"Chebyshev multislice: dz = {dz_slab:g}, degree {M} per slab, error {err[M]:.1e} "
           f"(tol {tol:g}{'; raise degree_cap' if err[M] > tol else ''})", verbose=verbose)
    # lo, hi, theta as Python floats: they are compile-time constants in the
    # forward, and numpy scalars confuse torch.compile's tracer
    return mu_mul, band_mask, torch.tensor(c[: M + 1], dtype=dtype, device=device), float(lo), float(hi), float(theta)


# Fused pointwise step of the Chebyshev recurrence, on float real views
# (shape (..., 2)). torch.compile cannot wrap the whole forward (inductor's
# complex-FFT lowering fails on the backward) AND cannot fuse complex pointwise
# (it silently falls back to eager -- slower than hand-rolled). On REAL views
# the arithmetic is real, so inductor fuses the two multiplies, the subtract
# and the complex accumulate into a single memory pass per degree. The FFTs
# stay outside; this coexists with make_graphed_callables (see models.py).
@torch.compile(dynamic=False)
def _cheb_step(mu_t, t1r, band_mask, convr, t0r, accr, cre, cim):
    t2r = mu_t * t1r + band_mask * convr - t0r  # B t1 - t0, real + imag channels
    re, im = t2r[..., 0], t2r[..., 1]
    acc_re = accr[..., 0] + cre * re - cim * im  # acc += (cre + i cim) * t2
    acc_im = accr[..., 1] + cre * im + cim * re
    return t2r, torch.stack((acc_re, acc_im), dim=-1)


# Plain eager function: speed comes from wrapping it with
# torch.cuda.make_graphed_callables at the call site (see models.py), which
# captures/replays the fwd+bwd CUDA graph.
def multislice_forward_chebyshev(
    object_patches, probe, mu_mul, band_mask, coeffs, lo, hi, k0_dz_slab, groups,
    omode_occu=None, eps=1e-10,
):
    """
    Multislice diffraction patterns with the Chebyshev slab model and incoherent
    probe and object modes.

    Args:
        object_patches (torch.Tensor): (N, omode, Nz, Ny, Nx, 2) float32 amplitude and phase.
        probe (torch.Tensor): (N, pmode, Ny, Nx) complex64 probe(s); N may be 1.
        mu_mul, band_mask, coeffs, lo, hi, k0_dz_slab: from chebyshev_setup.
        groups (tuple[tuple[int, int]]): slice ranges [s0, s1) per slab of thickness dz_slab.
        omode_occu (torch.Tensor, optional): (omode,) occupancies summing to 1.
        eps (float, optional): numerical stability floor. Defaults to 1e-10.

    Returns:
        torch.Tensor: (N, Ky, Kx) float32 diffraction patterns.
    """
    # These .contiguous() are needed for torch.compile in Linux
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    if omode_occu is None:
        omode_occu = default_omode_occu(object_patches[..., 1])

    # t_{k+1} = M t_k - t_{k-1}, M = 2(aB + b), with the interval map folded into the multipliers
    a, b = 4.0 / (hi - lo), -2.0 * (hi + lo) / (hi - lo)
    mu_t = a * mu_mul
    mu_b, band_b = mu_t.unsqueeze(-1), band_mask.unsqueeze(-1)  # (Ky, Kx, 1) for real views
    cre, cim = coeffs.real, coeffs.imag
    T = band_mask * fft2(probe[:, :, None])  # (N, pmode, 1, Ky, Kx)
    for s0, s1 in groups:
        # n - 1 per slab; sum of logs, not log of prod (prod's backward breaks CUDA graphs)
        nm1 = (object_patches[:, :, s0:s1, ..., 1].sum(2)
               - 1j * object_patches[:, :, s0:s1, ..., 0].clamp_min(1e-10).log().sum(2)) / k0_dz_slab
        eps_t = (a * nm1 * (nm1 + 2.0) + b)[:, None]  # (N, 1, omode, Ny, Nx)
        t0, t1 = T, 0.5 * (mu_t * T + band_mask * fft2(eps_t * ifft2(T)))
        acc = coeffs[0] * t0 + coeffs[1] * t1
        t0r, t1r = torch.view_as_real(t0.contiguous()), torch.view_as_real(t1.contiguous())
        accr = torch.view_as_real(acc.contiguous())
        for k in range(2, coeffs.shape[0]):
            convr = torch.view_as_real(fft2(eps_t * ifft2(torch.view_as_complex(t1r))))
            t2r, accr = _cheb_step(mu_b, t1r, band_b, convr, t0r, accr, cre[k], cim[k])
            t0r, t1r = t1r, t2r
        T = torch.view_as_complex(accr)

    # T is already the (band-limited) Fourier-space exit wave, so the far field
    # is T itself: detector's fft2(norm='ortho') of ifft2(T) collapses to
    # T / sqrt(Ny*Nx), i.e. |.|^2 scaled by 1/(Ny*Nx). Skipping the ifft2/fft2
    # round trip saves two FFTs per forward.
    scale = 1.0 / (T.shape[-2] * T.shape[-1])
    dp = torch.sum(fftshift2(T).abs().square() * omode_occu[:, None, None], dim=(1, 2)) * scale + eps
    return dp

"""
Chebyshev exact-generator multislice forward model.

Evaluates the one-way slab exponential

    U(dz) = exp( i * k0 * dz * ( sqrt(1 + mu + eps) - 1 ) )

per object slab EXACTLY (to a truncation tolerance) via a Chebyshev
operator polynomial, so that FEWER, THICKER object slabs can be used with
zero splitting error inside each slab. Here

    mu  = -K_perp^2 / k0^2            (Fourier multiplier, band-projected to
                                       the hard antialias band
                                       k_perp < band_frac * K_Nyq (default 2/3
                                       Nyquist; the data contains nothing
                                       beyond it), zero outside; the SAME hard
                                       mask is baked into the multiplier AND
                                       applied to the field at the entrance of
                                       every polynomial application and at the
                                       exit, so nothing lives off-interval),
    eps = 2*(n-1) + (n-1)^2,          (n-1) = phase_slab / (k0 * dz_slab),

with phase_slab the SUM of the constituent slice phases in the slab (the
slab is the z-homogenised column of its slices).

Shift stabilisation (critical in float32)
-----------------------------------------
The three-term recurrence is built on B = mu + eps, NOT on 1 + mu + eps,
and the scalar function expanded over the SHIFTED interval [a-1, b-1] is

    f(x) = exp( i * k0 * dz * x / (1 + sqrt(1 + x)) )
         = exp( i * k0 * dz * (sqrt(1 + x) - 1) ),

which is evaluated without forming sqrt(1+x)-1 by subtraction. The
unshifted form (recurrence on 1 + mu + eps with g(y) = exp(i k0 dz
(sqrt(y)-1))) loses about an order of magnitude to cancellation in f32
because the interval is a sliver around y = 1 (measured; see
markdowns/chebyshev_forward.md).

Interval management
-------------------
With the band projection, P mu P is negative semidefinite with spectrum in
[mu_min, 0], and eps is a pointwise real multiplier in [eps_min, eps_max],
so by Weyl's inequalities the spectrum of B lies in

    [a - 1, b - 1] = [mu_min + eps_min, eps_max].

The interval is computed from the CURRENT object each time coefficients
are built, padded by pad_frac (default 30%) of the span on each side, and
the coefficients are rebuilt ONLY when the object's eps range escapes the
padded interval (a cheap host-side check once per iteration; see
ChebyshevSlabKernel.refresh).

The band choice dominates the cost: the degree scales as
k0*dz*(sqrt(b)-sqrt(a))/2 ~ dz*k_band^2/(4*k0), i.e. quadratically in
k_band. Cropping from the historical wide band 0.9*min(k0, sqrt(2)*K_Nyq)
~ 1.27 K_Nyq to the 2/3-Nyquist antialias band cuts the degree ~3.6x.

Coefficients are computed host-side in numpy float64 (Chebyshev
interpolation at first-kind nodes, truncated at tail tolerance `tol`,
degree capped at `degree_cap`), cast to the compute dtype, and treated as
constants (detached). Gradients flow through the B applications only: the
fixed linear three-term recurrence is cleanly autodiff-able, which is the
point of Chebyshev over Lanczos.

Hybrid absorption screen
------------------------
The Chebyshev interval argument must stay real (B Hermitian), so the
amplitude (absorption) channel does NOT enter B. Instead the slab
amplitude screen A = prod(slice amplitudes) is applied pointwise at the
SLAB CENTRE: one slab is

    U(dz/2) * A * U(dz/2),

i.e. two half-slab Chebyshev applications with the real-space amplitude
multiplication between them (a Strang-style split of propagation against
absorption; exact for amp == 1). The coefficients stored by the manager
are therefore those of the HALF-slab exponential (theta = k0*dz_slab/2).

Cost
----
One B application = one FFT pair + pointwise multiplies. The polynomial
degree per slab scales as ~ k0*dz_slab*(sqrt(b)-sqrt(a))/2 plus a cushion
(printed at every coefficient build); the two half-slab applications
together cost about the same number of FFT pairs as one full-slab
application of twice the degree.

Conventions: the probe enters at the front face of the first slab and the
exit wave leaves at the back face of the last slab, i.e. the total drift
is n_slab*dz_slab (like the Chin 4A/4B slab models; plain Lie-Trotter in
this codebase drifts (n_slices-1)*dz — a pure defocus convention offset).
The carrier exp(i*k0*z) is removed (the -1 in the generator); far-field
intensities are unaffected.
"""

import numpy as np
import torch
from torch.fft import fft2, ifft2

from ptyrad.utils import fftshift2, vprint


def chebyshev_coeffs(theta, lo, hi, tol=1e-4, degree_cap=512):
    """
    Host-side numpy (float64) Chebyshev coefficients of
    f(x) = exp(i*theta*x/(1+sqrt(1+x))) over [lo, hi].

    Interpolates at first-kind Chebyshev nodes with the node count doubled
    until the coefficient tail is resolved, then truncates at the first
    index from the top with |c_k| >= tol (absolute; |f| = 1 so the
    coefficients are O(1)).

    Args:
        theta (float): k0 * dz of the propagated (half-)slab.
        lo, hi (float): padded SHIFTED interval [a-1-pad, b-1+pad].
        tol (float): tail truncation tolerance. Defaults to 1e-4.
        degree_cap (int): hard cap on the polynomial degree. Defaults to 512.

    Returns:
        np.ndarray: complex128 coefficients c_0..c_M (M = degree).
    """
    if not (1.0 + lo > 0.0):
        raise ValueError(f"Padded interval reaches the sqrt branch cut: 1+lo = {1.0 + lo:g} <= 0")
    n = 128
    while True:
        j = np.arange(n)
        nodes = np.cos(np.pi * (j + 0.5) / n)
        xm = 0.5 * (hi + lo) + 0.5 * (hi - lo) * nodes
        fj = np.exp(1j * theta * xm / (1.0 + np.sqrt(1.0 + xm)))
        k = np.arange(n)
        C = np.cos(np.pi * np.outer(k, j + 0.5) / n)
        c = (2.0 / n) * (C @ fj)
        c[0] *= 0.5
        # resolved when the top-of-spectrum tail is far below tol
        if np.max(np.abs(c[-8:])) < 0.01 * tol or n >= 8 * degree_cap:
            break
        n *= 2
    keep = np.nonzero(np.abs(c) >= tol)[0]
    M = int(keep[-1]) if keep.size else 0
    if M > degree_cap:
        vprint(
            f"WARNING: Chebyshev degree {M} exceeds cap {degree_cap}; truncating at the cap "
            f"(tail max |c| = {np.max(np.abs(c[degree_cap + 1:])):.3e} > tol = {tol:g})"
        )
        M = degree_cap
    return c[: M + 1]


def _apply_cheb_poly(psi, coeffs, lo, hi, mu_mul, eps_r):
    """
    y = sum_k c_k T_k(S) psi with S = (2B - (hi+lo)) / (hi - lo),
    B psi = ifft2(mu_mul * fft2(psi)) + eps_r * psi.

    coeffs are detached constants; gradients flow through the B
    applications (eps_r and psi). One FFT pair per recurrence step.
    """
    a = 2.0 / (hi - lo)
    b = -(hi + lo) / (hi - lo)

    def S_op(v):
        Bv = ifft2(mu_mul * fft2(v)) + eps_r * v
        return a * Bv + b * v

    deg = coeffs.shape[0] - 1
    t0 = psi
    acc = coeffs[0] * t0
    if deg >= 1:
        t1 = S_op(psi)
        acc = acc + coeffs[1] * t1
        for k in range(2, deg + 1):
            t2 = 2.0 * S_op(t1) - t0
            acc = acc + coeffs[k] * t2
            t0, t1 = t1, t2
    return acc


def _cheb_pointwise(Gr, t1r, t0r, accr, eps_b, cre, cim):
    """
    Fused pointwise tail of one recurrence step, on float real views:
    t2 = G + eps_t*t1 - t0;  acc += (cre + i*cim) * t2.
    Compiled (real arithmetic only, so Inductor can fuse it into one kernel);
    mathematically identical to the reference eager path.
    """
    t2r = Gr + eps_b * t1r - t0r
    re = t2r[..., 0]
    im = t2r[..., 1]
    acc_re = accr[..., 0] + cre * re - cim * im
    acc_im = accr[..., 1] + cre * im + cim * re
    return t2r, torch.stack([acc_re, acc_im], dim=-1)


_cheb_pointwise_compiled = None


def _apply_cheb_poly_fused(psi, coeffs, lo, hi, mu_mul, eps_r):
    """
    Fast path of _apply_cheb_poly (identical math): the recurrence
    t_{k+1} = M t_k - t_{k-1} with M = 2S is applied with the interval map
    pre-folded into the multipliers (mu_t = (4/(hi-lo))*mu_mul in k-space,
    eps_t = (4/(hi-lo))*eps_r - 2(hi+lo)/(hi-lo) in real space) and the
    pointwise tail of each step fused into a single compiled kernel on
    real views. Set PTYRAD_CHEB_EAGER=1 to force the reference path.
    """
    global _cheb_pointwise_compiled
    if _cheb_pointwise_compiled is None:
        _cheb_pointwise_compiled = torch.compile(_cheb_pointwise, dynamic=False)

    deg = coeffs.shape[0] - 1
    if deg < 2:
        return _apply_cheb_poly(psi, coeffs, lo, hi, mu_mul, eps_r)

    a = 2.0 / (hi - lo)
    b = -(hi + lo) / (hi - lo)
    mu_t = (2.0 * a) * mu_mul
    eps_t = (2.0 * a) * eps_r + (2.0 * b)
    rdt = eps_t.dtype
    cre = coeffs.real.to(rdt)
    cim = coeffs.imag.to(rdt)

    t0 = psi
    G = ifft2(mu_t * fft2(psi))
    t1 = 0.5 * (G + eps_t * psi)  # S psi = (M psi)/2
    acc = coeffs[0] * t0 + coeffs[1] * t1

    t0r = torch.view_as_real(t0.contiguous())
    t1r = torch.view_as_real(t1)
    accr = torch.view_as_real(acc)
    eps_b = eps_t.unsqueeze(-1)
    for k in range(2, deg + 1):
        G = ifft2(mu_t * fft2(torch.view_as_complex(t1r)))
        t2r, accr = _cheb_pointwise_compiled(
            torch.view_as_real(G), t1r, t0r, accr, eps_b, cre[k], cim[k]
        )
        t0r, t1r = t1r, t2r
    return torch.view_as_complex(accr)


def _select_poly_impl():
    import os

    return _apply_cheb_poly if os.environ.get("PTYRAD_CHEB_EAGER") else _apply_cheb_poly_fused


def chebyshev_exit_wave(
    object_patches, psi, mu_mul, band_mask, coeffs_half, lo, hi, k0_dz_slab, groups
):
    """
    Sweep the wave through all slabs: per slab U(dz/2) * A * U(dz/2) with
    U the Chebyshev half-slab exponential and A the pointwise amplitude
    screen at the slab centre.

    The hard band mask (the same one baked into mu_mul) is applied to the
    field at the ENTRANCE of every polynomial application and to the exit
    wave: the real-space eps and amplitude multiplies scatter a little
    content off-band inside each application, and the projection kills it
    before it can accumulate or reach the detector mis-propagated (its
    mu is zeroed, i.e. it would drift as if k_perp = 0). Costs one FFT
    pair per application plus one at the exit.

    Args:
        object_patches (torch.Tensor): (N, omode, Nz, Ny, Nx, 2) pseudo-complex
            amp/phase object patches (slice resolution).
        psi (torch.Tensor): (N, pmode, omode|1, Ny, Nx) wave at the entrance face.
        mu_mul (torch.Tensor): (Ny, Nx) real band-projected Fourier multiplier
            -K_perp^2/k0^2 (unshifted fft layout).
        band_mask (torch.Tensor): (Ny, Nx) real hard band mask (1 in-band,
            0 outside; unshifted fft layout).
        coeffs_half (torch.Tensor): (M+1,) complex HALF-slab Chebyshev
            coefficients (detached constants).
        lo, hi (float): padded shifted expansion interval.
        k0_dz_slab (float): k0 * dz_slab normalisation for n-1 = phase/(k0*dz).
        groups (tuple[tuple[int, int]]): slice index ranges [s0, s1) per slab.

    Returns:
        torch.Tensor: (N, pmode, omode, Ny, Nx) complex exit wave at the back
        face of the last slab (total drift n_slab*dz_slab, carrier removed).
    """
    apply_poly = _select_poly_impl()

    def band_project(v):
        return ifft2(band_mask * fft2(v))

    amp = object_patches[..., 0]
    phase = object_patches[..., 1]
    for s0, s1 in groups:
        phase_slab = phase[:, :, s0:s1].sum(dim=2)  # (N, omode, Ny, Nx)
        nm1 = phase_slab / k0_dz_slab
        eps_r = (nm1 * (2.0 + nm1))[:, None]  # (N, 1, omode, Ny, Nx)
        # exp(sum(log)) instead of prod: amplitudes are strictly positive and
        # prod's backward breaks CUDA-graph capture (host-side zero check)
        amp_slab = amp[:, :, s0:s1].clamp_min(1e-10).log().sum(dim=2).exp()[:, None]
        psi = apply_poly(band_project(psi), coeffs_half, lo, hi, mu_mul, eps_r)
        psi = psi * amp_slab
        psi = apply_poly(band_project(psi), coeffs_half, lo, hi, mu_mul, eps_r)
    return band_project(psi)


def multislice_forward_chebyshev(
    object_patches,
    probe,
    mu_mul,
    band_mask,
    coeffs_half,
    lo,
    hi,
    k0_dz_slab,
    groups,
    omode_occu=None,
    eps=1e-10,
):
    """
    Computes the multislice electron diffraction pattern with the Chebyshev
    exact-generator slab forward model (see module docstring).

    Args:
        object_patches (torch.Tensor): (N, omode, Nz, Ny, Nx, 2) pseudo-complex
            object patches with float amplitude and phase components.
        probe (torch.Tensor): (N|1, pmode, Ny, Nx) complex probe(s).
        mu_mul, band_mask, coeffs_half, lo, hi, k0_dz_slab, groups: precomputed
            kernel state from ChebyshevSlabKernel (coeffs are detached constants).
        omode_occu (torch.Tensor, optional): (omode,) occupancies summing to 1.
        eps (float, optional): numerical stability floor. Defaults to 1e-10.

    Returns:
        torch.Tensor: (N, Ky, Kx) float forward diffraction patterns.
    """
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()

    if omode_occu is None:
        objp = object_patches[..., 1]
        omode_occu = torch.ones(objp.size(1), dtype=objp.dtype, device=objp.device) / objp.size(1)

    psi = probe[:, :, None, :, :]  # (N, pmode, omode, Ny, Nx)
    psi = chebyshev_exit_wave(
        object_patches, psi, mu_mul, band_mask, coeffs_half, lo, hi, k0_dz_slab, groups
    )

    dp_fwd = (
        torch.sum(
            (fftshift2(fft2(psi, norm="ortho"))).abs().square() * omode_occu[:, None, None],
            dim=(1, 2),
        )
        + eps
    )
    return dp_fwd


class ChebyshevSlabKernel:
    """
    Host-side manager for the Chebyshev slab propagator state: the
    band-projected mu multiplier, the padded expansion interval, and the
    half-slab coefficients. Owns the cheap per-iteration interval check
    (refresh) that rebuilds the coefficients only when the object's eps
    range escapes the padded interval.
    """

    def __init__(
        self,
        shape,
        dx,
        lambd,
        dz_slab,
        groups,
        device,
        dtype=torch.complex64,
        tol=1e-4,
        degree_cap=512,
        pad_frac=0.3,
        band_frac=2.0 / 3.0,
        verbose=True,
    ):
        """
        Args:
            shape (tuple): (Ny, Nx) of the probe/patch grid.
            dx (float): pixel size (physical units, same as dz/lambda).
            lambd (float): wavelength.
            dz_slab (float): slab thickness = group size * slice thickness.
            groups (tuple[tuple[int, int]]): slice index ranges per slab.
            device: torch device for the kernel tensors.
            dtype: complex compute dtype (complex64 or complex128).
            tol (float): coefficient tail truncation tolerance.
            degree_cap (int): hard cap on the polynomial degree.
            pad_frac (float): interval padding fraction per side.
            band_frac (float): hard antialias band as a fraction of K_Nyquist
                (default 2/3, the antialias standard; safety-clamped below k0).
        """
        self.verbose = verbose
        self.tol = float(tol)
        self.degree_cap = int(degree_cap)
        self.pad_frac = float(pad_frac)
        self.band_frac = float(band_frac)
        self.device = device
        self.cdtype = dtype
        self.rdtype = torch.float64 if dtype == torch.complex128 else torch.float32

        self.k0 = 2.0 * np.pi / float(lambd)
        self.dz_slab = float(dz_slab)
        self.k0_dz_slab = self.k0 * self.dz_slab
        self.theta_half = 0.5 * self.k0_dz_slab
        self.groups = tuple((int(s0), int(s1)) for s0, s1 in groups)

        Ny, Nx = shape
        ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=float(dx))
        kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=float(dx))
        K2 = ky[:, None] ** 2 + kx[None, :] ** 2
        k_nyq = np.pi / float(dx)
        # hard antialias band: fraction of K_Nyquist (2/3 by default); the
        # sqrt generator needs 1 + mu + eps > 0, so stay safely below k0
        k_band = min(self.band_frac * k_nyq, 0.95 * self.k0)
        band = K2 < k_band**2
        mu = np.where(band, -K2 / self.k0**2, 0.0)
        self.mu_min = float(mu.min())
        self.mu_mul = torch.tensor(mu, dtype=self.rdtype, device=device)
        self.band_mask = torch.tensor(band.astype(np.float64), dtype=self.rdtype, device=device)

        self.lo = None
        self.hi = None
        self.coeffs = None
        self.n_rebuilds = 0
        vprint(
            f"Chebyshev slab kernel: k0 = {self.k0:.4f}, dz_slab = {self.dz_slab:.4f} "
            f"({len(self.groups)} slab(s) of {self.groups[0][1] - self.groups[0][0]} slice(s)), "
            f"hard band k_perp < {k_band:.4f} (frac {self.band_frac:.4f} of K_Nyq = {k_nyq:.4f}), "
            f"mu_min = {self.mu_min:.3e}",
            verbose=self.verbose,
        )

    @torch.no_grad()
    def eps_bounds(self, objp):
        """
        Min/max of eps = 2*(n-1) + (n-1)^2 over all slabs of the CURRENT
        object phase canvas (any leading dims; slices on dim -3).
        """
        nm1_min = np.inf
        nm1_max = -np.inf
        for s0, s1 in self.groups:
            phase_slab = objp[..., s0:s1, :, :].sum(dim=-3)
            nm1_min = min(nm1_min, float(phase_slab.min()) / self.k0_dz_slab)
            nm1_max = max(nm1_max, float(phase_slab.max()) / self.k0_dz_slab)
        # eps(x) = x*(2+x) is monotonically increasing for x > -1
        return nm1_min * (2.0 + nm1_min), nm1_max * (2.0 + nm1_max)

    def refresh(self, objp, verbose=None):
        """
        Cheap host-side per-iteration check: rebuild the coefficients only
        when the object's current Weyl interval escapes the padded one.

        Returns:
            bool: True if the coefficients were (re)built.
        """
        verbose = self.verbose if verbose is None else verbose
        eps_lo, eps_hi = self.eps_bounds(objp.detach())
        core_lo = self.mu_min + eps_lo  # a - 1
        core_hi = eps_hi  # b - 1 (Weyl: mu_max = 0)
        if self.coeffs is not None and core_lo >= self.lo and core_hi <= self.hi:
            return False

        span = core_hi - core_lo
        lo = core_lo - self.pad_frac * span
        hi = core_hi + self.pad_frac * span
        c = chebyshev_coeffs(self.theta_half, lo, hi, tol=self.tol, degree_cap=self.degree_cap)
        self.lo = lo
        self.hi = hi
        self.coeffs = torch.tensor(c, dtype=self.cdtype, device=self.device)
        self.n_rebuilds += 1
        deg_half = self.coeffs.shape[0] - 1
        est_full = self.k0_dz_slab * (np.sqrt(1.0 + core_hi) - np.sqrt(1.0 + core_lo)) / 2.0
        vprint(
            f"Chebyshev coefficients built (#{self.n_rebuilds}): interval "
            f"[{lo:.4e}, {hi:.4e}] (core [{core_lo:.4e}, {core_hi:.4e}], pad {self.pad_frac}), "
            f"half-slab degree {deg_half} (2x per slab), degree estimate per slab "
            f"k0*dz*(sqrt(b)-sqrt(a))/2 = {est_full:.1f} + cushion, tol {self.tol:g}",
            verbose=verbose,
        )
        return True

    def kernels(self):
        """Forward-model kernel state (coeffs are detached constants)."""
        return (
            self.mu_mul,
            self.band_mask,
            self.coeffs,
            self.lo,
            self.hi,
            self.k0_dz_slab,
            self.groups,
        )

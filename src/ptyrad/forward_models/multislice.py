"""
Physical forwad model that generates diffraction patterns from mixed-state probe/object in a fully vectorized way

"""

import torch
from torch.fft import fft2, ifft2

from ptyrad.utils import fftshift2

# The forward model takes a batch of object patches and probes with their mixed states
# By introducing and aligning the singleton dimensions carefully,
# we can vectorize all the operations except the serial z-dimension propagation
# For 3D object with n_slices, the for loop would go through n-1 loops and multiply the last slice without further Fresnel propagaiton
# This way we can skip the if statement and make it slightly faster
# For 2D object (n_slices = 1), the entire for loop is skipped
# Note that element-wise multiplication of tensor (*) is defaulted as out-of-place operation
# So new tensor is being created and referenced to the old graph to keep the gradient flowing


@torch.compile(mode="max-autotune")
def multislice_forward(object_patches, probe, H, omode_occu=None, eps=1e-10):
    """
    Computes the multislice electron diffraction pattern with multiple incoherent probe
    and object modes using a vectorized forward model.

    Args:
        object_patches (torch.Tensor): Tensor of shape (N, omode, Nz, Ny, Nx, 2), representing
            pseudo-complex object patches with float32 amplitude and phase components.
            N is the number of samples in a batch, omode is the number of object modes,
            Nz, Ny, Nx are the dimensions of the object patches.
        omode_occu (torch.Tensor): Tensor of shape (omode,) with float32 values, representing
            the occupancy/expectation for each object mode. The sum of all elements should be 1.
        probe (torch.Tensor): Tensor of shape (N, pmode, Ny, Nx) with complex64 values,
            representing the probe(s). N is the number of samples in the batch, pmode is the
            number of probe modes. By default, N is 1, assuming the same probe for all samples.
        H (torch.Tensor): Tensor of shape (N, Ky, Kx) with complex64 values, representing the Fresnel
            propagator that propagates the wave by a slice thickness.
        eps (float, optional): A small value added for numerical stability. Defaults to 1e-10.

    Returns:
        torch.Tensor: Tensor of shape (N, Ky, Kx) with float32 positive values, representing the
        forward diffraction pattern for each sample in the batch.
    """

    # These .contiguous() are needed for torch.compile in Linux
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    H = H.contiguous()

    # Initialize omode_occu if it's not specified
    if omode_occu is None:
        objp = object_patches[..., 1]
        device = objp.device
        dtype = objp.dtype
        omode = objp.size(1)
        omode_occu = torch.ones(omode, dtype=dtype, device=device) / omode

    # Cast the object back to actual complex tensor
    object_cplx = torch.polar(
        object_patches[..., 0], object_patches[..., 1]
    ).contiguous()  # (N, omode, Nz, Ny, Nx)
    n_slices = object_cplx.shape[2]

    # Expand psi to include omode dimension
    psi = probe[:, :, None, :, :].contiguous()  # (N, pmode, Ny, Nx) -> (N, pmode, omode, Ny, Nx)

    # Propagating each object layer using broadcasting
    for n in range(n_slices - 1):
        object_slice = object_cplx[:, :, n, :, :]  # object_slice -> (N, omode, Ny, Nx)
        psi = (
            psi * object_slice[:, None, :, :, :]
        )  # psi -> (N, pmode, omode, Ny, Nx). Note that psi is always centered in real space
        psi = ifft2(
            H[:, None, None] * fft2(psi)
        )  # Note that fft2 and ifft2 are applying to the last 2 axes. Although preshift psi before fft2 would seem more natural, it's nearly 50% slower to do it as fftshift2(ifft2(fft2(ifftshift2(psi))))

    # Interacting with the last layer, and no propagation is needed afterward
    object_slice = object_cplx[:, :, n_slices - 1, :, :]
    psi = psi * object_slice[:, None, :, :, :]

    # Propagate the object-modified exit wave psi(r) to detector plane into psi(k)
    # The contribution from probe / object modes are incoherently summed together

    # Breaking down the steps for clarity, while combine all of these for lower peak memory consumption
    # psi_k = fftshift(fft2(psi))
    # |psi_k|^2 = psi_k.abs().square()
    # weighted_psi_k = |psi_k|^2 * omode_occu
    # dp_fwd = sum(weighted_psi_k)
    # Note that norm = 'ortho' is needed to ensure that for each sample, sum(|psi|^2) and sum(dp) has the same scale (should be 1)

    dp_fwd = (
        torch.sum(
            (fftshift2(fft2(psi, norm="ortho"))).abs().square() * omode_occu[:, None, None],
            dim=(1, 2),
        )
        + eps
    )  # Add eps for numerical stability
    return dp_fwd


@torch.compile(mode="max-autotune")
def linduda_forward(object_patches, probe, H, delta, L_op=None, omode_occu=None, eps=1e-10, M=1):
    """
    Computes the multislice electron diffraction pattern with multiple incoherent probe
    and object modes using a vectorized forward model, extended with a Lie-Trotter
    higher-order correction solver.

    ... [Args Docstring unchanged, but note L_op addition] ...
        L_op (torch.Tensor, optional): The exact unwrapped analytical L operator (e.g., i * k_z * delta).
            Must be provided if M > 0 to avoid phase wrapping artifacts from torch.log(H).
    """

    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    H = H.contiguous()

    if omode_occu is None:
        objp = object_patches[..., 1]
        device = objp.device
        dtype = objp.dtype
        omode = objp.size(1)
        omode_occu = torch.ones(omode, dtype=dtype, device=device) / omode

    object_cplx = torch.polar(object_patches[..., 0], object_patches[..., 1]).contiguous()
    n_slices = object_cplx.shape[2]

    psi = probe[:, :, None, :, :].contiguous()

    # Precompute the L operator WITHOUT torch.log(H) to avoid phase wrapping
    if M > 0:
        if L_op is None:
            # Fallback (DANGEROUS for high angles/thick slices)
            L_op_expanded = torch.log(H)[:, None, None]
        else:
            # Safe analytic operator
            L_op_expanded = L_op[:, None, None].contiguous()

    for n in range(n_slices - 1):
        object_slice = object_cplx[:, :, n, :, :]

        # --- Extended Solver: Lie-Trotter Cross Term Application ---
        if M > 0:
            # Construct N_op directly from raw patches to avoid phase wrapping!
            amp_slice = object_patches[:, :, n, :, :, 0]
            phase_slice = object_patches[:, :, n, :, :, 1]

            # N_op = ln(amp) + i * phase
            N_op = torch.complex(torch.log(amp_slice + eps), phase_slice)[:, None, :, :, :]

            dpsi = psi
            psi_out = psi.clone()

            # Taylor series expansion for exp(Omega_{Delta z})
            for m in range(1, M + 1):
                # 1. Apply N then L
                N_dpsi = N_op * dpsi
                L_N_dpsi = ifft2(L_op_expanded * fft2(N_dpsi))

                # 2. Apply L then N
                L_dpsi = ifft2(L_op_expanded * fft2(dpsi))
                N_L_dpsi = N_op * L_dpsi

                # Calculate Omega_{Delta z} * dpsi
                Omega_dpsi = -0.5 / delta * (L_N_dpsi + N_L_dpsi)

                # Update current term and accumulate
                dpsi = Omega_dpsi / m
                psi_out = psi_out + dpsi

            psi = psi_out
        # -----------------------------------------------------------

        psi = psi * object_slice[:, None, :, :, :]
        psi = ifft2(H[:, None, None] * fft2(psi))

    # Last layer interaction
    object_slice = object_cplx[:, :, n_slices - 1, :, :]
    psi = psi * object_slice[:, None, :, :, :]

    dp_fwd = (
        torch.sum(
            (fftshift2(fft2(psi, norm="ortho"))).abs().square() * omode_occu[:, None, None],
            dim=(1, 2),
        )
        + eps
    )

    return dp_fwd


@torch.compile(mode="max-autotune")
def strang_forward(object_patches, probe, H_tuple, omode_occu=None, eps=1e-10):
    """
    Computes the multislice electron diffraction pattern using 2nd-order Strang Splitting.
    Maintains the exact interface and output of the 1st-order vectorized forward model.
    """
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    H, H_half = H_tuple

    if omode_occu is None:
        objp = object_patches[..., 1]
        device = objp.device
        dtype = objp.dtype
        omode = objp.size(1)
        omode_occu = torch.ones(omode, dtype=dtype, device=device) / omode

    object_cplx = torch.polar(object_patches[..., 0], object_patches[..., 1]).contiguous()
    n_slices = object_cplx.shape[2]

    psi = probe[:, :, None, :, :].contiguous()

    # --- STRANG SPLITTING aka Conjugate Lie Trotter ---
    # Lie trotter:
    # psi = (e^A e^B)^N = e^A e^B ... e^A e^B P
    # Strang:
    # psi = (e^A/2 e^B e^A/2)^N P = e^A/2 e^B e^A ... e^B e^A/2 P = e^-A/2 (e^A e^B)^N e^A/2 P
    # where P is the initial condition.
    # e^A = Fresnel propagator, e^B = object transmission
    # e^A = ifft2(H * fft2()), with |H| = 1
    # |F[psi]|^2 = |F[e^-A/2 (e^A e^B)^N e^A/2 P]|^2 = |F[(e^A e^B)^N e^A/2 P]|^2
    #            = |F[(e^A e^B)^N P']|^2
    # where P' = e^A/2 P is the initial condition after a half-step Fresnel propagation.

    # Is the detector wave error second order?

    # Initial half-drift into the first slice
    psi = ifft2(H_half[:, None, None] * fft2(psi))

    for n in range(n_slices - 1):
        object_slice = object_cplx[:, :, n, :, :]
        psi = psi * object_slice[:, None, :, :, :]
        psi = ifft2(H[:, None, None] * fft2(psi))

    # Interacting with the last layer
    object_slice = object_cplx[:, :, n_slices - 1, :, :]
    psi = psi * object_slice[:, None, :, :, :]

    # Final conjugate half-drift out of the last slice is cancelled out by the final FFT and modulus.

    dp_fwd = (
        torch.sum(
            (fftshift2(fft2(psi, norm="ortho"))).abs().square() * omode_occu[:, None, None],
            dim=(1, 2),
        )
        + eps
    )
    return dp_fwd


# ---------------------------------------------------------------------------
# Chin fourth-order gradient splittings (4A and 4B)
#
# References: S. A. Chin, Phys. Lett. A 226, 344 (1997);
#             S. A. Chin and C. R. Chen, J. Chem. Phys. 117, 1409 (2002).
#
# Each object slice k is treated as one layered slab of thickness dz, uniform
# in z, with chi_k = phase_k - i*log(amp_k) so that O_k = exp(i*chi_k). Both
# schemes reach fourth order by adding the double-commutator correction
# [V,[T,V]], which for the PARAXIAL (Fresnel) kinetic operator
# T = -(1/(2 k0)) * Laplacian is the LOCAL multiplication
# g_k = grad(chi_k) . grad(chi_k) (transverse gradient, complex dot product,
# no conjugation, so absorbing objects are covered). With the angular-spectrum
# kernel the double commutator is NOT a local multiplication and the schemes
# drop to second order — hence both functions require the Fresnel kernel
# K(d) = ifftshift(exp(-1j * d * (Kx^2 + Ky^2) / (2 k0))) (carrier removed).
# With this kernel sign convention and O = amp * exp(+i*phase), the correction
# sign is s = +1 (a wrong sign drops the schemes to second order; verified by
# the order-of-convergence tests in test/test_chin_splitting.py).
#
# Cost: both schemes use two FFT pairs per slice — the same as Lie-Trotter
# with twice as many slices.
# ---------------------------------------------------------------------------

# 4B Gauss-Legendre drift fractions (a1 + a2 + a1 = 1)
CHIN_4B_A1 = 0.5 * (1.0 - 1.0 / 3.0**0.5)
CHIN_4B_A2 = 1.0 / 3.0**0.5


def _chin_transmission(object_patches, g_patches, chi_frac, g_coeff, eps):
    """
    Build the corrected slice transmission exp(i*chi_frac*chi + i*g_coeff*g)
    for all slices at once.

    chi = phase - i*log(amp) and g = grad(chi).grad(chi) enter as
    W = chi_frac*chi + g_coeff*g, and the factor exp(i*W) is assembled as
    torch.polar(exp(-Im W), Re W). Fractional powers/logs are applied to the
    real amplitude/phase components before the polar cast (Inductor-safe).

    Args:
        object_patches: (N, omode, Nz, Ny, Nx, 2) pseudo-complex amp/phase.
        g_patches: (N, omode, Nz, Ny, Nx, 2) pseudo-complex Re(g)/Im(g),
            or None to drop the gradient term (g = 0).
        chi_frac (float): fraction of chi in the exponent (2/3 for 4A, 1/2 for 4B).
        g_coeff (torch.Tensor): real scalar s*dz/(72*k0) for 4A or
            s*(2-sqrt(3))*dz/(48*k0) for 4B, with s = +1.
        eps (float): numerical floor inside log(amp).

    Returns:
        (N, omode, Nz, Ny, Nx) complex transmission per slice.
    """
    amp = object_patches[..., 0]
    phase = object_patches[..., 1]
    w_re = chi_frac * phase
    w_im = -chi_frac * torch.log(amp.clamp_min(eps))
    if g_patches is not None:
        w_re = w_re + g_coeff * g_patches[..., 0]
        w_im = w_im + g_coeff * g_patches[..., 1]
    return torch.polar(torch.exp(-w_im), w_re).contiguous()


def _multislice_forward_chin4a_impl(
    object_patches, probe, H_half, g_patches, dz, k0, omode_occu=None, eps=1e-10
):
    """
    Multislice diffraction forward model using Chin's 4A fourth-order gradient
    splitting (potential outermost; Simpson weights 1/6, 2/3, 1/6).

    One slice of thickness dz is

        exp(i chi_k/6) K(dz/2) T_Ak K(dz/2) exp(i chi_k/6),
        T_Ak = exp( i (2/3) chi_k + i dz g_k / (72 k0) ),

    with K the FRESNEL propagator (required; see module comment) and
    g_k = grad(chi_k).grad(chi_k). Between slices the neighbouring boundary
    phases are merged into a single pointwise factor exp(i (chi_k+chi_{k+1})/6);
    the two end phases exp(i chi_1/6), exp(i chi_S/6) are applied explicitly
    (pointwise, free). The probe stays at the entrance plane of the first
    slab, and the exit wave leaves at the back face of the last slab, i.e. the
    total drift is n_slices*dz (Lie-Trotter in this codebase drifts
    (n_slices-1)*dz — the extra dz/2 at each end is the slab-vs-plane
    convention, a pure defocus offset).

    Cost: two FFT pairs (one K(dz/2) each) per slice — the same as
    Lie-Trotter with twice as many slices.

    Args:
        object_patches (torch.Tensor): (N, omode, Nz, Ny, Nx, 2) pseudo-complex
            object patches (float amplitude and phase).
        probe (torch.Tensor): (N, pmode, Ny, Nx) complex probe(s) at the
            entrance plane.
        H_half (torch.Tensor): (N, Ky, Kx) complex Fresnel kernel K(dz/2),
            built with the same tilt terms and batching as the full-dz H.
        g_patches (torch.Tensor): (N, omode, Nz, Ny, Nx, 2) pseudo-complex
            Re(g_k)/Im(g_k), computed with non-periodic finite differences and
            the physical pixel size (see PtychoAD.get_g_patches).
        dz (torch.Tensor): real scalar slice thickness (physical units).
        k0 (torch.Tensor): real scalar wavenumber 2*pi/lambda.
        omode_occu (torch.Tensor, optional): (omode,) occupancies summing to 1.
        eps (float, optional): numerical stability floor. Defaults to 1e-10.

    Returns:
        torch.Tensor: (N, Ky, Kx) float forward diffraction patterns.
    """
    # These .contiguous() are needed for torch.compile in Linux
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    H_half = H_half.contiguous()
    if g_patches is not None:
        g_patches = g_patches.contiguous()

    if omode_occu is None:
        objp = object_patches[..., 1]
        omode_occu = torch.ones(objp.size(1), dtype=objp.dtype, device=objp.device) / objp.size(1)

    amp = object_patches[..., 0]
    phase = object_patches[..., 1]
    n_slices = object_patches.shape[2]

    # T_Ak = exp(i (2/3) chi_k + i dz g_k/(72 k0)), correction sign s = +1
    T_cplx = _chin_transmission(object_patches, g_patches, 2.0 / 3.0, dz / (72.0 * k0), eps)

    # Boundary phases exp(i chi_k/6): end factors and merged interior pairs
    # exp(i (chi_k + chi_{k+1})/6). Fractional powers on real components
    # before the polar cast (Inductor-safe).
    amp6 = amp.clamp_min(eps) ** (1.0 / 6.0)
    ph6 = phase * (1.0 / 6.0)
    B_in = torch.polar(amp6[:, :, 0], ph6[:, :, 0]).contiguous()  # (N, omode, Ny, Nx)
    B_out = torch.polar(amp6[:, :, -1], ph6[:, :, -1]).contiguous()
    B_mid = torch.polar(
        amp6[:, :, 1:] * amp6[:, :, :-1], ph6[:, :, 1:] + ph6[:, :, :-1]
    ).contiguous()  # (N, omode, Nz-1, Ny, Nx)

    psi = probe[:, :, None, :, :].contiguous()  # (N, pmode, omode, Ny, Nx)
    psi = psi * B_in[:, None]

    for n in range(n_slices):
        psi = ifft2(H_half[:, None, None] * fft2(psi))
        psi = psi * T_cplx[:, None, :, n]
        psi = ifft2(H_half[:, None, None] * fft2(psi))
        if n < n_slices - 1:
            psi = psi * B_mid[:, None, :, n]

    psi = psi * B_out[:, None]

    dp_fwd = (
        torch.sum(
            (fftshift2(fft2(psi, norm="ortho"))).abs().square() * omode_occu[:, None, None],
            dim=(1, 2),
        )
        + eps
    )
    return dp_fwd


def _multislice_forward_chin4b_impl(
    object_patches,
    probe,
    H_a1,
    H_2a1,
    H_a2,
    g_patches,
    dz,
    k0,
    omode_occu=None,
    eps=1e-10,
    apply_end_props=True,
):
    """
    Multislice diffraction forward model using Chin's 4B fourth-order gradient
    splitting (kinetic outermost; Gauss-Legendre points a1 = (1-1/sqrt(3))/2,
    a2 = 1/sqrt(3)).

    One slice of thickness dz is

        K(a1 dz) T_Bk K(a2 dz) T_Bk K(a1 dz),
        T_Bk = exp( i chi_k/2 + i (2 - sqrt(3)) dz g_k / (48 k0) ),

    with K the FRESNEL propagator (required; see module comment). Between
    slices the adjoining drifts K(a1 dz) K(a1 dz) are merged into K(2 a1 dz),
    so the interior costs two FFT pairs (K(a2 dz) and K(2 a1 dz)) per slice —
    the same as Lie-Trotter with twice as many slices. The probe enters at
    the front face of the first slab and the exit wave leaves at the back
    face of the last slab (total drift n_slices*dz).

    End drifts (apply_end_props):
        True (default): the entrance and exit K(a1 dz) are applied explicitly
        (one extra FFT pair each, once per forward pass).
        False (far-field data only): both end drifts are dropped. The exit
        K(a1 dz) is a unit-modulus multiplication in k-space, so it cannot
        change far-field intensities; the entrance K(a1 dz) is absorbed into
        the probe, WHOSE PLANE THEREBY SHIFTS by a1*dz ~ 0.2113*dz into the
        first slab (i.e. the optimized probe converges to K(a1 dz)*probe; a
        pure defocus offset to account for when comparing probes).

    Args:
        object_patches (torch.Tensor): (N, omode, Nz, Ny, Nx, 2) pseudo-complex
            object patches (float amplitude and phase).
        probe (torch.Tensor): (N, pmode, Ny, Nx) complex probe(s).
        H_a1 (torch.Tensor): (N, Ky, Kx) Fresnel kernel K(a1*dz).
        H_2a1 (torch.Tensor): (N, Ky, Kx) Fresnel kernel K(2*a1*dz).
        H_a2 (torch.Tensor): (N, Ky, Kx) Fresnel kernel K(a2*dz).
            All three must be built with the same tilt terms and batching as
            the full-dz H (never as powers of H).
        g_patches (torch.Tensor): (N, omode, Nz, Ny, Nx, 2) pseudo-complex
            Re(g_k)/Im(g_k) (see PtychoAD.get_g_patches).
        dz (torch.Tensor): real scalar slice thickness (physical units).
        k0 (torch.Tensor): real scalar wavenumber 2*pi/lambda.
        omode_occu (torch.Tensor, optional): (omode,) occupancies summing to 1.
        eps (float, optional): numerical stability floor. Defaults to 1e-10.
        apply_end_props (bool, optional): apply the entrance/exit K(a1 dz)
            explicitly. Defaults to True.

    Returns:
        torch.Tensor: (N, Ky, Kx) float forward diffraction patterns.
    """
    # These .contiguous() are needed for torch.compile in Linux
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    H_a1 = H_a1.contiguous()
    H_2a1 = H_2a1.contiguous()
    H_a2 = H_a2.contiguous()
    if g_patches is not None:
        g_patches = g_patches.contiguous()

    if omode_occu is None:
        objp = object_patches[..., 1]
        omode_occu = torch.ones(objp.size(1), dtype=objp.dtype, device=objp.device) / objp.size(1)

    n_slices = object_patches.shape[2]

    # T_Bk = exp(i chi_k/2 + i (2-sqrt(3)) dz g_k/(48 k0)), correction sign s = +1
    g_coeff = (2.0 - 3.0**0.5) / 48.0 * dz / k0
    T_cplx = _chin_transmission(object_patches, g_patches, 0.5, g_coeff, eps)
    T_in = T_out = T_cplx

    psi = probe[:, :, None, :, :].contiguous()  # (N, pmode, omode, Ny, Nx)

    if apply_end_props:
        psi = ifft2(H_a1[:, None, None] * fft2(psi))

    for n in range(n_slices):
        psi = psi * T_in[:, None, :, n]
        psi = ifft2(H_a2[:, None, None] * fft2(psi))
        psi = psi * T_out[:, None, :, n]
        if n < n_slices - 1:
            psi = ifft2(H_2a1[:, None, None] * fft2(psi))

    if apply_end_props:
        psi = ifft2(H_a1[:, None, None] * fft2(psi))

    dp_fwd = (
        torch.sum(
            (fftshift2(fft2(psi, norm="ortho"))).abs().square() * omode_occu[:, None, None],
            dim=(1, 2),
        )
        + eps
    )
    return dp_fwd


# Compiled entry points; the _impl functions stay reachable for float64
# gradcheck and reference tests.
multislice_forward_chin4a = torch.compile(_multislice_forward_chin4a_impl, mode="max-autotune")
multislice_forward_chin4b = torch.compile(_multislice_forward_chin4b_impl, mode="max-autotune")

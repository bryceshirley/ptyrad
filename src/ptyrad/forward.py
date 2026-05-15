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
def multislice_forward_model_vec_all(object_patches, probe, H, omode_occu=None, eps=1e-10):
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
def multislice_forward_model_vec_strang(object_patches, probe, H_tuple, omode_occu=None, eps=1e-10):
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

    # --- STRANG SPLITTING ---
    # Initial half-drift into the first slice
    psi = ifft2(H_half[:, None, None] * fft2(psi))

    for n in range(n_slices - 1):
        object_slice = object_cplx[:, :, n, :, :]
        psi = psi * object_slice[:, None, :, :, :]
        psi = ifft2(H[:, None, None] * fft2(psi))

    # Interacting with the last layer
    object_slice = object_cplx[:, :, n_slices - 1, :, :]
    psi = psi * object_slice[:, None, :, :, :]

    # Final half-drift to the exit surface
    psi = ifft2(H_half[:, None, None] * fft2(psi))
    # --- END SPLITTING ---

    dp_fwd = (
        torch.sum(
            (fftshift2(fft2(psi, norm="ortho"))).abs().square() * omode_occu[:, None, None],
            dim=(1, 2),
        )
        + eps
    )
    return dp_fwd


# @torch.compile(mode="max-autotune")
def multislice_forward_model_vec_suzuki_trotter(
    object_patches, probe, H, omode_occu=None, eps=1e-10
):
    """
    Computes the multislice electron diffraction pattern using 4th-order Suzuki-Trotter Splitting.
    Maintains the exact interface and output of the 1st-order vectorized forward model.
    """
    # Force memory to be C-contiguous for CUDAGraph safety
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    H = H.contiguous()

    if omode_occu is None:
        objp = object_patches[..., 1]
        omode_occu = torch.ones(objp.size(1), dtype=objp.dtype, device=objp.device) / objp.size(1)

    omode_occu = omode_occu.contiguous()

    # Suzuki-Trotter fractional constants
    p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    w2 = 1.0 - 4.0 * p

    # Precompute fractional propagators
    H_p_half = torch.pow(H, p / 2.0).contiguous()
    H_p = torch.pow(H, p).contiguous()
    H_mid = torch.pow(H, (p + w2) / 2.0).contiguous()

    # Extract full amplitude and phase volumes
    amp = object_patches[..., 0]
    phase = object_patches[..., 1]

    # Pre-calculate the fractional object transmissions for ALL slices at once.
    # Applying fractional powers to real components before polar casting ensures
    # Inductor doesn't crash on the backward pass.
    O_p_cplx = torch.polar(amp**p, phase * p).contiguous()
    O_w2_cplx = torch.polar(amp**w2, phase * w2).contiguous()

    n_slices = object_patches.shape[2]
    psi = probe[:, :, None, :, :].contiguous()

    for n in range(n_slices):
        # Slice the precomputed complex volumes and add the 'pmode' singleton dimension
        # Shape: (N, omode, Nz, Ny, Nx) -> (N, 1, omode, Ny, Nx)
        O_p = O_p_cplx[:, None, :, n, :, :]
        O_w2 = O_w2_cplx[:, None, :, n, :, :]

        # --- 4th ORDER S-T FRACTAL ---

        # 1. Drift p/2
        psi = ifft2(H_p_half[:, None, None] * fft2(psi))

        # 2. Kick p, Drift p, Kick p
        psi = psi * O_p
        psi = ifft2(H_p[:, None, None] * fft2(psi))
        psi = psi * O_p

        # 3. Drift (p + w2)/2 (Merged boundary)
        psi = ifft2(H_mid[:, None, None] * fft2(psi))

        # 4. Kick w2 (The negative step)
        psi = psi * O_w2

        # 5. Drift (p + w2)/2 (Merged boundary)
        psi = ifft2(H_mid[:, None, None] * fft2(psi))

        # 6. Kick p, Drift p, Kick p
        psi = psi * O_p
        psi = ifft2(H_p[:, None, None] * fft2(psi))
        psi = psi * O_p

        # 7. Drift p/2
        psi = ifft2(H_p_half[:, None, None] * fft2(psi))

        # --- END FRACTAL ---

    dp_fwd = (
        torch.sum(
            (fftshift2(fft2(psi, norm="ortho"))).abs().square() * omode_occu[:, None, None],
            dim=(1, 2),
        )
        + eps
    )
    return dp_fwd


@torch.compile(mode="max-autotune")
def multislice_forward_model_vec_all_born(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    omode_occu: torch.Tensor,
    eps: float = 1e-10,
    n_max: int = 1,
) -> torch.Tensor:
    """
    Parallel Born Series Forward Model for Multislice Ptychography.

    This function implements a parallelized formulation of the multislice algorithm.
    By expanding the multislice operator into a Born series and factoring out the vacuum
    transmission, the traditional O(N_z) sequential propagation loop is replaced with
    parallel prefix sums (`cumsum`).

    Scattering Regimes (`n_max`):
    -----------------------------
    * `n_max = 1` (First-Order Born Approximation):
        Models a single scattering event. The wave scatters at each slice and
        propagates directly to the detector without further object interactions.
        This completely bypasses the 3D prefix sum, utilizing a single,
        parallel reduction directly to the 2D exit plane.

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
        H (torch.Tensor): Tuple of tensors of shape (N, Ky, Kx) or (1, Ky, Kx) with complex64 values,
            representing the Fresnel propagator that propagates the wave by a slice thickness.
        omode_occu (torch.Tensor): Tensor of shape (omode,) with float32 values.
        eps (float, optional): A small value added for numerical stability. Defaults to 1e-10.
        n_max (int): Maximum order of the Born series iterations (orders of scattering).

    Returns:
        torch.Tensor: Tensor of shape (N, Ny, Nx) with float32 positive values, representing the
        forward diffraction pattern for each sample in the batch.
    """
    # Ensure contiguity of incoming base tensors
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    kernel_fwd = H[0].contiguous()
    kernel_inv = H[1].contiguous()

    N_batch, omode, Nz, Ny, Nx, _ = object_patches.shape

    if omode_occu is None:
        omode_occu = (
            torch.ones(omode, dtype=object_patches.dtype, device=object_patches.device) / omode
        )
    norm_weight = omode_occu / (Nx * Ny)

    # ==========================================
    # 1. Scattering Operator (Object)
    # ==========================================
    amplitude = object_patches[..., 0]
    phase = object_patches[..., 1]

    # In the Bidiagonal formulation, obj acts as the spatial scattering potential
    obj = (torch.polar(amplitude, phase) - 1.0).unsqueeze(1)

    # ============================================
    # 2. Compute 0th Order Field and Detector Wave
    # ============================================
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)

    # Extract the 0th order detector wave in k-space
    Psi_M_hat = probe_k.squeeze(3)

    # Compute the 0th order internal spatial wave
    Psi_0_hat_3D = kernel_fwd * probe_k
    Psi_state_active = ifft2(Psi_0_hat_3D)

    # ==========================================
    # 3. Combinatorial Born Series Loop
    # ==========================================
    for n in range(1, n_max + 1):
        # Object physically cannot scatter at or below slice 'n' for the n-th bounce.
        W_spatial_active = obj[:, :, :, n:, :, :] * Psi_state_active[:, :, :, :-1, :, :]
        W_hat_active = fft2(W_spatial_active)

        k_inv_active = kernel_inv[:, :, :, n:, :, :]
        scattered_k = W_hat_active * k_inv_active

        if n < n_max:
            # INTERMEDIATE BORN ORDER:
            # 1. Compute the cumulative sum for the internal 3D state
            cumsum_scattered = torch.cumsum(scattered_k, dim=3)

            # 2. The detector contribution is just the last slice of the cumsum (free sum!)
            Psi_M_hat = Psi_M_hat + cumsum_scattered[:, :, :, -1, :, :]

            # 3. Propagate the internal 3D state forward for the next bounce
            k_fwd_active = kernel_fwd[:, :, :, n:, :, :]
            Psi_state_active = ifft2(cumsum_scattered * k_fwd_active)
        else:
            # Contribution to the detector plane from this highest order n_max
            Psi_M_hat = Psi_M_hat + torch.sum(scattered_k, dim=3)

    # ==========================================
    # 5. Detector Measurement (Incoherent Sum)
    # ==========================================
    dp_fwd = fftshift2(
        torch.sum(Psi_M_hat.abs().square() * norm_weight.view(1, 1, -1, 1, 1), dim=(1, 2)) + eps
    )

    return dp_fwd


@torch.compile(mode="max-autotune")
def multislice_forward_model_vec_all_first_born(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    omode_occu: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Parallel First-Born Forward Model.
    """
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    kernel_fwd = H[0].contiguous()
    kernel_inv = H[1].contiguous()

    N_batch, omode, Nz, Ny, Nx, _ = object_patches.shape

    if omode_occu is None:
        omode_occu = (
            torch.ones(omode, dtype=object_patches.dtype, device=object_patches.device) / omode
        )
    norm_weight = omode_occu / (Nx * Ny)

    # ============================================
    # 1. Compute Probe in K-Space
    # ============================================
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    Psi_M_hat = probe_k.squeeze(3)

    # ==========================================
    # 2. 0th Order Spatial Field
    # ==========================================
    k_fwd_active = kernel_fwd[:, :, :, :-1, :, :]
    Psi_state_active = ifft2(k_fwd_active * probe_k)

    # ==========================================
    # 3. Object Potential
    # ==========================================
    amplitude = object_patches[:, :, 1:, :, :, 0]
    phase = object_patches[:, :, 1:, :, :, 1]

    obj_active = (torch.polar(amplitude, phase) - 1.0).unsqueeze(1)

    # ==========================================
    # 4. First Born Scattering
    # ==========================================
    W_spatial_active = obj_active * Psi_state_active
    W_hat_active = fft2(W_spatial_active)

    k_inv_active = kernel_inv[:, :, :, 1:, :, :]
    scattered_k = W_hat_active * k_inv_active

    # ==========================================
    # 5. Detector Measurement
    # ==========================================
    Psi_M_hat = Psi_M_hat + torch.sum(scattered_k, dim=3)

    dp_fwd = fftshift2(
        torch.sum(Psi_M_hat.abs().square() * norm_weight.view(1, 1, -1, 1, 1), dim=(1, 2)) + eps
    )

    return dp_fwd


@torch.compile(mode="max-autotune")
def multislice_forward_model_vec_all_gmres1(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    omode_occu: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Parallel Forward Model using 1 Iteration of GMRES.
    Highly optimized: Mathematically simplified to bypass explicit Arnoldi orthogonalization.
    """
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    kernel_fwd = H[0].contiguous()
    kernel_inv = H[1].contiguous()

    N_batch, omode, Nz, Ny, Nx, _ = object_patches.shape

    if omode_occu is None:
        omode_occu = (
            torch.ones(omode, dtype=object_patches.dtype, device=object_patches.device) / omode
        )
    norm_weight = omode_occu / (Nx * Ny)

    # ============================================
    # 1. Compute Probe in K-Space (x_0)
    # ============================================
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    Psi_M_hat = probe_k.squeeze(3)  # This is our initial guess: x_0_hat

    # ==========================================
    # 2. 0th Order Spatial Field
    # ==========================================
    k_fwd_active = kernel_fwd[:, :, :, :-1, :, :]
    Psi_state_active = ifft2(k_fwd_active * probe_k)

    # ==========================================
    # 3. Object Potential
    # ==========================================
    amplitude = object_patches[:, :, 1:, :, :, 0]
    phase = object_patches[:, :, 1:, :, :, 1]
    obj_active = (torch.polar(amplitude, phase) - 1.0).unsqueeze(1)
    # ==========================================
    # 4. Initial Residual (r_0 = S * x_0)
    # ==========================================
    W_spatial_active = obj_active * Psi_state_active
    W_hat_active = fft2(W_spatial_active)

    k_inv_active = kernel_inv[:, :, :, 1:, :, :]
    scattered_k = W_hat_active * k_inv_active

    # r_0 = c - A(x_0) = x_0 - (x_0 - Scatter(x_0)) = Scatter(x_0)
    r0_hat = torch.sum(scattered_k, dim=3)

    # ==========================================
    # 5. Apply Operator directly to Initial Residual
    # ==========================================
    # We apply A(r_0) = r_0 - Scatter(r_0) without normalizing r_0 first
    r0_spatial = ifft2(k_fwd_active * r0_hat.unsqueeze(3))
    W_r0_spatial = obj_active * r0_spatial
    W_r0_hat = fft2(W_r0_spatial)

    scatter_r0 = torch.sum(W_r0_hat * k_inv_active, dim=3)
    z_hat = r0_hat - scatter_r0  # This represents A(r0_hat)

    # ==========================================
    # 6. GMRES 1D Least Squares (Direct Projection)
    # ==========================================
    # Mathematically identical to Arnoldi for k=1, but skips w_perp and normalization.
    # alpha = <z_hat, r0_hat> / ||z_hat||^2
    num = torch.sum(torch.conj(z_hat) * r0_hat, dim=(-2, -1), keepdim=True)
    den = torch.sum(z_hat.abs().square(), dim=(-2, -1), keepdim=True) + eps
    alpha = num / den

    # Update solution
    x1_hat = Psi_M_hat + alpha * r0_hat

    # ==========================================
    # 7. Detector Measurement
    # ==========================================
    dp_fwd = fftshift2(
        torch.sum(x1_hat.abs().square() * norm_weight.view(1, 1, -1, 1, 1), dim=(1, 2)) + eps
    )

    return dp_fwd

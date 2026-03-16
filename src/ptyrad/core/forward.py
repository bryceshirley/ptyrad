"""
Physical forward model that generates diffraction patterns from mixed-state probe/object in a fully vectorized way

"""

# ruff: noqa: N806

import torch
from torch.fft import fft2, ifft2

from ptyrad.core.functional import fftshift2

# The forward model takes a batch of object patches and probes with their mixed states
# By introducing and aligning the singleton dimensions carefully,
# we can vectorize all the operations except the serial z-dimension propagation
# For 3D object with n_slices, the for loop would go through n-1 loops and multiply the last slice without further Fresnel propagaiton
# This way we can skip the if statement and make it slightly faster
# For 2D object (n_slices = 1), the entire for loop is skipped
# Note that element-wise multiplication of tensor (*) is defaulted as out-of-place operation
# So new tensor is being created and referenced to the old graph to keep the gradient flowing


def multislice_forward(
    obja_patches: torch.Tensor,
    objp_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,  # noqa: N803
    omode_occu: torch.Tensor | None = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Computes the multislice electron diffraction pattern with multiple incoherent probe
    and object modes using a vectorized forward model.

    Args:
        obja_patches (torch.Tensor): Tensor of shape (N, omode, Nz, Ny, Nx), representing
            object amplitude patches with float32.
            N is the number of samples in a batch, omode is the number of object modes,
            Nz, Ny, Nx are the dimensions of the object patches.
        objp_patches (torch.Tensor): Tensor of shape (N, omode, Nz, Ny, Nx), representing
            object phase patches with float32.
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

    assert obja_patches.shape == objp_patches.shape

    # Initialize omode_occu if it's not specified
    if omode_occu is None:
        device = objp_patches.device
        dtype = objp_patches.dtype
        omode = objp_patches.size(1)
        omode_occu = torch.ones(omode, dtype=dtype, device=device) / omode

    # Unbind the Z-dimension (dim=2) BEFORE the loop
    # This returns a tuple of n_slices independent tensors of shape (N, omode, Ny, Nx)
    # This is critical for efficient torch.compile triton code generation during .backward(), especially for pytorch >= 2.8.0
    obja_slices = torch.unbind(obja_patches, dim=2)
    objp_slices = torch.unbind(objp_patches, dim=2)
    n_slices = len(obja_slices)

    # Expand psi to include omode dimension
    psi = probe[:, :, None, :, :]  # (N, pmode, Ny, Nx) -> (N, pmode, omode, Ny, Nx)

    # Propagating each object layer using broadcasting
    for n in range(n_slices - 1):
        object_slice = torch.polar(
            obja_slices[n], objp_slices[n]
        )  # object_slice -> (N, omode, Ny, Nx)
        psi = (
            psi * object_slice[:, None, :, :]
        )  # psi -> (N, pmode, omode, Ny, Nx). Note that psi is always centered in real space
        psi = ifft2(
            H[:, None, None] * fft2(psi)
        )  # Note that fft2 and ifft2 are applying to the last 2 axes. Although preshift psi before fft2 would seem more natural, it's nearly 50% slower to do it as fftshift2(ifft2(fft2(ifftshift2(psi))))

    # Interacting with the last layer, and no propagation is needed afterward
    object_slice = torch.polar(obja_slices[-1], objp_slices[-1])
    psi = psi * object_slice[:, None, :, :]

    # Propagate the object-modified exit wave psi(r) to detector plane into psi(k)
    # The contribution from probe / object modes are incoherently summed together
    # Chained all operations for lower peak memory consumption
    # Doing fftshift2 last reduces the needed memory moves
    # Note that norm = 'ortho' is needed to ensure that for each sample, sum(|psi|^2) and sum(dp) has the same scale (should be 1)

    dp_fwd = (
        fftshift2(
            torch.sum(
                fft2(psi, norm="ortho").abs().square() * omode_occu[:, None, None],
                dim=(1, 2),
            ),
        )
        + eps
    )  # Add eps for numerical stability
    return dp_fwd  # type: ignore[no-any-return]


@torch.compile(mode="max-autotune")
def multislice_forward_born(
    obja_patches: torch.Tensor,
    objp_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,  # noqa: N803
    omode_occu: torch.Tensor | None = None,
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
        obja_patches (torch.Tensor): Tensor of shape (N, omode, Nz, Ny, Nx), representing
            object amplitude patches with float32.
            N is the number of samples in a batch, omode is the number of object modes,
            Nz, Ny, Nx are the dimensions of the object patches.
        objp_patches (torch.Tensor): Tensor of shape (N, omode, Nz, Ny, Nx), representing
            object phase patches with float32.
            N is the number of samples in a batch, omode is the number of object modes,
            Nz, Ny, Nx are the dimensions of the object patches.
        probe (torch.Tensor): Tensor of shape (N, pmode, Ny, Nx) with complex64 values,
            representing the probe(s). N is the number of samples in the batch, pmode is the
            number of probe modes. By default, N is 1, assuming the same probe for all samples.
        H (torch.Tensor): Tensor of shape (N, Ky, Kx) with complex64 values, representing the Fresnel
            propagator that propagates the wave by a slice thickness.
        omode_occu (torch.Tensor): Tensor of shape (omode,) with float32 values, representing
            the occupancy/expectation for each object mode. The sum of all elements should be 1.
        eps (float, optional): A small value added for numerical stability. Defaults to 1e-10.
        n_max (int): Maximum order of the Born series iterations (orders of scattering).

    Returns:
        torch.Tensor: Tensor of shape (N, Ky, Kx) with float32 positive values, representing the
        forward diffraction pattern for each sample in the batch.
    """
    assert obja_patches.shape == objp_patches.shape

    # Ensure contiguity of incoming base tensors
    obja_patches = obja_patches.contiguous()
    objp_patches = objp_patches.contiguous()
    probe = probe.contiguous()
    H = H.contiguous()

    _N_batch, omode, Nz, Ny, Nx = obja_patches.shape

    if omode_occu is None:
        device = objp_patches.device
        dtype = objp_patches.dtype
        omode = objp_patches.size(1)
        omode_occu = torch.ones(omode, dtype=dtype, device=device) / omode

    # ==========================================
    # 1. Precompute Nilpotent Scattering (Object)
    # ==========================================
    # In the Bidiagonal formulation, obj acts as the spatial scattering potential
    obj = 1.0 - torch.polar(obja_patches, objp_patches).contiguous()

    # Broadcast to (N, 1, omode, Nz, Ny, Nx) to allow cross-mode broadcasting with probe
    obj = obj.unsqueeze(1)

    # ==========================================
    # 2. Compute Propagation Kernels
    # ==========================================
    # Elevate H to 6D: (N or 1, 1, 1, 1, Ny, Nx)
    z_idx = torch.arange(Nz, device=H.device).view(1, 1, 1, Nz, 1, 1)
    H_view = H.view(H.shape[0], 1, 1, 1, Ny, Nx)

    kernel_fwd = H_view.pow(z_idx + 1)
    kernel_inv = H_view.conj().pow(z_idx)

    # 2D Exit kernel for the final reduction to the detector plane
    k_fwd_exit = kernel_fwd[:, :, :, -1:, :, :]

    # ============================================
    # 3. Compute 0th Order Field and Detector Wave
    # ============================================
    # Elevate probe to 6D: (N or 1, pmode, 1, 1, Ny, Nx)
    probe_view = probe.view(probe.shape[0], probe.shape[1], 1, 1, Ny, Nx)
    probe_k = fft2(probe_view)

    # If probe is (1,...) and kernel_fwd is (N,...), this automatically broadcasts them
    Psi_0_hat_3D = kernel_fwd * probe_k

    # Extract the 0th order detector wave in k-space
    Psi_M_hat = Psi_0_hat_3D[:, :, :, -1, :, :]

    # Compute the 0th order internal spatial wave
    Psi_0_spatial = ifft2(Psi_0_hat_3D)

    # ==========================================
    # 4. Combinatorial Born Series Loop
    # ==========================================

    # Track exactly ONE state variable: the spatial 3D wave.
    Psi_state_3D = Psi_0_spatial

    for n in range(1, n_max + 1):
        # Spatial Scattering Source and project to k-space
        # The negative sign correctly mirrors the physics: Object = I - S -> scattered wave must be subtracted!
        W_spatial = -obj * torch.roll(Psi_state_3D, shifts=1, dims=3)

        # Enforce Nilpotency IN SPATIAL DOMAIN before FFT
        W_spatial[:, :, :, 0, :, :] = 0.0

        # Project to k-space
        W_hat = fft2(W_spatial)

        if n == n_max:
            # HIGHEST BORN ORDER requested:
            # Skip the 3D cumsum. Run parallel reduction directly to the 2D detector.
            Psi_n_M_hat = (
                torch.sum(W_hat * kernel_inv, dim=3, keepdim=True) * k_fwd_exit
            )

            # Contribution to the detector plane from this highest order n_max
            Psi_M_hat = Psi_M_hat + Psi_n_M_hat.squeeze(3)

            # No IFFT needed since there are no more scattering events to propagate through.
            # We are already in k-space at the far field.
        else:
            # INTERMEDIATE BORN ORDER:
            # Compute full 3D internal field via parallel cumsum.
            Psi_n_hat_3D = torch.cumsum(W_hat * kernel_inv, dim=3) * kernel_fwd

            # Contribution to the detector plane from this intermediate order n
            Psi_M_hat = Psi_M_hat + Psi_n_hat_3D[:, :, :, -1, :, :]

            # OVERWRITE STATE: Immediately transform to spatial domain for the next loop.
            Psi_state_3D = ifft2(Psi_n_hat_3D)

    # ==========================================
    # 5. Detector Measurement (Corrected Incoherent Sum)
    # ==========================================
    # 1. Take absolute square of the k-space wave
    # Shape: (N, pmode, omode, Ny, Nx)
    intensity_k = Psi_M_hat.abs().square()

    # 2. Weight by object mode occupancy
    # Shape: (N, pmode, omode, Ny, Nx) * (omode,) -> (N, pmode, omode, Ny, Nx)
    weighted_intensity = intensity_k * omode_occu.view(1, 1, -1, 1, 1)

    # 3. Sum over ALL incoherent modes (pmode and omode)
    # Result: (N, Ny, Nx)
    dp_fwd = torch.sum(weighted_intensity, dim=(1, 2))

    # 4. Final Shift and Normalization
    dp_fwd = fftshift2(dp_fwd)
    dp_fwd = (dp_fwd / (Nx * Ny)) + eps

    return dp_fwd  # type: ignore[no-any-return]

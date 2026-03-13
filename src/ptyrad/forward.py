"""
Physical forwad model that generates diffraction patterns from mixed-state probe/object in a fully vectorized way

"""

import torch
import triton
import triton.language as tl
from torch.fft import fft2, ifft2

from ptyrad.utils import fftshift2


import math


# The forward model takes a batch of object patches and probes with their mixed states
# By introducing and aligning the singleton dimensions carefully,
# we can vectorize all the operations except the serial z-dimension propagation
# For 3D object with n_slices, the for loop would go through n-1 loops and multiply the last slice without further Fresnel propagaiton
# This way we can skip the if statement and make it slightly faster
# For 2D object (n_slices = 1), the entire for loop is skipped
# Note that element-wise multiplication of tensor (*) is defaulted as out-of-place operation
# So new tensor is being created and referenced to the old graph to keep the gradient flowing


#@torch.compile(mode="max-autotune")
def multislice_forward_model_vec_all(
    object_patches, probe, H, omode_occu=None, eps=1e-10
):
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
        probe (torch.Tensor): Tensor of shape (N, pmode, Ny, Nx) or (1, pmode, Ny, Nx) with complex64 values,
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
    psi = probe[
        :, :, None, :, :
    ].contiguous()  # (N, pmode, Ny, Nx) -> (N, pmode, omode, Ny, Nx)

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
            (fftshift2(fft2(psi, norm="ortho"))).abs().square()
            * omode_occu[:, None, None],
            dim=(1, 2),
        )
        + eps
    )  # Add eps for numerical stability
    return dp_fwd


#@torch.compile(mode="max-autotune")
def multislice_forward_model_vec_all_parallel(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    omode_occu: torch.Tensor = None,
    eps: float = 1e-10,
    n_iter: int = 1,
) -> torch.Tensor:
    """
    Parallel Multislice Forward Model (Bidiagonal ASM Solver).
    """
    # Ensure contiguity
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    H = H.contiguous()

    N_batch, omode, Nz, Ny, Nx, _ = object_patches.shape

    if omode_occu is None:
        omode_occu = (
            torch.ones(omode, dtype=object_patches.dtype, device=object_patches.device)
            / omode
        )

    # ==========================================
    # 1. Half-Object Transmissions
    # ==========================================
    amplitude = object_patches[..., 0]
    phase = object_patches[..., 1]
    half_obj = torch.polar(torch.sqrt(amplitude), phase / 2.0).contiguous()

    # ==========================================
    # 2. Precompute Fresnel Propagation Kernels
    # ==========================================
    z_idx = torch.arange(Nz, device=H.device).view(1, Nz, 1, 1)
    H_view = H.view(H.shape[0], 1, Ny, Nx)
    kernel_fwd = H_view.pow(z_idx + 1)
    kernel_inv = H_view.conj().pow(z_idx)

    # ==========================================
    # 3. Precompute Scattering Error (E1)
    # ==========================================
    shifted_half_obj = torch.roll(half_obj, shifts=1, dims=2)
    error_coeff = 1.0 - (half_obj * shifted_half_obj)

    # Enforce boundary
    boundary_mask = torch.ones((1, 1, Nz, 1, 1), device=kernel_fwd.device)
    boundary_mask[:, :, 0, :, :] = 0.0
    error_kern = (error_coeff * boundary_mask).unsqueeze(1)

    # ==========================================
    # 4. Bidiagonal Solver Operator (M^-1)
    # ==========================================
    def apply_m_inv(psi_spatial: torch.Tensor) -> torch.Tensor:
        psi_k = fft2(psi_spatial)
        # Kernels are already precomputed! Just unsqueeze and multiply.
        k_inv_6d = kernel_inv.unsqueeze(1).unsqueeze(1)
        k_fwd_6d = kernel_fwd.unsqueeze(1).unsqueeze(1)

        psi_k = torch.cumsum(psi_k * k_inv_6d, dim=3) * k_fwd_6d
        return ifft2(psi_k)

    # ==========================================
    # 5. Initial Source Propagation
    # ==========================================
    half_obj_entrance = half_obj[:, None, :, 0:1, :, :]
    source_spatial = probe.unsqueeze(2).unsqueeze(3) * half_obj_entrance
    source_k = fft2(source_spatial)
    u_sol = ifft2(source_k * kernel_fwd.unsqueeze(1).unsqueeze(1))

    # ==========================================
    # 6. Richardson Iteration Solver
    # ==========================================
    for _ in range(n_iter):
        scattered_field = torch.roll(u_sol, shifts=1, dims=3) * error_kern
        u_sol = u_sol - apply_m_inv(scattered_field)

    # ==========================================
    # 7. Final Split-Step & Exit Wave
    # ==========================================
    u_sol = u_sol * half_obj.unsqueeze(1)
    psi_exit = u_sol[:, :, :, -1, :, :].contiguous()

    # ==========================================
    # 8. Detector Measurement
    # ==========================================
    exit_intensity_k = (fftshift2(fft2(psi_exit, norm="ortho"))).abs().square()
    dp_fwd = torch.sum(exit_intensity_k * omode_occu[:, None, None], dim=(1, 2)) + eps

    return dp_fwd

#@torch.compile(mode="max-autotune")
def multislice_forward_model_vec_all_born(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    omode_occu: torch.Tensor = None,
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
        H (torch.Tensor): Tensor of shape (N, Ky, Kx) or (1, Ky, Kx) with complex64 values, representing the Fresnel
            propagator that propagates the wave by a slice thickness.
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
    H = H.contiguous()

    N_batch, omode, Nz, Ny, Nx, _ = object_patches.shape

    if omode_occu is None:
        omode_occu = (
            torch.ones(omode, dtype=object_patches.dtype, device=object_patches.device)
            / omode
        )

    # ==========================================
    # 1. Precompute Nilpotent Scattering (Object)
    # ==========================================
    amplitude = object_patches[..., 0]
    phase = object_patches[..., 1]
    
    # In the Bidiagonal formulation, obj acts as the spatial scattering potential
    obj = 1.0 - torch.polar(torch.sqrt(amplitude), phase).contiguous()
    
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
            Psi_n_M_hat = torch.sum(W_hat * kernel_inv, dim=3, keepdim=True) * k_fwd_exit

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

    return dp_fwd
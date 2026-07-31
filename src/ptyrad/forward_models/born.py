"""
Physical forwad model that generates diffraction patterns from mixed-state probe/object in a fully vectorized way

"""

# The forward model takes a batch of object patches and probes with their mixed states
# By introducing and aligning the singleton dimensions carefully,
# we can vectorize all the operations except the serial z-dimension propagation
# For 3D object with n_slices, the for loop would go through n-1 loops and multiply the last slice without further Fresnel propagaiton
# This way we can skip the if statement and make it slightly faster
# For 2D object (n_slices = 1), the entire for loop is skipped
# Note that element-wise multiplication of tensor (*) is defaulted as out-of-place operation
# So new tensor is being created and referenced to the old graph to keep the gradient flowing

import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift
from ptyrad.utils import fftshift2, ifftshift2


@torch.compile(mode="max-autotune")
def firstborn_forward(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    omode_occu: torch.Tensor = None,
    eps: float = 1e-10,
    linearise_obj: bool = True,
) -> torch.Tensor:
    """
    Fully Vectorized First-Born Forward Model.
    """
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()

    _, omode, _, Ny, Nx, _ = object_patches.shape

    if omode_occu is None:
        omode_occu = (
            torch.ones(omode, dtype=object_patches.dtype, device=object_patches.device) / omode
        )

    norm_weight = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)

    # ============================================
    # 1. Compute Probe in K-Space
    # ============================================
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)

    # ==========================================
    # 2. 0th Order Spatial Field
    # ==========================================
    Psi_state_active = ifft2(H * probe_k)

    # ==========================================
    # 3. Object Potential (Scattering Perturbation)
    # ==========================================
    amplitude = object_patches[..., 0]
    phase = object_patches[..., 1]

    if linearise_obj:
        # Linearised weak-phase/weak-amplitude approximation: (A - 1) + iφ
        real_imag_stacked = torch.stack([amplitude - 1.0, phase], dim=-1).contiguous()
        obj_active = torch.view_as_complex(real_imag_stacked).unsqueeze(1)
    else:
        # Exact polar representation: (A * e^(iφ) - 1)
        obj_active = (torch.polar(amplitude, phase) - 1.0).unsqueeze(1)

    # ==========================================
    # 4. Scattering and Detector Measurement
    # ==========================================
    scattered_k_sum = torch.sum(fft2(obj_active * Psi_state_active) * H.conj(), dim=3)

    # ==========================================
    # 5. Intensity Reduction
    # ==========================================
    dp_fwd = fftshift2(
        torch.sum((probe_k.squeeze(3) + scattered_k_sum).abs().square() * norm_weight, dim=(1, 2))
        + eps
    )

    return dp_fwd


def fftshift2(x: torch.Tensor) -> torch.Tensor:
    return fftshift(x, dim=(-2, -1))


def ifftshift2(x: torch.Tensor) -> torch.Tensor:
    return ifftshift(x, dim=(-2, -1))


class FirstBornForwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, object_patches, probe, H, omode_occu, eps, linearise_obj):
        object_patches = object_patches.contiguous()
        probe = probe.contiguous()

        _, omode, _, Ny, Nx, _ = object_patches.shape

        if omode_occu is None:
            omode_occu = (
                torch.ones(omode, dtype=object_patches.dtype, device=object_patches.device) / omode
            )

        omode_weight = omode_occu.view(1, 1, -1, 1, 1)
        norm_weight = omode_weight / (Nx * Ny)

        # 1. Compute Probe in K-Space
        probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)

        # 2. 0th Order Spatial Field
        Psi_state_active = ifft2(H * probe_k)

        # 3. Object Potential (Scattering Perturbation)
        amplitude = object_patches[..., 0]
        phase = object_patches[..., 1]

        if linearise_obj:
            real_imag_stacked = torch.stack([amplitude - 1.0, phase], dim=-1).contiguous()
            obj_active = torch.view_as_complex(real_imag_stacked).unsqueeze(1)
        else:
            obj_active = (torch.polar(amplitude, phase) - 1.0).unsqueeze(1)

        # 4. Scattering and Detector Measurement
        fft_obj_psi = fft2(obj_active * Psi_state_active)
        scattered_k_sum = torch.sum(fft_obj_psi * H.conj(), dim=3)

        # 5. Intensity Reduction
        Psi_hat_k = probe_k.squeeze(3) + scattered_k_sum
        I_k = Psi_hat_k.abs().square() * norm_weight
        dp_fwd = fftshift2(torch.sum(I_k, dim=(1, 2)) + eps)

        ctx.save_for_backward(object_patches, probe, H, omode_weight, Psi_hat_k, Psi_state_active)
        ctx.eps = eps
        ctx.linearise_obj = linearise_obj
        ctx.Ny = Ny
        ctx.Nx = Nx

        return dp_fwd

    @staticmethod
    def backward(ctx, grad_output):
        object_patches, probe, H, omode_weight, Psi_hat_k, Psi_state_active = ctx.saved_tensors
        linearise_obj = ctx.linearise_obj
        Ny, Nx = ctx.Ny, ctx.Nx
        N = Ny * Nx

        amplitude = object_patches[..., 0]
        phase = object_patches[..., 1]

        if linearise_obj:
            real_imag_stacked = torch.stack([amplitude - 1.0, phase], dim=-1).contiguous()
            obj_active = torch.view_as_complex(real_imag_stacked).unsqueeze(1)
        else:
            obj_active = (torch.polar(amplitude, phase) - 1.0).unsqueeze(1)

        # 1-3. Detector Plane to K-Space
        grad_dp_unshifted = ifftshift2(grad_output)
        grad_I_k = grad_dp_unshifted.unsqueeze(1).unsqueeze(2) * omode_weight
        grad_Psi_hat = grad_I_k * Psi_hat_k
        grad_probe_k_dir = grad_Psi_hat.unsqueeze(3)

        # 4. First-Born Scattering Sum Adjoint
        grad_scattered_unsqueezed = grad_Psi_hat.unsqueeze(3)
        grad_fft_obj_Psi = grad_scattered_unsqueezed * H

        # 5. Adjoint of fft2
        grad_obj_Psi_unnorm = ifft2(grad_fft_obj_Psi)

        # 6. Branch gradients to Object Potential and 0th Order Field
        grad_obj_active_unsqueezed = grad_obj_Psi_unnorm * Psi_state_active.conj()
        grad_Psi_state_unnorm = grad_obj_Psi_unnorm * obj_active.conj()

        # 7. Evaluate Object Parameter Gradients
        grad_obj_active = grad_obj_active_unsqueezed.sum(dim=1)

        if linearise_obj:
            grad_amplitude = grad_obj_active.real
            grad_phase = grad_obj_active.imag
        else:
            u = grad_obj_active.real
            v = grad_obj_active.imag
            cos_phi = torch.cos(phase)
            sin_phi = torch.sin(phase)

            grad_amplitude = u * cos_phi + v * sin_phi
            grad_phase = amplitude * (v * cos_phi - u * sin_phi)

        grad_object_patches = torch.stack([grad_amplitude, grad_phase], dim=-1)

        # 8. Backprop through the 0th Order Spatial Field creation
        grad_H_probe_k = fft2(grad_Psi_state_unnorm)
        grad_probe_k_indir = grad_H_probe_k * H.conj()

        # 9. Consolidate Probe Gradients
        grad_probe_k_dir += grad_probe_k_indir.sum(dim=3, keepdim=True)
        grad_probe_k = grad_probe_k_dir.view(-1, probe.shape[1], Ny, Nx)

        # Initial probe_k = fft2(probe) -> adjoint is ifft2
        grad_probe = ifft2(grad_probe_k)

        if grad_probe.shape[0] != probe.shape[0]:
            grad_probe = grad_probe.sum(dim=0, keepdim=True)

        grad_probe = grad_probe.reshape(probe.shape)

        return grad_object_patches, grad_probe, None, None, None, None


@torch.compile(mode="max-autotune")
def firstborn_forward_analytical(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    omode_occu: torch.Tensor = None,
    eps: float = 1e-10,
    linearise_obj: bool = True,
) -> torch.Tensor:
    """
    Fully Vectorized First-Born Forward Model with Optimized Analytical Custom Autograd.
    """
    return FirstBornForwardFunction.apply(object_patches, probe, H, omode_occu, eps, linearise_obj)


@torch.compile(mode="max-autotune")
def born_forward(
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

    _, omode, _, Ny, Nx, _ = object_patches.shape

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

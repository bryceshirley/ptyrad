"""
Physical forward model that generates diffraction patterns from mixed-state probe/object
in a fully vectorized way using Stochastic Block Coordinate Descent (SBCD).
"""

import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift
from ptyrad.utils import fftshift2, ifftshift2


# =============================================================================
# Helper functions for SBCD
# =============================================================================


@torch.compile(mode="default")
def prepare_object_complex(
    object_tensor: torch.Tensor,
    linearise_obj: bool = True,
) -> torch.Tensor:
    """Helper function to convert [amplitude, phase] patch tensor to complex potential perturbation (O - 1)."""
    amplitude = object_tensor[..., 0]
    phase = object_tensor[..., 1]

    if linearise_obj:
        real_imag_stacked = torch.stack([amplitude - 1.0, phase], dim=-1).contiguous()
        return torch.view_as_complex(real_imag_stacked).unsqueeze(1)
    else:
        return (torch.polar(amplitude, phase) - 1.0).unsqueeze(1)

@torch.compile(mode="default")
def detector(
    Psi_total_k: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute detector intensity projection from total field in Fourier space."""
    return fftshift2(torch.sum(Psi_total_k.abs().square() * norm_weight, dim=(1, 2)) + eps)

@torch.compile(mode="default")
def _compiled_probe_forward_core(
    obj_active_detached: torch.Tensor,
    probe: torch.Tensor,
    H_3d: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Pure autograd graph execution for probe optimization (No graph breaks)."""
    Ny, Nx = probe.shape[-2], probe.shape[-1]
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)

    # Compute slice illuminations (Gradients flow back to probe_k through H_3d)
    Psi_state_active = ifft2(H_3d * probe_k)

    # Total scattered field summed across all z-slices
    u_stack = fft2(obj_active_detached * Psi_state_active) * H_3d.conj()
    U_total = u_stack.sum(dim=3)

    # Total field interference & detector intensity projection
    Psi_total_k = probe_k.squeeze(3) + U_total
    return detector(Psi_total_k, norm_weight, eps)

# =============================================================================
# Forward Pass for SBCD
# =============================================================================

def stochastic_born_probe_forward(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H_3d: torch.Tensor,
    omode_occu: torch.Tensor = None,
    eps: float = 1e-10,
    linearise_obj: bool = True,
) -> torch.Tensor:
    """Evaluates the 3D forward pass for probe updates while keeping object detached."""
    _, omode, Nz, Ny, Nx, _ = object_patches.shape

    if omode_occu is None:
        omode_occu = (
            torch.ones(omode, dtype=object_patches.dtype, device=object_patches.device) / omode
        )

    norm_weight = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)

    # Explicitly detach object outside compiled autograd context to avoid graph breaks
    with torch.no_grad():
        obj_active_detached = prepare_object_complex(object_patches, linearise_obj)

    return _compiled_probe_forward_core(obj_active_detached, probe, H_3d, norm_weight, eps)


@torch.compile(mode="default")
@torch.no_grad()
def stochastic_born_components(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H_3d: torch.Tensor,
    omode_occu: torch.Tensor = None,
    eps: float = 1e-10,
    linearise_obj: bool = True,
):
    """Phase 2: Pre-compute detached background wavefield stack required for SBCD."""
    omode = object_patches.shape[1]
    Ny, Nx = probe.shape[-2], probe.shape[-1]

    if omode_occu is None:
        omode_occu = (
            torch.ones(omode, dtype=object_patches.dtype, device=object_patches.device) / omode
        )
    norm_weight = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)

    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    Psi_state_active = ifft2(H_3d * probe_k)
    obj_active = prepare_object_complex(object_patches, linearise_obj)
    u_stack = fft2(obj_active * Psi_state_active) * H_3d.conj()

    U_total = u_stack.sum(dim=3)
    Psi_total_k = probe_k.squeeze(3) + U_total
    dp_fwd = detector(Psi_total_k, norm_weight, eps)

    return (
        dp_fwd,
        probe_k.squeeze(3).detach(),
        Psi_state_active.detach(),
        u_stack.detach()
    )


@torch.compile(mode="default", dynamic=True)
def stochastic_born_forward_block(
    obj_block: torch.Tensor,
    Psi_block: torch.Tensor,
    H_3d_block: torch.Tensor,
    cache_k: torch.Tensor,
    omode_occu: torch.Tensor = None,
    eps: float = 1e-10,
    linearise_obj: bool = True,
):
    """Block Forward pass computing multi-slice wavefields and output intensity patterns."""
    _, omode, N_block, Ny, Nx, _ = obj_block.shape

    if omode_occu is None:
        omode_occu = torch.ones(omode, dtype=obj_block.dtype, device=obj_block.device) / omode

    norm_weight = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)

    # 1. Compute wavefield for active block slices
    obj_active = prepare_object_complex(obj_block, linearise_obj)
    u_block = fft2(obj_active * Psi_block) * H_3d_block.conj()

    # 2. Sum scattering contribution within active block & apply interference
    block_u_sum = u_block.sum(dim=3)
    Psi_hat_k = cache_k + block_u_sum

    # 3. Detector intensity projection
    dp_fwd = detector(Psi_hat_k, norm_weight, eps)

    return dp_fwd


@torch.compile(mode="default", dynamic=True)
def stochastic_born_single_block_u(
    obj_block: torch.Tensor,
    Psi_block: torch.Tensor,
    H_3d_block: torch.Tensor,
    linearise_obj: bool = True,
) -> torch.Tensor:
    """Compute scattered wavefields (u_block) for a specific active block of Z-slices."""
    obj_active = prepare_object_complex(obj_block, linearise_obj)
    return fft2(obj_active * Psi_block) * H_3d_block.conj()

# =============================================================================
# ANALYTICAL GRADIENT CALCULATION FOR SBCD
# =============================================================================

@torch.compile(mode="default")
def stochastic_born_analytical_block_grad(
    residual: torch.Tensor,
    Psi_hat_k: torch.Tensor,
    Psi_block: torch.Tensor,
    H_3d_block: torch.Tensor,
    object_block: torch.Tensor,
    omode_occu: torch.Tensor = None,
    linearise_obj: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    Ny, Nx = residual.shape[-2], residual.shape[-1]
    omode = object_block.shape[1]

    if omode_occu is None:
        omode_occu = torch.ones(omode, dtype=residual.dtype, device=residual.device) / omode

    # 1. Weight matrix matching detector projection: [1, 1, O_modes, 1, 1]
    norm_weight = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)

    # 2. Shift loss derivative back from detector space: [B, 1, 1, Y, X]
    grad_dp = ifftshift2(residual).unsqueeze(1).unsqueeze(2)

    # 3. Wirtinger Derivative of Intensity w.r.t Psi_hat_k: [B, P_modes, O_modes, Y, X]
    d_Psi_k = 2.0 * Psi_hat_k * norm_weight * grad_dp

    # 4. Backprop through propagator H_3d_block: [B, P_modes, O_modes, N_block, Y, X]
    d_u_block = d_Psi_k.unsqueeze(3) * H_3d_block

    # 5. Backprop through Inverse FFT: [B, P_modes, O_modes, N_block, Y, X]
    d_spatial = ifft2(d_u_block) * (Nx * Ny)

    # 6. Backprop through object-probe multiplication
    grad_O_complex = (d_spatial * Psi_block.conj()).sum(dim=1)

    # 7. Convert complex potential gradient to real [amplitude, phase] gradients
    if linearise_obj:
        grad_obja = grad_O_complex.real
        grad_objp = grad_O_complex.imag
    else:
        amp = object_block[..., 0]
        phase = object_block[..., 1]

        grad_obja = (grad_O_complex * torch.exp(torch.complex(torch.zeros_like(phase), -phase))).real
        # Fixed mathematical sign error in the imaginary component for polar exponentiation
        grad_objp = (grad_O_complex * torch.complex(-amp * torch.sin(phase), -amp * torch.cos(phase))).real

    return grad_obja, grad_objp

@torch.compile(mode="default")
def stochastic_born_analytical_probe_grad(
    residual: torch.Tensor,
    Psi_hat_k: torch.Tensor,
    H_3d: torch.Tensor,
    object_patches: torch.Tensor,
    omode_occu: torch.Tensor = None,
    linearise_obj: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    Ny, Nx = residual.shape[-2], residual.shape[-1]
    omode = object_patches.shape[1]

    if omode_occu is None:
        omode_occu = torch.ones(omode, dtype=residual.dtype, device=residual.device) / omode

    # Weight matrix matching detector projection: [1, 1, O_modes, 1, 1]
    norm_weight = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)

    # 1. Backprop through Detector (Shift and Weight)
    grad_dp = ifftshift2(residual).unsqueeze(1).unsqueeze(2)

    # D(k) - The total field gradient [Batch, P_modes, O_modes, Y, X]
    d_Psi_k = 2.0 * Psi_hat_k * norm_weight * grad_dp

    # 2. Backprop scattered waves through free-space propagator and inverse FFT
    d_u_z = d_Psi_k.unsqueeze(3) * H_3d
    d_spatial = ifft2(d_u_z) * (Nx * Ny)

    # 3. Multiply by Object conjugate
    obj_active = prepare_object_complex(object_patches, linearise_obj)
    grad_incident = d_spatial * obj_active.conj()

    # 4. Propagate incident gradient back to k-space probe
    d_probe_k_scattered = fft2(grad_incident) / (Nx * Ny) * H_3d.conj()
    
    # 5. Sum across object slices and modes, and add the direct unscattered path D(k)
    d_probe_k_total = d_Psi_k.sum(dim=2) + d_probe_k_scattered.sum(dim=(2, 3))
    
    # 6. Backprop into real-space probe, sum across batch instances
    grad_probe_spatial = (ifft2(d_probe_k_total) * (Nx * Ny)).sum(dim=0)
    
    # Return as real tensor to match opt_probe shape [P_modes, Ny, Nx, 2]
    return torch.view_as_real(grad_probe_spatial).contiguous()
    
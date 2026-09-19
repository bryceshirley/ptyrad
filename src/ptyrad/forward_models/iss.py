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
from torch.fft import fft2, ifft2

from ptyrad.utils import fftshift2, ifftshift2


@torch.compile(mode="max-autotune")
def iss_forward(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    omode_occu: torch.Tensor | None = None,
    eps: float = 1e-10,
    linearise_obj: bool = False,
) -> torch.Tensor:
    """
    Fully Vectorized ISS Forward Model.
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


class ISSLowMemFunction(torch.autograd.Function):
    """Slice-looped ISS with a low-memory hand adjoint.

    The parallel formulation (iss_forward + autograd) materialises
    (B, pmode, omode, Nz, Ny, Nx)
    intermediates in forward and backward, so peak memory grows as
    O(batch x slices). Here both passes loop over slices: the only stored
    per-slice quantity is the unscattered illumination phi_j = IFFT[H_j FFT P]
    — ONE field per slice, batch-free when the probe is shared — plus the
    O(batch) exit field. Peak workspace is O(batch) + O(slices), matching the
    low-memory Born of the ptypy reference engine.

    Gradients follow torch's convention (z.grad = 2 dL/dz*) and are validated
    against autograd of iss_forward in test/test_iss_lowmem.py.
    (The former iss_forward_analytical / ISSForwardFunction hand adjoint was
    removed: it failed that check — object gradient exactly half, probe
    gradient mishandling the omode dimension — and had no callers.)
    """

    @staticmethod
    def forward(
        ctx,
        object_patches,
        probe,
        H,
        omode_occu=None,
        eps=1e-10,
        linearise_obj=False,
        slice_chunk=4,
    ):
        object_patches = object_patches.contiguous()
        probe = probe.contiguous()
        B, omode, Nz, Ny, Nx, _ = object_patches.shape

        if omode_occu is None:
            omode_occu = (
                torch.ones(omode, dtype=object_patches.dtype, device=object_patches.device) / omode
            )
        omode_weight = omode_occu.view(1, 1, -1, 1, 1)
        C = max(int(slice_chunk), 1)

        probe_k = fft2(probe)  # (Bp, pmode, Ny, Nx)
        # phi: one unscattered field per slice, batch-free for a shared probe
        Hz = H[:, 0, 0]  # (Bh, Nz, Ny, Nx)
        phi = ifft2(Hz.unsqueeze(1) * probe_k.unsqueeze(2))  # (Bmax, pmode, Nz, Ny, Nx)

        amplitude = object_patches[..., 0]
        phase = object_patches[..., 1]
        acc = None
        # chunked slice loop: workspace O(batch x C); C trades the loop's
        # launch/traffic overhead against memory (C = Nz ~ the parallel model)
        for j0 in range(0, Nz, C):
            sl = slice(j0, min(j0 + C, Nz))
            if linearise_obj:
                obj_c = torch.complex(amplitude[:, :, sl] - 1.0, phase[:, :, sl])
            else:
                obj_c = torch.polar(amplitude[:, :, sl], phase[:, :, sl]) - 1.0
            # (B, 1, omode, C, Ny, Nx) * (Bp, pmode, 1, C, Ny, Nx)
            term = fft2(obj_c.unsqueeze(1) * phi[:, :, None, sl]) * Hz[:, None, None, sl].conj()
            term = term.sum(dim=3)  # (B, pmode, omode, Ny, Nx)
            acc = term if acc is None else acc + term

        Psi_hat_k = probe_k.unsqueeze(2) + acc  # (B, pmode, omode, Ny, Nx)
        norm_weight = omode_weight / (Nx * Ny)
        dp_fwd = fftshift2(torch.sum(Psi_hat_k.abs().square() * norm_weight, dim=(1, 2)) + eps)

        ctx.save_for_backward(object_patches, probe, H, omode_weight, Psi_hat_k, phi)
        ctx.linearise_obj = linearise_obj
        ctx.slice_chunk = C
        return dp_fwd

    @staticmethod
    def backward(ctx, grad_output):  # ty: ignore[invalid-method-override]  # autograd.Function convention
        object_patches, probe, H, omode_weight, Psi_hat_k, phi = ctx.saved_tensors
        linearise_obj = ctx.linearise_obj
        B, omode, Nz, Ny, Nx, _ = object_patches.shape
        Hz = H[:, 0, 0]  # (Bh, Nz, Ny, Nx)

        amplitude = object_patches[..., 0]
        phase = object_patches[..., 1]

        # torch convention seed on the exit field: z.grad = 2 dL/dz*; the
        # 1/(Nx*Ny) of the intensity normalisation cancels against the
        # unnormalised-FFT adjoints below, so it is deliberately absent here.
        Wn = (
            2.0 * ifftshift2(grad_output).unsqueeze(1).unsqueeze(2) * omode_weight * Psi_hat_k
        )  # (B, pmode, omode, Ny, Nx)
        C = ctx.slice_chunk

        pk_acc = Wn.sum(dim=2)  # direct probe path, summed over omode
        grad_amp = torch.empty_like(amplitude)
        grad_phs = torch.empty_like(phase)
        Wn4 = Wn.unsqueeze(3)  # (B, pmode, omode, 1, Ny, Nx)
        for j0 in range(0, Nz, C):
            sl = slice(j0, min(j0 + C, Nz))
            T = ifft2(Hz[:, None, None, sl] * Wn4)  # (B, pmode, omode, C, Ny, Nx)
            og = (phi[:, :, None, sl].conj() * T).sum(dim=1)  # (B, omode, C, Ny, Nx)
            if linearise_obj:
                obj_c = torch.complex(amplitude[:, :, sl] - 1.0, phase[:, :, sl])
                grad_amp[:, :, sl] = og.real
                grad_phs[:, :, sl] = og.imag
            else:
                obj_c = torch.polar(amplitude[:, :, sl], phase[:, :, sl]) - 1.0
                u, v = og.real, og.imag
                cos_p = torch.cos(phase[:, :, sl])
                sin_p = torch.sin(phase[:, :, sl])
                grad_amp[:, :, sl] = u * cos_p + v * sin_p
                grad_phs[:, :, sl] = amplitude[:, :, sl] * (v * cos_p - u * sin_p)
            # indirect probe path through phi_j (object broadcast over pmode,
            # phi broadcast over omode -> sum omode; then sum the chunk)
            phig = (obj_c.unsqueeze(1).conj() * T).sum(dim=2)  # (B, pmode, C, Ny, Nx)
            pk_acc = pk_acc + (Hz[:, None, sl].conj() * fft2(phig)).sum(dim=2)

        grad_object_patches = torch.stack([grad_amp, grad_phs], dim=-1)
        grad_probe = ifft2(pk_acc)  # (B, pmode, Ny, Nx)
        if grad_probe.shape[0] != probe.shape[0]:
            grad_probe = grad_probe.sum(dim=0, keepdim=True)
        return (grad_object_patches, grad_probe.reshape(probe.shape), None, None, None, None, None)


def iss_forward_lowmem(
    object_patches, probe, H, omode_occu=None, eps=1e-10, linearise_obj=False, slice_chunk=4
):
    """Low-memory ISS: O(batch x slice_chunk) + O(slices) peak
    workspace (see ISSLowMemFunction). slice_chunk trades the slice
    loop's launch/traffic overhead against memory: 1 = minimum memory,
    larger chunks approach the parallel model's wall clock (at large batch
    the per-chunk FFTs already saturate the GPU, so a modest chunk closes
    most of the gap). Same interface and output as iss_forward."""
    return ISSLowMemFunction.apply(
        object_patches, probe, H, omode_occu, eps, linearise_obj, slice_chunk
    )

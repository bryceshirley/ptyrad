"""Parallel Born-series forward model for multislice ptychography.

The Born series is a combinatorial expansion of the multislice operator, and
the parallel formulation replaces the sequential propagation loop with a
parallel prefix sum (`cumsum`) over the slices. The series is nilpotent
across slices, so it terminates exactly at the number of slices, and higher
`n_max` adds nothing. The recursion primitives and the residual-minimizing
(GMRES) coefficient machinery live in born_helpers (born_krylov_gram /
born_gmres_coeffs); the coefficients passed in here only reweight the
per-order detector sum.
"""

import torch

from ptyrad.utils import fftshift2

from .born_helpers import _born_advance, _born_init, _born_scatter


@torch.compile(mode="max-autotune")
def born_forward(
    object_patches: torch.Tensor,
    probe: torch.Tensor,
    H: torch.Tensor,
    omode_occu: torch.Tensor,
    eps: float = 1e-10,
    n_max: int = 1,
    coeffs: torch.Tensor | None = None,
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
        H (torch.Tensor): Tensor of shape (N|1, 1, 1, Nz, Ny, Nx) with complex64 values —
            the stack of Fresnel propagator powers H^j (entrance to slice j), identical to
            the H accepted by iss_forward. The detector-path kernel is derived internally
            as its conjugate (the vacuum propagator is unimodular).
        omode_occu (torch.Tensor): Tensor of shape (omode,) with float32 values.
        eps (float, optional): A small value added for numerical stability. Defaults to 1e-10.
        n_max (int): Maximum order of the Born series iterations (orders of scattering).
        coeffs (torch.Tensor, optional): Tensor of shape (n_max, 2) with float32 values,
            a pseudo-complex (real, imag) coefficient per scattering order. The detector
            field becomes Psi_0 + sum_n c_n D_n, so coeffs full of (1, 0) reproduces the
            plain series exactly. The coefficients reweight only the detector sum — the
            internal recursion stays the plain series, so each D_n remains the true
            n-th order field. Register as an nn.Parameter to tune the coefficients by
            autograd during data fitting alongside the object and probe. For a coefficient
            update that stays identifiable against the object scale, freeze c_1 at (1, 0)
            and optimize only the top order(s). Defaults to None (plain series, and no
            per-order detector fields are retained for a coefficient gradient).

    Returns:
        torch.Tensor: Tensor of shape (N, Ny, Nx) with float32 positive values, representing the
        forward diffraction pattern for each sample in the batch.
    """
    # ---------------------------------------------------------------
    # 1. Setup: contiguity and mode weights
    # ---------------------------------------------------------------
    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    kernel_fwd = H.contiguous()

    _, omode, Nz, Ny, Nx, _ = object_patches.shape

    if omode_occu is None:
        omode_occu = (
            torch.ones(omode, dtype=object_patches.dtype, device=object_patches.device) / omode
        )
    norm_weight = omode_occu / (Nx * Ny)

    # ---------------------------------------------------------------
    # 2. Scattering potential, 0th order field and detector wave
    # ---------------------------------------------------------------
    obj, probe_k, Psi_state_active = _born_init(object_patches, probe, kernel_fwd)
    Psi_M_hat = probe_k.squeeze(3)  # 0th order detector wave

    # ---------------------------------------------------------------
    # 3. Born series loop: scatter -> advance, reweight detector orders
    # ---------------------------------------------------------------
    n_orders = min(n_max, Nz)  # nilpotent termination
    for n in range(n_orders):  # n = order - 1 = first allowed scattering site
        scattered_k = _born_scatter(obj, Psi_state_active, kernel_fwd, n)
        if n < n_orders - 1:
            Psi_state_active, D_n = _born_advance(scattered_k, kernel_fwd, n)
        else:
            # final order: no next bounce
            D_n = torch.sum(scattered_k, dim=3)
        if coeffs is None:
            Psi_M_hat = Psi_M_hat + D_n
        else:
            Psi_M_hat = Psi_M_hat + torch.complex(coeffs[n, 0], coeffs[n, 1]) * D_n

    # ---------------------------------------------------------------
    # 4. Detector measurement (incoherent sum over modes)
    # ---------------------------------------------------------------
    return fftshift2(
        torch.sum(Psi_M_hat.abs().square() * norm_weight.view(1, 1, -1, 1, 1), dim=(1, 2)) + eps
    )

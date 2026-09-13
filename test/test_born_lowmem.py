"""Gradient parity of the low-memory first-Born adjoint against autograd.

FirstBornLowMemFunction must reproduce autograd of firstborn_forward to
float64 precision — forward values, object gradients, and probe gradients —
including omode > 1 with unequal occupancies (the case that exposes the
omode-scrambling bug in the older FirstBornForwardFunction), shared and
per-view probes, and both object parameterisations.
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import pytest
import torch

torch._dynamo.config.disable = True

from ptyrad.forward_models import firstborn_forward
from ptyrad.forward_models.born import firstborn_forward_lowmem


def _setup(B, omode, Nz, Ny, Nx, pmode, Bp, seed=0):
    g = torch.Generator().manual_seed(seed)
    patches = torch.rand(B, omode, Nz, Ny, Nx, 2, generator=g, dtype=torch.float64)
    patches[..., 0] = 1.0 + 0.2 * (patches[..., 0] - 0.5)
    patches[..., 1] *= 0.4
    probe = torch.randn(Bp, pmode, Ny, Nx, generator=g, dtype=torch.float64) \
        + 1j * torch.randn(Bp, pmode, Ny, Nx, generator=g, dtype=torch.float64)
    ky = torch.fft.fftfreq(Ny, dtype=torch.float64)
    kx = torch.fft.fftfreq(Nx, dtype=torch.float64)
    H1 = torch.exp(-1j * 0.3 * (ky[:, None] ** 2 + kx[None, :] ** 2) * Ny * Nx)
    H = (H1 ** torch.arange(Nz, dtype=torch.float64).view(Nz, 1, 1)).view(
        1, 1, 1, Nz, Ny, Nx)
    occu = torch.linspace(1.0, 0.5, omode, dtype=torch.float64)
    occu = occu / occu.sum()
    cot = torch.randn(B, Ny, Nx, generator=g, dtype=torch.float64)
    return patches, probe, H, occu, cot


@pytest.mark.parametrize("linearise", [False, True])
@pytest.mark.parametrize("Bp", [1, 3])
def test_lowmem_matches_autograd(linearise, Bp):
    patches, probe, H, occu, cot = _setup(3, 2, 4, 16, 16, 2, Bp)

    p1 = patches.clone().requires_grad_(True)
    pr1 = probe.clone().requires_grad_(True)
    dp1 = firstborn_forward(p1, pr1, H, occu, 1e-10, linearise)
    (dp1 * cot).sum().backward()

    p2 = patches.clone().requires_grad_(True)
    pr2 = probe.clone().requires_grad_(True)
    dp2 = firstborn_forward_lowmem(p2, pr2, H, occu, 1e-10, linearise)
    (dp2 * cot).sum().backward()

    assert torch.allclose(dp1, dp2, rtol=1e-12, atol=1e-14)
    assert torch.allclose(p1.grad, p2.grad, rtol=1e-9, atol=1e-12 * p1.grad.abs().max())
    assert torch.allclose(pr1.grad, pr2.grad, rtol=1e-9,
                          atol=1e-12 * pr1.grad.abs().max())


def test_lowmem_default_occupancy_and_float32():
    patches, probe, H, occu, cot = _setup(2, 1, 3, 16, 16, 2, 1, seed=1)
    patches, probe, H, cot = (patches.float(), probe.to(torch.complex64),
                              H.to(torch.complex64), cot.float())
    p1 = patches.clone().requires_grad_(True)
    pr1 = probe.clone().requires_grad_(True)
    dp1 = firstborn_forward(p1, pr1, H, None)
    (dp1 * cot).sum().backward()
    p2 = patches.clone().requires_grad_(True)
    pr2 = probe.clone().requires_grad_(True)
    dp2 = firstborn_forward_lowmem(p2, pr2, H, None)
    (dp2 * cot).sum().backward()
    assert torch.allclose(dp1, dp2, rtol=1e-5, atol=1e-8)
    assert torch.allclose(p1.grad, p2.grad, rtol=1e-4,
                          atol=1e-5 * p1.grad.abs().max())
    assert torch.allclose(pr1.grad, pr2.grad, rtol=1e-4,
                          atol=1e-5 * pr1.grad.abs().max())

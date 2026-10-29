#!/usr/bin/env python
"""Single-slice compile/smoke test for the full-B Chebyshev slab forward.
One slab, one object slice (Nz=1, groups=((0,1),)). Checks:
  1. kernel builds for a single slice;
  2. eager Chebyshev forward runs, finite, correct shape, grads flow;
  3. the CUDA-graph path (make_graphed_callables, the recon's PTYRAD_CHEB_GRAPH
     path) captures and replays bit-reproducibly, matching eager.
Run: PYTHONPATH=../src CUDA_VISIBLE_DEVICES=1 ~/ptyrad/.venv/bin/python cheb_singleslice_compile.py
"""
import torch

from ptyrad.forward_models import ChebyshevSlabKernel, multislice_forward_chebyshev

dev = "cuda:0"  # CUDA_VISIBLE_DEVICES pins the physical card
torch.manual_seed(0)
Ny = Nx = 256
Nz = 1
dx, lam, dz = 0.09359, 0.01969, 70.0
groups = ((0, 1),)          # single slab = the single slice
pmode = 4

objp = (0.02 * torch.randn(1, 1, Nz, Ny, Nx, device=dev)).float()
obja = torch.ones(1, 1, Nz, Ny, Nx, device=dev).float()
obj0 = torch.stack([obja, objp], dim=-1)
probe0 = torch.randn(1, pmode, Ny, Nx, dtype=torch.complex64, device=dev)

kern = ChebyshevSlabKernel((Ny, Nx), dx, lam, dz, groups, dev,
                           dtype=torch.complex64, tol=1e-4, verbose=True)
kern.refresh(objp)
mu_mul, band_mask, coeffs, lo, hi, k0dz, grp = kern.kernels()
print(f"single-slice Chebyshev half-slab degree {coeffs.shape[0]-1}, groups={grp}")

# 1. eager forward + grad
o = obj0.detach().clone().requires_grad_(True)
p = probe0.detach().clone().requires_grad_(True)
dp = multislice_forward_chebyshev(o, p, mu_mul, band_mask, coeffs, lo, hi, k0dz, grp)
dp.sum().backward()
print("eager shape:", tuple(dp.shape), "finite:", bool(torch.isfinite(dp).all()))
print("grad objp finite/nonzero:", bool(torch.isfinite(o.grad).all()),
      float(o.grad[..., 1].abs().sum()) > 0)
print("grad probe finite/nonzero:", bool(torch.isfinite(p.grad).all()),
      float(p.grad.abs().sum()) > 0)

# 2. CUDA-graph capture (recon's PTYRAD_CHEB_GRAPH path)
def fn(pp, pr):
    return multislice_forward_chebyshev(pp, pr, mu_mul, band_mask, coeffs, lo, hi, k0dz, grp)

sample = (obj0.detach().clone().requires_grad_(True),
          probe0.detach().clone().requires_grad_(True))
graphed = torch.cuda.make_graphed_callables(fn, sample)
g1 = graphed(obj0.detach().clone().requires_grad_(True),
             probe0.detach().clone().requires_grad_(True))
g2 = graphed(obj0.detach().clone().requires_grad_(True),
             probe0.detach().clone().requires_grad_(True))
eager = multislice_forward_chebyshev(obj0, probe0, mu_mul, band_mask, coeffs, lo, hi, k0dz, grp)
rel = (torch.linalg.vector_norm(g1 - eager) / torch.linalg.vector_norm(eager)).item()
print("CUDA graph captured; replay reproducible:", bool(torch.equal(g1, g2)),
      "finite:", bool(torch.isfinite(g1).all()))
print(f"graph vs eager rel-L2: {rel:.2e}")
print("OK")

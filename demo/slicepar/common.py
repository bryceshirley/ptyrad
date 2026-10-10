"""Shared problem builder for the slicepar carry-scheme study.

Synthetic but physically-shaped inputs: weak-phase object (amplitude ~1,
small phase), unimodular quadratic-phase propagator powers H^j, normalized
random probe. Batch N=1 (the PSO batch-1 regime), omode=1.

Shapes follow born_forward / born_forward_dist:
  object_patches (1, omode, Nz, Ny, Nx, 2)  pseudo-complex (amp, phase)
  probe          (1, pmode, Ny, Nx)         complex64
  H              (1, 1, 1, Nz, Ny, Nx)      complex64, H[j] = exp(i j phi)
"""

import math

import torch


def make_problem(Nz, n, pmode=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = 1.0 + 0.02 * torch.randn(1, 1, Nz, n, n, generator=g)
    p = 0.05 * torch.randn(1, 1, Nz, n, n, generator=g)
    pr = (torch.randn(1, pmode, n, n, generator=g)
          + 1j * torch.randn(1, pmode, n, n, generator=g))
    pr = pr / pr.abs().square().sum().sqrt()
    f = torch.fft.fftfreq(n)
    k2 = f[:, None] ** 2 + f[None, :] ** 2
    phi = -math.pi * 4.0 * k2  # bounded quadratic phase per hop
    j = torch.arange(Nz).view(-1, 1, 1)
    H = torch.exp(1j * j * phi).to(torch.complex64).view(1, 1, 1, Nz, n, n)
    occu = torch.ones(1)
    return a, p, pr.to(torch.complex64), H, occu


def even_bounds(Nz, P):
    base, b = Nz // P, [0]
    for i in range(P - 1):
        b.append(b[-1] + base + (1 if i < Nz % P else 0))
    b.append(Nz)
    return b


def build_dist(a, p, probe, H, bounds, devices, requires_grad=False):
    """Per-device leaves (a_k, p_k) + pre-resident block dicts.

    Returns (blocks, leaves) where leaves = list of (a_k, p_k); objc keeps
    the graph from the leaves when requires_grad=True.
    """
    blocks, leaves = [], []
    P = len(devices)
    for k in range(P):
        lo, hi = bounds[k], bounds[k + 1]
        dev = devices[k]
        ak = a[:, :, lo:hi].to(dev).requires_grad_(requires_grad)
        pk_ = p[:, :, lo:hi].to(dev).requires_grad_(requires_grad)
        leaves.append((ak, pk_))
        objc = (torch.polar(ak, pk_) - 1.0).unsqueeze(1)
        Hd = H[..., lo:hi, :, :].to(dev).contiguous()
        blocks.append(dict(objc=objc, H=Hd, Hc=Hd.conj().contiguous(),
                           probe=probe.to(dev), dev=dev))
    return blocks, leaves


def single_inputs(a, p, probe, H, device="cuda:0", requires_grad=False):
    """Leaves + object_patches for single-device born_forward."""
    a0 = a.to(device).requires_grad_(requires_grad)
    p0 = p.to(device).requires_grad_(requires_grad)
    op = torch.stack([a0, p0], dim=-1)
    return op, (a0, p0), probe.to(device), H.to(device)

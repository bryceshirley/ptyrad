#!/usr/bin/env python
"""
MICRO-BENCH ONLY — NOT the reconstruction protocol (which is batch 1).

Chebyshev 3-slab forward(+backward) wall per position vs batch size, on the
archived 21-slice PSO object patch (same specimen as
validate_chebyshev_forward.py), f32, fused path, optimised 2/3-Nyquist band.
Positions are replicated patches: identical math per position, so this
measures pure batching efficiency of the recurrence (FFT batching +
launch-overhead amortisation), nothing about reconstruction quality.

Run: CUDA_VISIBLE_DEVICES=<UUID> PYTHONPATH=src \
     ~/ptyrad/.venv/bin/python scripts/cheb_batch_microbench.py
"""

import glob

import h5py
import numpy as np
import torch

from ptyrad.forward_models.chebyshev import ChebyshevSlabKernel, multislice_forward_chebyshev
from ptyrad.utils.physics import get_EM_constants

ARCHIVE = sorted(
    glob.glob(
        "/home/dnz75396/ptyrad/demo/output/high_order_benchmarks/chin_results/"
        "20260927_lt21_ref_s42_full*/model_iter0100.hdf5"
    )
)[0]
NPATCH = 256
dev = torch.device("cuda")

with h5py.File(ARCHIVE, "r") as f:
    obja = f["optimizable_tensors/obja"][0]
    objp = f["optimizable_tensors/objp"][0]
    probe = f["optimizable_tensors/probe"][...]
    dx = float(f["model_attributes/dx"][...])
    dz = float(f["optimizable_tensors/slice_thickness"][...])
y0 = (obja.shape[-2] - NPATCH) // 2
obja = obja[:, y0 : y0 + NPATCH, y0 : y0 + NPATCH]
objp = objp[:, y0 : y0 + NPATCH, y0 : y0 + NPATCH]

lambd = get_EM_constants(300.0, "wavelength")
nz = objp.shape[0]
km = ChebyshevSlabKernel(
    (NPATCH, NPATCH), dx, lambd, 7 * dz, ((0, 7), (7, 14), (14, 21)),
    device=dev, dtype=torch.complex64,
)
ph_t = torch.tensor(objp, dtype=torch.float32, device=dev)
km.refresh(ph_t)
print(f"half-slab degree: {km.coeffs.shape[0] - 1}")

amp_t = torch.tensor(obja, dtype=torch.float32, device=dev)
probe_t = torch.tensor(probe, device=dev).to(torch.complex64)
patch1 = torch.stack([amp_t, ph_t], dim=-1)[None, None]  # (1,1,21,N,N,2)

print(f"{'batch':>6}{'fwd ms/pos':>12}{'fwd+bwd ms/pos':>16}")
for B in (1, 2, 4, 8, 16):
    patches = patch1.expand(B, -1, -1, -1, -1, -1).contiguous()
    pr = probe_t[None].expand(B, -1, -1, -1).contiguous()

    def run(backward):
        p = patches.clone().requires_grad_(backward)
        dp = multislice_forward_chebyshev(p, pr, *km.kernels())
        if backward:
            dp.sum().backward()

    out = []
    for backward in (False, True):
        for _ in range(3):
            run(backward)
        torch.cuda.synchronize()
        best = np.inf
        for _ in range(5):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            run(backward)
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
        out.append(best / B)
    print(f"{B:>6}{out[0]:>12.2f}{out[1]:>16.2f}")

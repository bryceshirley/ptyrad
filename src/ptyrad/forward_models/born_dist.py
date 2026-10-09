"""Multi-GPU depth-partitioned parallel Born series (two-level scan).

The slice axis of the parallel Born recursion is a prefix sum, so a depth
partition across P GPUs is a two-level scan. The naive ("ripple carry")
version serializes per order: block i's COMPUTE waits for block i-1's carry,
costing sum_i(compute_i) + (P-1) hops per order. But the carry enters the
local cumsum LINEARLY, so each order really has three phases:

  A. local scatter + scan on every block IN PARALLEL (no cross-GPU deps):
       sc_i = F[objc_i * psi_i] * conj(H_i);  cs_i = cumsum_z(sc_i);
       t_i = cs_i[..., -1, :, :]                      (local block total)
  B. carry combine, pure transfer+add along the chain (one 2D field/hop):
       p_1 = t_0 -> dev1;  p_{i+1} = (p_i + t_i) -> dev_{i+1}
       grand total on the last device = p_{P-1} + t_{P-1}  (detector D_n)
  C. psi update on every block IN PARALLEL:
       block 0:   psi_0 = F^-1[ cs_0[..., :-1] * H_0[..., win+1:] ]
       block i>0: psi_i = F^-1[ cat(p_i, cs_i[..., :-1] + p_i) * H_i ]

Per order the critical path drops from P*compute + (P-1)*hops interleaved to
max_i(compute_i) + (P-1) transfer-only hops. Phases A and C occupy all GPUs
simultaneously; the hops carry one (N, pmode, omode, Ny, Nx) field each and
overlap the tail of phase A on the upstream devices.

Blocks must be PRE-RESIDENT (object slices, H powers, probe replica placed on
their devices once per solve, not per call) — a depth-partitioned object also
keeps its gradients and optimizer state per-device, with zero per-step object
movement. Exactness vs the single-device `born_forward` is float-reassociation
level (~1e-6 forward, ~2e-4 object grad on the real PSO checkpoint).
"""

import torch
from torch.fft import fft2, ifft2

from ptyrad.utils import fftshift2


def make_blocks(objc_blocks, H_blocks, probe, devices):
    """Assemble pre-resident per-device block dicts.

    Args:
        objc_blocks: list of (N, 1, omode, nz_k, Ny, Nx) complex scattering
            potentials (O - 1), block k already on devices[k].
        H_blocks: list of (N|1, 1, 1, nz_k, Ny, Nx) propagator-power stacks,
            block k already on devices[k] (slice of the full H.pow(z) stack).
        probe: (N|1, pmode, Ny, Nx) complex probe on any device; replicated.
        devices: list of device strings, one per block.
    """
    blocks = []
    for oc, Hd, dev in zip(objc_blocks, H_blocks, devices):
        Hd = Hd.contiguous()
        blocks.append(dict(objc=oc, H=Hd, Hc=Hd.conj().contiguous(),
                           probe=probe.to(dev), dev=dev))
    return blocks


def born_forward_dist(blocks, omode_occu, n_max, coeffs=None, eps=1e-10):
    """Depth-partitioned parallel Born forward (two-level scan).

    Args:
        blocks: per-device block dicts from `make_blocks` (pre-resident).
        omode_occu: (omode,) occupancies ON THE LAST BLOCK'S DEVICE — a CPU
            tensor here would force a pageable H2D that blocks the host until
            the whole chain drains and serializes everything downstream.
        n_max: Born order M (nilpotent: effective order min(M, Nz)).
        coeffs: optional (M, 2) per-order detector reweights (pseudo-complex).
        eps: detector-intensity floor.

    Returns:
        (N, Ny, Nx) diffraction intensity on the last block's device.
    """
    P = len(blocks)
    Nz = sum(b["objc"].shape[3] for b in blocks)
    pmode = blocks[0]["probe"].shape[1]
    Ny, Nx = blocks[0]["probe"].shape[-2:]

    for b in blocks:
        pk = fft2(b["probe"]).view(-1, pmode, 1, 1, Ny, Nx)
        b["pk"] = pk
        b["psi"] = ifft2(b["H"] * pk)
    Psi_M = blocks[-1]["pk"].squeeze(3)

    c_all = None
    if coeffs is not None:
        c_all = torch.complex(coeffs[:, 0], coeffs[:, 1]).to(Psi_M.device)

    M = min(n_max, Nz)
    for n in range(M):
        last = n == M - 1
        # --- A: local scatter + scan, all blocks in parallel ---------------
        for i, b in enumerate(blocks):
            win0 = n if i == 0 else 0
            b["win0"] = win0
            sc = fft2(b["objc"][..., win0:, :, :] * b["psi"]) * b["Hc"][..., win0:, :, :]
            b["cs"] = torch.cumsum(sc, dim=3)
            b["t"] = b["cs"][..., -1, :, :]
        # --- B: carry combine, transfer + add only -------------------------
        prefix = None  # incoming prefix for block i, resident on its device
        for i in range(P - 1):
            total = blocks[i]["t"] if prefix is None else prefix + blocks[i]["t"]
            prefix = total.to(blocks[i + 1]["dev"], non_blocking=True)
            blocks[i + 1]["prefix"] = prefix
        D_n = blocks[-1]["t"] if prefix is None else prefix + blocks[-1]["t"]
        # --- C: psi update, all blocks in parallel --------------------------
        if not last:
            for i, b in enumerate(blocks):
                if i == 0:
                    b["psi"] = ifft2(b["cs"][..., :-1, :, :]
                                     * b["H"][..., b["win0"] + 1:, :, :])
                else:
                    p = b["prefix"].unsqueeze(3)
                    b["psi"] = ifft2(
                        torch.cat([p, b["cs"][..., :-1, :, :] + p], dim=3) * b["H"])
        Psi_M = Psi_M + D_n if c_all is None else Psi_M + c_all[n] * D_n

    nw = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(Psi_M.abs().square() * nw, dim=(1, 2)) + eps)

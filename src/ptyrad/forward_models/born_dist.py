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


def born_forward_dist(blocks, omode_occu, n_max, coeffs=None, eps=1e-10, carry="chain"):
    """Depth-partitioned parallel Born forward (two-level scan).

    Args:
        blocks: per-device block dicts from `make_blocks` (pre-resident).
        omode_occu: (omode,) occupancies ON THE LAST BLOCK'S DEVICE — a CPU
            tensor here would force a pageable H2D that blocks the host until
            the whole chain drains and serializes everything downstream.
        n_max: Born order M (nilpotent: effective order min(M, Nz)).
        coeffs: optional (M, 2) per-order detector reweights (pseudo-complex).
        eps: detector-intensity floor.
        carry: carry-combine schedule for phase B. "chain" ripples the running
            prefix through P-1 SEQUENTIAL hops (each hop waits for the
            upstream add). "direct" sends every block's local total straight
            to every downstream device — all P(P-1)/2 boundary transfers are
            issued concurrently and each block reduces its prefix locally, so
            the critical path is ~one hop instead of P-1 (identical to
            "chain" at P=2; more bytes, fewer serial latencies at P>2).

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
    if P > 1 and blocks[0]["objc"].shape[3] < M:
        raise ValueError(
            f"block 0 holds {blocks[0]['objc'].shape[3]} slices but carries the "
            f"shrinking window: it needs >= M={M} (give block 0 the larger share)"
        )
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
        if carry == "direct" and P > 2:
            # issue ALL boundary transfers up front (concurrent engines),
            # then reduce each block's prefix locally
            t_on = [
                [blocks[i]["t"].to(blocks[j]["dev"], non_blocking=True) for i in range(j)]
                for j in range(1, P)
            ]
            prefix = None
            for j in range(1, P):
                pj = t_on[j - 1][0]
                for ti in t_on[j - 1][1:]:
                    pj = pj + ti
                blocks[j]["prefix"] = pj
                prefix = pj
            D_n = prefix + blocks[-1]["t"]
        else:
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


def depth_bounds(nz: int, p: int, m: int = 0) -> list[int]:
    """Work-balanced depth-block bounds for a P-way slice split.

    Near-equal blocks with the remainder given to the FRONT blocks; block 0
    is then grown (at the expense of later blocks, each kept >= 1 slice) until
    it holds at least ``m`` slices — it carries the shrinking window, so the
    recursion needs nz_0 >= M (see born_forward_dist).
    """
    if not 1 <= p <= nz:
        raise ValueError(f"need 1 <= P={p} <= Nz={nz}")
    base, rem = divmod(nz, p)
    sizes = [base + (1 if i < rem else 0) for i in range(p)]
    need = min(m, nz - (p - 1)) - sizes[0]
    for i in range(p - 1, 0, -1):
        if need <= 0:
            break
        take = min(need, sizes[i] - 1)
        sizes[i] -= take
        sizes[0] += take
        need -= take
    bounds = [0]
    for s in sizes:
        bounds.append(bounds[-1] + s)
    return bounds


def mode_groups(pmode: int, n_groups: int) -> list[tuple[int, int]]:
    """Contiguous near-equal probe-mode [lo, hi) groups; empty groups dropped."""
    n_groups = min(n_groups, pmode)
    base, rem = divmod(pmode, n_groups)
    bounds = [0]
    for i in range(n_groups):
        bounds.append(bounds[-1] + base + (1 if i < rem else 0))
    return [(bounds[i], bounds[i + 1]) for i in range(n_groups)]


def make_modesplit_forward(devices, base_forward=None):
    """Drop-in replacement for `born_forward` splitting probe modes across devices.

    Probe modes are incoherent at the detector, so per-group intensities ADD
    and the split is exact to float reassociation. Group 0 stays on the
    caller's device (devices[0] is ignored for placement of the output);
    static H replicas are cached per device keyed by H.data_ptr(). The object
    patches/probe/coeffs are replicated per call — they change or carry grad,
    and that per-batch transfer is the known cost of this axis.

    Args:
        devices: device strings, one probe-mode group per device
            (e.g. ["cuda:0", "cuda:1", "cuda:2", "cuda:3"] for a 4-way split).
        base_forward: single-device forward to wrap (defaults to
            `ptyrad.forward_models.born.born_forward`).

    Returns:
        Callable with the `born_forward` signature.
    """
    if base_forward is None:
        from .born import born_forward as base_forward
    h_cache: dict[tuple[int, str], torch.Tensor] = {}

    def forward(object_patches, probe, H, omode_occu, eps=1e-10, n_max=1, coeffs=None):
        groups = mode_groups(probe.shape[1], len(devices))
        if len(groups) < 2:
            return base_forward(
                object_patches, probe, H, omode_occu, eps=eps, n_max=n_max, coeffs=coeffs
            )
        outs = []
        for gi, ((lo, hi), dev) in enumerate(zip(groups, devices)):
            if gi == 0:
                Hd = H  # group 0 stays on the caller's device/graph
            else:
                key = (H.data_ptr(), dev)
                Hd = h_cache.get(key)
                if Hd is None:
                    if len(h_cache) > 4 * len(devices):
                        h_cache.clear()
                    Hd = H.detach().to(dev)
                    h_cache[key] = Hd
            outs.append(
                base_forward(
                    object_patches.to(dev),
                    probe[:, lo:hi].to(dev),
                    Hd,
                    omode_occu.to(dev),
                    eps=eps,
                    n_max=n_max,
                    coeffs=coeffs.to(dev) if coeffs is not None else None,
                )
            )
        out = outs[0]
        for o in outs[1:]:
            out = out + o.to(out.device)
        return out - (len(outs) - 1) * eps  # eps was added once per group

    return forward

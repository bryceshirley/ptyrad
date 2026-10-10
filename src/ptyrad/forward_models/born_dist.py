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

Carry schemes (`carry_scheme=`, phase B only; A and C are identical):

  'chain'           (default) serial ripple of the block totals,
                    K_{p+1} = K_p + T_p: P-1 hops in series.
  'chain_reordered' same communication as 'chain'. Kept as a named scheme for
                    the launch-order experiment; the stock engine ALREADY
                    enqueues all of phase A before the first carry, so this is
                    the same code path.
  'direct'          every block sends T_p to every later block in one
                    concurrent round; block q sums its P-1 (at most) incoming
                    totals locally in fixed ascending-p order (deterministic).
                    1 hop of latency, fan-in bandwidth.
  'logscan'         Hillis-Steele inclusive scan across devices in
                    ceil(log2 P) concurrent rounds; the exclusive prefix is
                    recovered locally as K_p = S_p - T_p (one subtract, no
                    extra hop).
  'transpose'       no local cumsum: the scattered fields are all-to-all'd
                    from depth slabs to Ny-column shards, each device runs the
                    FULL-depth cumsum on its shard, and the exclusive rows are
                    all-to-all'd back for phase C.
"""

from contextlib import nullcontext

import torch
from torch.fft import fft2, ifft2

from ptyrad.utils import fftshift2

# flip from the profiling harness to get per-phase ranges in torch.profiler
PROFILE_RANGES = False


def _rf(name):
    return torch.profiler.record_function(name) if PROFILE_RANGES else nullcontext()


_PIN_CACHE = {}


def _pinned_copy(x, dst_dev):
    """Cross-device copy staged through a cached pinned bounce buffer.

    The driver's no-P2P peer-copy fallback on this class of host (KVM, PHB
    topology) costs ~2.5 ms per hop regardless of size; an explicit
    D2H(pinned)+H2D pair with event ordering costs ~0.08 ms at carry size.
    One bounce buffer per (shape, dtype, src, dst); stream-level events keep
    reuse safe without ever syncing the host.
    """
    key = (tuple(x.shape), x.dtype, str(x.device), str(dst_dev))
    ent = _PIN_CACHE.get(key)
    src_stream = torch.cuda.current_stream(x.device)
    dst_stream = torch.cuda.current_stream(dst_dev)
    if ent is None:
        ent = dict(pin=torch.empty(x.shape, dtype=x.dtype, pin_memory=True),
                   ev_d2h=torch.cuda.Event(), ev_h2d=torch.cuda.Event())
        ent["ev_h2d"].record(dst_stream)  # initialize both events
        ent["ev_d2h"].record(src_stream)
        _PIN_CACHE[key] = ent
    # don't overwrite the bounce before the previous consumer has read it
    src_stream.wait_event(ent["ev_h2d"])
    ent["pin"].copy_(x, non_blocking=True)          # D2H on src stream
    ent["ev_d2h"].record(src_stream)
    out = torch.empty(x.shape, dtype=x.dtype, device=dst_dev)
    dst_stream.wait_event(ent["ev_d2h"])
    out.copy_(ent["pin"], non_blocking=True)        # H2D on dst stream
    ent["ev_h2d"].record(dst_stream)
    return out


class _PinnedHop(torch.autograd.Function):
    """Differentiable cross-device move via the pinned bounce (the adjoint
    is the mirrored hop back to the source device)."""

    @staticmethod
    def forward(ctx, x, dst_dev):
        ctx.src_dev = x.device
        return _pinned_copy(x.contiguous(), dst_dev)

    @staticmethod
    def backward(ctx, g):
        return _pinned_copy(g.contiguous(), ctx.src_dev), None


def _hop(x, dst_dev, transport):
    """One cross-device move with the selected transport."""
    if transport == "pinned" and x.is_cuda and str(x.device) != str(dst_dev):
        return _PinnedHop.apply(x, torch.device(dst_dev))
    return x.to(dst_dev, non_blocking=True)


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


def _win(blocks, i, n, skip_zero):
    """Scattering-window start for block i at order n.

    Psi^(n) vanishes exactly (bitwise 0.0) on the first n planes, so those
    rows can be skipped. Stock behaviour (skip_zero=False): only block 0
    windows, clamped to its slab. skip_zero=True continues the window across
    slab boundaries — at most one block is partially windowed at a time, and
    fully covered blocks go dead.
    """
    nz_i = blocks[i]["objc"].shape[3]
    if skip_zero:
        z0 = sum(blocks[k]["objc"].shape[3] for k in range(i))
        return max(0, min(n - z0, nz_i))
    return min(n, nz_i) if i == 0 else 0


def _phase_a(blocks, n, skip_zero=False):
    """Local scatter + scan on every block; no cross-device dependency.

    A block whose window covers its whole slab contributes a zero total and
    is marked dead (its psi is never read again). For non-exhausted blocks
    this is byte-identical to the original inline loop.
    """
    dead_any = False
    for i, b in enumerate(blocks):
        nz_i = b["objc"].shape[3]
        win0 = _win(blocks, i, n, skip_zero)
        b["win0"] = win0
        if win0 == nz_i:
            b["dead"] = True
            b["cs"] = None
            b["t"] = None  # filled from a live block's shape below
            dead_any = True
            continue
        b["dead"] = False
        sc = fft2(b["objc"][..., win0:, :, :] * b["psi"]) * b["Hc"][..., win0:, :, :]
        b["cs"] = torch.cumsum(sc, dim=3)
        b["t"] = b["cs"][..., -1, :, :]
    if dead_any:
        ref = blocks[-1]["t"]  # last block never windows fully, always live
        for b in blocks:
            if b["dead"]:
                b["t"] = torch.zeros(ref.shape, dtype=ref.dtype,
                                     device=b["dev"])


def _phase_c(blocks):
    """psi update on every block in parallel (needs b['prefix'] for i > 0).

    A block whose window has entered its slab (i == 0, or win0 > 0 under
    skip_zero_planes) drops the exactly-zero head row instead of adding a
    prefix — every earlier block is fully below the window there.
    """
    for i, b in enumerate(blocks):
        if b["dead"]:
            continue
        if i == 0 or b["win0"] > 0:
            if b["cs"].shape[3] < 2:  # window empties next order; psi unused
                continue
            b["psi"] = ifft2(b["cs"][..., :-1, :, :]
                             * b["H"][..., b["win0"] + 1:, :, :])
        else:
            p = b["prefix"].unsqueeze(3)
            b["psi"] = ifft2(
                torch.cat([p, b["cs"][..., :-1, :, :] + p], dim=3) * b["H"])


def _carry_chain(blocks, transport="peer"):
    """Serial ripple: K_{p+1} = K_p + T_p, P-1 hops in series (the original)."""
    P = len(blocks)
    prefix = None  # incoming prefix for block i, resident on its device
    for i in range(P - 1):
        total = blocks[i]["t"] if prefix is None else prefix + blocks[i]["t"]
        prefix = _hop(total, blocks[i + 1]["dev"], transport)
        blocks[i + 1]["prefix"] = prefix
    return blocks[-1]["t"] if prefix is None else prefix + blocks[-1]["t"]


def _carry_direct(blocks, transport="peer"):
    """One concurrent round: every block sends T_p to every later block, the
    receiver sums in fixed ascending-p order (deterministic)."""
    P = len(blocks)
    if P == 1:
        return blocks[0]["t"]
    sends = {}
    for q in range(P - 1):  # enqueue ALL copies before any add
        tq = blocks[q]["t"]
        for j in range(q + 1, P):
            sends[(q, j)] = _hop(tq, blocks[j]["dev"], transport)
    for j in range(1, P):
        acc = sends[(0, j)]
        for q in range(1, j):
            acc = acc + sends[(q, j)]
        blocks[j]["prefix"] = acc
    return blocks[-1]["prefix"] + blocks[-1]["t"]


def _carry_logscan(blocks, transport="peer"):
    """Hillis-Steele inclusive scan in ceil(log2 P) concurrent rounds; the
    exclusive prefix is recovered locally as K_p = S_p - T_p."""
    P = len(blocks)
    if P == 1:
        return blocks[0]["t"]
    s = [b["t"] for b in blocks]
    d = 1
    while d < P:
        recv = [_hop(s[i - d], blocks[i]["dev"], transport)
                for i in range(d, P)]
        for i in range(d, P):
            s[i] = s[i] + recv[i - d]
        d *= 2
    for i in range(1, P):
        blocks[i]["prefix"] = s[i] - blocks[i]["t"]
    return s[-1]


def _order_transpose(blocks, n, last, skip_zero=False, transport="peer"):
    """One order pass via a global transpose: phase 1 computes the scattered
    fields only; an all-to-all moves depth slabs to Ny-column shards; each
    device cumsums the FULL depth on its shard; the exclusive rows go back by
    a second all-to-all for phase C. Returns D_n on the last device."""
    P = len(blocks)
    with _rf("phase1"):
        live = []
        for i, b in enumerate(blocks):
            nz_i = b["objc"].shape[3]
            win0 = _win(blocks, i, n, skip_zero)
            b["win0"] = win0
            if win0 == nz_i:
                b["dead"] = True
                b["sc"] = None
                continue
            b["dead"] = False
            b["sc"] = (fft2(b["objc"][..., win0:, :, :] * b["psi"])
                       * b["Hc"][..., win0:, :, :])
            live.append(i)
    Ny = blocks[0]["probe"].shape[-2]
    yb = [round(j * Ny / P) for j in range(P + 1)]  # Ny shard bounds
    z0, off = {}, 0
    for i in live:
        z0[i] = off
        off += blocks[i]["sc"].shape[3]
    with _rf("a2a_fwd"):
        # pack + send: P*P concurrent chunks, all enqueued before any cumsum
        chunks = [[] for _ in range(P)]
        for i in live:
            sc = blocks[i]["sc"]
            for j in range(P):
                part = sc[..., yb[j]:yb[j + 1], :].contiguous()
                chunks[j].append(_hop(part, blocks[j]["dev"], transport))
    with _rf("colscan"):
        cs_g = [torch.cumsum(torch.cat(chunks[j], dim=3), dim=3)
                for j in range(P)]
    with _rf("a2a_back"):
        dn = [_hop(cs_g[j][..., -1, :, :].contiguous(),
                   blocks[-1]["dev"], transport) for j in range(P)]
        D_n = torch.cat(dn, dim=-2)
        if not last:
            for i in live:
                nz_i = blocks[i]["objc"].shape[3]
                if i == 0 or blocks[i]["win0"] > 0:
                    lo = z0[i]
                    hi = z0[i] + blocks[i]["sc"].shape[3] - 1
                    zero_head = False
                else:
                    a = z0[i] - 1  # global exclusive rows for this slab
                    lo, hi, zero_head = max(a, 0), a + nz_i, a < 0
                parts = [_hop(cs_g[j][..., lo:hi, :, :].contiguous(),
                              blocks[i]["dev"], transport)
                         for j in range(P)]
                ex = torch.cat(parts, dim=-2)
                if zero_head:
                    z = torch.zeros_like(ex[..., :1, :, :])
                    ex = torch.cat([z, ex], dim=3)
                blocks[i]["ex"] = ex
    if not last:
        with _rf("phase2"):
            for i in live:
                b = blocks[i]
                if i == 0 or b["win0"] > 0:
                    if b["ex"].shape[3] == 0:  # window empties next order
                        continue
                    b["psi"] = ifft2(b["ex"] * b["H"][..., b["win0"] + 1:, :, :])
                else:
                    b["psi"] = ifft2(b["ex"] * b["H"])
    return D_n


_CARRIES = {"chain": _carry_chain, "chain_reordered": _carry_chain,
            "direct": _carry_direct, "logscan": _carry_logscan}


def born_forward_dist(blocks, omode_occu, n_max, coeffs=None, eps=1e-10,
                      carry_scheme="chain", skip_zero_planes=False,
                      carry_transport="peer"):
    """Depth-partitioned parallel Born forward (two-level scan).

    Args:
        blocks: per-device block dicts from `make_blocks` (pre-resident).
        omode_occu: (omode,) occupancies ON THE LAST BLOCK'S DEVICE — a CPU
            tensor here would force a pageable H2D that blocks the host until
            the whole chain drains and serializes everything downstream.
        n_max: Born order M (nilpotent: effective order min(M, Nz)).
        coeffs: optional (M, 2) per-order detector reweights (pseudo-complex).
        eps: detector-intensity floor.
        carry_scheme: cross-device combine for the block totals — 'chain'
            (default, the original serial ripple), 'chain_reordered',
            'direct', 'logscan', or 'transpose'. See module docstring.
        skip_zero_planes: continue the nilpotent scattering window across
            slab boundaries (Psi^(n) is exactly zero on the first n planes),
            instead of windowing block 0 only. Off by default (stock
            behaviour); the skipped rows are float-exact zeros, so enabling
            it changes results only at the reassociation level. Pays only
            when M is comparable to Nz/P.
        carry_transport: 'peer' (default, the stock non-blocking .to peer
            copy) or 'pinned' — explicit D2H(pinned)+H2D staging with event
            ordering, ~30x lower hop latency on hosts whose no-P2P peer-copy
            fallback is slow (KVM / PHB topology).

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

    if carry_scheme not in _CARRIES and carry_scheme != "transpose":
        raise ValueError(f"unknown carry_scheme {carry_scheme!r}")

    M = min(n_max, Nz)
    for n in range(M):
        last = n == M - 1
        if carry_scheme == "transpose":
            D_n = _order_transpose(blocks, n, last, skip_zero_planes,
                                   carry_transport)
        else:
            with _rf("phase1"):
                _phase_a(blocks, n, skip_zero_planes)
            with _rf("carry"):
                D_n = _CARRIES[carry_scheme](blocks, carry_transport)
            if not last:
                with _rf("phase2"):
                    _phase_c(blocks)
        Psi_M = Psi_M + D_n if c_all is None else Psi_M + c_all[n] * D_n

    nw = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(Psi_M.abs().square() * nw, dim=(1, 2)) + eps)

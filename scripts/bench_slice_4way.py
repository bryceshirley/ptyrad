"""P-way slice (depth) split across up to 4 A100s — does slice-split scale
past 2 GPUs? This is slice-split's reason to exist vs mode-split, which caps
at #probe-modes (4) and replicates the full object on every device.

Nz split into P contiguous blocks, one per GPU. The two-level scan
generalises to a carry CHAIN: block b's running prefix (sum of all upstream
block totals) is passed to block b+1 each order; the grand total at the last
block is the detector contribution D_n. Information flows strictly downstream,
so blocks form an offset wavefront (block b trails block b-1 by one order).

NOTRANSFER=1 replaces each chain carry with a cached zero (breaks correctness)
to expose the compute-only scaling ceiling, separating it from the host-staged
PCIe carry cost (this box is all-PHB, no P2P/NVLink).
"""
import os, sys, time
import torch
from torch.fft import fft2, ifft2
sys.path.insert(0, "/home/dnz75396/ptyrad/src")
from ptyrad.utils import fftshift2

torch.manual_seed(0)
Nz, Ny, Nx, PMODE, OMODE, M = 21, 256, 256, 4, 1, 6
EPS = 1e-10
NOTRANSFER = os.environ.get("NOTRANSFER", "0") == "1"
# CODEC compresses the carry on each PCIe hop. All three are TRAINING-SAFE: the
# induced object-gradient error is far below the photon shot-noise floor
# (bench_slice_compress.py). c64=8B/elem (exact), half=4B (2x), quarter=2B (4x,
# two fp8_e4m3 + a scalar scale).
CODEC = os.environ.get("CODEC", "c64")
_HI = 448.0  # fp8_e4m3 max magnitude
# work-balanced boundaries (block0 carries the shrinking window so is largest)
BOUNDS = {1: [0, 21], 2: [0, 12, 21], 4: [0, 7, 12, 17, 21]}
_ZERO = {}


def make_inputs(N):
    obj = torch.randn(N, OMODE, Nz, Ny, Nx, 2) * 0.02
    obj[..., 0] += 1.0
    probe = torch.randn(N, PMODE, Ny, Nx, dtype=torch.complex64)
    H = torch.exp(1j * 0.01 * torch.randn(1, 1, 1, Nz, Ny, Nx)).to(torch.complex64)
    occu = torch.ones(OMODE)
    return obj, probe, H, occu


def distribute(obj, probe, H, occu, P):
    """Pre-place each block's object/H/probe on its device ONCE (outside timing).
    In a real run these live distributed across the GPUs for the whole solve."""
    devs = [f"cuda:{i}" for i in range(P)]
    b = BOUNDS[P]
    O = torch.polar(obj[..., 0], obj[..., 1])
    objc_all = (O - 1.0).unsqueeze(1)
    blk = []
    for i in range(P):
        lo, hi = b[i], b[i + 1]
        d = devs[i]
        Hd = H[..., lo:hi, :, :].to(d).contiguous()
        blk.append(dict(
            objc=objc_all[..., lo:hi, :, :].to(d).contiguous(), H=Hd, Hc=Hd.conj().contiguous(),
            probe=probe.to(d), lo=lo, dev=d,
        ))
    return blk


def forward_pway(blk, occu, P):
    """P-way slice split on pre-distributed blocks. Returns intensity on last dev."""
    for s in blk:
        pk = fft2(s["probe"]).view(-1, PMODE, 1, 1, Ny, Nx)
        s["pk"] = pk
        s["psi"] = ifft2(s["H"] * pk)
    Psi_M = blk[-1]["pk"].squeeze(3)

    def xfer(carry, dev):
        if not NOTRANSFER:
            return carry.to(dev, non_blocking=True)
        key = (carry.shape, dev)
        z = _ZERO.get(key)
        if z is None:
            z = torch.zeros(carry.shape, dtype=carry.dtype, device=dev); _ZERO[key] = z
        return z

    for n in range(M):
        carry_in = None            # running prefix entering the current block
        for i in range(P):
            s = blk[i]
            win0 = n if i == 0 else 0            # only block0 has the shrinking window
            sc = fft2(s["objc"][..., win0:, :, :] * s["psi"]) * s["Hc"][..., win0:, :, :]
            cs = torch.cumsum(sc, dim=3)
            if carry_in is not None:
                cs = cs + carry_in.unsqueeze(3)
            running = cs[..., -1, :, :]           # prefix out of this block
            if n < M - 1:
                if i == 0:
                    s["psi"] = ifft2(cs[..., :-1, :, :] * s["H"][..., win0 + 1:, :, :])
                else:
                    head = carry_in.unsqueeze(3)
                    s["psi"] = ifft2(torch.cat([head, cs[..., :-1, :, :]], dim=3) * s["H"])
            if i < P - 1:
                carry_in = xfer(running, blk[i + 1]["dev"])
            else:
                D_n = running
        Psi_M = Psi_M + D_n
    nw = (occu.to(Psi_M.device) / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(Psi_M.abs().square() * nw, dim=(1, 2)) + EPS)


def make_chain_bufs(P):
    """Per-hop pinned double-buffers + copy streams for the carry chain.
    Hop h moves the running prefix from dev h to dev h+1. D2H on a src-side
    copy stream, H2D on a dst-side copy stream, gated by events, so the carry
    transfer overlaps compute on the other devices (the PHB box has no P2P)."""
    hops = []
    for h in range(P - 1):
        hops.append(dict(
            src=torch.cuda.Stream(device=f"cuda:{h}"),
            dst=torch.cuda.Stream(device=f"cuda:{h + 1}"),
            buf={},  # slot -> list of (pinned_host, dev_recv) per encoded part
        ))
    return hops


def forward_pway_pinned(blk, occu, P, hops):
    """P-way slice split with the carry chain overlapped via pinned double-buffers."""
    for s in blk:
        pk = fft2(s["probe"]).view(-1, PMODE, 1, 1, Ny, Nx)
        s["pk"] = pk
        s["psi"] = ifft2(s["H"] * pk)
    Psi_M = blk[-1]["pk"].squeeze(3)

    def encode(carry):  # -> list of real/complex tensors to move over PCIe
        if CODEC == "half":
            return [carry.to(torch.complex32)]
        if CODEC == "quarter":
            s = carry.abs().amax().clamp_min(1e-30)
            return [(carry.real / s * _HI).to(torch.float8_e4m3fn),
                    (carry.imag / s * _HI).to(torch.float8_e4m3fn), s]
        return [carry]

    def decode(parts):
        if CODEC == "half":
            return parts[0].to(torch.complex64)
        if CODEC == "quarter":
            re, im, s = parts
            return torch.complex(re.to(torch.float32), im.to(torch.float32)) * (s / _HI)
        return parts[0]

    def xfer(carry, h, slot):
        hb = hops[h]
        dev1 = f"cuda:{h + 1}"
        parts = encode(carry)
        bufs = hb["buf"].setdefault(slot, None)
        if bufs is None:
            bufs = [(torch.empty(p.shape, dtype=p.dtype, pin_memory=True),
                     torch.empty(p.shape, dtype=p.dtype, device=dev1)) for p in parts]
            hb["buf"][slot] = bufs
        hb["src"].wait_stream(torch.cuda.current_stream(f"cuda:{h}"))
        with torch.cuda.stream(hb["src"]):
            for p, (pin, _) in zip(parts, bufs):
                pin.copy_(p, non_blocking=True)
        ev = torch.cuda.Event(); ev.record(hb["src"])
        hb["dst"].wait_event(ev)
        with torch.cuda.stream(hb["dst"]):
            for (pin, recv) in bufs:
                recv.copy_(pin, non_blocking=True)
        ev2 = torch.cuda.Event(); ev2.record(hb["dst"])
        torch.cuda.current_stream(dev1).wait_event(ev2)
        return decode([recv for (_, recv) in bufs])

    for n in range(M):
        carry_in = None
        for i in range(P):
            s = blk[i]
            win0 = n if i == 0 else 0
            sc = fft2(s["objc"][..., win0:, :, :] * s["psi"]) * s["Hc"][..., win0:, :, :]
            cs = torch.cumsum(sc, dim=3)
            if carry_in is not None:
                cs = cs + carry_in.unsqueeze(3)
            running = cs[..., -1, :, :]
            if n < M - 1:
                if i == 0:
                    s["psi"] = ifft2(cs[..., :-1, :, :] * s["H"][..., win0 + 1:, :, :])
                else:
                    head = carry_in.unsqueeze(3)
                    s["psi"] = ifft2(torch.cat([head, cs[..., :-1, :, :]], dim=3) * s["H"])
            if i < P - 1:
                carry_in = xfer(running, i, n % 2)
            else:
                D_n = running
        Psi_M = Psi_M + D_n
    nw = (occu.to(Psi_M.device) / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(Psi_M.abs().square() * nw, dim=(1, 2)) + EPS)


def time_fwd(fn, devs, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    for d in devs:
        torch.cuda.synchronize(d)
    ts = []
    for _ in range(reps):
        for d in devs:
            torch.cuda.synchronize(d)
        t0 = time.perf_counter(); fn()
        for d in devs:
            torch.cuda.synchronize(d)
        ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2]


if __name__ == "__main__":
    print(f"GPUs visible: {torch.cuda.device_count()} | NOTRANSFER={NOTRANSFER}")
    # correctness: P=2 and P=4 vs P=1 (sync chain AND pinned-async chain)
    N = 2
    obj, probe, H, occu = make_inputs(N)
    I1 = forward_pway(distribute(obj, probe, H, occu, 1), occu, 1)
    for P in (2, 4):
        IP = forward_pway(distribute(obj, probe, H, occu, P), occu, P).to("cuda:0")
        Ipin = forward_pway_pinned(distribute(obj, probe, H, occu, P), occu, P,
                                   make_chain_bufs(P)).to("cuda:0")
        print(f" P={P} sync rel err {((IP - I1).norm()/I1.norm()).item():.2e} | "
              f"pinned rel err {((Ipin - I1).norm()/I1.norm()).item():.2e}")
    print()
    for N in (1, 32):
        obj, probe, H, occu = make_inputs(N)
        t, tp = {}, {}
        for P in (1, 2, 4):
            devs = [f"cuda:{i}" for i in range(P)]
            blk = distribute(obj, probe, H, occu, P)
            t[P] = time_fwd(lambda blk=blk, P=P: forward_pway(blk, occu, P), devs)
            if P == 1:
                tp[P] = t[P]
            else:
                blkp = distribute(obj, probe, H, occu, P)
                hops = make_chain_bufs(P)
                tp[P] = time_fwd(lambda blkp=blkp, P=P, hops=hops:
                                 forward_pway_pinned(blkp, occu, P, hops), devs)
        print(f" N={N:2d} sync   : 1-GPU {t[1]*1e3:7.2f} ms | "
              f"2-GPU {t[2]*1e3:7.2f} ms ({t[1]/t[2]:.2f}x) | "
              f"4-GPU {t[4]*1e3:7.2f} ms ({t[1]/t[4]:.2f}x)")
        print(f" N={N:2d} pinned : 1-GPU {tp[1]*1e3:7.2f} ms | "
              f"2-GPU {tp[2]*1e3:7.2f} ms ({tp[1]/tp[2]:.2f}x) | "
              f"4-GPU {tp[4]*1e3:7.2f} ms ({tp[1]/tp[4]:.2f}x)")

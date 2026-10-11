"""Slice-split carry schedule: CHAIN (ripple, P-1 serial hops/order) vs
DIRECT (all boundary totals shipped concurrently, local prefix reduce).

At P=2 the schedules are identical (one hop either way) — that is why the
slicepar round could not separate them. At P=4 the chain pays 3 SEQUENTIAL
hop latencies per order while direct issues all 6 transfers at once, so on
a latency-bound interconnect (PHB PCIe here; per-hop launch/sync latency is
the exposed cost per bench_results §L.6 hyp-4) direct should approach the
one-hop critical path at the price of more bytes.

Times `born_forward_dist` (the production slicedist path) at batch N:
  1-GPU reference | chain fwd / f+b | direct fwd / f+b   per Nz.

Env: NGPU (4), BORN_M (6), PMODE (12), OMODE (1), N (1),
     NZ_LIST ("42,64,128,256,512"), REPS (15).
"""

import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from ptyrad.forward_models import born_forward_dist, depth_bounds, make_blocks  # noqa: E402

torch.manual_seed(0)
P = int(os.environ.get("NGPU", "4"))
M = int(os.environ.get("BORN_M", "6"))
PMODE = int(os.environ.get("PMODE", "12"))
OMODE = int(os.environ.get("OMODE", "1"))
N = int(os.environ.get("N", "1"))
NZ_LIST = [int(z) for z in os.environ.get("NZ_LIST", "42,64,128,256,512").split(",")]
REPS = int(os.environ.get("REPS", "15"))
Ny = Nx = 256


def make_inputs(nz):
    obj = torch.randn(N, OMODE, nz, Ny, Nx, 2) * 0.02
    obj[..., 0] += 1.0
    objc = (torch.polar(obj[..., 0], obj[..., 1]) - 1.0).unsqueeze(1)
    probe = torch.randn(N, PMODE, Ny, Nx, dtype=torch.complex64)
    H = torch.exp(1j * 0.01 * torch.randn(1, 1, 1, nz, Ny, Nx)).to(torch.complex64)
    return objc, probe, H


def place_blocks(objc, H, probe, p, grad=False):
    b = depth_bounds(H.shape[3], p, m=M)
    devs = [f"cuda:{k}" for k in range(p)]
    oc_blocks, h_blocks = [], []
    for k in range(p):
        oc = objc[..., b[k] : b[k + 1], :, :].to(devs[k]).contiguous()
        if grad:
            oc = oc.detach().requires_grad_(True)
        oc_blocks.append(oc)
        h_blocks.append(H[..., b[k] : b[k + 1], :, :].to(devs[k]).contiguous())
    leaves = oc_blocks if grad else None
    return make_blocks(oc_blocks, h_blocks, probe, devs), leaves


def sync_all(p):
    for k in range(p):
        torch.cuda.synchronize(f"cuda:{k}")


def tmed(fn, p, warmup=3):
    for _ in range(warmup):
        fn()
    sync_all(p)
    ts = []
    for _ in range(REPS):
        sync_all(p)
        t0 = time.perf_counter()
        fn()
        sync_all(p)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    occu_last = torch.ones(OMODE, device=f"cuda:{P - 1}")
    occu_0 = torch.ones(OMODE, device="cuda:0")
    # correctness first: chain and direct must match the 1-GPU forward
    objc, probe, H = make_inputs(NZ_LIST[0])
    blk1, _ = place_blocks(objc, H, probe, 1)
    ref = born_forward_dist(blk1, occu_0, M).cpu()
    for carry in ("chain", "direct"):
        blk, _ = place_blocks(objc, H, probe, P)
        out = born_forward_dist(blk, occu_last, M, carry=carry).cpu()
        rel = ((out - ref).norm() / ref.norm()).item()
        print(f"correctness {carry}: rel err vs 1-GPU = {rel:.2e}")
        assert rel < 1e-5, f"{carry} carry is WRONG (rel {rel:.2e}) — do not trust timings"
    del objc, probe, H, blk1
    torch.cuda.empty_cache()
    print(
        f"P={P} | batch={N}, {Ny}x{Nx}, pmode={PMODE}, M={M} "
        f"(BORN_M/PMODE/NGPU/N/NZ_LIST env); times in ms"
    )
    print(
        f"{'Nz':>5} | {'1G fwd':>8} | {'chain fwd':>9} | {'direct fwd':>10} "
        f"| {'1G f+b':>8} | {'chain f+b':>9} | {'direct f+b':>10} | best f+b x"
    )
    for nz in NZ_LIST:
        objc, probe, H = make_inputs(nz)
        blk1, _ = place_blocks(objc, H, probe, 1)
        t1 = tmed(lambda: born_forward_dist(blk1, occu_0, M), 1)
        blk1g, lv1 = place_blocks(objc, H, probe, 1, grad=True)
        t1fb = tmed(lambda: born_forward_dist(blk1g, occu_0, M).sum().backward(), 1)
        for leaf in lv1:
            leaf.grad = None
        res = {}
        for carry in ("chain", "direct"):
            blk, _ = place_blocks(objc, H, probe, P)
            res[carry] = tmed(lambda: born_forward_dist(blk, occu_last, M, carry=carry), P)
            blkg, lvg = place_blocks(objc, H, probe, P, grad=True)

            def fb():
                born_forward_dist(blkg, occu_last, M, carry=carry).sum().backward()
                for leaf in lvg:
                    leaf.grad = None

            res[carry + "_fb"] = tmed(fb, P)
        best = min(res["chain_fb"], res["direct_fb"])
        print(
            f"{nz:>5} | {t1 * 1e3:8.2f} | {res['chain'] * 1e3:9.2f} "
            f"| {res['direct'] * 1e3:10.2f} | {t1fb * 1e3:8.2f} "
            f"| {res['chain_fb'] * 1e3:9.2f} | {res['direct_fb'] * 1e3:10.2f} "
            f"| {t1fb / best:6.2f}x"
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

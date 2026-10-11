"""Compare the three multi-GPU axes on the REAL born6 PSO checkpoint:
  * slice  — depth-plane split (carry chain; bench_slice_real)
  * mode   — split the 4 probe modes across GPUs (incoherent detector sum, so
             per-group born_forward outputs just ADD; keeps compiled born;
             batch-1 capable; caps at #modes=4)
  * batch  — data-parallel over scan positions (embarrassingly parallel fwd;
             one object-grad all-reduce per step; needs N>=G so UNAVAILABLE at
             the real BATCH_SIZE=1)

Reuses helpers + the real tensors from bench_slice_real. Forward is the clean
apples-to-apples metric; fwd+bw reported with each axis's natural comm.
"""
import os, sys, time
import numpy as np
import torch
torch._dynamo.config.recompile_limit = 64  # per-device born_forward recompiles
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
sys.path.insert(0, _HERE)
from bench_slice_real import (
    load_ckpt, build_inputs, stock_forward, distribute, forward_pway_pinned,
    make_chain_bufs, time_fn, sync_all, born_forward, gaussian_blur,
    PMODE, M, BLUR_SIGMA, EPS)


# ---- mode split ----------------------------------------------------------
def mode_setup(ck, N, G):
    """Replicate object/H on each GPU; give each a contiguous group of modes."""
    devs = [f"cuda:{i}" for i in range(G)]
    groups = [list(a) for a in np.array_split(range(PMODE), G)]
    patches, probe, H3d, occu, coeffs = build_inputs(ck, N, dev="cuda:0")
    reps = []
    for g, dev in enumerate(devs):
        reps.append(dict(patches=patches.to(dev), H3d=H3d.to(dev),
                         occu=occu.to(dev), coeffs=coeffs.to(dev),
                         probe=probe[:, groups[g]].to(dev).contiguous(), dev=dev))
    return reps


def mode_forward(reps):
    parts = [born_forward(r["patches"], r["probe"], r["H3d"], omode_occu=r["occu"],
                          n_max=M, coeffs=r["coeffs"]).to("cuda:0") for r in reps]
    I = torch.stack(parts).sum(0)
    return gaussian_blur(I, kernel_size=[5, 5], sigma=BLUR_SIGMA)


def mode_fb(reps):
    # identical .sum() loss as the 1-GPU/batch baselines, + object-grad all-reduce
    for r in reps:
        r["p"] = r["patches"].detach().requires_grad_(True)
    parts = [born_forward(r["p"], r["probe"], r["H3d"], omode_occu=r["occu"],
                          n_max=M, coeffs=r["coeffs"]).to("cuda:0") for r in reps]
    Ib = gaussian_blur(torch.stack(parts).sum(0), kernel_size=[5, 5], sigma=BLUR_SIGMA)
    Ib.sum().backward()
    g = sum(r["p"].grad.to("cuda:0") for r in reps)   # object-grad all-reduce
    return g


# ---- batch (data-parallel) split ----------------------------------------
def batch_setup(ck, N, G):
    devs = [f"cuda:{i}" for i in range(G)]
    patches, probe, H3d, occu, coeffs = build_inputs(ck, N, dev="cuda:0")
    sh = N // G
    reps = []
    for g, dev in enumerate(devs):
        reps.append(dict(patches=patches[g * sh:(g + 1) * sh].to(dev),
                         probe=probe.to(dev), H3d=H3d.to(dev),
                         occu=occu.to(dev), coeffs=coeffs.to(dev), dev=dev))
    return reps


def batch_forward(reps):
    return [stock_forward(r["patches"], r["probe"], r["H3d"], r["occu"], r["coeffs"])
            for r in reps]


def batch_fb(reps):
    # per-shard fwd+bw (shards are disjoint positions -> independent patch leaves);
    # the shared-object all-reduce is quoted separately (allreduce_cost).
    for r in reps:
        p = r["patches"].detach().requires_grad_(True)
        stock_forward(p, r["probe"], r["H3d"], r["occu"], r["coeffs"]).sum().backward()


def allreduce_cost(G, reps=20):
    """One full-object (obja+objp = 2 x 1x21x639x639 f32 ~68MB) sum-reduce to
    cuda:0, host-staged (no P2P). The per-step comm of data-parallel."""
    if G == 1:
        return 0.0
    gs = [torch.randn(1, 21, 639, 639, 2, device=f"cuda:{i}") for i in range(G)]
    def red():
        _ = sum(g.to("cuda:0") for g in gs)
    for _ in range(5):
        red()
    sync_all(G)
    ts = []
    for _ in range(reps):
        sync_all(G); t0 = time.perf_counter(); red(); sync_all(G)
        ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2]


if __name__ == "__main__":
    print(f"GPUs visible: {torch.cuda.device_count()}")
    ck = load_ckpt()

    # correctness of mode-split vs stock (real object)
    patches, probe, H3d, occu, coeffs = build_inputs(ck, 2)
    I0 = stock_forward(patches, probe, H3d, occu, coeffs)
    for G in (2, 4):
        Im = mode_forward(mode_setup(ck, 2, G)).to("cuda:0")
        print(f"mode-split G={G} forward rel err {((Im - I0).norm()/I0.norm()).item():.2e}")
    for G in (2, 4):
        print(f"object all-reduce (68MB) G={G}: {allreduce_cost(G)*1e3:.2f} ms/step")

    print("\n=== forward speedup vs 1-GPU (compiled born_forward baseline) ===")
    hdr = f"{'N':>4} | {'1-GPU':>8} | {'slice2':>14} | {'slice4':>14} | {'mode2':>14} | {'mode4':>14} | {'batch2':>14} | {'batch4':>14}"
    print(hdr)
    for N in (1, 8, 32):
        patches, probe, H3d, occu, coeffs = build_inputs(ck, N)
        t1 = time_fn(lambda: stock_forward(patches, probe, H3d, occu, coeffs), 1)
        row = {"1": t1}
        # slice 2/4
        for P in (2, 4):
            blk = distribute(patches, probe, H3d, P)
            hops = make_chain_bufs(P)
            row[f"s{P}"] = time_fn(lambda blk=blk, P=P, hops=hops:
                                   forward_pway_pinned(blk, occu, coeffs, P, hops, "half"), P)
        # mode 2/4
        for G in (2, 4):
            reps = mode_setup(ck, N, G)
            row[f"m{G}"] = time_fn(lambda reps=reps: mode_forward(reps), G)
        # batch 2/4 (only if divisible)
        for G in (2, 4):
            if N % G == 0 and N >= G:
                reps = batch_setup(ck, N, G)
                row[f"b{G}"] = time_fn(lambda reps=reps: batch_forward(reps), G)
            else:
                row[f"b{G}"] = None
        def fmt(key):
            t = row.get(key)
            if t is None:
                return f"{'N/A':>14}"
            return f"{t*1e3:6.1f}ms {t1/t:4.2f}x"
        print(f"{N:>4} | {t1*1e3:6.1f}ms | {fmt('s2')} | {fmt('s4')} | "
              f"{fmt('m2')} | {fmt('m4')} | {fmt('b2')} | {fmt('b4')}")

    # The COMPILED backward collides on torch.compile's metrics context when run
    # across devices (a dynamo bug, not our logic); eager==compiled for this born
    # (both 3.6 ms fwd), so measure fwd+bw eager -> ratios stay fair.
    torch.compiler.set_stance("force_eager")
    print("\n=== forward+backward speedup vs 1-GPU (eager backend, all axes) ===")
    for N in (1, 8, 32):
        patches, probe, H3d, occu, coeffs = build_inputs(ck, N)
        def s1():
            p = patches.detach().requires_grad_(True)
            stock_forward(p, probe, H3d, occu, coeffs).sum().backward()
        t1 = time_fn(s1, 1)
        # mode 2/4 (incl object all-reduce)
        m = {}
        for G in (2, 4):
            reps = mode_setup(ck, N, G)
            m[G] = time_fn(lambda reps=reps: mode_fb(reps), G)
        # batch 2/4 compute (+ separate all-reduce quote)
        b = {}
        for G in (2, 4):
            if N % G == 0 and N >= G:
                reps = batch_setup(ck, N, G)
                b[G] = time_fn(lambda reps=reps: batch_fb(reps), G) + allreduce_cost(G)
            else:
                b[G] = None
        def fmt(t):
            return f"{'N/A':>14}" if t is None else f"{t*1e3:6.1f}ms {t1/t:4.2f}x"
        print(f"{N:>4} | 1-GPU {t1*1e3:6.1f}ms | mode2 {fmt(m[2])} | mode4 {fmt(m[4])} | "
              f"batch2 {fmt(b[2])} | batch4 {fmt(b[4])}")

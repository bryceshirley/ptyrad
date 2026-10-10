"""Step 5: timing grid for the carry schemes.

Each cell times (a) one forward pass and (b) one optimizer step
(objc-from-leaves + forward + scalar loss + backward + Adam step), with
5 warm-ups and median/IQR over 20 repeats, host-synced only around the
timed region. P=1 is single-device born_forward (eager by default; the
'compiled' arm uses the stock @torch.compile(max-autotune) wrapper).

Usage: bench_grid.py --n 256 [--arms main|compiled|pmode4|skipz] [--out f.csv]
Appends rows: mode,P,scheme,transport,skipz,Nz,n,M,pmode,reps,ms_med,ms_iqr,status
"""

import argparse
import csv
import os
import statistics as st
import time

import torch

from common import build_dist, even_bounds, make_problem, single_inputs

from ptyrad.forward_models import born_forward
from ptyrad.forward_models.born_dist import _hop, born_forward_dist

DEVS = ["cuda:0", "cuda:1"]
NZS = [4, 8, 16, 32, 64, 128, 256]
REPS, WARM = 20, 5


def sync_all():
    for d in DEVS:
        torch.cuda.synchronize(d)


def run_cell(fn, reps=REPS, warm=WARM):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        sync_all()
        t0 = time.perf_counter()
        fn()
        sync_all()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return st.median(ts), ts[(3 * len(ts)) // 4] - ts[len(ts) // 4]


def p1_cell(Nz, n, M, pmode, mode, compiled=False):
    a, p, probe, H, occu = make_problem(Nz, n, pmode=pmode)
    pr0 = probe.to("cuda:0").requires_grad_(mode == "step")
    occ0 = occu.to("cuda:0")
    H0 = H.to("cuda:0")
    a0 = a.to("cuda:0").requires_grad_(mode == "step")
    p0 = p.to("cuda:0").requires_grad_(mode == "step")
    w = torch.randn(1, n, n).to("cuda:0")
    if mode == "step":
        opt = torch.optim.Adam([a0, p0, pr0], lr=1e-4)

    def fwd():
        op = torch.stack([a0, p0], dim=-1)
        return born_forward(op, pr0, H0, occ0, n_max=M)

    def step():
        opt.zero_grad(set_to_none=True)
        (fwd() * w).sum().backward()
        opt.step()

    with torch.compiler.set_stance("default" if compiled else "force_eager"):
        return run_cell(fwd if mode == "fwd" else step)


def p2_cell(Nz, n, M, pmode, mode, scheme, transport, skipz):
    a, p, probe, H, occu = make_problem(Nz, n, pmode=pmode)
    bounds = even_bounds(Nz, 2)
    occd = occu.to(DEVS[-1])
    grad = mode == "step"
    pr0 = probe.to("cuda:0").requires_grad_(grad)
    # pre-resident leaves + H; objc rebuilt from leaves inside the step
    leaves, Hb = [], []
    for k in range(2):
        lo, hi = bounds[k], bounds[k + 1]
        d = DEVS[k]
        leaves.append((a[:, :, lo:hi].to(d).requires_grad_(grad),
                       p[:, :, lo:hi].to(d).requires_grad_(grad)))
        Hd = H[..., lo:hi, :, :].to(d).contiguous()
        Hb.append((Hd, Hd.conj().contiguous()))
    w = torch.randn(1, n, n).to(DEVS[-1])
    if grad:
        opt = torch.optim.Adam([t for pair in leaves for t in pair] + [pr0],
                               lr=1e-4)

    def fwd():
        blocks = []
        for k in range(2):
            ak, pk = leaves[k]
            blocks.append(dict(objc=(torch.polar(ak, pk) - 1.0).unsqueeze(1),
                               H=Hb[k][0], Hc=Hb[k][1],
                               probe=_hop(pr0, DEVS[k], transport),
                               dev=DEVS[k]))
        return born_forward_dist(blocks, occd, M, carry_scheme=scheme,
                                 skip_zero_planes=skipz,
                                 carry_transport=transport)

    def step():
        opt.zero_grad(set_to_none=True)
        (fwd() * w).sum().backward()
        opt.step()

    with torch.compiler.set_stance("force_eager"):
        return run_cell(fwd if mode == "fwd" else step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--arms", default="main")
    ap.add_argument("--out", default="slicepar_results.csv")
    ap.add_argument("--budget", type=float, default=1800.0)
    args = ap.parse_args()
    n = args.n
    t_start = time.time()

    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    wcsv = csv.writer(f)
    if new:
        wcsv.writerow(["mode", "P", "scheme", "transport", "skipz", "Nz", "n",
                       "M", "pmode", "reps", "ms_med", "ms_iqr", "status"])

    def emit(mode, P, scheme, transport, skipz, Nz, M, pmode, fn):
        if time.time() - t_start > args.budget:
            wcsv.writerow([mode, P, scheme, transport, skipz, Nz, n, M, pmode,
                           0, "", "", "SKIP_BUDGET"])
            return
        try:
            med, iqr = fn()
            wcsv.writerow([mode, P, scheme, transport, skipz, Nz, n, M, pmode,
                           REPS, f"{med:.4f}", f"{iqr:.4f}", "ok"])
            print(f"{mode:4s} P{P} {scheme:15s} {transport:6s} skipz={skipz} "
                  f"Nz={Nz:3d} n={n} M={M} pmode={pmode}: {med:9.3f} ms "
                  f"(iqr {iqr:.3f})")
        except torch.OutOfMemoryError:
            wcsv.writerow([mode, P, scheme, transport, skipz, Nz, n, M, pmode,
                           0, "", "", "OOM"])
            print(f"{mode} P{P} {scheme} Nz={Nz} n={n} M={M}: OOM")
        for d in DEVS:
            torch.cuda.empty_cache()
        f.flush()

    if args.arms == "main":
        arms = [("chain", "peer", False), ("chain_reordered", "peer", False),
                ("direct", "peer", False), ("logscan", "peer", False),
                ("transpose", "peer", False), ("chain", "pinned", False),
                ("transpose", "pinned", False)]
        for Nz in NZS:
            for M in (1, 4):
                for mode in ("fwd", "step"):
                    emit(mode, 1, "single", "-", False, Nz, M, 1,
                         lambda: p1_cell(Nz, n, M, 1, mode))
                    for scheme, tr, sz in arms:
                        emit(mode, 2, scheme, tr, sz, Nz, M, 1,
                             lambda: p2_cell(Nz, n, M, 1, mode, scheme, tr, sz))
    elif args.arms == "compiled":
        for Nz in (32, 128):
            for M in (1, 4):
                for mode in ("fwd", "step"):
                    emit(mode, 1, "single_compiled", "-", False, Nz, M, 1,
                         lambda: p1_cell(Nz, n, M, 1, mode, compiled=True))
    elif args.arms == "pmode4":
        for Nz in (32, 128):
            for M in (1, 4):
                for mode in ("fwd", "step"):
                    emit(mode, 1, "single", "-", False, Nz, M, 4,
                         lambda: p1_cell(Nz, n, M, 4, mode))
                    for scheme, tr in (("chain", "peer"), ("chain", "pinned"),
                                       ("transpose", "pinned")):
                        emit(mode, 2, scheme, tr, False, Nz, M, 4,
                             lambda: p2_cell(Nz, n, M, 4, mode, scheme, tr,
                                             False))
    elif args.arms == "skipz":
        for Nz in (4, 8, 16, 32):
            for scheme, tr in (("chain", "peer"), ("chain", "pinned")):
                for sz in (False, True):
                    for mode in ("fwd", "step"):
                        emit(mode, 2, scheme, tr, sz, Nz, 4, 1,
                             lambda: p2_cell(Nz, n, 4, 1, mode, scheme, tr, sz))
    f.close()


if __name__ == "__main__":
    main()

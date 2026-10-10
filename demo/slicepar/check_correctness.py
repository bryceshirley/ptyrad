"""Correctness gates for carry schemes vs single-device born_forward.

Gate 1: forward detector intensity rel err < 1e-6 (fp32).
Gate 2: object (amp, phase) and probe grads rel err < 1e-5.
Gate 3 (mode=hash): 50 repeated forwards bitwise identical; driver script
        compares the printed digest against a CUDA_LAUNCH_BLOCKING=1 run.

P=4 runs logically on 2 physical GPUs (blocks on cuda:0,1,0,1) when only two
devices are visible — same scheme code paths, real cross-device hops.

Usage: check_correctness.py [hash]
"""

import hashlib
import sys

import torch

torch.compiler.set_stance("force_eager")

from common import build_dist, even_bounds, make_problem, single_inputs  # noqa: E402

from ptyrad.forward_models import born_forward  # noqa: E402
from ptyrad.forward_models.born_dist import born_forward_dist  # noqa: E402

SCHEMES = ["chain", "chain_reordered", "direct", "logscan", "transpose"]
NDEV = torch.cuda.device_count()
SKIPZ = "skipz" in sys.argv  # gate the skip_zero_planes variant
TRANSPORT = "pinned" if "pinned" in sys.argv else "peer"


def devices_for(P):
    return [f"cuda:{k % NDEV}" for k in range(P)]


def rel(x, y):
    return ((x - y).norm() / y.norm()).item()


def run_case(P, Nz, M, pmode, n=256, seed=0):
    a, p, probe, H, occu = make_problem(Nz, n, pmode=pmode, seed=seed)
    devs = devices_for(P)
    bounds = even_bounds(Nz, P)

    # single-device reference (eager) with grads
    pr0 = probe.to("cuda:0").requires_grad_(True)
    op, (a0, p0), prs, Hs = single_inputs(a, p, pr0, H, requires_grad=True)
    w = torch.randn(1, n, n, generator=torch.Generator().manual_seed(1)).to("cuda:0")
    dp_ref = born_forward(op, prs, Hs, occu.to("cuda:0"), n_max=M)
    (dp_ref * w).sum().backward()
    ga_ref = a0.grad.clone()
    gp_ref = p0.grad.clone()
    gpr_ref = pr0.grad.clone()

    rows = []
    for scheme in SCHEMES:
        prd = probe.to("cuda:0").requires_grad_(True)
        blocks, leaves = build_dist(a, p, prd, H, bounds, devs, requires_grad=True)
        dp = born_forward_dist(blocks, occu.to(devs[-1]), M, carry_scheme=scheme,
                               skip_zero_planes=SKIPZ,
                               carry_transport=TRANSPORT)
        e_fwd = rel(dp.to("cuda:0"), dp_ref)
        (dp * w.to(devs[-1])).sum().backward()
        ga = torch.cat([ak.grad.to("cuda:0") for ak, _ in leaves], dim=2)
        gp = torch.cat([pk.grad.to("cuda:0") for _, pk in leaves], dim=2)
        e_ga = rel(ga, ga_ref)
        e_gp = rel(gp, gp_ref)
        e_gpr = rel(prd.grad.to("cuda:0"), gpr_ref)
        ok = e_fwd < 1e-6 and max(e_ga, e_gp, e_gpr) < 1e-5
        rows.append((P, Nz, M, pmode, scheme, e_fwd, e_ga, e_gp, e_gpr, ok))
    return rows


def main():
    print("P,Nz,M,pmode,scheme,relerr_fwd,relerr_grad_obja,relerr_grad_objp,"
          "relerr_grad_probe,pass")
    all_ok = True
    for P in (2, 4):
        for Nz in (7, 32, 129):
            for M in (1, 4):
                for pmode in (1, 4):
                    for r in run_case(P, Nz, M, pmode):
                        print(",".join(str(x) for x in r[:5])
                              + "," + ",".join(f"{v:.3e}" for v in r[5:9])
                              + f",{r[9]}")
                        all_ok &= r[9]
    print("ALL_PASS" if all_ok else "SOME_FAIL")


def main_hash():
    """50 repeated forwards per scheme; print one digest per scheme."""
    for P in (2, 4):
        a, p, probe, H, occu = make_problem(32, 256, pmode=1, seed=0)
        devs = devices_for(P)
        bounds = even_bounds(32, P)
        for scheme in SCHEMES:
            digests = set()
            for _ in range(50):
                blocks, _ = build_dist(a, p, probe.to("cuda:0"), H, bounds, devs)
                dp = born_forward_dist(blocks, occu.to(devs[-1]), 4,
                                       carry_scheme=scheme,
                                       carry_transport=TRANSPORT)
                digests.add(hashlib.sha256(
                    dp.cpu().numpy().tobytes()).hexdigest())
            print(f"P{P} {scheme} unique={len(digests)} digest={min(digests)[:16]}")


if __name__ == "__main__":
    if "hash" in sys.argv[1:]:
        main_hash()
    else:
        main()

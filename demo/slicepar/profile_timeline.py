"""Step 2: per-GPU phase timeline of born_forward_dist via CUDA events.

Replicates the engine's order loop with per-(order, block, phase) CUDA
events on each device's current stream; cross-device clocks are aligned by
recording a t0 event on every device back-to-back after a full sync (skew
~ a few us, fine at ms scale). The replica's output is checked against the
library forward each run.

Also dumps a torch.profiler chrome trace of forward+backward+step to verify
the BACKWARD is concurrent across GPUs (autograd uses one worker thread per
device).

Usage: profile_timeline.py <scheme> <Nz> <M> [n=256] [tag]
Writes timeline_<tag>.json (segments) and trace_<tag>.json.gz (chrome).
"""

import gzip
import json
import sys

import torch
from torch.fft import fft2, ifft2

torch.compiler.set_stance("force_eager")

from common import build_dist, even_bounds, make_problem  # noqa: E402

from ptyrad.forward_models.born_dist import _hop, born_forward_dist  # noqa: E402
from ptyrad.utils import fftshift2  # noqa: E402

SCHEME = sys.argv[1]
NZ = int(sys.argv[2])
M = int(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 256
TAG = sys.argv[5] if len(sys.argv) > 5 else f"{SCHEME}_Nz{NZ}_M{M}_n{N}"
import os
TRANSPORT = os.environ.get("SLICEPAR_TRANSPORT", "peer")
P = 2
DEVS = [f"cuda:{k}" for k in range(P)]


def instrumented_forward(blocks, occu, segs):
    """Chain/direct/logscan order loop with CUDA events per phase+block."""
    pmode = blocks[0]["probe"].shape[1]
    Ny, Nx = blocks[0]["probe"].shape[-2:]
    Nz = sum(b["objc"].shape[3] for b in blocks)

    def ev(dev):
        e = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(dev):
            e.record()
        return e

    for b in blocks:
        pk = fft2(b["probe"]).view(-1, pmode, 1, 1, Ny, Nx)
        b["pk"] = pk
        b["psi"] = ifft2(b["H"] * pk)
    Psi_M = blocks[-1]["pk"].squeeze(3)
    Mo = min(M, Nz)
    for n in range(Mo):
        last = n == Mo - 1
        # phase A, per block
        for i, b in enumerate(blocks):
            e0 = ev(b["dev"])
            win0 = n if i == 0 else 0
            b["win0"] = win0
            sc = (fft2(b["objc"][..., win0:, :, :] * b["psi"])
                  * b["Hc"][..., win0:, :, :])
            b["cs"] = torch.cumsum(sc, dim=3)
            b["t"] = b["cs"][..., -1, :, :]
            segs.append((b["dev"], f"A{n}", e0, ev(b["dev"])))
        # carry (chain at P=2 == direct == logscan)
        e_src = ev(blocks[0]["dev"])
        prefix = _hop(blocks[0]["t"], blocks[1]["dev"], TRANSPORT)
        blocks[1]["prefix"] = prefix
        e_arr = ev(blocks[1]["dev"])  # fires when dst stream has the prefix
        D_n = prefix + blocks[1]["t"]
        segs.append((blocks[0]["dev"], f"B{n}.src", e_src, ev(blocks[0]["dev"])))
        segs.append((blocks[1]["dev"], f"B{n}.dst", e_arr, ev(blocks[1]["dev"])))
        # phase C, per block
        if not last:
            for i, b in enumerate(blocks):
                e0 = ev(b["dev"])
                if i == 0:
                    if b["cs"].shape[3] >= 2:
                        b["psi"] = ifft2(b["cs"][..., :-1, :, :]
                                         * b["H"][..., b["win0"] + 1:, :, :])
                else:
                    p = b["prefix"].unsqueeze(3)
                    b["psi"] = ifft2(
                        torch.cat([p, b["cs"][..., :-1, :, :] + p], dim=3)
                        * b["H"])
                segs.append((b["dev"], f"C{n}", e0, ev(b["dev"])))
        Psi_M = Psi_M + D_n
    nw = (occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(Psi_M.abs().square() * nw, dim=(1, 2)) + 1e-10)


def main():
    a, p, probe, H, occu = make_problem(NZ, N, pmode=1)
    bounds = even_bounds(NZ, P)
    occd = occu.to(DEVS[-1])

    def fresh():
        return build_dist(a, p, probe.to("cuda:0"), H, bounds, DEVS)[0]

    # warm-up + library reference
    for _ in range(5):
        ref = born_forward_dist(fresh(), occd, M, carry_scheme=SCHEME,
                                carry_transport=TRANSPORT)
    torch.cuda.synchronize(0), torch.cuda.synchronize(1)

    blocks = fresh()
    torch.cuda.synchronize(0), torch.cuda.synchronize(1)
    t0 = {}
    for d in DEVS:  # aligned origin events, recorded back-to-back
        e = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(d):
            e.record()
        t0[d] = e
    segs = []
    dp = instrumented_forward(blocks, occd, segs)
    torch.cuda.synchronize(0), torch.cuda.synchronize(1)
    err = ((dp - ref).norm() / ref.norm()).item()

    out = [dict(dev=d, name=nm, start=t0[d].elapsed_time(e0),
                end=t0[d].elapsed_time(e1))
           for d, nm, e0, e1 in segs]
    with open(f"timeline_{TAG}.json", "w") as f:
        json.dump(dict(scheme=SCHEME, Nz=NZ, M=M, n=N, relerr_vs_lib=err,
                       segments=out), f, indent=1)

    # overlap fraction of phase A across GPUs
    byo = {}
    for s in out:
        if s["name"].startswith("A"):
            byo.setdefault(s["name"], {})[s["dev"]] = (s["start"], s["end"])
    fr = []
    for nm, dd in sorted(byo.items()):
        iv = list(dd.values())
        inter = max(0.0, min(e for _, e in iv) - max(s for s, _ in iv))
        mean = sum(e - s for s, e in iv) / len(iv)
        fr.append((nm, inter / mean if mean > 0 else 0.0))
    print(f"relerr vs library: {err:.2e}")
    for nm, f_ in fr:
        print(f"phase1 overlap {nm}: {f_:.3f}")

    # chrome trace of one optimizer step for backward concurrency
    leaves_blocks, leaves = build_dist(a, p, probe.to("cuda:0"), H, bounds,
                                       DEVS, requires_grad=True)
    w = torch.randn(1, N, N).to(DEVS[-1])
    opt = torch.optim.Adam([t for pair in leaves for t in pair], lr=1e-3)
    for _ in range(3):  # warm-up incl. backward
        blk, lv = build_dist(a, p, probe.to("cuda:0"), H, bounds, DEVS,
                             requires_grad=True)
        (born_forward_dist(blk, occd, M, carry_scheme=SCHEME,
                           carry_transport=TRANSPORT) * w).sum().backward()
    torch.cuda.synchronize(0), torch.cuda.synchronize(1)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
        blk, lv = build_dist(a, p, probe.to("cuda:0"), H, bounds, DEVS,
                             requires_grad=True)
        loss = (born_forward_dist(blk, occd, M, carry_scheme=SCHEME,
                                  carry_transport=TRANSPORT) * w).sum()
        loss.backward()
        opt.step()
        torch.cuda.synchronize(0), torch.cuda.synchronize(1)
    pr.export_chrome_trace(f"trace_{TAG}.json")
    with open(f"trace_{TAG}.json", "rb") as f:
        data = f.read()
    with gzip.open(f"trace_{TAG}.json.gz", "wb") as f:
        f.write(data)
    import os
    os.remove(f"trace_{TAG}.json")
    print(f"wrote timeline_{TAG}.json and trace_{TAG}.json.gz")


if __name__ == "__main__":
    main()

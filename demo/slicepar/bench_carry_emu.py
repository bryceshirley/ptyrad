"""Carry-phase emulation for logical P>=3 on 2 physical GPUs.

Blocks are placed round-robin on [cuda:0, cuda:1], so at P=4 the chain's
3 hops are all cross-device and serial — the same latency chain as
physical P=4 on an equal-alpha (all-PHB) topology. Known distortions,
stated with the results:
  * direct: the (0->2) and (1->3) sends are same-device (free) -> 4 of 6
    sends are real; mild optimism, but fan-in sharing one link is
    representative of a PHB box where everything funnels through the host.
  * logscan: round 2's sends (stride 2) are same-device -> emulates only
    round 1 of 2; treat as a LOWER bound (model says ceil(log2 P) rounds).
Times the actual library carries on fake block dicts (t pre-resident), so
the enqueue pattern is exactly the engine's. Compute phases are NOT
emulated (two blocks share a GPU).
"""

import statistics as st
import time

import torch

from ptyrad.forward_models.born_dist import (_carry_chain, _carry_direct,
                                             _carry_logscan)

DEVS = ["cuda:0", "cuda:1"]
REPS, WARM = 40, 8
CARRIES = {"chain": _carry_chain, "direct": _carry_direct,
           "logscan": _carry_logscan}


def sync_all():
    for d in DEVS:
        torch.cuda.synchronize(d)


def run(P, n, scheme, transport):
    devs = [DEVS[k % 2] for k in range(P)]
    blocks = [dict(dev=d, t=torch.randn(1, 1, 1, n, n, dtype=torch.complex64,
                                        device=d)) for d in devs]
    fn = CARRIES[scheme]

    def once():
        return fn(blocks, transport)

    for _ in range(WARM):
        once()
    ts = []
    for _ in range(REPS):
        sync_all()
        t0 = time.perf_counter()
        once()
        sync_all()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return st.median(ts), ts[(3 * len(ts)) // 4] - ts[len(ts) // 4], ts[0]


if __name__ == "__main__":
    print(f"{'n':>5} {'transport':>9} {'P':>2} | "
          + " | ".join(f"{s:>22}" for s in CARRIES))
    for n in (256, 2048):
        for tr in ("peer", "pinned"):
            for P in (2, 3, 4):
                cells = []
                for s in CARRIES:
                    med, iqr, mn = run(P, n, s, tr)
                    cells.append(f"{med:7.3f} (iqr {iqr:5.3f}, min {mn:5.3f})"[:22])
                print(f"{n:>5} {tr:>9} {P:>2} | " + " | ".join(f"{c:>22}" for c in cells))

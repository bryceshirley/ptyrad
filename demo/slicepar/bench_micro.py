"""Step 3: communication microbenchmarks on the visible GPU pair.

Measures, for complex64 planes n in {256,512,1024,2048,4096}:
  1. sync .to(dev)                      (host-blocking)
  2. .to(dev, non_blocking=True)        (event-ordered, wall to completion)
  3. bidirectional exchange 0<->1       (the 2-GPU analog of fan-in)
  4. volume all-to-all (slabs -> column shards), P^2 chunks on side streams
  5. complex32 (half) carries
  6. pinned-staged manual copy: D2H to a pinned bounce + H2D on the peer,
     plus a 4-chunk pipelined version (candidate replacement transport)
  7. tiny copies (8 B, 4 KB) for the latency intercept alpha

Writes micro_results.csv (case,n,bytes,ms_median,ms_iqr).
"""

import csv
import statistics as st
import time

import torch

DEVS = ["cuda:0", "cuda:1"]
NS = [256, 512, 1024, 2048, 4096]
REPS = 30
WARM = 5


def sync_all():
    for d in DEVS:
        torch.cuda.synchronize(d)


def timeit(fn, reps=REPS, warm=WARM):
    for _ in range(warm):
        fn()
    sync_all()
    ts = []
    for _ in range(reps):
        sync_all()
        t0 = time.perf_counter()
        fn()
        sync_all()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    med = st.median(ts)
    iqr = ts[(3 * len(ts)) // 4] - ts[len(ts) // 4]
    return med, iqr


def main():
    rows = []

    def rec(case, n, nbytes, med, iqr):
        rows.append((case, n, nbytes, f"{med:.4f}", f"{iqr:.4f}"))
        print(f"{case:24s} n={n:5d} {nbytes/1e6:9.2f} MB  {med:8.4f} ms "
              f"(iqr {iqr:.4f})")

    # 7. latency intercept
    for nbytes, shape in ((8, (1,)), (4096, (512,))):
        x = torch.randn(shape, dtype=torch.complex64, device=DEVS[0])
        med, iqr = timeit(lambda: x.to(DEVS[1]))
        rec("tiny_sync_to", 0, nbytes, med, iqr)

    for n in NS:
        b = 8 * n * n
        x = torch.randn(n, n, dtype=torch.complex64, device=DEVS[0])
        y = torch.randn(n, n, dtype=torch.complex64, device=DEVS[1])

        med, iqr = timeit(lambda: x.to(DEVS[1]))
        rec("sync_to", n, b, med, iqr)

        med, iqr = timeit(lambda: x.to(DEVS[1], non_blocking=True))
        rec("nonblocking_to", n, b, med, iqr)

        def bidir():
            a = x.to(DEVS[1], non_blocking=True)
            c = y.to(DEVS[0], non_blocking=True)
            return a, c
        med, iqr = timeit(bidir)
        rec("bidir_exchange", n, b, med, iqr)

        # 5. half-precision carry
        xh = x.to(torch.complex32)
        med, iqr = timeit(lambda: xh.to(DEVS[1], non_blocking=True))
        rec("chalf_to", n, b // 2, med, iqr)

        # 6. pinned-staged
        pin = torch.empty(n, n, dtype=torch.complex64, pin_memory=True)
        dst = torch.empty(n, n, dtype=torch.complex64, device=DEVS[1])
        s1 = torch.cuda.Stream(DEVS[1])
        ev = torch.cuda.Event()

        def staged():
            pin.copy_(x, non_blocking=True)  # D2H on dev0 current stream
            ev.record(torch.cuda.current_stream(DEVS[0]))
            with torch.cuda.stream(s1):
                s1.wait_event(ev)
                dst.copy_(pin, non_blocking=True)  # H2D on dev1 side stream
        med, iqr = timeit(staged)
        rec("pinned_staged", n, b, med, iqr)

        # 6b. pipelined in 4 chunks
        nch = 4
        pins = [torch.empty(n // nch, n, dtype=torch.complex64,
                            pin_memory=True) for _ in range(nch)]
        evs = [torch.cuda.Event() for _ in range(nch)]
        xs = list(x.chunk(nch, 0))
        ds = list(dst.chunk(nch, 0))

        def staged_pipe():
            for k in range(nch):
                pins[k].copy_(xs[k], non_blocking=True)
                evs[k].record(torch.cuda.current_stream(DEVS[0]))
            with torch.cuda.stream(s1):
                for k in range(nch):
                    s1.wait_event(evs[k])
                    ds[k].copy_(pins[k], non_blocking=True)
        med, iqr = timeit(staged_pipe)
        rec("pinned_staged_pipe4", n, b, med, iqr)

    # 4. volume all-to-all: N planes of n^2, slabs->column shards, P=2
    for n in (256, 1024, 2048):
        for N in (32, 128):
            vol = [torch.randn(N // 2, n, n, dtype=torch.complex64,
                               device=d) for d in DEVS]
            side = [torch.cuda.Stream(d) for d in DEVS]

            def a2a():
                out = [[None, None], [None, None]]
                for i in range(2):
                    for j in range(2):
                        chunk = vol[i][..., j * (n // 2):(j + 1) * (n // 2), :]
                        if i == j:
                            out[j][i] = chunk
                        else:
                            with torch.cuda.stream(side[i]):
                                out[j][i] = chunk.contiguous().to(
                                    DEVS[j], non_blocking=True)
                return out
            med, iqr = timeit(a2a, reps=20)
            nbytes = 2 * (N // 2) * (n // 2) * n * 8  # cross-device bytes
            rec(f"a2a_vol_N{N}", n, nbytes, med, iqr)

    with open("micro_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "n", "bytes", "ms_median", "ms_iqr"])
        w.writerows(rows)
    print("wrote micro_results.csv")


if __name__ == "__main__":
    main()

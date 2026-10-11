"""Slice-split optimisation experiments (GPUs 1&2, PHB/no-NVLink).

Part 1 — ACCURACY of the half-precision carry.
    The carry is a correction term: the detector field is
    Psi = Psi0 + sum_{m>=1} c_m D_m, and the cross-device carry only enters the
    D_m (m>=1), i.e. the scattered part, never the dominant unscattered Psi0.
    Shipping it at complex32 perturbs only that correction by ~2^-11. We show
    the induced forward AND object-gradient error is orders of magnitude below
    the Born truncation error delta_psi(M) that the model already accepts, and
    that it does not accumulate / bias over reconstruction iterations.

Part 2 — OFFSET PIPELINE with the carry transfer on a side stream.
    The compute pipeline already overlaps (P0 order n+1 || P1 order n). The
    only thing left on the critical path is moving carry_n. Put that copy on a
    dedicated dev0 stream + event so it overlaps P0's next-order compute, and
    measure how close forward time gets to the no-transfer floor.
"""
import os, sys, time
import torch
from torch.fft import fft2, ifft2
sys.path.insert(0, "/home/dnz75396/ptyrad/src")
from ptyrad.forward_models.born import born_forward
from ptyrad.utils import fftshift2

torch.manual_seed(0)
Nz, Ny, Nx, PMODE, OMODE = 21, 256, 256, 4, 1
S = 12
DEV0, DEV1 = "cuda:0", "cuda:1"
EPS = 1e-10


def make_inputs(N, device, scatter=0.02):
    obj = torch.randn(N, OMODE, Nz, Ny, Nx, 2, device=device) * scatter
    obj[..., 0] += 1.0
    probe = torch.randn(N, PMODE, Ny, Nx, dtype=torch.complex64, device=device)
    H = torch.exp(1j * 0.01 * torch.randn(1, 1, 1, Nz, Ny, Nx, device=device)).to(torch.complex64)
    occu = torch.ones(OMODE, device=device)
    target = torch.rand(N, Ny, Nx, device=device) + 0.1
    return obj, probe, H, occu, target


def loss_fn(I, target):
    return ((I.clamp_min(1e-10).sqrt() - target.sqrt()) ** 2).mean()


# ---------------------------------------------------------------------------
# slice-split forward (single device variant for the accuracy test, so we can
# compare carry precision cleanly without cross-device noise), n_max = M.
# carry_cast: identity (c64) or a complex32 round-trip (the half-precision path)
# ---------------------------------------------------------------------------
def slice_forward_1dev(obj, probe, H, occu, M, carry_cast):
    O = torch.polar(obj[..., 0], obj[..., 1])
    objc = (O - 1.0).unsqueeze(1)
    pk = fft2(probe).view(-1, PMODE, 1, 1, Ny, Nx)
    psi0 = ifft2(H[..., :S, :, :] * pk)          # block0 [0,S)
    psi1 = ifft2(H[..., S:, :, :] * pk)          # block1 [S,Nz)
    objc0, objc1 = objc[..., :S, :, :], objc[..., S:, :, :]
    H0, H0c = H[..., :S, :, :], H[..., :S, :, :].conj()
    H1, H1c = H[..., S:, :, :], H[..., S:, :, :].conj()
    Psi_M = pk.squeeze(3)
    for n in range(M):
        sc0 = fft2(objc0[..., n:, :, :] * psi0) * H0c[..., n:, :, :]
        cs0 = torch.cumsum(sc0, dim=3)
        carry = carry_cast(cs0[..., -1, :, :]).unsqueeze(3)   # <-- precision knob
        sc1 = fft2(objc1 * psi1) * H1c
        cs1 = torch.cumsum(sc1, dim=3) + carry
        D_n = cs1[..., -1, :, :]
        Psi_M = Psi_M + D_n
        if n < M - 1:
            psi0 = ifft2(cs0[..., :-1, :, :] * H0[..., n + 1:, :, :])
            psi1 = ifft2(torch.cat([carry, cs1[..., :-1, :, :]], dim=3) * H1)
    nw = (occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(Psi_M.abs().square() * nw, dim=(1, 2)) + EPS)


_id = lambda c: c
_half = lambda c: c.to(torch.complex32).to(torch.complex64)


def part1_accuracy():
    print("=" * 72)
    print("PART 1 — half-precision carry accuracy (single device, M=6)")
    print("=" * 72)
    M = 6
    for scatter, tag in [(0.02, "weak (PSO-like)"), (0.06, "3x stronger")]:
        N = 8
        obj, probe, H, occu, target = make_inputs(N, DEV0, scatter=scatter)
        obj_g = obj.clone().requires_grad_(True)
        obj_h = obj.clone().requires_grad_(True)
        # exact multislice (n_max = Nz) as ground truth
        with torch.no_grad():
            I_exact = born_forward(obj, probe, H, occu, n_max=Nz).clone()
        # forward: full vs half carry
        I_full = slice_forward_1dev(obj_g, probe, H, occu, M, _id)
        I_half = slice_forward_1dev(obj_h, probe, H, occu, M, _half)
        trunc = ((I_full.detach() - I_exact).norm() / I_exact.norm()).item()
        fwd_half = ((I_half.detach() - I_full.detach()).norm() / I_full.detach().norm()).item()
        # gradients wrt object
        loss_fn(I_full, target).backward()
        loss_fn(I_half, target).backward()
        g_full, g_half = obj_g.grad, obj_h.grad
        grad_err = ((g_half - g_full).norm() / g_full.norm()).item()
        # bias check: mean signed component of the grad difference vs its magnitude
        dg = (g_half - g_full)
        bias = (dg.mean().abs() / dg.abs().mean().clamp_min(1e-30)).item()
        print(f"\n [{tag}, scatter={scatter}]")
        print(f"   Born truncation error delta_psi(6) vs multislice : {trunc:.3e}")
        print(f"   half-carry forward error                         : {fwd_half:.3e}"
              f"   ({trunc/max(fwd_half,1e-30):.0f}x below truncation)")
        print(f"   half-carry OBJECT-GRADIENT rel error             : {grad_err:.3e}")
        print(f"   gradient-error bias (mean/|mean|, ~0 = unbiased) : {bias:.3e}")

    # accumulation over iterations: run GD with full vs half carry, compare drift
    print("\n [accumulation over 50 GD steps, weak object]")
    N = 4
    obj0, probe, H, occu, target = make_inputs(N, DEV0, scatter=0.02)
    xs = {}
    for cast, name in [(_id, "full"), (_half, "half")]:
        x = obj0.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=1e-3)
        for _ in range(50):
            opt.zero_grad()
            loss_fn(slice_forward_1dev(x, probe, H, occu, M, cast), target).backward()
            opt.step()
        xs[name] = x.detach()
    drift = ((xs["half"] - xs["full"]).norm() / xs["full"].norm()).item()
    print(f"   object drift after 50 steps (half vs full carry) : {drift:.3e}")


# ---------------------------------------------------------------------------
# Part 2 — two-device offset pipeline, carry copy on default vs side stream.
# forward-only (no autograd) so manual streams are safe; isolates scan+transfer.
# ---------------------------------------------------------------------------
def build_2dev(N, half):
    obj, probe, H, occu, _ = make_inputs(N, DEV0)
    O = torch.polar(obj[..., 0], obj[..., 1])
    objc = (O - 1.0).unsqueeze(1)
    pk0 = fft2(probe).view(-1, PMODE, 1, 1, Ny, Nx)
    d = dict(
        objc0=objc[..., :S, :, :].contiguous(),
        H0=H[..., :S, :, :].contiguous(), H0c=H[..., :S, :, :].conj().contiguous(),
        psi0_init=ifft2(H[..., :S, :, :] * pk0),
        objc1=objc[..., S:, :, :].to(DEV1).contiguous(),
        H1=H[..., S:, :, :].to(DEV1).contiguous(), H1c=H[..., S:, :, :].conj().to(DEV1).contiguous(),
        psi1_init=ifft2(H[..., S:, :, :].to(DEV1) * pk0.to(DEV1)),
        pk1=pk0.to(DEV1), occu1=occu.to(DEV1), half=half,
    )
    return d


def forward_2dev(d, M, overlap, copy_stream):
    objc0, H0, H0c, psi0 = d["objc0"], d["H0"], d["H0c"], d["psi0_init"]
    objc1, H1, H1c, psi1 = d["objc1"], d["H1"], d["H1c"], d["psi1_init"]
    Psi_M = d["pk1"].squeeze(3)

    def cast(c):
        return c.to(torch.complex32) if d["half"] else c

    def xfer(carry0):
        if not overlap:
            return cast(carry0).to(DEV1, non_blocking=True).to(torch.complex64)
        copy_stream.wait_stream(torch.cuda.current_stream(DEV0))
        with torch.cuda.stream(copy_stream):
            c1 = cast(carry0).to(DEV1, non_blocking=True)
        ev = torch.cuda.Event(); ev.record(copy_stream)
        torch.cuda.current_stream(DEV1).wait_event(ev)
        return c1.to(torch.complex64)

    def blk0(n, psi0):
        sc0 = fft2(objc0[..., n:, :, :] * psi0) * H0c[..., n:, :, :]
        cs0 = torch.cumsum(sc0, dim=3)
        psi0n = ifft2(cs0[..., :-1, :, :] * H0[..., n + 1:, :, :]) if n < M - 1 else None
        return cs0[..., -1, :, :], psi0n

    def blk1(n, psi1, c1):
        c = c1.unsqueeze(3)
        cs1 = torch.cumsum(fft2(objc1 * psi1) * H1c, dim=3) + c
        psi1n = ifft2(torch.cat([c, cs1[..., :-1, :, :]], dim=3) * H1) if n < M - 1 else None
        return cs1[..., -1, :, :], psi1n

    carry, psi0 = blk0(0, psi0)
    c1 = xfer(carry)
    for n in range(M):
        D_n, psi1n = blk1(n, psi1, c1)
        Psi_M = Psi_M + D_n
        if n < M - 1:
            carry, psi0 = blk0(n + 1, psi0)
            c1 = xfer(carry)
            psi1 = psi1n
    nw = (d["occu1"] / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(Psi_M.abs().square() * nw, dim=(1, 2)) + EPS)


def forward_2dev_pinned(d, M, bufs):
    """Offset pipeline with a TRUE async host-staged carry transfer:
    D2H on a dev0 copy stream (overlaps P0's next-order compute) then H2D on a
    dev1 copy stream (overlaps P1 compute), double-buffered. This is the
    overlap your offset pipeline needs on a PHB box with no P2P/NVLink."""
    objc0, H0, H0c, psi0 = d["objc0"], d["H0"], d["H0c"], d["psi0_init"]
    objc1, H1, H1c, psi1 = d["objc1"], d["H1"], d["H1c"], d["psi1_init"]
    Psi_M = d["pk1"].squeeze(3)
    cs0_stream, pins, recvs, cs1_copy = bufs["s0"], bufs["pin"], bufs["recv"], bufs["s1"]

    def blk0(n, psi0):
        sc0 = fft2(objc0[..., n:, :, :] * psi0) * H0c[..., n:, :, :]
        cs0 = torch.cumsum(sc0, dim=3)
        psi0n = ifft2(cs0[..., :-1, :, :] * H0[..., n + 1:, :, :]) if n < M - 1 else None
        return cs0[..., -1, :, :], psi0n

    def blk1(n, psi1, c1):
        c = c1.unsqueeze(3)
        cs1 = torch.cumsum(fft2(objc1 * psi1) * H1c, dim=3) + c
        psi1n = ifft2(torch.cat([c, cs1[..., :-1, :, :]], dim=3) * H1) if n < M - 1 else None
        return cs1[..., -1, :, :], psi1n

    def xfer(carry0, slot):
        # D2H on dev0 copy stream (overlaps dev0 default-stream compute)
        cs0_stream.wait_stream(torch.cuda.current_stream(DEV0))
        with torch.cuda.stream(cs0_stream):
            pins[slot].copy_(carry0, non_blocking=True)
        ev = torch.cuda.Event(); ev.record(cs0_stream)
        # H2D on dev1 copy stream, gated on the D2H event
        cs1_copy.wait_event(ev)
        with torch.cuda.stream(cs1_copy):
            recvs[slot].copy_(pins[slot], non_blocking=True)
        ev2 = torch.cuda.Event(); ev2.record(cs1_copy)
        torch.cuda.current_stream(DEV1).wait_event(ev2)
        return recvs[slot]

    carry, psi0 = blk0(0, psi0)
    c1 = xfer(carry, 0)
    for n in range(M):
        D_n, psi1n = blk1(n, psi1, c1)
        Psi_M = Psi_M + D_n
        if n < M - 1:
            carry, psi0 = blk0(n + 1, psi0)
            c1 = xfer(carry, (n + 1) % 2)
            psi1 = psi1n
    nw = (d["occu1"] / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(Psi_M.abs().square() * nw, dim=(1, 2)) + EPS)


def make_bufs(N):
    shape = (N, PMODE, OMODE, Ny, Nx)
    return dict(
        s0=torch.cuda.Stream(device=DEV0), s1=torch.cuda.Stream(device=DEV1),
        pin=[torch.empty(shape, dtype=torch.complex64, pin_memory=True) for _ in range(2)],
        recv=[torch.empty(shape, dtype=torch.complex64, device=DEV1) for _ in range(2)],
    )


def time_fwd(fn, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(DEV0); torch.cuda.synchronize(DEV1)
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(DEV0); torch.cuda.synchronize(DEV1)
        t0 = time.perf_counter(); fn()
        torch.cuda.synchronize(DEV0); torch.cuda.synchronize(DEV1)
        ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2]


def time_1dev(N, M):
    obj, probe, H, occu, _ = make_inputs(N, DEV0)
    def f(): slice_forward_1dev(obj, probe, H, occu, M, _id)
    # reuse the 2-device timer's sync (both devices) for consistency
    for _ in range(5): f()
    torch.cuda.synchronize(DEV0)
    ts = []
    for _ in range(30):
        torch.cuda.synchronize(DEV0)
        t0 = time.perf_counter(); f(); torch.cuda.synchronize(DEV0)
        ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2]


def part2_pipeline():
    print("\n" + "=" * 72)
    print("PART 2 — offset pipeline, carry copy default vs side-stream (forward only)")
    print("=" * 72)
    M = 6
    copy_stream = torch.cuda.Stream(device=DEV0)
    # correctness of the pinned-async path vs the plain default-stream forward
    dchk = build_2dev(1, half=False)
    bchk = make_bufs(1)
    I_def = forward_2dev(dchk, M, False, copy_stream)
    I_pin = forward_2dev_pinned(dchk, M, bchk)
    print(f" pinned-async correctness: rel err vs default = "
          f"{((I_pin - I_def).norm() / I_def.norm()).item():.2e}")
    for N in (1, 32):
        t1 = time_1dev(N, M)
        d64 = build_2dev(N, half=False)
        bufs = make_bufs(N)
        t_def = time_fwd(lambda: forward_2dev(d64, M, False, copy_stream))
        t_pin = time_fwd(lambda: forward_2dev_pinned(d64, M, bufs))
        print(f"\n N={N}: 1-GPU {t1*1e3:6.2f} ms")
        print(f"   c64 default-stream (sync copy) {t_def*1e3:6.2f} ms ({t1/t_def:.2f}x)")
        print(f"   c64 pinned-async overlap       {t_pin*1e3:6.2f} ms ({t1/t_pin:.2f}x)")


if __name__ == "__main__":
    print(f"GPUs visible: {torch.cuda.device_count()} | split S={S}")
    part1_accuracy()
    part2_pipeline()

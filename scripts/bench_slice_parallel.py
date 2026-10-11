"""Micro-benchmark: born6 forward+backward, slice-parallel across 2 GPUs.

Splits the SLICE axis (depth) across two devices: planes [0, s) on dev0,
planes [s, Nz) on dev1. This is the paper's distributed-Sigma regime
(paper-1.tex sec. "Scattering Operator Decomposition", ~line 514): the
cumulative sum Sigma that assembles the downstream field couples the planes,
so a slice split needs the running total carried across the device boundary
EVERY Born order. We use the cheap two-level scan (one carry plane per order,
dev0->dev1) rather than a full all-to-all transpose.

The point of the benchmark: unlike probe-mode parallelism (two independent
compiled born_forward calls, cudagraphs intact), a slice split has a
per-order cross-device dependency (block1 order n needs block0's order-n
carry, and block1's next state at plane s IS that carry). That serial carry
chain cannot live in one compiled/cudagraph region, so the forward is driven
by per-order Python orchestration. We measure whether overlapping the two
device streams hides the carry cost, and compare to:
  - 1-GPU compiled born_forward (production floor),
  - 1-GPU eager recursion (fair eager baseline for the parallelism ratio),
  - 2-GPU mode-split (bench_mode_parallel.py: 7.03 / 198 ms).

Correctness is checked against born_forward on identical inputs.
"""
import os, sys, time
import torch
from torch.fft import fft2, ifft2
sys.path.insert(0, "/home/dnz75396/ptyrad/src")
from ptyrad.forward_models.born import born_forward
from ptyrad.utils import fftshift2

torch.manual_seed(0)
Nz, Ny, Nx, PMODE, OMODE, NMAX = 21, 256, 256, 4, 1, 6
S = int(os.environ.get("SPLIT_S", 12))  # block0=[0,S), block1=[S,Nz)
# NOTRANSFER=1 replaces the cross-device carry with a cached zero on DEV1:
# BREAKS correctness, but isolates the compute-only critical path (no data
# movement) to separate "how fast data moves" from serialization/launch cost.
NOTRANSFER = os.environ.get("NOTRANSFER", "0") == "1"
HALFCARRY = os.environ.get("HALFCARRY", "0") == "1"  # ship carry at complex32
DEV0, DEV1 = "cuda:0", "cuda:1"
EPS = 1e-10
assert NMAX <= S, "block1 stays the full [s,Nz) window only while every order n < s"
_ZERO1 = {}  # device-1 zero-carry cache for NOTRANSFER diagnostic


def make_inputs(N, device):
    obj = torch.randn(N, OMODE, Nz, Ny, Nx, 2, device=device) * 0.02
    obj[..., 0] += 1.0  # near-vacuum transmission, realistic
    probe = torch.randn(N, PMODE, Ny, Nx, dtype=torch.complex64, device=device)
    H = torch.exp(1j * 0.01 * torch.randn(1, 1, 1, Nz, Ny, Nx, device=device)).to(torch.complex64)
    occu = torch.ones(OMODE, device=device)
    coeffs = torch.zeros(NMAX, 2, device=device); coeffs[:, 0] = 1.0
    target = torch.rand(N, Ny, Nx, device=device) + 0.1
    return obj, probe, H, occu, coeffs, target


def loss_fn(I, target):
    return ((I.clamp_min(1e-10).sqrt() - target.sqrt()) ** 2).mean()


def time_fn(fn, warmup=5, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(DEV0); torch.cuda.synchronize(DEV1)
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(DEV0); torch.cuda.synchronize(DEV1)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(DEV0); torch.cuda.synchronize(DEV1)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


# ---------------------------------------------------------------------------
# Slice-split forward. All ops eager so device placement is explicit; the two
# device streams run concurrently because we never synchronize mid-forward.
# ---------------------------------------------------------------------------
def _slice_forward(obj0, p0, H0, obj1, p1, H1, occu1, coeffs0, coeffs1):
    """obj{0,1}: pseudo-complex (N,OMODE,nz_blk,Ny,Nx,2) leaves on their device.
    H0/H1: Fresnel power stacks (1,1,1,nz_blk,Ny,Nx) for planes [0,S)/[S,Nz).
    Returns intensity (N,Ny,Nx) on DEV1.

    PIPELINED: block0 order n+1 (DEV0) is launched right after block1 order n
    (DEV1) so the two device streams overlap; each order's carry transfers on
    its own (host-staged, no P2P). The only serial point is block1 order n
    waiting on block0 order n's carry."""
    H0c = H0.conj()
    O0 = torch.polar(obj0[..., 0], obj0[..., 1])
    objc0 = (O0 - 1.0).unsqueeze(1)                      # (N,1,OMODE,S,Ny,Nx)
    pk0 = fft2(p0).view(-1, PMODE, 1, 1, Ny, Nx)
    psi0 = ifft2(H0 * pk0)                               # block0 active [0,S)

    H1c = H1.conj()
    O1 = torch.polar(obj1[..., 0], obj1[..., 1])
    objc1 = (O1 - 1.0).unsqueeze(1)                      # (N,1,OMODE,Nz-S,Ny,Nx)
    pk1 = fft2(p1).view(-1, PMODE, 1, 1, Ny, Nx)
    psi1 = ifft2(H1 * pk1)                               # block1 active [S,Nz)
    Psi_M = pk1.squeeze(3)                               # 0th-order detector wave

    def blk0_step(n, psi0):
        sc0 = fft2(objc0[..., n:, :, :] * psi0) * H0c[..., n:, :, :]
        cs0 = torch.cumsum(sc0, dim=3)
        carry = cs0[..., -1, :, :]                        # block0 partial total over [n,S)
        psi0n = ifft2(cs0[..., :-1, :, :] * H0[..., n + 1:, :, :]) if n < NMAX - 1 else None
        return carry, psi0n

    def blk1_step(n, psi1, carry1):
        c = carry1.unsqueeze(3)                           # (N,PMODE,OMODE,1,Ny,Nx)
        sc1 = fft2(objc1 * psi1) * H1c
        cs1 = torch.cumsum(sc1, dim=3) + c                # add block0 prefix
        D_n = cs1[..., -1, :, :]                          # grand-total detector contribution
        psi1n = ifft2(torch.cat([c, cs1[..., :-1, :, :]], dim=3) * H1) if n < NMAX - 1 else None
        return D_n, psi1n

    # prologue: block0 order 0 (critical path head)
    def _xfer(carry):
        if not NOTRANSFER:
            if HALFCARRY:  # ship the carry at half precision: half the bytes
                return carry.to(torch.complex32).to(DEV1, non_blocking=True).to(torch.complex64)
            return carry.to(DEV1, non_blocking=True)
        z = _ZERO1.get(carry.shape)
        if z is None:
            z = torch.zeros(carry.shape, dtype=carry.dtype, device=DEV1)
            _ZERO1[carry.shape] = z
        return z

    carry, psi0 = blk0_step(0, psi0)
    carry1 = _xfer(carry)
    for n in range(NMAX):
        D_n, psi1_next = blk1_step(n, psi1, carry1)       # DEV1
        Psi_M = Psi_M + torch.complex(coeffs1[n, 0], coeffs1[n, 1]) * D_n
        if n < NMAX - 1:
            carry, psi0 = blk0_step(n + 1, psi0)          # DEV0, overlaps blk1_step(n)
            carry1 = _xfer(carry)
            psi1 = psi1_next
    norm_weight = (occu1 / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(Psi_M.abs().square() * norm_weight, dim=(1, 2)) + EPS)


def _eager_forward(obj, probe, H, occu, coeffs):
    """1-GPU eager born recursion (no compile) — fair eager baseline."""
    O = torch.polar(obj[..., 0], obj[..., 1])
    objc = (O - 1.0).unsqueeze(1)
    pk = fft2(probe).view(-1, PMODE, 1, 1, Ny, Nx)
    psi = ifft2(H * pk)
    Psi_M = pk.squeeze(3)
    for n in range(NMAX):
        sc = fft2(objc[..., n:, :, :] * psi) * H[..., n:, :, :].conj()
        cs = torch.cumsum(sc, dim=3)
        if n < NMAX - 1:
            D_n = cs[..., -1, :, :]
            psi = ifft2(cs[..., :-1, :, :] * H[..., n + 1:, :, :])
        else:
            D_n = sc.sum(dim=3)
        Psi_M = Psi_M + torch.complex(coeffs[n, 0], coeffs[n, 1]) * D_n
    norm_weight = (occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(Psi_M.abs().square() * norm_weight, dim=(1, 2)) + EPS)


def verify():
    N = 1
    obj, probe, H, occu, coeffs, _ = make_inputs(N, DEV0)
    torch.compiler.cudagraph_mark_step_begin()
    I_ref = born_forward(obj, probe, H, occu, n_max=NMAX, coeffs=coeffs).clone()
    I_eager = _eager_forward(obj, probe, H, occu, coeffs)
    # split tensors across devices
    obj0, obj1 = obj[:, :, :S].contiguous(), obj[:, :, S:].to(DEV1).contiguous()
    H0, H1 = H[:, :, :, :S].contiguous(), H[:, :, :, S:].to(DEV1).contiguous()
    p1 = probe.to(DEV1)
    I_split = _slice_forward(obj0, probe, H0, obj1, p1, H1, occu.to(DEV1), coeffs, coeffs.to(DEV1))
    r_eager = (I_ref - I_eager).norm() / I_ref.norm()
    r_split = (I_ref - I_split.to(DEV0)).norm() / I_ref.norm()
    print(f"eager-recursion vs born_forward : rel err = {r_eager.item():.2e}")
    print(f"slice-split    vs born_forward  : rel err = {r_split.item():.2e}")


def single_gpu_compiled(N):
    obj, probe, H, occu, coeffs, target = make_inputs(N, DEV0)
    obj.requires_grad_(True); probe.requires_grad_(True)

    def step():
        obj.grad = None; probe.grad = None
        torch.compiler.cudagraph_mark_step_begin()
        I = born_forward(obj, probe, H, occu, n_max=NMAX, coeffs=coeffs).clone()
        loss_fn(I, target).backward()
    return time_fn(step)


def single_gpu_eager(N):
    obj, probe, H, occu, coeffs, target = make_inputs(N, DEV0)
    obj.requires_grad_(True); probe.requires_grad_(True)

    def step():
        obj.grad = None; probe.grad = None
        I = _eager_forward(obj, probe, H, occu, coeffs)
        loss_fn(I, target).backward()
    return time_fn(step)


def slice_parallel(N):
    obj, probe, H, occu, coeffs, target = make_inputs(N, DEV0)
    obj0 = obj[:, :, :S].detach().contiguous().requires_grad_(True)
    obj1 = obj[:, :, S:].detach().to(DEV1).contiguous().requires_grad_(True)
    H0 = H[:, :, :, :S].contiguous(); H1 = H[:, :, :, S:].to(DEV1).contiguous()
    p0 = probe.detach().clone().requires_grad_(True)
    p1 = probe.detach().to(DEV1).requires_grad_(True)
    occu1 = occu.to(DEV1); c0 = coeffs; c1 = coeffs.to(DEV1)
    target1 = target.to(DEV1)

    def step():
        for t in (obj0, obj1, p0, p1):
            t.grad = None
        I = _slice_forward(obj0, p0, H0, obj1, p1, H1, occu1, c0, c1)
        loss_fn(I, target1).backward()
        _ = obj0.grad  # realized on dev0 via the carry path
    return time_fn(step)


if __name__ == "__main__":
    print(f"GPUs: {torch.cuda.device_count()} visible | split plane S={S} "
          f"(block0 {S} planes, block1 {Nz - S} planes)")
    verify()
    print()
    for N in (1, 32):
        t_c = single_gpu_compiled(N)
        t_e = single_gpu_eager(N)
        t_s = slice_parallel(N)
        print(f"N={N:2d}: 1-GPU compiled {t_c*1e3:7.2f} ms | 1-GPU eager {t_e*1e3:7.2f} ms "
              f"| 2-GPU slice-split {t_s*1e3:7.2f} ms "
              f"| split vs eager {t_e/t_s:.2f}x | split vs compiled {t_c/t_s:.2f}x")

"""Micro-benchmark: born6 forward+backward at batch 1 / 32,
single-GPU (all 4 probe modes) vs 2-GPU probe-mode-parallel (2 modes/GPU).

Mode parallelism is exact because born_forward sums probe modes incoherently
only at the detector (born.py:125). Each GPU computes the partial intensity
for its mode subset; the two (N,Ny,Nx) maps are added on dev0. A single
backward populates the object-patch gradient on each device (cross-device .to
is differentiable); we sum them to get the true gradient.
"""
import sys, time, os
import torch
sys.path.insert(0, "/home/dnz75396/ptyrad/src")
from ptyrad.forward_models.born import born_forward

torch.manual_seed(0)
Nz, Ny, Nx, PMODE, OMODE, NMAX = 21, 256, 256, 4, 1, 6
DEV0, DEV1 = "cuda:0", "cuda:1"


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


def single_gpu(N):
    obj, probe, H, occu, coeffs, target = make_inputs(N, DEV0)
    obj.requires_grad_(True); probe.requires_grad_(True)

    def step():
        obj.grad = None; probe.grad = None
        torch.compiler.cudagraph_mark_step_begin()
        I = born_forward(obj, probe, H, occu, n_max=NMAX, coeffs=coeffs).clone()
        loss_fn(I, target).backward()
    return time_fn(step), step, (obj, probe, H, occu, coeffs, target)


def mode_parallel(N):
    # shared object + H on both devices; probe modes split 2/2
    obj0, probe, H0, occu0, coeffs0, target = make_inputs(N, DEV0)
    obj1 = obj0.detach().to(DEV1); H1 = H0.to(DEV1)
    occu1 = occu0.to(DEV1); coeffs1 = coeffs0.to(DEV1)
    obj0.requires_grad_(True); obj1.requires_grad_(True)
    p0 = probe[:, :2].detach().clone().requires_grad_(True)
    p1 = probe[:, 2:].detach().clone().to(DEV1).requires_grad_(True)

    def step():
        for t in (obj0, obj1, p0, p1):
            t.grad = None
        torch.compiler.cudagraph_mark_step_begin()
        I0 = born_forward(obj0, p0, H0, occu0, n_max=NMAX, coeffs=coeffs0).clone()
        I1 = born_forward(obj1, p1, H1, occu1, n_max=NMAX, coeffs=coeffs1).clone()
        I = I0 + I1.to(DEV0)
        loss_fn(I, target).backward()
        # true object grad = obj0.grad + obj1.grad(->dev0); done outside timing-critical? keep in.
        _ = obj0.grad + obj1.grad.to(DEV0)
    return time_fn(step), step


def verify():
    N = 1
    obj, probe, H, occu, coeffs, target = make_inputs(N, DEV0)
    torch.compiler.cudagraph_mark_step_begin(); I_full = born_forward(obj, probe, H, occu, n_max=NMAX, coeffs=coeffs).clone()
    torch.compiler.cudagraph_mark_step_begin(); I0 = born_forward(obj, probe[:, :2], H, occu, n_max=NMAX, coeffs=coeffs).clone()
    torch.compiler.cudagraph_mark_step_begin(); I1 = born_forward(obj, probe[:, 2:], H, occu, n_max=NMAX, coeffs=coeffs).clone()
    rel = (I_full - (I0 + I1)).norm() / I_full.norm()
    print(f"mode-split correctness: rel err = {rel.item():.2e}")


if __name__ == "__main__":
    print(f"GPUs: {torch.cuda.device_count()} visible")
    verify()
    for N in (1, 32):
        t1, _, _ = single_gpu(N)
        t2, _ = mode_parallel(N)
        print(f"N={N:2d}: 1-GPU(4 modes) {t1*1e3:7.2f} ms | "
              f"2-GPU(2+2 modes) {t2*1e3:7.2f} ms | speedup {t1/t2:.2f}x | "
              f"per-iter extrap x4096/N: 1GPU {t1*4096/N:6.1f}s 2GPU {t2*4096/N:6.1f}s")

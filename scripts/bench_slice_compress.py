"""Carry-compression accuracy + gradient-safety sweep for the slice split.

The cross-device carry is a CORRECTION term (the scattered field), so forward
error from compressing it is tiny. The open question (raised: "2.2e-5 forward
error is below the noise floor, why would the gradient care?") is whether the
*object gradient* it induces is actually harmful. The decisive yardstick is not
the raw relative gradient error but whether that error sits BELOW the gradient's
own Poisson shot-noise floor at realistic dose. If quantisation moves the
gradient less than photon noise already does, it is training-safe.

Schemes (each halves/quarters the PCIe bytes per carry hop):
  half  c32  (2x)   bf16 RN/SR (2x)   fp8 e4m3 / e5m2  ("quarter", 4x, RN & SR)
  int8 RN/SR (~4x)   low-rank-r (SVD)   zfp fixed-tolerance (real lossy codec)

Each quantiser is applied with a straight-through estimator (forward = the
compressed carry actually shipped; backward = identity), which is exactly how a
distributed impl treats a compressed activation, and is how torch already treats
a dtype cast. We then compare the induced object-gradient error to the
shot-noise floor across a dose sweep.
"""
import os, sys, time
import numpy as np
import torch
from torch.fft import fft2, ifft2
sys.path.insert(0, "/home/dnz75396/ptyrad/src")
from ptyrad.forward_models.born import born_forward
from ptyrad.utils import fftshift2
try:
    import zfpy
    HAVE_ZFP = True
except Exception:
    HAVE_ZFP = False

torch.manual_seed(0)
Nz, Ny, Nx, PMODE, OMODE = 21, 256, 256, 4, 1
S, M = 12, 6
DEV = "cuda:0"
EPS = 1e-10


def make_inputs(N, scatter=0.02):
    obj = torch.randn(N, OMODE, Nz, Ny, Nx, 2, device=DEV) * scatter
    obj[..., 0] += 1.0
    probe = torch.randn(N, PMODE, Ny, Nx, dtype=torch.complex64, device=DEV)
    H = torch.exp(1j * 0.01 * torch.randn(1, 1, 1, Nz, Ny, Nx, device=DEV)).to(torch.complex64)
    occu = torch.ones(OMODE, device=DEV)
    return obj, probe, H, occu


# ---- straight-through wrapper: forward uses q(x), backward is identity --------
def ste(q):
    return lambda x: x + (q(x) - x).detach()


# ---- compression schemes: complex64 carry -> complex64 (lossy) ---------------
def _id(c):
    return c


def _half(c):  # complex32, deterministic (round-to-nearest) -- the known-bad one
    return c.to(torch.complex32).to(torch.complex64)


def _sr_bf16_comp(x):  # float32 -> bf16-precision, STOCHASTIC rounding (bit trick)
    xi = x.contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, x.shape, device=x.device, dtype=torch.int32)
    return (xi + r).bitwise_and(-65536).view(torch.float32)


def _bf16_sr(c):
    return torch.complex(_sr_bf16_comp(c.real), _sr_bf16_comp(c.imag))


def _bf16_rn(c):
    return torch.complex(c.real.to(torch.bfloat16).to(torch.float32),
                         c.imag.to(torch.bfloat16).to(torch.float32))


def _fp8(c, dt, hi):  # quarter precision, round-to-nearest, per-tensor scaled
    def q(x):
        s = x.abs().amax().clamp_min(1e-30)
        return (x / s * hi).to(dt).to(torch.float32) * s / hi
    return torch.complex(q(c.real), q(c.imag))


def _fp8_e4m3(c):
    return _fp8(c, torch.float8_e4m3fn, 448.0)


def _fp8_e5m2(c):
    return _fp8(c, torch.float8_e5m2, 57344.0)


def _int8_rn(c):
    def q(x):
        s = x.abs().amax().clamp_min(1e-30) / 127
        return torch.round(x / s).clamp(-127, 127) * s
    return torch.complex(q(c.real), q(c.imag))


def _int8_sr(c):
    def q(x):
        s = x.abs().amax().clamp_min(1e-30) / 127
        y = x / s
        f = torch.floor(y)
        y = f + (torch.rand_like(y) < (y - f)).to(y.dtype)
        return y.clamp(-127, 127) * s
    return torch.complex(q(c.real), q(c.imag))


def _lowrank(c, r):
    U, Sg, Vh = torch.linalg.svd(c, full_matrices=False)
    return (U[..., :r] * Sg[..., None, :r]) @ Vh[..., :r, :]


def _zfp(c, tol):
    def q(x):
        a = np.ascontiguousarray(x.detach().cpu().numpy().astype(np.float32).reshape(-1, Ny, Nx))
        comp = zfpy.compress_numpy(a, tolerance=tol)
        dec = zfpy.decompress_numpy(comp).reshape(x.shape)
        return torch.from_numpy(dec).to(x.device), len(comp), a.nbytes
    re, br, ob = q(c.real)
    im, bi, _ = q(c.imag)
    _zfp.ratio = (2 * ob) / (br + bi)
    return torch.complex(re, im)


# ---- single-device slice-split forward with a carry cast knob ----------------
def slice_forward(obj, probe, H, occu, carry_cast):
    O = torch.polar(obj[..., 0], obj[..., 1])
    objc = (O - 1.0).unsqueeze(1)
    pk = fft2(probe).view(-1, PMODE, 1, 1, Ny, Nx)
    psi0 = ifft2(H[..., :S, :, :] * pk)
    psi1 = ifft2(H[..., S:, :, :] * pk)
    objc0, objc1 = objc[..., :S, :, :], objc[..., S:, :, :]
    H0, H0c = H[..., :S, :, :], H[..., :S, :, :].conj()
    H1, H1c = H[..., S:, :, :], H[..., S:, :, :].conj()
    Psi_M = pk.squeeze(3)
    for n in range(M):
        cs0 = torch.cumsum(fft2(objc0[..., n:, :, :] * psi0) * H0c[..., n:, :, :], dim=3)
        carry = carry_cast(cs0[..., -1, :, :]).unsqueeze(3)
        cs1 = torch.cumsum(fft2(objc1 * psi1) * H1c, dim=3) + carry
        Psi_M = Psi_M + cs1[..., -1, :, :]
        if n < M - 1:
            psi0 = ifft2(cs0[..., :-1, :, :] * H0[..., n + 1:, :, :])
            psi1 = ifft2(torch.cat([carry, cs1[..., :-1, :, :]], dim=3) * H1)
    nw = (occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(Psi_M.abs().square() * nw, dim=(1, 2)) + EPS)


def amp_loss(I, tgt):
    return ((I.clamp_min(1e-10).sqrt() - tgt.sqrt()) ** 2).mean()


def obj_grad(obj, probe, H, occu, carry_cast, tgt):
    x = obj.clone().requires_grad_(True)
    amp_loss(slice_forward(x, probe, H, occu, carry_cast), tgt).backward()
    return x.grad


if __name__ == "__main__":
    print(f"zfp available: {HAVE_ZFP}")
    N = 8
    # true object -> measured data; current ESTIMATE is perturbed, so the
    # gradient is a real descent direction (nonzero), as during reconstruction.
    obj_true, probe, H, occu = make_inputs(N, scatter=0.02)
    with torch.no_grad():
        I_data = slice_forward(obj_true, probe, H, occu, _id).clone()   # "measured"
        I_exact = born_forward(obj_true, probe, H, occu, n_max=Nz).clone()
    obj = obj_true + 0.3 * torch.randn_like(obj_true) * 0.02            # estimate
    tgt = I_data                                                        # clean data target
    g_full = obj_grad(obj, probe, H, occu, ste(_id), tgt)
    trunc = ((slice_forward(obj_true, probe, H, occu, _id) - I_exact).norm()
             / I_exact.norm()).item()
    print(f"Born truncation err M=6 vs multislice: {trunc:.2e}  "
          f"(this is the error the model already accepts)")
    print(f"operating point: |grad| = {g_full.norm().item():.3e} (nonzero, mid-recon)\n")

    # ---- shot-noise gradient floor: gradient variation from Poisson noise ----
    print("=== shot-noise gradient floor (reference yardstick) ===")
    print("dose = total detected counts per diffraction pattern; floor = rel.")
    print("gradient difference between two independent Poisson realisations.\n")
    noise_floor = {}
    for dose in (1e4, 1e6, 1e8):
        cpp = dose / (Ny * Nx)              # mean counts per pixel
        lam = (I_data / I_data.mean() * cpp).clamp_min(0)
        errs = []
        for seed in range(1, 6):           # avg over realisations vs clean-data grad
            torch.manual_seed(seed)
            tgt_n = (torch.poisson(lam) / cpp * I_data.mean()).clamp_min(1e-10)
            g_n = obj_grad(obj, probe, H, occu, ste(_id), tgt_n)
            errs.append(((g_n - g_full).norm() / g_full.norm()).item())
        nf = float(np.mean(errs))
        noise_floor[dose] = nf
        print(f"  dose={dose:.0e}:  grad shot-noise floor = {nf:.3e}  "
              f"(photon noise moves the gradient this much, relative)")
    print()

    # ---- per-scheme forward + gradient error + compression -------------------
    with torch.no_grad():
        I_full = slice_forward(obj, probe, H, occu, _id).clone()  # full-carry @ estimate
    print("=== compression schemes (fwd err, object-grad err, bias, bytes) ===")
    print(f"{'scheme':18s} {'fwd err':>10s} {'grad err':>10s} {'bias':>8s} {'xfer':>7s}")
    schemes = [
        ("half c32 RN",   _half,     2.0),
        ("bf16 RN",       _bf16_rn,  2.0),
        ("bf16 SR",       _bf16_sr,  2.0),
        ("fp8 e4m3 (1/4)", _fp8_e4m3, 4.0),
        ("fp8 e5m2 (1/4)", _fp8_e5m2, 4.0),
        ("int8 RN",       _int8_rn,  4.0),
        ("int8 SR",       _int8_sr,  4.0),
        ("low-rank r=64", lambda c: _lowrank(c, 64), (Ny * Nx) / (64 * (Ny + Nx))),
        ("low-rank r=16", lambda c: _lowrank(c, 16), (Ny * Nx) / (16 * (Ny + Nx))),
    ]
    if HAVE_ZFP:
        schemes += [("zfp tol=1e-4", lambda c: _zfp(c, 1e-4), None),
                    ("zfp tol=1e-3", lambda c: _zfp(c, 1e-3), None)]
    for name, fn, ratio in schemes:
        torch.manual_seed(0)
        with torch.no_grad():
            I_q = slice_forward(obj, probe, H, occu, fn)
        fwd = ((I_q - I_full).norm() / I_full.norm()).item()
        torch.manual_seed(0)
        g_q = obj_grad(obj, probe, H, occu, ste(fn), tgt)
        gerr = ((g_q - g_full).norm() / g_full.norm()).item()
        dg = g_q - g_full
        bias = (dg.mean().abs() / dg.abs().mean().clamp_min(1e-30)).item()
        if ratio is None:
            ratio = _zfp.ratio
        print(f"{name:18s} {fwd:10.2e} {gerr:10.2e} {bias:8.2f} {ratio:6.1f}x")

    # ---- end-to-end drift: 50-step reconstruction, half vs full carry --------
    # vs the drift photon noise alone causes (two noisy datasets, full carry).
    print("\n=== 50-step reconstruction drift (the decisive test) ===")

    def recon(carry_cast, tgt, steps=50):
        x = obj.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=1e-3)
        for _ in range(steps):
            opt.zero_grad()
            amp_loss(slice_forward(x, probe, H, occu, carry_cast), tgt).backward()
            opt.step()
        return x.detach()

    x_full = recon(ste(_id), I_data)
    x_half = recon(ste(_half), I_data)
    x_fp8 = recon(ste(_fp8_e4m3), I_data)
    drift_half = ((x_half - x_full).norm() / x_full.norm()).item()
    drift_fp8 = ((x_fp8 - x_full).norm() / x_full.norm()).item()
    # noise-induced drift: same recon, full carry, two noisy datasets @ dose 1e6
    cpp = 1e6 / (Ny * Nx)
    lam = (I_data / I_data.mean() * cpp).clamp_min(0)
    xs = []
    for seed in (10, 11):
        torch.manual_seed(seed)
        xs.append(recon(ste(_id), (torch.poisson(lam) / cpp * I_data.mean()).clamp_min(1e-10)))
    drift_noise = ((xs[0] - xs[1]).norm() / xs[0].norm()).item()
    print(f"  object drift, half-carry   vs full : {drift_half:.3e}")
    print(f"  object drift, fp8/quarter  vs full : {drift_fp8:.3e}")
    print(f"  object drift, photon noise (dose1e6): {drift_noise:.3e}  <- the real floor")
    print(f"\n  => half-carry is {drift_noise/max(drift_half,1e-30):.0f}x below the "
          f"photon-noise floor; fp8 is {drift_noise/max(drift_fp8,1e-30):.0f}x below.")

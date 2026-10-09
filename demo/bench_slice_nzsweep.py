"""Where does the flat Born curve bend, and where does batch-1 slice-split win?

Context: vs multislice (A100_results cost plots), Born/ISS is flat in Nz at
batch 1 because the parallel-over-slices cumsum soaks up idle GPU capacity.
Once Nz*M saturates the device, Born goes linear too. Slice-split over P GPUs
is the ONLY multi-GPU axis multislice cannot copy (its slices are sequential),
and it extends the flat region by ~P -- but only past the bend.

This sweep measures, at BATCH 1 (the regime where Born beats multislice),
PSO-like config (256^2, 4 pmodes, M=6), synthetic volumes of Nz slices:
  * 1-GPU born fwd and fwd+bw (the flat curve and its bend)
  * 4-GPU slice-split compute floor (NOTRANSFER) -- NVLink-attainable ceiling
  * 4-GPU slice-split real transfer on this PHB box (sync + pinned-half)
Crossover Nz = where the 4-GPU floor beats 1 GPU at batch 1.
"""
import os, sys, time
import torch
from torch.fft import fft2, ifft2
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from ptyrad.utils import fftshift2

torch.manual_seed(0)
Ny, Nx, PMODE, OMODE = 256, 256, 4, 1
M = int(os.environ.get("BORN_M", "6"))  # scattering order: 6=born6, 2=double, 1=ISS
EPS = 1e-10
N = 1  # batch 1: the ONLY regime where Born beats multislice on one GPU
_ZERO = {}


def make_inputs(Nz):
    obj = torch.randn(N, OMODE, Nz, Ny, Nx, 2) * 0.02
    obj[..., 0] += 1.0
    probe = torch.randn(N, PMODE, Ny, Nx, dtype=torch.complex64)
    H = torch.exp(1j * 0.01 * torch.randn(1, 1, 1, Nz, Ny, Nx)).to(torch.complex64)
    occu = torch.ones(OMODE)
    return obj, probe, H, occu


def bounds(Nz, P):
    # block0 carries the shrinking window; give it a slightly larger share
    if P == 1:
        return [0, Nz]
    base = Nz // P
    b = [0]
    for i in range(P - 1):
        b.append(b[-1] + base + (1 if i < Nz % P else 0))
    b.append(Nz)
    return b


def distribute(obj, probe, H, P):
    b = bounds(H.shape[3], P)
    O = torch.polar(obj[..., 0], obj[..., 1])
    objc_all = (O - 1.0).unsqueeze(1)
    blk = []
    for i in range(P):
        lo, hi = b[i], b[i + 1]
        d = f"cuda:{i}"
        Hd = H[..., lo:hi, :, :].to(d).contiguous()
        blk.append(dict(objc=objc_all[..., lo:hi, :, :].to(d).contiguous(),
                        H=Hd, Hc=Hd.conj().contiguous(), probe=probe.to(d), dev=d))
    return blk


def forward_pway(blk, occu, P, notransfer=False, codec="c64"):
    for s in blk:
        pk = fft2(s["probe"]).view(-1, PMODE, 1, 1, Ny, Nx)
        s["pk"] = pk
        s["psi"] = ifft2(s["H"] * pk)
    Psi_M = blk[-1]["pk"].squeeze(3)

    def xfer(c, dev):
        if notransfer:
            key = (c.shape, dev)
            z = _ZERO.get(key)
            if z is None:
                z = torch.zeros(c.shape, dtype=c.dtype, device=dev); _ZERO[key] = z
            return z
        if codec == "half":
            return c.to(torch.complex32).to(dev, non_blocking=True).to(torch.complex64)
        return c.to(dev, non_blocking=True)

    for n in range(M):
        carry_in = None
        for i in range(P):
            s = blk[i]
            win0 = n if i == 0 else 0
            sc = fft2(s["objc"][..., win0:, :, :] * s["psi"]) * s["Hc"][..., win0:, :, :]
            cs = torch.cumsum(sc, dim=3)
            if carry_in is not None:
                cs = cs + carry_in.unsqueeze(3)
            running = cs[..., -1, :, :]
            if n < M - 1:
                if i == 0:
                    s["psi"] = ifft2(cs[..., :-1, :, :] * s["H"][..., win0 + 1:, :, :])
                else:
                    head = carry_in.unsqueeze(3)
                    s["psi"] = ifft2(torch.cat([head, cs[..., :-1, :, :]], dim=3) * s["H"])
            if i < P - 1:
                carry_in = xfer(running, blk[i + 1]["dev"])
            else:
                D_n = running
        Psi_M = Psi_M + D_n
    nw = (occu.to(Psi_M.device) / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(Psi_M.abs().square() * nw, dim=(1, 2)) + EPS)


def sync_all(P):
    for i in range(P):
        torch.cuda.synchronize(f"cuda:{i}")


def tmed(fn, P, warmup=3, reps=15):
    for _ in range(warmup):
        fn()
    sync_all(P)
    ts = []
    for _ in range(reps):
        sync_all(P); t0 = time.perf_counter(); fn(); sync_all(P)
        ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2]


if __name__ == "__main__":
    print(f"batch={N}, {Ny}x{Nx}, pmode={PMODE}, M={M} (BORN_M env); times in ms")
    print(f"{'Nz':>5} | {'1-GPU fwd':>10} | {'1-GPU f+b':>10} | {'4G floor':>9} "
          f"| {'floor x':>7} | {'4G sync':>8} | {'4G half':>8}")
    for Nz in (21, 32, 64, 128, 256, 512):
        obj, probe, H, occu = make_inputs(Nz)
        blk1 = distribute(obj, probe, H, 1)
        t1 = tmed(lambda: forward_pway(blk1, occu, 1), 1)
        # fwd+bw on 1 GPU (object gradient, the recon cost)
        O = torch.polar(obj[..., 0], obj[..., 1])
        oc = (O - 1.0).unsqueeze(1).to("cuda:0")
        Hd = H.to("cuda:0"); pr = probe.to("cuda:0")
        def fb():
            o = oc.detach().requires_grad_(True)
            b = [dict(objc=o, H=Hd, Hc=Hd.conj(), probe=pr, dev="cuda:0")]
            forward_pway(b, occu, 1).sum().backward()
        t1fb = tmed(fb, 1)
        blk4 = distribute(obj, probe, H, 4)
        tf = tmed(lambda: forward_pway(blk4, occu, 4, notransfer=True), 4)
        _ZERO.clear()
        blk4b = distribute(obj, probe, H, 4)
        ts_ = tmed(lambda: forward_pway(blk4b, occu, 4), 4)
        blk4c = distribute(obj, probe, H, 4)
        th = tmed(lambda: forward_pway(blk4c, occu, 4, codec="half"), 4)
        print(f"{Nz:>5} | {t1*1e3:10.2f} | {t1fb*1e3:10.2f} | {tf*1e3:9.2f} "
              f"| {t1/tf:6.2f}x | {ts_*1e3:8.2f} | {th*1e3:8.2f}")
        del obj, probe, H, blk1, blk4, blk4b, blk4c, oc, Hd
        torch.cuda.empty_cache()

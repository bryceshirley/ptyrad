"""Slice (depth) split on 4 A100s against the REAL born6 PSO checkpoint.

Loads the actual reconstructed object/probe/H/born_coeffs/crop_pos from
model_iter0100.hdf5 (the run named in the task) and runs the exact ptyrad
forward model two ways:

  * 1-GPU stock  : the real `born_forward` (torch.compile max-autotune) +
                   the detector gaussian_blur, i.e. what `ptyrad run` executes.
  * 4-GPU slice  : Nz=21 split into P contiguous depth blocks, one per GPU,
                   with the two-level-scan carry CHAIN (block b's running
                   prefix -> block b+1), coeffs applied to the per-order
                   detector field, same detector blur. Carry hops use the
                   pinned-async double-buffered path with a training-safe
                   codec (c64/half/quarter) for the forward-only timing, and
                   plain autograd-safe `.to()` for the gradient check + f+b.

Checks (on the real object):
  - forward rel err  (slice vs stock)  for P=2,4 and each codec
  - object-gradient rel err (slice vs stock) for P=4
Then times 1-GPU vs 4-GPU at N=1 (the real BATCH_SIZE) and N=8.

The real config is BATCH_SIZE=1 -> the latency-bound regime, so the honest
expectation is that the slice split does NOT beat 1 GPU at N=1; it is reported
alongside N=8 where the compute floor is reachable.
"""
import os, sys, time
import h5py
import torch
from torch.fft import fft2, ifft2
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from ptyrad.utils import fftshift2
try:
    from torchvision.transforms.functional import gaussian_blur
except Exception:
    from ptyrad.utils import gaussian_blur_2d as gaussian_blur
from ptyrad.forward_models import born_forward

CKPT = os.environ.get("PSO_CKPT", os.path.join(
    REPO, "demo/output/PSO/"
    "20260929_pso_born6_qr_msT_noreg_full_N4096_dp256_random1_p4_1obj_21slice_dz10_"
    "plr1e-4_oalr5e-4_oplr5e-4_dpblur1.0_orblur0.4_ozblur1.0_mamp0.03_4.0_oathr0.96_"
    "oposc_sng1.0_spr0.1/model_iter0100.hdf5"))

Nz, Ny, Nx, PMODE, OMODE, M = 21, 256, 256, 4, 1, 6
EPS = 1e-10
BLUR_SIGMA = 1.0
BOUNDS = {1: [0, 21], 2: [0, 12, 21], 4: [0, 7, 12, 17, 21]}
_HI = 448.0
# NOTRANSFER replaces every carry with a cached zero (BREAKS correctness) to
# expose the pure compute-split ceiling, isolating batch-1 granularity/launch
# overhead from the PHB host-staged carry cost.
NOTRANSFER = os.environ.get("NOTRANSFER", "0") == "1"
_ZERO = {}


def load_ckpt():
    with h5py.File(CKPT, "r") as f:
        obja = torch.as_tensor(f["optimizable_tensors/obja"][()])        # (1,21,639,639)
        objp = torch.as_tensor(f["optimizable_tensors/objp"][()])
        probe = torch.as_tensor(f["optimizable_tensors/probe"][()])      # (4,256,256) c64
        coeffs = torch.as_tensor(f["optimizable_tensors/born_coeffs"][()])  # (6,2)
        H2d = torch.as_tensor(f["model_attributes/H"][()])               # (256,256) c64
        crop_pos = torch.as_tensor(f["model_attributes/crop_pos"][()]).long()  # (4096,2)
        occu = torch.as_tensor(f["model_attributes/omode_occu"][()])     # (1,)
    return dict(obja=obja, objp=objp, probe=probe, coeffs=coeffs,
                H2d=H2d, crop_pos=crop_pos, occu=occu)


def build_inputs(ck, N, dev="cuda:0", start=0):
    """Crop N object patches exactly as models.get_obj_ROI does, build the H
    power-stack and (1,pmode,Ny,Nx) probe. Returns everything on `dev`."""
    obja, objp = ck["obja"].to(dev), ck["objp"].to(dev)
    opt_obj = torch.stack([obja, objp], dim=-1)                 # (1,21,639,639,2)
    cp = ck["crop_pos"][start:start + N].to(dev)
    ry = torch.arange(Ny, device=dev)
    rx = torch.arange(Nx, device=dev)
    gy = ry[None, :, None] + cp[:, None, None, 0]              # (N,Ny,1) -> broadcast
    gx = rx[None, None, :] + cp[:, None, None, 1]
    gy = ry.view(1, Ny, 1) + cp[:, None, None, 0]
    gx = rx.view(1, 1, Nx) + cp[:, None, None, 1]
    # opt_obj[:, :, gy, gx, :] -> (1,21,N,Ny,Nx,2) then permute to (N,1,21,Ny,Nx,2)
    patches = opt_obj[:, :, gy, gx, :].permute(2, 0, 1, 3, 4, 5).contiguous()
    probe = ck["probe"].to(dev).unsqueeze(0)                    # (1,4,256,256)
    z_idx = torch.arange(Nz, device=dev).view(1, 1, 1, Nz, 1, 1)
    H3d = ck["H2d"].to(dev).pow(z_idx)                          # (1,1,1,21,256,256)
    occu = ck["occu"].to(dev)
    coeffs = ck["coeffs"].to(dev)
    return patches, probe, H3d, occu, coeffs


def stock_forward(patches, probe, H3d, occu, coeffs):
    dp = born_forward(patches, probe, H3d, omode_occu=occu, n_max=M, coeffs=coeffs)
    return gaussian_blur(dp, kernel_size=[5, 5], sigma=BLUR_SIGMA)


# ---- slice split ---------------------------------------------------------

def distribute(patches, probe, H3d, P, grad=False):
    """Place each depth block's objc/H/probe on its GPU. If grad=True the block
    objc tensors stay connected (via .to) to the `patches` leaf on cuda:0 so a
    backward accumulates the full object gradient; else they are detached and
    pre-placed (timing)."""
    devs = [f"cuda:{i}" for i in range(P)]
    b = BOUNDS[P]
    O = torch.polar(patches[..., 0], patches[..., 1])
    objc_all = (O - 1.0).unsqueeze(1)                          # (N,1,omode,Nz,Ny,Nx)
    blk = []
    for i in range(P):
        lo, hi = b[i], b[i + 1]
        d = devs[i]
        Hd = H3d[..., lo:hi, :, :].to(d).contiguous()
        oc = objc_all[..., lo:hi, :, :].to(d)
        if not grad:
            oc = oc.detach().contiguous()
        blk.append(dict(objc=oc, H=Hd, Hc=Hd.conj().contiguous(),
                        probe=probe.to(d), lo=lo, dev=d))
    return blk


def _detector(Psi_M, occu, coeffs=None):
    nw = (occu.to(Psi_M.device) / (Nx * Ny)).view(1, 1, -1, 1, 1)
    dp = fftshift2(torch.sum(Psi_M.abs().square() * nw, dim=(1, 2)) + EPS)
    return gaussian_blur(dp, kernel_size=[5, 5], sigma=BLUR_SIGMA)


def forward_pway(blk, occu, coeffs, P):
    """Autograd-safe P-way slice split (plain .to carry). Matches born_forward."""
    for s in blk:
        pk = fft2(s["probe"]).view(-1, PMODE, 1, 1, Ny, Nx)
        s["pk"] = pk
        s["psi"] = ifft2(s["H"] * pk)
    Psi_M = blk[-1]["pk"].squeeze(3)
    for n in range(M):
        cf = torch.complex(coeffs[n, 0], coeffs[n, 1])
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
                carry_in = running.to(blk[i + 1]["dev"])
            else:
                D_n = running
        Psi_M = Psi_M + cf.to(Psi_M.device) * D_n
    return _detector(Psi_M, occu, coeffs)


# ---- pinned-async carry (forward-only timing) ----------------------------

def make_chain_bufs(P):
    hops = []
    for h in range(P - 1):
        hops.append(dict(src=torch.cuda.Stream(device=f"cuda:{h}"),
                         dst=torch.cuda.Stream(device=f"cuda:{h + 1}"), buf={}))
    return hops


def forward_pway_pinned(blk, occu, coeffs, P, hops, codec):
    for s in blk:
        pk = fft2(s["probe"]).view(-1, PMODE, 1, 1, Ny, Nx)
        s["pk"] = pk
        s["psi"] = ifft2(s["H"] * pk)
    Psi_M = blk[-1]["pk"].squeeze(3)

    def encode(c):
        if codec == "half":
            return [c.to(torch.complex32)]
        if codec == "quarter":
            sc = c.abs().amax().clamp_min(1e-30)
            return [(c.real / sc * _HI).to(torch.float8_e4m3fn),
                    (c.imag / sc * _HI).to(torch.float8_e4m3fn), sc]
        return [c]

    def decode(parts):
        if codec == "half":
            return parts[0].to(torch.complex64)
        if codec == "quarter":
            re, im, sc = parts
            return torch.complex(re.to(torch.float32), im.to(torch.float32)) * (sc / _HI)
        return parts[0]

    def xfer(carry, h, slot):
        if NOTRANSFER:
            key = (carry.shape, f"cuda:{h + 1}")
            z = _ZERO.get(key)
            if z is None:
                z = torch.zeros(carry.shape, dtype=carry.dtype, device=f"cuda:{h + 1}")
                _ZERO[key] = z
            return z
        hb = hops[h]
        dev1 = f"cuda:{h + 1}"
        parts = encode(carry)
        bufs = hb["buf"].get(slot)
        if bufs is None:
            bufs = [(torch.empty(p.shape, dtype=p.dtype, pin_memory=True),
                     torch.empty(p.shape, dtype=p.dtype, device=dev1)) for p in parts]
            hb["buf"][slot] = bufs
        hb["src"].wait_stream(torch.cuda.current_stream(f"cuda:{h}"))
        with torch.cuda.stream(hb["src"]):
            for p, (pin, _) in zip(parts, bufs):
                pin.copy_(p, non_blocking=True)
        ev = torch.cuda.Event(); ev.record(hb["src"])
        hb["dst"].wait_event(ev)
        with torch.cuda.stream(hb["dst"]):
            for (pin, recv) in bufs:
                recv.copy_(pin, non_blocking=True)
        ev2 = torch.cuda.Event(); ev2.record(hb["dst"])
        torch.cuda.current_stream(dev1).wait_event(ev2)
        return decode([recv for (_, recv) in bufs])

    for n in range(M):
        cf = torch.complex(coeffs[n, 0], coeffs[n, 1])
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
                carry_in = xfer(running, i, n % 2)
            else:
                D_n = running
        Psi_M = Psi_M + cf.to(Psi_M.device) * D_n
    return _detector(Psi_M, occu, coeffs)


def sync_all(P):
    for i in range(P):
        torch.cuda.synchronize(f"cuda:{i}")


def time_fn(fn, P, warmup=5, reps=20):
    for _ in range(warmup):
        fn()
    sync_all(P)
    ts = []
    for _ in range(reps):
        sync_all(P)
        t0 = time.perf_counter(); fn(); sync_all(P)
        ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2]


if __name__ == "__main__":
    nd = torch.cuda.device_count()
    print(f"GPUs visible: {nd}")
    p2p = [[int(torch.cuda.can_device_access_peer(i, j)) if i != j else 1
            for j in range(nd)] for i in range(nd)]
    print(f"P2P matrix (1=direct GPU<->GPU, the NVLink/P2P path): {p2p}")
    ck = load_ckpt()
    print(f"obja {tuple(ck['obja'].shape)} probe {tuple(ck['probe'].shape)} "
          f"coeffs {tuple(ck['coeffs'].shape)} occu={ck['occu'].tolist()}")
    print(f"born_coeffs=\n{ck['coeffs']}")

    # ---- correctness: forward (real object) ----
    N = 2
    patches, probe, H3d, occu, coeffs = build_inputs(ck, N)
    I0 = stock_forward(patches, probe, H3d, occu, coeffs)
    print(f"\nforward correctness (N={N}, real object, vs stock compiled born+blur):")
    for P in (2, 4):
        Isl = forward_pway(distribute(patches, probe, H3d, P), occu, coeffs, P).to("cuda:0")
        print(f"  P={P} sync .to      rel err {((Isl - I0).norm()/I0.norm()).item():.2e}")
        for codec in ("c64", "half", "quarter"):
            Ip = forward_pway_pinned(distribute(patches, probe, H3d, P), occu, coeffs,
                                     P, make_chain_bufs(P), codec).to("cuda:0")
            print(f"  P={P} pinned {codec:7s} rel err {((Ip - I0).norm()/I0.norm()).item():.2e}")

    # ---- correctness: object gradient (P=4) ----
    print("\ngradient correctness (N=2, P=4, amplitude loss vs self-consistent data):")
    pg = patches.detach().clone().requires_grad_(True)
    tgt = torch.sqrt(stock_forward(pg, probe, H3d, occu, coeffs).detach() + 0)
    # perturbed estimate -> realistic nonzero gradient
    est = (patches.detach() + 0.3 * torch.randn_like(patches) * 0.02).requires_grad_(True)
    loss_s = (torch.sqrt(stock_forward(est, probe, H3d, occu, coeffs) + EPS) - tgt).pow(2).mean()
    loss_s.backward()
    g_stock = est.grad.detach().clone()

    est2 = est.detach().clone().requires_grad_(True)
    blk = distribute(est2, probe, H3d, 4, grad=True)
    Isl = forward_pway(blk, occu, coeffs, 4).to("cuda:0")
    loss_sl = (torch.sqrt(Isl + EPS) - tgt).pow(2).mean()
    loss_sl.backward()
    g_slice = est2.grad.detach().clone()
    print(f"  loss stock {loss_s.item():.6e} | slice {loss_sl.item():.6e}")
    print(f"  object-grad rel err {((g_slice - g_stock).norm()/g_stock.norm()).item():.2e}")

    # ---- timing ----
    print("\ntiming (median; 1-GPU stock compiled born+blur vs 4-GPU slice):")
    for N in (1, 8, 32):
        patches, probe, H3d, occu, coeffs = build_inputs(ck, N)
        # 1-GPU stock forward
        t1_f = time_fn(lambda: stock_forward(patches, probe, H3d, occu, coeffs), 1)
        # 1-GPU EAGER baseline (slice-split machinery, P=1) — the apples-to-apples
        # reference for the 4-GPU eager slice; exposes the compiled-vs-eager gap
        blk1 = distribute(patches, probe, H3d, 1)
        t1e_f = time_fn(lambda: forward_pway(blk1, occu, coeffs, 1), 1)
        # 1-GPU stock forward+backward
        def stock_fb():
            p = patches.detach().requires_grad_(True)
            stock_forward(p, probe, H3d, occu, coeffs).sum().backward()
        t1_fb = time_fn(stock_fb, 1)
        # 4-GPU slice forward (pinned, half codec = training-safe best on PHB)
        blkp = distribute(patches, probe, H3d, 4)
        hops = make_chain_bufs(4)
        t4_f = time_fn(lambda: forward_pway_pinned(blkp, occu, coeffs, 4, hops, "half"), 4)
        # 4-GPU slice forward, sync .to carry (uncompressed; on an NVLink/P2P
        # box this .to is a direct GPU->GPU copy and should approach the floor)
        blks = distribute(patches, probe, H3d, 4)
        t4_fs = time_fn(lambda: forward_pway(blks, occu, coeffs, 4), 4)
        # 4-GPU slice forward+backward (autograd-safe sync .to carry)
        def slice_fb():
            p = patches.detach().requires_grad_(True)
            b = distribute(p, probe, H3d, 4, grad=True)
            forward_pway(b, occu, coeffs, 4).sum().backward()
        t4_fb = time_fn(slice_fb, 4)
        print(f"  N={N}: fwd   1-GPU(compiled) {t1_f*1e3:7.2f} ms | "
              f"1-GPU(eager) {t1e_f*1e3:7.2f} ms | 4-GPU pinned-half {t4_f*1e3:7.2f} ms "
              f"({t1_f/t4_f:.2f}x) | 4-GPU sync-c64 {t4_fs*1e3:7.2f} ms ({t1_f/t4_fs:.2f}x)")
        print(f"  N={N}: fwd+bw 1-GPU {t1_fb*1e3:7.2f} ms | 4-GPU {t4_fb*1e3:7.2f} ms "
              f"({t1_fb/t4_fb:.2f}x)")

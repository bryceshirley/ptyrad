"""Multislice vs parallel-Born at BATCH 1 across depth — the paper's actual
comparison, plus the multi-GPU question: can born6 + NVLink + P GPUs beat
multislice at batch 1?

Same config as bench_slice_nzsweep (256^2, 4 pmodes, 1 omode, batch 1).
Measures on this box:
  * multislice fwd and f+b, 1 GPU, eager AND compiled (its tiny sequential
    per-slice kernels are the best case for torch.compile fusion)
  * born(M) fwd and f+b, 1 GPU (eager, = production setting)
  * born(M) 4-GPU slice-split NOTRANSFER floor, fwd AND f+b (the
    NVLink-attainable ceiling; carries replaced by cached zeros)
Multislice CANNOT slice-split (sequential slices), so its 1-GPU number is its
batch-1 number, while born's floor is reachable with NVLink.
"""
import os, sys, time
import torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_slice_nzsweep import (make_inputs, distribute, forward_pway, tmed,
                                 bounds, M, Ny, Nx, PMODE, OMODE, N)
from ptyrad.forward_models import multislice_forward

torch.manual_seed(0)
SWEEP = (21, 32, 64, 128, 256, 512)


def ms_inputs(Nz, dev="cuda:0"):
    obj = (torch.randn(N, OMODE, Nz, Ny, Nx, 2) * 0.02)
    obj[..., 0] += 1.0
    probe = torch.randn(N, PMODE, Ny, Nx, dtype=torch.complex64)
    H2 = torch.exp(1j * 0.01 * torch.randn(1, Ny, Nx)).to(torch.complex64)
    occu = torch.ones(OMODE)
    return obj.to(dev), probe.to(dev), H2.to(dev), occu.to(dev)


if __name__ == "__main__":
    print(f"batch={N}, {Ny}x{Nx}, pmode={PMODE}, born M={M} (BORN_M env); ms")
    hdr = (f"{'Nz':>5} | {'MS fwd eag':>10} | {'MS fwd cmp':>10} | {'MS f+b eag':>10} "
           f"| {'born1G fwd':>10} | {'born1G f+b':>10} | {'b4G flr fwd':>11} | {'b4G flr f+b':>11}")
    print(hdr)
    for Nz in SWEEP:
        # ---- multislice, 1 GPU ----
        obj, probe, H2, occu = ms_inputs(Nz)
        with torch.compiler.set_stance("force_eager"):
            t_ms_e = tmed(lambda: multislice_forward(obj, probe, H2, occu), 1)
            def ms_fb():
                o = obj.detach().requires_grad_(True)
                multislice_forward(o, probe, H2, occu).sum().backward()
            t_ms_fbe = tmed(ms_fb, 1)
        t_ms_c = tmed(lambda: multislice_forward(obj, probe, H2, occu), 1)
        # ---- born(M), 1 GPU eager ----
        bobj, bprobe, bH, boccu = make_inputs(Nz)
        O = torch.polar(bobj[..., 0], bobj[..., 1]); objc = (O - 1).unsqueeze(1)
        oc = objc.to("cuda:0"); Hd = bH.to("cuda:0"); pr = bprobe.to("cuda:0")
        blk1 = [dict(objc=oc.detach(), H=Hd, Hc=Hd.conj(), probe=pr, dev="cuda:0")]
        t_b1 = tmed(lambda: forward_pway(blk1, boccu, 1), 1)
        def b_fb():
            o = oc.detach().requires_grad_(True)
            forward_pway([dict(objc=o, H=Hd, Hc=Hd.conj(), probe=pr, dev="cuda:0")],
                         boccu, 1).sum().backward()
        t_b1fb = tmed(b_fb, 1)
        # ---- born(M), 4-GPU NOTRANSFER floor (NVLink-attainable), pre-placed ----
        b = bounds(Nz, 4); blk = []
        for i in range(4):
            lo, hi = b[i], b[i + 1]; d = f"cuda:{i}"
            Hb = bH[..., lo:hi, :, :].to(d).contiguous()
            blk.append(dict(objc=objc[..., lo:hi, :, :].to(d).contiguous().requires_grad_(True),
                            H=Hb, Hc=Hb.conj().contiguous(), probe=bprobe.to(d), dev=d))
        t_f4 = tmed(lambda: forward_pway(blk, boccu, 4, notransfer=True), 4)
        def f4_fb():
            for s in blk:
                s["objc"].grad = None
            forward_pway(blk, boccu, 4, notransfer=True).sum().backward()
        t_f4fb = tmed(f4_fb, 4)
        print(f"{Nz:>5} | {t_ms_e*1e3:10.2f} | {t_ms_c*1e3:10.2f} | {t_ms_fbe*1e3:10.2f} "
              f"| {t_b1*1e3:10.2f} | {t_b1fb*1e3:10.2f} | {t_f4*1e3:11.2f} | {t_f4fb*1e3:11.2f}")
        del obj, probe, H2, bobj, bprobe, bH, oc, Hd, blk
        torch.cuda.empty_cache()

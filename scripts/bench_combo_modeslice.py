"""Combined mode x slice split: P_m probe-mode groups x P_z slice blocks on
P_m*P_z GPUs -- the 8-GPU NVLink configuration (2x4), validated here as 2x2
on 4 GPUs. Both axes are DIFFERENTIAL vs multislice at batch 1 (bench_results
SSH.2): MS is pmode-insensitive (launch-bound) and cannot slice-split.

Group g holds probe modes [g*pm/P_m : (g+1)*pm/P_m] and a FULL copy of the
object/H, slice-chained across its own GPU subset [g*P_z .. g*P_z+P_z-1].
Detector intensities are incoherent over modes, so group outputs ADD (minus
the duplicated eps). The mode axis replicates the object, so f+b needs a
cross-group gradient reduce per block (reported separately: it is ~free on
NVLink, ~Nz-proportional on PCIe).

Env: PM (default 2), PZ (default 2), BORN_M (default 6), NOTRANSFER=1 floor.
"""
import os, sys
import torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
sys.path.insert(0, _HERE)
from bench_slice_nzsweep import (make_inputs, bounds, forward_pway, tmed,
                                 M, PMODE, EPS)

PM = int(os.environ.get("PM", "2"))
PZ = int(os.environ.get("PZ", "2"))
NOTRANSFER = os.environ.get("NOTRANSFER", "0") == "1"
NGPU = PM * PZ
assert PMODE % PM == 0
GM = PMODE // PM  # modes per group


def build_groups(obj, probe, H, occu, grad=False):
    """Per-group pre-placed slice chains on disjoint GPU subsets."""
    O = torch.polar(obj[..., 0], obj[..., 1])
    objc = (O - 1.0).unsqueeze(1)
    b = bounds(H.shape[3], PZ)
    groups = []
    for g in range(PM):
        devs = [f"cuda:{g * PZ + i}" for i in range(PZ)]
        pg = probe[:, g * GM:(g + 1) * GM]
        blk = []
        for i in range(PZ):
            lo, hi = b[i], b[i + 1]
            d = devs[i]
            Hd = H[..., lo:hi, :, :].to(d).contiguous()
            oc = objc[..., lo:hi, :, :].to(d).contiguous()
            if grad:
                oc.requires_grad_(True)
            blk.append(dict(objc=oc, H=Hd, Hc=Hd.conj().contiguous(),
                            probe=pg.to(d), dev=d))
        # occu PRE-PLACED on the group's last device: a CPU occu would force a
        # pageable H2D at each chain's end, blocking the host until that whole
        # GPU chain completes and SERIALIZING the groups (measured: PM=4 0.9x)
        groups.append(dict(blk=blk, occu=occu.to(devs[-1])))
    return groups


def combo_forward(groups):
    outs = [forward_pway(g["blk"], g["occu"], PZ, notransfer=NOTRANSFER) for g in groups]
    out = outs[0].to("cuda:0")
    for o in outs[1:]:
        out = out + o.to("cuda:0")
    return out - (PM - 1) * EPS


def grad_reduce(groups):
    """Sum each block's object grad across the PM replicas (-> group 0)."""
    for i in range(PZ):
        g0 = groups[0]["blk"][i]["objc"]
        acc = g0.grad if g0.grad is not None else torch.zeros_like(g0)
        for g in range(1, PM):
            gg = groups[g]["blk"][i]["objc"].grad  # None under NOTRANSFER (graph cut)
            if gg is not None:
                acc = acc + gg.to(g0.device)
    return acc


if __name__ == "__main__":
    print(f"PM={PM} mode-groups x PZ={PZ} slice-blocks = {NGPU} GPUs | "
          f"M={M} | NOTRANSFER={NOTRANSFER}")
    # correctness vs 1-GPU
    obj, probe, H, occu = make_inputs(64)
    blk1 = [dict(objc=(torch.polar(obj[..., 0], obj[..., 1]) - 1).unsqueeze(1).to("cuda:0"),
                 H=H.to("cuda:0"), Hc=H.conj().to("cuda:0"),
                 probe=probe.to("cuda:0"), dev="cuda:0")]
    I1 = forward_pway(blk1, occu, 1)
    if not NOTRANSFER:
        Ic = combo_forward(build_groups(obj, probe, H, occu))
        print(f"combo correctness (Nz=64): rel err {((Ic - I1).norm() / I1.norm()).item():.2e}")
    print(f"{'Nz':>5} | {'1-GPU fwd':>9} | {'combo fwd':>9} | {'x':>5} "
          f"| {'1-GPU f+b':>9} | {'combo f+b':>9} | {'x':>5} | {'+gradred':>8} | {'x':>5}")
    for Nz in tuple(int(x) for x in os.environ.get("NZ_LIST", "64,128,256,512").split(",")):
        obj, probe, H, occu = make_inputs(Nz)
        oc1 = (torch.polar(obj[..., 0], obj[..., 1]) - 1).unsqueeze(1).to("cuda:0")
        Hd = H.to("cuda:0"); pr = probe.to("cuda:0")
        blk1 = [dict(objc=oc1.detach(), H=Hd, Hc=Hd.conj(), probe=pr, dev="cuda:0")]
        t1 = tmed(lambda: forward_pway(blk1, occu, 1), 1)
        def fb1():
            o = oc1.detach().requires_grad_(True)
            forward_pway([dict(objc=o, H=Hd, Hc=Hd.conj(), probe=pr, dev="cuda:0")],
                         occu, 1).sum().backward()
        t1fb = tmed(fb1, 1)
        groups = build_groups(obj, probe, H, occu)
        tc = tmed(lambda: combo_forward(groups), NGPU)
        groups_g = build_groups(obj, probe, H, occu, grad=True)
        def fbc():
            for g in groups_g:
                for s in g["blk"]:
                    s["objc"].grad = None
            combo_forward(groups_g).sum().backward()
        tcfb = tmed(fbc, NGPU)
        def fbcr():
            fbc(); grad_reduce(groups_g)
        tcfbr = tmed(fbcr, NGPU)
        print(f"{Nz:>5} | {t1*1e3:9.2f} | {tc*1e3:9.2f} | {t1/tc:4.2f}x "
              f"| {t1fb*1e3:9.2f} | {tcfb*1e3:9.2f} | {t1fb/tcfb:4.2f}x "
              f"| {tcfbr*1e3:8.2f} | {t1fb/tcfbr:4.2f}x")
        del obj, probe, H, groups, groups_g, oc1, Hd
        torch.cuda.empty_cache()

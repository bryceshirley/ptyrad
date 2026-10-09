"""20-iteration reconstruction comparison on the real PSO born6 config:
stock 1-GPU vs 2-GPU probe-mode split, fresh from simu init, same seed.

Usage: python run_cmp_modesplit.py baseline|modesplit

The mode split is injected by patching `ptyrad.forward_models.born_forward`
(get_forward_meas re-imports it from the package each call): probe modes
[0:2] run on cuda:0, modes [2:4] on cuda:1 with the SAME object patches and
H replicated; the per-group detector intensities simply ADD (incoherent mode
sum), so the split is exact (microbench rel err 3.9e-8). Autograd flows back
through the cross-device `.to`, so probe/object gradients are the full-mode
gradients. Static H is cached on cuda:1; patches/probe/coeffs move per step
(they change / carry grad).

Both arms run with torch.compiler force_eager: the user's production setting
disables the step compile already, eager==compiled for this born (3.6 ms fwd,
bench_results §D), and the compiled backward has a dynamo multi-device
metrics bug — so eager keeps both arms on the identical code path.
"""
import sys

import torch

torch.compiler.set_stance("force_eager")

ROLE = sys.argv[1]
assert ROLE in ("baseline", "modesplit", "smoke", "slicesplit", "slicesmoke",
                "slicedist", "distsmoke")
PARAMS = f"params/cmp_{ROLE}.yml"

if ROLE in ("slicedist", "distsmoke"):
    # DISTRIBUTED-OBJECT slice split: the object itself lives partitioned as
    # per-GPU optimizer LEAVES (blocks [0,7)[7,12)[12,17)[17,21) of Nz=21 on
    # cuda:0-3). Gradients STAY on their GPU and Adam updates run locally —
    # zero per-step object movement (the SSI structural benefit). Per step only
    # the carry chain (0.5 MB/hop) and the probe replica cross GPUs. The
    # master opt_obja/objp on cuda:0 is reassembled ONCE PER ITER around the
    # constraint pass (and so stays current for refit/canvas/saving), then
    # scattered back to the leaves.
    import numpy as np
    import ptyrad.models as pmodels
    from ptyrad.forward_models import born_forward_dist
    try:
        from torchvision.transforms.functional import gaussian_blur
    except Exception:
        from ptyrad.utils import gaussian_blur_2d as gaussian_blur

    _P = 4
    _B = [0, 7, 12, 17, 21]
    _DEVS = [f"cuda:{k}" for k in range(_P)]

    _orig_init_opt = pmodels.PtychoAD.create_optimizable_params_dict

    def init_opt_dist(self, *a, **kw):
        _orig_init_opt(self, *a, **kw)
        lra = self.lr_params.get("obja", 0)
        lrp = self.lr_params.get("objp", 0)
        self.obja_blks, self.objp_blks = [], []
        for k in range(_P):
            ab = self.opt_obja.data[:, _B[k]:_B[k + 1]].clone().to(_DEVS[k]).requires_grad_(True)
            pb = self.opt_objp.data[:, _B[k]:_B[k + 1]].clone().to(_DEVS[k]).requires_grad_(True)
            self.obja_blks.append(ab)
            self.objp_blks.append(pb)
        new = []
        for g in self.optimizable_params:
            t = g["params"][0]
            if t is self.opt_obja or t is self.opt_objp:
                continue
            new.append(g)
        for k in range(_P):
            new.append({"params": [self.obja_blks[k]], "lr": lra})
            new.append({"params": [self.objp_blks[k]], "lr": lrp})
        self.optimizable_params = new
        self._dist = {
            "cp": [self.crop_pos.to(d) for d in _DEVS],
            "rpy": [self.rpy_grid.to(d) for d in _DEVS],
            "rpx": [self.rpx_grid.to(d) for d in _DEVS],
            "occ3": self.omode_occu.to(_DEVS[-1]),
            "Hkey": None, "Hb": None,
        }
        print(f"[cmp] object partitioned into {_P} per-GPU leaves {_B}; "
              f"grads+Adam stay on-device")

    def forward_dist(self, indices):
        d = self._dist
        probes = self.get_probes(indices)
        H3 = self.get_propagators_3d(self.get_propagators(indices))
        if d["Hkey"] != H3.data_ptr():
            d["Hb"] = []
            for k in range(_P):
                Hd = H3[..., _B[k]:_B[k + 1], :, :].detach().to(_DEVS[k]).contiguous()
                d["Hb"].append((Hd, Hd.conj().contiguous()))
            d["Hkey"] = H3.data_ptr()
        idx = indices if isinstance(indices, torch.Tensor) else torch.as_tensor(np.asarray(indices))
        pmode = probes.shape[1]
        Ny_, Nx_ = probes.shape[-2:]
        blk, patch_blocks = [], []
        for k in range(_P):
            dev = _DEVS[k]
            ik = idx.to(dev)
            oo = torch.stack([self.obja_blks[k], self.objp_blks[k]], dim=-1)
            gy = d["rpy"][k][None] + d["cp"][k][ik, None, None, 0]
            gx = d["rpx"][k][None] + d["cp"][k][ik, None, None, 1]
            pk = oo[:, :, gy, gx, :].permute(2, 0, 1, 3, 4, 5)  # (N,omode,nz_k,Ny,Nx,2)
            patch_blocks.append(pk)
            objc = (torch.polar(pk[..., 0], pk[..., 1]) - 1.0).unsqueeze(1)
            blk.append(dict(objc=objc, H=d["Hb"][k][0], Hc=d["Hb"][k][1],
                            probe=probes.to(dev), dev=dev))
        # two-level scan engine from ptyrad source (born_dist): phase-A local
        # scans run on all GPUs in parallel, carries are transfer+add only
        coeffs = self.opt_born_coeffs if self.use_born_coeffs else None
        dp = born_forward_dist(blk, d["occ3"], self.born_iterations, coeffs=coeffs)
        if self.detector_blur_std is not None and self.detector_blur_std != 0:
            dp = gaussian_blur(dp, kernel_size=[5, 5], sigma=self.detector_blur_std)
        self._current_object_patches = patch_blocks
        return dp.to("cuda:0")

    pmodels.PtychoAD.create_optimizable_params_dict = init_opt_dist
    pmodels.PtychoAD.forward = forward_dist

    class DistLoss:
        """Sparse (L1 objp) loss computed per block on its device — scalars
        cross GPUs, gradients stay local. Other losses via the original."""
        def __init__(self, orig):
            self.orig = orig
            self.loss_params = orig.loss_params
            self.device = orig.device
            self.sp = dict(orig.loss_params["loss_sparse"])
            orig.loss_params["loss_sparse"]["state"] = False  # we add it back
            self._dummy = torch.zeros(1, 1, 1, 1, 1, 2, device="cuda:0")
            self._w = [(_B[k + 1] - _B[k]) / _B[-1] for k in range(_P)]

        def __call__(self, model_DP, measured_DP, patch_blocks, omode_occu):
            loss, losses = self.orig(model_DP, measured_DP, self._dummy, omode_occu)
            if self.sp["state"]:
                ln = self.sp["ln_order"]
                per_omode = sum(
                    (w * pk[..., 1].abs().pow(ln).mean(dim=(0, 2, 3, 4))).to("cuda:0")
                    for w, pk in zip(self._w, patch_blocks))
                sp = self.sp["weight"] * (per_omode.pow(1.0 / ln) * omode_occu).sum()
                loss = loss + sp
                losses = list(losses)
                losses[list(self.loss_params.keys()).index("loss_sparse")] = sp
            return loss, losses

    class DistConstraint:
        """Gather leaves -> master, run the stock constraints (full-z blurs
        etc. need the whole object), scatter back. Once per iteration."""
        def __init__(self, orig):
            self.orig = orig

        def __call__(self, model, niter):
            m = model
            while hasattr(m, "module") or hasattr(m, "_orig_mod"):
                m = getattr(m, "module", m)
                m = getattr(m, "_orig_mod", m)
            with torch.no_grad():
                for k in range(_P):
                    m.opt_obja.data[:, _B[k]:_B[k + 1]].copy_(m.obja_blks[k].data.to("cuda:0"))
                    m.opt_objp.data[:, _B[k]:_B[k + 1]].copy_(m.objp_blks[k].data.to("cuda:0"))
            out = self.orig(model, niter)
            with torch.no_grad():
                for k in range(_P):
                    m.obja_blks[k].data.copy_(m.opt_obja.data[:, _B[k]:_B[k + 1]].to(_DEVS[k]))
                    m.objp_blks[k].data.copy_(m.opt_objp.data[:, _B[k]:_B[k + 1]].to(_DEVS[k]))
            return out

if ROLE in ("slicesplit", "slicesmoke"):
    # 4-GPU depth (slice) split: Nz=21 split into blocks [0,7)[7,12)[12,17)
    # [17,21) across cuda:0-3 with the carry CHAIN (two-level scan), exactly
    # the recursion validated against the real checkpoint in bench_results SSD
    # (fwd rel err 1.8e-6, object-grad 1.9e-4). Static H blocks cached; object
    # slices/probe move per call with autograd intact (the known PCIe tax).
    import ptyrad.forward_models as fm
    from ptyrad.forward_models import born_forward_dist

    _orig_born = fm.born_forward
    _P = 4
    _H_CACHE = {}

    def _bounds(Nz):
        if Nz == 21:
            return [0, 7, 12, 17, 21]  # work-balanced (block0 carries the window)
        base = Nz // _P
        b = [0]
        for i in range(_P - 1):
            b.append(b[-1] + base + (1 if i < Nz % _P else 0))
        return b + [Nz]

    def born_forward_sliced(object_patches, probe, H, omode_occu,
                            eps=1e-10, n_max=1, coeffs=None):
        Nz = H.shape[3]
        if Nz < 2 * _P or not probe.is_cuda:
            return _orig_born(object_patches, probe, H, omode_occu,
                              eps=eps, n_max=n_max, coeffs=coeffs)
        b = _bounds(Nz)
        key = H.data_ptr()
        Hb = _H_CACHE.get(key)
        if Hb is None:
            _H_CACHE.clear()
            Hb = []
            for i in range(_P):
                Hd = H[..., b[i]:b[i + 1], :, :].detach().to(f"cuda:{i}").contiguous()
                Hb.append((Hd, Hd.conj().contiguous()))
            _H_CACHE[key] = Hb
        pmode = probe.shape[1]
        Ny_, Nx_ = probe.shape[-2:]
        O = torch.polar(object_patches[..., 0], object_patches[..., 1])
        objc = (O - 1.0).unsqueeze(1)                     # (N,1,omode,Nz,Ny,Nx)
        blk = []
        for i in range(_P):
            d = f"cuda:{i}"
            blk.append(dict(objc=objc[..., b[i]:b[i + 1], :, :].to(d),
                            H=Hb[i][0], Hc=Hb[i][1], probe=probe.to(d), dev=d))
        for s in blk:
            pk = fft2(s["probe"]).view(-1, pmode, 1, 1, Ny_, Nx_)
            s["pk"] = pk
            s["psi"] = ifft2(s["H"] * pk)
        Psi_M = blk[-1]["pk"].squeeze(3)
        M = min(n_max, Nz)
        c_all = None
        if coeffs is not None:
            c_all = torch.complex(coeffs[:, 0], coeffs[:, 1]).to(Psi_M.device)
        for n in range(M):
            carry_in = None
            for i in range(_P):
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
                if i < _P - 1:
                    carry_in = running.to(blk[i + 1]["dev"])
                else:
                    D_n = running
            Psi_M = Psi_M + D_n if c_all is None else Psi_M + c_all[n] * D_n
        nw = (omode_occu.to(Psi_M.device) / (Nx_ * Ny_)).view(1, 1, -1, 1, 1)
        dp = fftshift2(torch.sum(Psi_M.abs().square() * nw, dim=(1, 2)) + eps)
        return dp.to(object_patches.device)

    fm.born_forward = born_forward_sliced
    print("[cmp] born_forward patched: 4-GPU slice (depth) split, carry chain")

if ROLE in ("modesplit", "smoke"):
    import ptyrad.forward_models as fm

    _orig_born = fm.born_forward
    _DEV1 = "cuda:1"
    _H_CACHE = {}

    def born_forward_modesplit(object_patches, probe, H, omode_occu,
                               eps=1e-10, n_max=1, coeffs=None):
        pmode = probe.shape[1]
        if pmode < 2 or not probe.is_cuda:
            return _orig_born(object_patches, probe, H, omode_occu,
                              eps=eps, n_max=n_max, coeffs=coeffs)
        g = pmode // 2
        key = H.data_ptr()
        H1 = _H_CACHE.get(key)
        if H1 is None:
            H1 = H.detach().to(_DEV1)
            _H_CACHE.clear()
            _H_CACHE[key] = H1
        occu1 = omode_occu.to(_DEV1) if omode_occu is not None else None
        c1 = coeffs.to(_DEV1) if coeffs is not None else None
        out1 = _orig_born(object_patches.to(_DEV1), probe[:, g:].to(_DEV1), H1,
                          occu1, eps=eps, n_max=n_max, coeffs=c1)
        out0 = _orig_born(object_patches, probe[:, :g], H, omode_occu,
                          eps=eps, n_max=n_max, coeffs=coeffs)
        return out0 + out1.to(out0.device) - eps  # eps was added once per group

    fm.born_forward = born_forward_modesplit
    print("[cmp] born_forward patched: probe-mode split across cuda:0/cuda:1")

from ptyrad.load import load_params
from ptyrad.reconstruction import PtyRADSolver
from ptyrad.utils import CustomLogger, print_system_info, set_accelerator, set_gpu_device

logger = CustomLogger(log_file="ptyrad_log.txt", log_dir="auto",
                      prefix_time="datetime", prefix_jobid=0,
                      append_to_file=True, show_timestamp=True)
accelerator = set_accelerator()
print_system_info()
params = load_params(PARAMS)
device = set_gpu_device(0)
solver = PtyRADSolver(params, device=device, seed=42, acc=accelerator, logger=logger)
if ROLE in ("slicedist", "distsmoke"):
    solver.loss_fn = DistLoss(solver.loss_fn)
    solver.constraint_fn = DistConstraint(solver.constraint_fn)
    print("[cmp] loss/constraint wrapped for distributed object leaves")
solver.run()

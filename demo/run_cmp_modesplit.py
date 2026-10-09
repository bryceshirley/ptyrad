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
assert ROLE in ("baseline", "modesplit", "smoke")
PARAMS = f"params/cmp_{ROLE}.yml"

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
solver.run()

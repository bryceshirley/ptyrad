"""
Analytic adjoint-solve ePIE update vs Adam + autodiff for the Chebyshev
exact-generator slab model, on the PSO 3-slab x 70 A problem (256^2, 4 probe
modes, iter-50 checkpoint). Per-update wall time and peak CUDA memory at
batch sizes 1 and 32.

Arms:
  adam-eag : loss.backward() through multislice_forward_chebyshev, eager
             autograd (shows the raw tape), + Adam step on patch/probe leaves.
  adam-gr  : same wrapped in torch.cuda.make_graphed_callables (the
             production autodiff path, models._get_cheb_na_graphed).
  pie-eag  : analytic adjoint ePIE update (_cheb_epie), eager + fused steps.
  pie-gr   : same, CUDA-graph capture/replay (cheb_epie_update, as shipped).

Peak MiB is the ABSOLUTE torch.cuda.max_memory_allocated over the arm's
build + updates (tape and CUDA-graph static buffers included); the shared
problem tensors (~60 MiB at B=32) are inside every arm's number.

Usage: CUDA_VISIBLE_DEVICES=<UUID> python benchmarks/cheb_adjoint_vs_adam.py
"""

import gc
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, "/home/dnz75396/ptyrad/src")
from ptyrad.forward_models.chebyshev import chebyshev_setup, multislice_forward_chebyshev  # noqa: E402
from ptyrad.forward_models.chebyshev_adjoint import _cheb_epie, _CudaGraphedEpie  # noqa: E402

CKPT = (
    "/home/dnz75396/ptyrad/demo/output/high_order_benchmarks/chebyshev/"
    "20261022_cheb3_matched_n50_full_N4096_dp256_random1_p4_1obj_3slice_dz70_plr1e-4_"
    "oalr1e-3_oplr1e-3_dpblur1.0_orblur0.4_ozblur0.5238_mamp0.03_4.0_oathr0.96_oposc_"
    "sng1.0_spr0.05/model_iter0050.hdf5"
)
MEAS = "/home/dnz75396/ptyrad/demo/data/PSO/sample_data_PrScO3.mat"

dev = "cuda"
NPIX = 256
DZ = 70.0
DX = 0.09359515458345413
NZ = 3


def wavelength_A(kv):
    V = kv * 1e3
    return 12.2639 / np.sqrt(V * (1.0 + 0.97845e-6 * V))


LAMBD = wavelength_A(300.0)

with h5py.File(CKPT, "r") as f:
    obja = f["optimizable_tensors/obja"][()]  # (1, 3, 639, 639)
    objp = f["optimizable_tensors/objp"][()]
    probe_np = f["optimizable_tensors/probe"][()]  # (4, 256, 256) c64
    crop_pos = f["model_attributes/crop_pos"][()]

with h5py.File(MEAS, "r") as f:
    n_meas = f["dp"].shape[0]

kernel = chebyshev_setup(
    (NPIX, NPIX), dx=DX, lambd=LAMBD, dz_slab=DZ, phase_cap=(0.0, 14.0),
    amp_cap=(0.8, 1.2), tol=1e-3, degree_cap=512, band_frac=0.9,
    device=dev, dtype=torch.complex64,
)
groups = ((0, 1), (1, 2), (2, 3))
deg = kernel[2].shape[0] - 1
print(f"degree {deg} per slab, {len(groups)} slabs")

probe0 = torch.tensor(probe_np, device=dev)[None]  # (1,4,256,256) c64
occu = torch.ones(1, device=dev)


def make_batch(B, seed):
    r = np.random.default_rng(seed)
    sel = r.choice(len(crop_pos), B, replace=False)
    patches = np.empty((B, 1, NZ, NPIX, NPIX, 2), dtype=np.float32)
    for i, (y, x) in enumerate(crop_pos[sel]):
        patches[i, 0, :, :, :, 0] = obja[0, :, y : y + NPIX, x : x + NPIX]
        patches[i, 0, :, :, :, 1] = objp[0, :, y : y + NPIX, x : x + NPIX]
    obj = torch.tensor(patches, device=dev)
    with h5py.File(MEAS, "r") as f:
        idx = np.sort(r.choice(n_meas, B, replace=False))
        meas = torch.tensor(f["dp"][idx], dtype=torch.float32, device=dev)
    meas = meas / meas.sum((-2, -1), keepdim=True)  # arbitrary normalisation
    return obj, meas


def timeit(fn, n_warm=3, n_time=20):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_time):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3  # ms


def arm(build):
    """Absolute peak over build + timed updates of one arm."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn, teardown = build()
    ms = timeit(fn)
    peak = torch.cuda.max_memory_allocated() / 2**20
    teardown()
    gc.collect()
    torch.cuda.empty_cache()
    return ms, peak


results = {}
for B in (1, 32):
    obj, meas = make_batch(B, seed=B)
    probes = probe0.expand(B, -1, -1, -1).contiguous()

    def build_adam(graphed_fwd):
        def build():
            obj_leaf = obj.clone().requires_grad_(True)
            probe_leaf = torch.view_as_real(probe0.clone()).requires_grad_(True)
            opt = torch.optim.Adam([obj_leaf, probe_leaf], lr=1e-3)

            def fwd(p, pr):
                return multislice_forward_chebyshev(
                    p, torch.view_as_complex(pr), *kernel, groups, omode_occu=occu
                )

            f = fwd
            if graphed_fwd:
                sample = (
                    obj.detach().clone().requires_grad_(True),
                    torch.view_as_real(probe0).detach().clone().requires_grad_(True),
                )
                f = torch.cuda.make_graphed_callables(fwd, sample)

            def step():
                opt.zero_grad(set_to_none=False)
                dp = f(obj_leaf, probe_leaf)
                loss = ((dp + 1e-10).sqrt() - meas.sqrt()).square().sum()
                loss.backward()
                opt.step()

            return step, lambda: None

        return build

    def build_pie_eager():
        def step():
            with torch.no_grad():
                _cheb_epie(obj, probes, *kernel, groups, occu, meas,
                           0.1, 0.1, True, 1e-10)
        return step, lambda: None

    def build_pie_graphed():
        gr = _CudaGraphedEpie()

        def step():
            with torch.no_grad():
                gr(obj, probes, *kernel, groups, occu, meas, 0.1, 0.1, True, 1e-10)
        return step, lambda: None

    results[(B, "adam-eag")] = arm(build_adam(False))
    results[(B, "adam-gr")] = arm(build_adam(True))
    results[(B, "pie-eag")] = arm(build_pie_eager)
    results[(B, "pie-gr")] = arm(build_pie_graphed)

    # agreement eager vs graphed
    gr = _CudaGraphedEpie()
    with torch.no_grad():
        d1 = _cheb_epie(obj, probes, *kernel, groups, occu, meas, 0.1, 0.1, True, 1e-10)
        d2 = gr(obj, probes, *kernel, groups, occu, meas, 0.1, 0.1, True, 1e-10)
    rel = [(a - b).abs().max().item() / (a.abs().max().item() + 1e-30)
           for a, b in zip(d1, d2)]
    print(f"  (B={B} eager-vs-graphed max rel: dp {rel[0]:.1e} obj {rel[1]:.1e} dP {rel[2]:.1e})")
    del gr
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n{'batch':>5s} {'arm':>9s} {'ms/update':>10s} {'peak MiB':>9s}")
for (B, a), (ms, mib) in results.items():
    print(f"{B:5d} {a:>9s} {ms:10.2f} {mib:9.0f}")

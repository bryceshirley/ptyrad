"""PtyRAD counterpart of ptypy's borndiagnostics cost benchmark
(draft_paper/07_cost.png): wall clock and peak memory of forward + adjoint
per batch, against slice count, for the tBL-WSe2 geometry.

Methodology mirrors borndiagnostics._benchmark/_time_it:
  - real reconstructed slices, repeated cyclically to extend N (cost, not
    accuracy, is the axis) — object windows, probe, H from a real checkpoint;
  - per config: one warmup call, then `REPS` timed calls; peak CUDA
    allocation above the resting baseline (inputs + propagator stacks are
    allocated BEFORE the baseline, like ptypy charges its O(N) illumination
    cache separately);
  - configs that OOM are recorded as NaN and skipped.

Series (all eager PyTorch — the compiled variants would specialise per
shape and time compilation, not the maths):
  multislice            plain sequential multislice (one propagation per
                        slice; local impl — the tree's multislice_forward is
                        Strang-subsliced and suzukitrotter is 4th order,
                        both do several propagations per slice)
  Born, autograd        forward_models.born.firstborn_forward + autograd
  Born, analytical adj  FirstBornForwardFunction (hand adjoint; PtyRAD's
                        closest analogue of ptypy's low-memory Born — it
                        saves phi and the exit field instead of the autograd
                        tape, but still holds the O(N x batch) illumination,
                        so don't expect ptypy's O(batch) flat memory curve)

Adjoint = torch.autograd.grad of dp.sum() w.r.t. (object_patches, probe).

Outputs: demo/cost_ptyrad.png, demo/cost_ptyrad.csv.
"""

import csv
import glob
import os
import time

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import h5py
import numpy as np
import torch

torch._dynamo.config.disable = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.fft import fft2, fftshift, ifft2

from ptyrad.forward_models import firstborn_forward
from ptyrad.forward_models.born import FirstBornForwardFunction

DEMO = "/home/dnz75396/ptyrad/demo"
CKPT = sorted(glob.glob(
    f"{DEMO}/output/test_100/tBL_WSe2_born/20260913_*random32*/model_iter0100.hdf5"))[-1]
BATCHES = (1, 16, 32, 64)
SLICES = (1, 2, 4, 8, 16, 32)
REPS = 10
EPS = 1e-10


def plain_multislice(object_patches, probe, H, omode_occu, eps=EPS):
    """Standard 1st-order multislice: transmit each slice, propagate by H
    between slices (none after the last), detector intensity in PtyRAD units.
    One FFT pair per slice — the baseline the ptypy cost figure uses."""
    Ny, Nx = object_patches.shape[-3:-1]
    O = torch.polar(object_patches[..., 0], object_patches[..., 1])  # (B,omode,Nz,Ny,Nx)
    n = O.shape[2]
    psi = probe[:, :, None]  # (B, pmode, 1, Ny, Nx), broadcast over omode
    for j in range(n - 1):
        psi = ifft2(H[:, None, None] * fft2(psi * O[:, None, :, j]))
    psi = psi * O[:, None, :, n - 1]
    nw = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    dp = torch.sum(fft2(psi).abs().square() * nw, dim=(1, 2)) + eps
    return fftshift(dp, dim=(-2, -1))


def born_autograd(patches, probe, H3, occu):
    return firstborn_forward(patches, probe, H3, occu)


def born_analytical(patches, probe, H3, occu):
    return FirstBornForwardFunction.apply(patches, probe, H3, occu, EPS, False)


def time_one(fn, patches, probe, reps):
    """Warm up once; time `reps` forward+adjoint calls; peak GB above the
    resting baseline (inputs/propagators already resident)."""
    def call():
        dp = fn()
        torch.autograd.grad(dp.sum(), (patches, probe))

    call()  # warmup
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        call()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    mem = (torch.cuda.max_memory_allocated() - base) / 1e9
    return dt * 1e3, mem  # ms, GB


def main():
    dev = "cuda"
    with h5py.File(CKPT, "r") as f:
        obja = torch.tensor(f["optimizable_tensors/obja"][...], device=dev)
        objp = torch.tensor(f["optimizable_tensors/objp"][...], device=dev)
        probe0 = torch.tensor(f["optimizable_tensors/probe"][...], device=dev)
        crop_pos = torch.tensor(
            f["model_attributes/crop_pos"][...].astype(np.int64), device=dev)
        H1 = torch.tensor(f["model_attributes/H"][...], device=dev)  # (Ny, Nx)
    pmode, Ny, Nx = probe0.shape
    N_true = obja.shape[1]
    occu = torch.ones(1, device=dev)
    probe_in = probe0.unsqueeze(0)  # (1, pmode, Ny, Nx)
    print(f"checkpoint: {CKPT.split('/')[-2][:40]}... | frame {Ny}x{Nx}, "
          f"{pmode} probe modes, {N_true} real slices | eager, reps={REPS}")

    rows = []
    for B in BATCHES:
        # real windows for B views
        wins_a, wins_p = [], []
        for v in range(B):
            y0, x0 = crop_pos[v]
            wins_a.append(obja[:, :, y0:y0 + Ny, x0:x0 + Nx])
            wins_p.append(objp[:, :, y0:y0 + Ny, x0:x0 + Nx])
        base_a = torch.stack(wins_a)  # (B, omode, N_true, Ny, Nx)
        base_p = torch.stack(wins_p)

        for Nz in SLICES:
            rep_ix = [j % N_true for j in range(Nz)]  # repeat slices to extend N
            patches = torch.stack(
                [torch.stack([base_a[:, :, j], base_p[:, :, j]], dim=-1)
                 for j in rep_ix], dim=2)  # (B, omode, Nz, Ny, Nx, 2)
            patches = patches.contiguous().requires_grad_(True)
            probe = probe_in.clone().requires_grad_(True)
            zj = torch.arange(Nz, device=dev).view(1, 1, 1, Nz, 1, 1)
            H3 = (H1 ** zj).contiguous()          # (1,1,1,Nz,Ny,Nx), Born
            H2 = H1.unsqueeze(0).contiguous()     # (1,Ny,Nx), multislice

            for name, fn in (
                ("multislice", lambda: plain_multislice(patches, probe, H2, occu)),
                ("Born, autograd", lambda: born_autograd(patches, probe, H3, occu)),
                ("Born, analytical adjoint",
                 lambda: born_analytical(patches, probe, H3, occu)),
            ):
                try:
                    ms, gb = time_one(fn, patches, probe, REPS)
                except RuntimeError as e:  # OOM etc.
                    print(f"  B={B:3d} N={Nz:3d} {name}: skipped ({str(e)[:50]})")
                    ms, gb = float("nan"), float("nan")
                    torch.cuda.empty_cache()
                rows.append((name, B, Nz, ms, gb))
                if np.isfinite(ms):
                    print(f"  B={B:3d} N={Nz:3d} {name:26s} {ms:8.2f} ms  {gb:6.3f} GB")
            del patches, probe, H3
            torch.cuda.empty_cache()

    with open(f"{DEMO}/cost_ptyrad.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["series", "batch", "slices", "fwd_adj_ms", "peak_GB"])
        wr.writerows(rows)

    # ---- figure: rows = (wall clock, peak memory), cols = batch sizes -----
    colors = {"multislice": "#2a78d6", "Born, autograd": "#eb6834",
              "Born, analytical adjoint": "#1baf7a"}
    markers = {"multislice": "o", "Born, autograd": "s",
               "Born, analytical adjoint": "^"}
    ink, muted = "#1a1a19", "#6b6a60"
    fig, axes = plt.subplots(2, len(BATCHES), figsize=(3.1 * len(BATCHES), 6.4),
                             dpi=160, sharex=True, sharey="row")
    for c, B in enumerate(BATCHES):
        for r, key, ylabel in ((0, 3, "forward + adjoint (ms per batch)"),
                               (1, 4, "peak allocation (GB)")):
            ax = axes[r, c]
            for name in colors:
                pts = [(nz, row[key]) for row in rows
                       for nz in [row[2]]
                       if row[0] == name and row[1] == B and np.isfinite(row[key])]
                if pts:
                    xs, ys = zip(*pts, strict=True)
                    ax.loglog(xs, ys, color=colors[name], marker=markers[name],
                              ms=5, lw=1.8, label=name)
            if r == 0:
                ax.set_title(f"batch {B}", fontsize=10, color=ink)
            if r == 1:
                ax.set_xlabel("slices $N$", color=ink)
            if c == 0:
                ax.set_ylabel(ylabel, color=ink)
            ax.grid(True, which="both", color="#e8e7de", lw=0.5)
            ax.tick_params(colors=muted, labelsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=ink)
    fig.suptitle(
        "PtyRAD cost against depth, tBL-WSe$_2$ geometry (128$^2$ frames, 6 probe "
        "modes, slices repeated to extend $N$; eager PyTorch, RTX A4000)",
        fontsize=10, color=ink)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{DEMO}/cost_ptyrad.png", facecolor="white")
    print(f"saved {DEMO}/cost_ptyrad.png and cost_ptyrad.csv")


if __name__ == "__main__":
    main()

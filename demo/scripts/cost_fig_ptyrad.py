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
  Born, parallel        forward_models.born.firstborn_forward + autograd
                        (materialises the O(batch x N) slice stacks)
  Born, low memory      firstborn_forward_lowmem: slice-looped hand adjoint,
                        stores ONE unscattered field per slice (batch-free
                        for a shared probe) + the O(batch) exit field —
                        the counterpart of ptypy's low-memory Born
                        (gradients == autograd, test_born_lowmem.py)
  Born + line search    ONE FULL exact-line-search update (spec §3 steps
                        1-7: forward + backward for both gradients, K
                        preconditioner, per-slice direction response, both
                        quartic solves, probe response) — the per-batch cost
                        of a line-search update, vs. the others' single
                        forward + adjoint.

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

import ptyrad.linesearch as ls
from ptyrad.forward_models import firstborn_forward
from ptyrad.forward_models.born import firstborn_forward_lowmem

DEMO = "/home/dnz75396/ptyrad/demo"
CKPT = sorted(glob.glob(
    f"{DEMO}/output/test_100/tBL_WSe2_born/20260913_*random32*/model_iter0100.hdf5"))[-1]
BATCHES = (1, 16, 32, 64)
SLICES = (1, 2, 4, 8, 16, 32, 64)
REPS = 10
EPS = 1e-10


def slices_for(B):
    return SLICES


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


def born_parallel(patches, probe, H3, occu):
    return firstborn_forward(patches, probe, H3, occu)


def born_lowmem(patches, probe, H3, occu, chunk):
    return firstborn_forward_lowmem(patches, probe, H3, occu, EPS, False, chunk)


def time_one(call, reps):
    """Warm up once; time `reps` self-contained calls; peak GB above the
    resting baseline (inputs/propagators already resident)."""
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


def make_fwd_adj(fn, patches, probe):
    def call():
        dp = fn()
        torch.autograd.grad(dp.sum(), (patches, probe))
    return call


def make_ls_update(O0, probe0, H3, I_dat, occu, Nz):
    """One full exact-line-search batch update (§3 steps 1-7) at the tensor
    level: same maths as linesearch_batch_update, benchmark-shaped (B-sized
    patches, storage write-back excluded — it is O(canvas), not per-batch)."""
    omega = 1.0 / (I_dat + 1.0)

    def call():
        O = O0.detach().requires_grad_(True)
        P = probe0.detach().requires_grad_(True)
        L, F, u = ls._fwd_loss(O, P, H3, I_dat, None, occu, "amplitude")
        gO, gP = torch.autograd.grad(L, (O, P))
        phi = ls.unscattered_illumination(probe0, H3)
        dn = ls.object_denominator(phi)
        d = (-gO) / dn
        D = ls.direction_response(None, d, probe0, H3, per_slice=True).sum(dim=3)
        F = F.detach()
        u = u.detach()
        v, w = ls.response_terms(F, D, occu)
        a = ls.line_search(u - I_dat, v, w, omega, fallback=1.0 / Nz)
        F2 = F + a * D
        u2 = u + (2.0 * a) * v + (a * a) * w
        dn_p = ls.probe_denominator(O.detach())
        q = (-gP) / dn_p
        D_P = ls._fields_from_complex(O.detach() + a * d, q, H3)
        v2, w2 = ls.response_terms(F2, D_P, occu)
        ls.line_search(u2 - I_dat, v2, w2, omega, fallback=1.0 / Nz)
    return call


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

        for Nz in slices_for(B):
            rep_ix = [j % N_true for j in range(Nz)]  # repeat slices to extend N
            patches = torch.stack(
                [torch.stack([base_a[:, :, j], base_p[:, :, j]], dim=-1)
                 for j in rep_ix], dim=2)  # (B, omode, Nz, Ny, Nx, 2)
            patches = patches.contiguous().requires_grad_(True)
            probe = probe_in.clone().requires_grad_(True)
            zj = torch.arange(Nz, device=dev).view(1, 1, 1, Nz, 1, 1)
            H3 = (H1 ** zj).contiguous()          # (1,1,1,Nz,Ny,Nx), Born
            H2 = H1.unsqueeze(0).contiguous()     # (1,Ny,Nx), multislice

            # line-search inputs: complex object, synthetic data with residual
            with torch.no_grad():
                O0 = torch.polar(patches[..., 0], patches[..., 1]).contiguous()
                try:
                    I_dat = 1.02 * firstborn_forward(
                        patches.detach(), probe_in, H3, occu)
                except RuntimeError:
                    I_dat = None
                torch.cuda.empty_cache()

            series = [
                ("multislice",
                 make_fwd_adj(lambda: plain_multislice(patches, probe, H2, occu),
                              patches, probe)),
                ("Born, parallel",
                 make_fwd_adj(lambda: born_parallel(patches, probe, H3, occu),
                              patches, probe)),
                ("Born, low memory (chunk 1)",
                 make_fwd_adj(lambda: born_lowmem(patches, probe, H3, occu, 1),
                              patches, probe)),
                ("Born, low memory (chunk 4)",
                 make_fwd_adj(lambda: born_lowmem(patches, probe, H3, occu, 4),
                              patches, probe)),
            ]
            if I_dat is not None:
                series.append(
                    ("Born + line search",
                     make_ls_update(O0, probe_in, H3, I_dat, occu, Nz)))
            else:
                rows.append(("Born + line search", B, Nz, float("nan"), float("nan")))

            for name, call in series:
                try:
                    ms, gb = time_one(call, REPS)
                except RuntimeError as e:  # OOM etc.
                    print(f"  B={B:3d} N={Nz:3d} {name}: skipped ({str(e)[:50]})")
                    ms, gb = float("nan"), float("nan")
                    torch.cuda.empty_cache()
                rows.append((name, B, Nz, ms, gb))
                if np.isfinite(ms):
                    print(f"  B={B:3d} N={Nz:3d} {name:26s} {ms:8.2f} ms  {gb:6.3f} GB")
            del patches, probe, H3, O0, I_dat
            torch.cuda.empty_cache()

    with open(f"{DEMO}/cost_ptyrad.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["series", "batch", "slices", "fwd_adj_ms", "peak_GB"])
        wr.writerows(rows)

    plot_figs(rows)


def plot_figs(rows):
    """Two paper-ready figures: wall clock and peak memory, each a 2x2 grid
    of the four batch panels (larger panels than the combined 2x5 layout).
    Linear y per panel — log y hides how much one model beats another; log x
    keeps the doubling grid of N readable. No suptitle: the caption belongs
    to the paper."""
    colors = {"multislice": "#2a78d6", "Born, parallel": "#eb6834",
              "Born, low memory (chunk 1)": "#1baf7a",
              "Born, low memory (chunk 4)": "#e87ba4",
              "Born + line search": "#eda100"}
    markers = {"multislice": "o", "Born, parallel": "s",
               "Born, low memory (chunk 1)": "^",
               "Born, low memory (chunk 4)": "v",
               "Born + line search": "D"}
    ink, muted = "#1a1a19", "#6b6a60"
    for key, ylabel, fname in (
        (3, "forward + adjoint (ms per batch)", "cost_ptyrad_time.png"),
        (4, "peak allocation (GB)", "cost_ptyrad_mem.png"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.0), dpi=200)
        for k, B in enumerate(BATCHES):
            ax = axes[k // 2, k % 2]
            for name in colors:
                pts = [(row[2], row[key]) for row in rows
                       if row[0] == name and row[1] == B
                       and row[2] in slices_for(B) and np.isfinite(row[key])]
                if pts:
                    xs, ys = zip(*pts, strict=True)
                    ax.plot(xs, ys, color=colors[name], marker=markers[name],
                            ms=6, lw=2.0, label=name)
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(slices_for(B)))
            ax.set_xticklabels([str(s) for s in slices_for(B)])
            ax.set_xlim(0.8, max(slices_for(B)) * 1.35)
            ax.set_ylim(bottom=0)
            ax.set_title(f"batch {B}", fontsize=11, color=ink)
            if k // 2 == 1:
                ax.set_xlabel("slices $N$", color=ink, fontsize=10)
            if k % 2 == 0:
                ax.set_ylabel(ylabel, color=ink, fontsize=10)
            ax.grid(True, which="major", color="#e8e7de", lw=0.5)
            ax.tick_params(colors=muted, labelsize=9)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left",
                          labelcolor=ink)
        fig.tight_layout()
        fig.savefig(f"{DEMO}/{fname}", facecolor="white")
        plt.close(fig)
        print(f"saved {DEMO}/{fname}")


def plot_from_csv():
    rows = []
    with open(f"{DEMO}/cost_ptyrad.csv") as f:
        rd = csv.reader(f)
        next(rd)
        for name, B, Nz, ms, gb in rd:
            rows.append((name, int(B), int(Nz), float(ms), float(gb)))
    plot_figs(rows)


if __name__ == "__main__":
    import sys

    if "--plot-only" in sys.argv:
        plot_from_csv()
    else:
        main()

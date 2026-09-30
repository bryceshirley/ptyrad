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
                        slice; local eager impl of the tree's
                        multislice_forward, which is torch.compile'd)
  Born, parallel        forward_models.iss.iss_forward + autograd
                        (materialises the O(batch x N) slice stacks)
  Born, low memory      iss_forward_lowmem: slice-looped hand adjoint,
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
from torch.fft import fft2, fftshift, ifft2, ifftshift

import ptyrad.linesearch as ls
import ptyrad.mliss as mlm
from ptyrad.forward_models import born_forward, iss_forward
from ptyrad.forward_models.iss import iss_forward_lowmem

DEMO = "/home/dnz75396/ptyrad/demo"
CKPT = sorted(
    glob.glob(f"{DEMO}/output/test_100/tBL_WSe2_born/2026*_*random32*/model_iter0100.hdf5")
)[-1]
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
    return iss_forward(patches, probe, H3, occu)


def born_lowmem(patches, probe, H3, occu, chunk):
    return iss_forward_lowmem(patches, probe, H3, occu, EPS, False, chunk)


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


def ls_lowmem_update(O0, probe0, H3, I_dat, omega, occu, Nz, chunk, debug=False):
    """Low-memory §3 steps 1-7: the same maths as make_ls_update, with every
    O(batch x pmode x N) stack replaced by a chunked slice loop, mirroring
    ISSLowMemFunction at the complex-object level (same seed/adjoint algebra,
    validated against the parallel path — gradients, responses, and both
    steps agree to float32 accuracy). Resident workspace: the batch-free
    O(N x pmode) illumination phi, O(batch) detector fields, the inherent
    O(batch x N) direction d, and O(batch x pmode x chunk) loop temporaries."""
    Hz = H3[:, 0, 0]  # (1, Nz, Ny, Nx)
    probe_k = fft2(probe0)  # (1, pmode, Ny, Nx)
    phi = ifft2(Hz.unsqueeze(1) * probe_k.unsqueeze(2))  # (1, pmode, Nz, Ny, Nx)
    sls = [slice(j0, min(j0 + chunk, Nz)) for j0 in range(0, Nz, chunk)]

    # step 1 forward field, chunked
    acc = 0.0
    for sl in sls:
        g = O0[:, :, sl] - 1.0
        acc = acc + (fft2(g.unsqueeze(1) * phi[:, :, None, sl]) * Hz[:, None, None, sl].conj()).sum(
            dim=3
        )
    F = probe_k.unsqueeze(2) + acc  # (B, pmode, omode, Ny, Nx)
    u = ls.dp_from_fields(F, occu)

    # amplitude-objective gradients (torch Wirtinger convention, identical to
    # autograd of _fwd_loss): detector-plane seed, then one chunked adjoint
    # sweep; the 1/(Nx*Ny) of the unit map cancels against the unnormalised
    # FFT adjoints exactly as in ISSLowMemFunction.backward.
    dLdu = 1.0 - I_dat.clamp_min(0).sqrt() / u.clamp_min(ls.SQRT_FLOOR).sqrt()
    ow = occu.view(1, 1, -1, 1, 1)
    Wn = 2.0 * ifftshift(dLdu, dim=(-2, -1)).unsqueeze(1).unsqueeze(2) * ow * F
    gO = torch.empty_like(O0)
    pk_acc = Wn.sum(dim=2)
    Wn4 = Wn.unsqueeze(3)
    for sl in sls:
        T = ifft2(Hz[:, None, None, sl] * Wn4)  # (B, pmode, omode, C, Ny, Nx)
        gO[:, :, sl] = (phi[:, :, None, sl].conj() * T).sum(dim=1)
        gc = (O0[:, :, sl] - 1.0).unsqueeze(1).conj()
        pk_acc = pk_acc + (Hz[:, None, sl].conj() * fft2((gc * T).sum(dim=2))).sum(dim=2)
    gP = ifft2(pk_acc).sum(dim=0, keepdim=True)  # (1, pmode, Ny, Nx)

    # steps 2-5: preconditioned direction, chunked response, object step
    dn = ls.object_denominator(phi)
    d = (-gO) / dn
    D = 0.0
    for sl in sls:
        D = D + (fft2(d[:, None, :, sl] * phi[:, :, None, sl]) * Hz[:, None, None, sl].conj()).sum(
            dim=3
        )
    v, w = ls.response_terms(F, D, occu)
    a = ls.line_search(u - I_dat, v, w, omega, fallback=1.0 / Nz)

    # steps 6-7: post-step field by the response identities, probe step
    F2 = F + a * D
    u2 = u + (2.0 * a) * v + (a * a) * w
    dn_p = ls.probe_denominator(O0)
    q = (-gP) / dn_p
    qk = fft2(q)
    accq = 0.0
    for sl in sls:
        psi_q = ifft2(Hz[:, sl].unsqueeze(1) * qk.unsqueeze(2))
        g2 = O0[:, :, sl] + a * d[:, :, sl] - 1.0
        accq = accq + (
            fft2(g2.unsqueeze(1) * psi_q[:, :, None]) * Hz[:, None, None, sl].conj()
        ).sum(dim=3)
    D_P = qk.unsqueeze(2) + accq
    v2, w2 = ls.response_terms(F2, D_P, occu)
    a2 = ls.line_search(u2 - I_dat, v2, w2, omega, fallback=1.0 / Nz)
    if debug:
        return dict(
            F=F, u=u, gO=gO, gP=gP, d=d, D=D, v=v, w=w, a=a, q=q, D_P=D_P, v2=v2, w2=w2, a2=a2
        )


def make_ls_update_lowmem(O0, probe0, H3, I_dat, occu, Nz, chunk=4):
    omega = 1.0 / (I_dat + 1.0)

    def call():
        ls_lowmem_update(O0, probe0, H3, I_dat, omega, occu, Nz, chunk)

    return call


def make_mliss_update(O0, probe0, H3, I_dat, occu, Nz, c=558.0):
    """One full ML-ISS (Gaussian, alternating) batch update at the tensor
    level — mirror of make_ls_update with the L_G objective and its own
    exact quartic steps (same transform count as the BLISS update)."""
    sigma2 = I_dat + 1.0 / c
    omega = 1.0 / sigma2

    def call():
        O = O0.detach().requires_grad_(True)
        P = probe0.detach().requires_grad_(True)
        L, F, u = mlm._fwd_lg(O, P, H3, I_dat, None, sigma2, occu)
        gO, gP = torch.autograd.grad(L, (O, P))
        phi = ls.unscattered_illumination(probe0, H3)
        dn = ls.object_denominator(phi)
        d = (-gO) / dn
        D = ls.direction_response(None, d, probe0, H3, per_slice=True).sum(dim=3)
        F2 = F.detach()
        u2 = u.detach()
        v, w = ls.response_terms(F2, D, occu)
        a, _ = mlm.ml_line_search(u2 - I_dat, v, w, omega, fallback=1.0 / Nz)
        F3 = F2 + a * D
        u3 = u2 + (2.0 * a) * v + (a * a) * w
        dn_p = ls.probe_denominator(O.detach())
        q = (-gP) / dn_p
        D_P = ls._fields_from_complex(O.detach() + a * d, q, H3)
        v2, w2 = ls.response_terms(F3, D_P, occu)
        mlm.ml_line_search(u3 - I_dat, v2, w2, omega, fallback=1.0 / Nz)

    return call


def main():
    dev = "cuda"
    with h5py.File(CKPT, "r") as f:
        obja = torch.tensor(f["optimizable_tensors/obja"][...], device=dev)
        objp = torch.tensor(f["optimizable_tensors/objp"][...], device=dev)
        probe0 = torch.tensor(f["optimizable_tensors/probe"][...], device=dev)
        crop_pos = torch.tensor(f["model_attributes/crop_pos"][...].astype(np.int64), device=dev)
        H1 = torch.tensor(f["model_attributes/H"][...], device=dev)  # (Ny, Nx)
    pmode, Ny, Nx = probe0.shape
    N_true = obja.shape[1]
    occu = torch.ones(1, device=dev)
    probe_in = probe0.unsqueeze(0)  # (1, pmode, Ny, Nx)
    print(
        f"checkpoint: {CKPT.split('/')[-2][:40]}... | frame {Ny}x{Nx}, "
        f"{pmode} probe modes, {N_true} real slices | eager, reps={REPS}"
    )

    rows = []
    for B in BATCHES:
        # real windows for B views
        wins_a, wins_p = [], []
        for v in range(B):
            y0, x0 = crop_pos[v]
            wins_a.append(obja[:, :, y0 : y0 + Ny, x0 : x0 + Nx])
            wins_p.append(objp[:, :, y0 : y0 + Ny, x0 : x0 + Nx])
        base_a = torch.stack(wins_a)  # (B, omode, N_true, Ny, Nx)
        base_p = torch.stack(wins_p)

        for Nz in slices_for(B):
            rep_ix = [j % N_true for j in range(Nz)]  # repeat slices to extend N
            patches = torch.stack(
                [torch.stack([base_a[:, :, j], base_p[:, :, j]], dim=-1) for j in rep_ix], dim=2
            )  # (B, omode, Nz, Ny, Nx, 2)
            patches = patches.contiguous().requires_grad_(True)
            probe = probe_in.clone().requires_grad_(True)
            zj = torch.arange(Nz, device=dev).view(1, 1, 1, Nz, 1, 1)
            H3 = (H1**zj).contiguous()  # (1,1,1,Nz,Ny,Nx), Born
            H2 = H1.unsqueeze(0).contiguous()  # (1,Ny,Nx), multislice

            # line-search inputs: complex object, synthetic data with residual
            with torch.no_grad():
                O0 = torch.polar(patches[..., 0], patches[..., 1]).contiguous()
                try:
                    I_dat = 1.02 * iss_forward(patches.detach(), probe_in, H3, occu)
                except RuntimeError:
                    I_dat = None
                torch.cuda.empty_cache()

            series = [
                (
                    "multislice",
                    make_fwd_adj(
                        lambda: plain_multislice(patches, probe, H2, occu), patches, probe
                    ),
                ),
                (
                    "ISS, parallel",
                    make_fwd_adj(lambda: born_parallel(patches, probe, H3, occu), patches, probe),
                ),
                (
                    "ISS, double scattering",
                    make_fwd_adj(
                        lambda: born_forward(patches, probe, H3, occu, EPS, 2), patches, probe
                    ),
                ),
                (
                    "ISS, triple scattering",
                    make_fwd_adj(
                        lambda: born_forward(patches, probe, H3, occu, EPS, 3), patches, probe
                    ),
                ),
                (
                    "ISS, quadruple scattering",
                    make_fwd_adj(
                        lambda: born_forward(patches, probe, H3, occu, EPS, 4), patches, probe
                    ),
                ),
                (
                    "ISS, low memory (chunk 1)",
                    make_fwd_adj(lambda: born_lowmem(patches, probe, H3, occu, 1), patches, probe),
                ),
                (
                    "ISS, low memory (chunk 4)",
                    make_fwd_adj(lambda: born_lowmem(patches, probe, H3, occu, 4), patches, probe),
                ),
            ]
            if I_dat is not None:
                series.append(
                    ("ISS + line search", make_mliss_update(O0, probe_in, H3, I_dat, occu, Nz))
                )
                series.append(
                    (
                        "ISS + line search (low mem)",
                        make_ls_update_lowmem(O0, probe_in, H3, I_dat, occu, Nz, 4),
                    )
                )
            else:
                rows.append(("ISS + line search", B, Nz, float("nan"), float("nan")))
                rows.append(("ISS + line search (low mem)", B, Nz, float("nan"), float("nan")))

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
            patches = probe = H3 = O0 = I_dat = (
                None  # free GPU memory (rebind, not del: closures reference these)
            )
            torch.cuda.empty_cache()

    with open("/home/dnz75396/reproduce/cost_ptyrad_mliss_A100.csv", "w", newline="") as f:
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
    colors = {
        "multislice": "#2a78d6",
        "ISS, parallel": "#eb6834",
        "ISS, double scattering": "#b8442c",
        "ISS, triple scattering": "#7a2f1d",
        "ISS, quadruple scattering": "#4a1c10",
        "ISS, low memory (chunk 1)": "#1baf7a",
        "ISS, low memory (chunk 4)": "#e87ba4",
        "ISS + line search": "#eda100",
        "ISS + line search (low mem)": "#8d6cd9",
    }
    markers = {
        "multislice": "o",
        "ISS, parallel": "s",
        "ISS, double scattering": "<",
        "ISS, triple scattering": ">",
        "ISS, quadruple scattering": "X",
        "ISS, low memory (chunk 1)": "^",
        "ISS, low memory (chunk 4)": "v",
        "ISS + line search": "D",
        "ISS + line search (low mem)": "P",
    }
    ink, muted = "#1a1a19", "#6b6a60"
    time_series = ("multislice", "ISS, parallel", "ISS, low memory (chunk 1)",
               "ISS, low memory (chunk 4)", "ISS + line search")
    time_born_series = (
        "multislice",
        "ISS, parallel",
        "ISS, double scattering",
        "ISS, triple scattering",
        "ISS, quadruple scattering",
    )
    # legend text for the scattering-order figure: "ISS" means single
    # scattering, so the higher orders are labelled by order alone
    born_labels = {
        "ISS, parallel": "single scattering",
        "ISS, double scattering": "double scattering",
        "ISS, triple scattering": "triple scattering",
        "ISS, quadruple scattering": "quadruple scattering",
    }
    mem_lowmem_series = ("multislice", "ISS, low memory (chunk 4)", "ISS + line search (low mem)")
    lowmem_labels = {
        "ISS, low memory (chunk 4)": "ISS, low memory",
        "ISS + line search (low mem)": "ISS + line search, low memory",
    }
    for key, ylabel, fname, series, leg in (
        (3, "forward + adjoint (ms per batch)", "cost_ptyrad_time_mliss_A100.png", time_series, None),
        (
            3,
            "forward + adjoint (ms per batch)",
            "cost_ptyrad_time_born.png",
            time_born_series,
            born_labels,
        ),
        (4, "peak allocation (GB)", "cost_ptyrad_mem.png", tuple(colors), None),
        (4, "peak allocation (GB)", "cost_ptyrad_mem_lowmem.png", mem_lowmem_series, lowmem_labels),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.0), dpi=200)
        for k, B in enumerate(BATCHES):
            ax = axes[k // 2, k % 2]
            for name in series:
                pts = [
                    (row[2], row[key])
                    for row in rows
                    if row[0] == name
                    and row[1] == B
                    and row[2] in slices_for(B)
                    and np.isfinite(row[key])
                ]
                if pts:
                    xs, ys = zip(*pts, strict=True)
                    ax.plot(
                        xs,
                        ys,
                        color=colors[name],
                        marker=markers[name],
                        ms=6,
                        lw=2.0,
                        label=(leg or {}).get(name) or LABEL_MAP.get(name, name),
                    )
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
        axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=ink)
        fig.tight_layout()
        fig.savefig(f"/home/dnz75396/reproduce/{fname}", facecolor="white")
        plt.close(fig)
        print(f"saved /home/dnz75396/reproduce/{fname}")


LABEL_MAP = {
    "Born, parallel": "ISS, parallel",
    "Born, low memory (chunk 1)": "ISS, low memory (chunk 1)",
    "Born, low memory (chunk 4)": "ISS, low memory (chunk 4)",
    "Born + line search": "ISS + line search",
    "ISS + line search": "ML-ISS",
}


def plot_from_csv():
    rows = []
    with open("/home/dnz75396/reproduce/cost_ptyrad_mliss_A100.csv") as f:
        rd = csv.reader(f)
        next(rd)
        for name, B, Nz, ms, gb in rd:
            rows.append((LABEL_MAP.get(name, name), int(B), int(Nz), float(ms), float(gb)))
    plot_figs(rows)


if __name__ == "__main__":
    import sys

    if "--plot-only" in sys.argv:
        plot_from_csv()
    else:
        main()

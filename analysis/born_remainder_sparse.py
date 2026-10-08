"""Relative remainder of the truncated Born series — sparse strong-phase
synthetic specimen (ptypy chord-vs-tangent demonstration sample).

The specimen of paper-1.tex Sec. (sparse): three pure-phase layers 1.58 mm
apart on the scan-434 geometry (670 positions, 512 px frames, 8 keV,
4.18 m), particles rescaled to ~-10 rad with |O| = 1 and placed so that no
scan position illuminates particles on two layers within the beam cone.
Ground truth (layers, probe, z) comes from the ptypy simulation
(run_sparse_strong_sim_10rad.py); the real potential phi for the tangent is
phi_true_unwrapped.npy.

Same analysis as born_remainder_bounds_wse2.py, restricted to what is
plotted there: measured (unit weights) and tuned (per-view fit, batch 1)
remainders and detector intensity errors for the chord (Delta O = O - 1)
and tangent (Delta O = i phi) perturbations, truncation orders M = 1..3,
energy-weighted RMS over ALL 670 positions. Nz = 3 keeps the per-view
least squares small, so the tuned coefficients are solved by batched
column-normalized normal equations with an eigenvalue cutoff, in
complex128 on the GPU. Exit-plane and detector norms coincide (unimodular
gauge, F unitary).

The position -> pixel mapping (axis order and motor flips) is recovered by
matching a synthetic coverage map against the simulation's coverage.npy.

Run (from the repo root):
  CUDA_VISIBLE_DEVICES=0 ~/ptyrad/.venv/bin/python analysis/born_remainder_sparse.py
"""

import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ptyrad.forward_models.born_helpers import (  # noqa: E402
    _born_advance,
    _born_scatter,
    born_multislice_target,
)
from torch.fft import fft2, ifft2  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
SIM_DIR = (
    "/home/dnz75396/ptypy/data/nanomax/process/0002_multislice/scan_000434/"
    "ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/sim/"
    "sim10rad_20260920_082901_slices-3_seed-7_entref/"
)
PTYD = SIM_DIR + "sim10rad_20260920_082901_slices-3_seed-7.ptyd"
RESULT_DIR = "/home/dnz75396/ptypy/ISS-work/result/fixedstep_phase_20260920/"
PHI_TRUE = RESULT_DIR + "phi_true_unwrapped.npy"
PROBE_TRUE = RESULT_DIR + "truth_probe.npy"
COVERAGE = RESULT_DIR + "coverage.npy"
OUT_PNG = os.path.join(REPO, "demo", "born_remainder_sparse.png")
OUT_CSV = os.path.join(REPO, "demo", "born_remainder_sparse.csv")

ENERGY_KEV = 8.0
DIST_M = 4.18
DET_PSIZE = 75e-6
NPIX = 512
DZ_M = 1.58e-3
N_ORDER = 3

C_PLAIN = "#2a78d6"
C_TUNED = "#eb6834"
C_GRID = "#e1e0d9"
C_SURFACE = "#fcfcfb"


def detector_orders(obj, Psi_state, H, n_max):
    orders = []
    for k in range(n_max):
        scat = _born_scatter(obj, Psi_state, H, k)
        if k == n_max - 1:
            orders.append(scat.sum(dim=3))
        else:
            Psi_state, D_k = _born_advance(scat, H, k)
            orders.append(D_k.clone())
    return torch.stack(orders)


def pixel_offsets(positions, dx, canvas, coverage):
    """Recover the position -> pixel-offset mapping (axis order and flips)
    by matching a box coverage map against the simulation's coverage.npy."""
    best = None
    for swap in (False, True):
        pos = positions[:, ::-1] if swap else positions
        for fy in (1, -1):
            for fx in (1, -1):
                p = pos * np.array([fy, fx])
                pix = (p - p.min(axis=0)) / dx
                off = np.round(pix).astype(int)
                if (off.max(axis=0) + NPIX > np.array(canvas)).any():
                    continue
                cov = np.zeros(canvas)
                for oy, ox in off:
                    cov[oy : oy + NPIX, ox : ox + NPIX] += 1.0
                c = np.corrcoef(cov.ravel(), coverage.ravel())[0, 1]
                if best is None or c > best[0]:
                    best = (c, off, swap, fy, fx)
    c, off, swap, fy, fx = best
    print(f"position mapping: swap={swap} flip_y={fy} flip_x={fx} "
          f"coverage correlation {c:.4f}")
    assert c > 0.98, "could not recover the position -> pixel mapping"
    return off


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phi = np.load(PHI_TRUE)  # (3, 1224, 1225) real potential, ~-10 rad
    probe = np.load(PROBE_TRUE)  # (2, 512, 512) complex
    coverage = np.load(COVERAGE)
    with h5py.File(PTYD) as f:
        positions = f["chunks/0/positions"][()]  # (670, 2) metres

    lam = 1.239842e-9 / ENERGY_KEV
    dx = lam * DIST_M / (NPIX * DET_PSIZE)
    off = pixel_offsets(positions, dx, phi.shape[1:], coverage)
    n_pos = off.shape[0]

    # unimodular angular-spectrum propagator for one 1.58 mm gap
    fy = np.fft.fftfreq(NPIX, dx)
    k0 = 2 * np.pi / lam
    kx2 = (2 * np.pi * fy[None, :]) ** 2 + (2 * np.pi * fy[:, None]) ** 2
    kz = np.sqrt(np.clip(k0**2 - kx2, 0.0, None))
    H1 = torch.from_numpy(np.exp(1j * DZ_M * (kz - k0))).to(device)
    H3 = torch.stack(
        [torch.ones_like(H1), H1, H1 * H1], dim=0
    ).view(1, 1, 1, N_ORDER, NPIX, NPIX)  # H^j, |H| = 1 exactly

    probes = torch.from_numpy(probe).to(device, torch.complex128).unsqueeze(0)
    probe_k = fft2(probes).view(1, probe.shape[0], 1, 1, NPIX, NPIX)
    Psi0 = ifft2(H3 * probe_k)
    D0_single = probe_k.squeeze(3)  # (1, pmode, 1, Ny, Nx)
    phi_t = torch.from_numpy(phi).to(device, torch.float64)

    def vnorm(field):  # plain 2-norm over (pmode, omode, y, x), per view
        return field.abs().square().sum(dim=(1, 2, 3, 4)).sqrt()

    rel = {("chord", "plain"): [], ("chord", "tuned"): [],
           ("chord", "plainI"): [], ("chord", "tunedI"): [],
           ("tangent", "plain"): [], ("tangent", "tuned"): [],
           ("tangent", "plainI"): [], ("tangent", "tunedI"): []}
    nT_l, nI_l = [], []
    with torch.no_grad():
        for lo in range(0, n_pos, 8):
            sel = off[lo : lo + 8]
            B = sel.shape[0]
            pp = torch.stack(
                [phi_t[:, oy : oy + NPIX, ox : ox + NPIX] for oy, ox in sel]
            ).unsqueeze(1)  # (B, omode=1, Nz, Ny, Nx)
            patches = torch.stack(
                [torch.ones_like(pp), pp], dim=-1
            )  # pseudo-complex, |O| = 1
            obj_chord = (torch.polar(patches[..., 0], patches[..., 1]) - 1.0
                         ).unsqueeze(1)
            obj_tan = (1j * pp.to(torch.complex128)).unsqueeze(1)

            T = born_multislice_target(patches, probes, H3)
            n_T = vnorm(T)
            I_ref = T.abs().square().sum(dim=(1, 2))
            n_I = I_ref.flatten(1).norm(dim=1)
            nT_l.append(n_T.cpu())
            nI_l.append(n_I.cpu())
            D0 = D0_single.expand(B, -1, -1, -1, -1)

            def intensity_err(R):
                dI = (R.abs().square() - 2 * (T.conj() * R).real).sum(dim=(1, 2))
                return (dI.flatten(1).norm(dim=1) / n_I).cpu()

            for name, obj in (("chord", obj_chord), ("tangent", obj_tan)):
                D = detector_orders(obj, Psi0, H3, N_ORDER)
                term = torch.stack([vnorm(D[m]) for m in range(N_ORDER)])
                plain, tuned, plain_I, tuned_I = [], [], [], []
                psi_leq = D0.clone()
                for M in range(1, N_ORDER + 1):
                    psi_leq = psi_leq + D[M - 1]
                    R_plain = T - psi_leq
                    plain.append((vnorm(R_plain) / n_T).cpu())
                    plain_I.append(intensity_err(R_plain))
                    # batch-1 fits via column-normalized normal equations
                    # (M <= 3), eigenvalue cutoff at 1e-12 of the largest.
                    # Sparse specimen: many views have D_m = 0 exactly (no
                    # particle in the beam), so clamp the column normalizer
                    # — the zero columns then land in the eigenvalue cutoff
                    term_c = torch.maximum(
                        term[:M], 1e-30 * n_T.view(1, -1)
                    ).to(torch.complex128)
                    A = (D[:M] / term_c.view(M, B, 1, 1, 1, 1)).reshape(M, B, -1)
                    b = R_plain.reshape(B, -1)
                    G = torch.einsum("mbl,nbl->bmn", A.conj(), A)
                    p = torch.einsum("mbl,bl->bm", A.conj(), b)
                    lam_e, V = torch.linalg.eigh(G.cpu())  # tiny; CPU is robust
                    lam_e, V = lam_e.to(device), V.to(device)
                    lam_max = lam_e[:, -1].clamp(min=1e-300)
                    inv = torch.where(
                        lam_e > 1e-12 * lam_max[:, None],
                        1.0 / lam_e, torch.zeros_like(lam_e),
                    ).to(torch.complex128)
                    y = torch.einsum(
                        "bmn,bn,bn->bm", V, inv,
                        torch.einsum("bnm,bn->bm", V.conj(), p),
                    )
                    # guard at the float64 floor (see the WSe2 script)
                    fitted = (vnorm(R_plain) >= 1e-14 * n_T).view(
                        1, B, 1, 1, 1, 1
                    )
                    x = (y.T / term_c).view(M, B, 1, 1, 1, 1)
                    R_tuned = R_plain - (x * D[:M] * fitted).sum(dim=0)
                    tuned.append((vnorm(R_tuned) / n_T).cpu())
                    tuned_I.append(intensity_err(R_tuned))
                rel[name, "plain"].append(torch.stack(plain))
                rel[name, "tuned"].append(torch.stack(tuned))
                rel[name, "plainI"].append(torch.stack(plain_I))
                rel[name, "tunedI"].append(torch.stack(tuned_I))

    curves = {k: torch.cat(v, dim=1).numpy() for k, v in rel.items()}
    n_T_all = torch.cat(nT_l).numpy()
    n_I_all = torch.cat(nI_l).numpy()

    # energy-weighted RMS over all positions
    def rms(v, w):
        return np.sqrt((v**2 * w[None, :]).sum(axis=1) / w.sum())

    agg = {}
    for (name, kind), v in curves.items():
        agg[name, kind] = rms(v, n_I_all**2 if kind.endswith("I") else n_T_all**2)

    m_axis = np.arange(1, N_ORDER + 1)
    kinds = ["plain", "tuned", "plainI", "tunedI"]
    with open(OUT_CSV, "w") as f:
        cols = [f"{n}_{k}_{s}" for n in ("chord", "tangent") for k in kinds
                for s in ("rms", "mean", "min", "max")]
        f.write("M," + ",".join(cols) + "\n")
        for m in range(N_ORDER):
            row = [str(m + 1)]
            for n in ("chord", "tangent"):
                for k in kinds:
                    v = curves[n, k][m]
                    row += [f"{agg[n, k][m]:.8g}", f"{v.mean():.8g}",
                            f"{v.min():.8g}", f"{v.max():.8g}"]
            f.write(",".join(row) + "\n")

    fig, (ax, axI) = plt.subplots(
        1, 2, figsize=(11.0, 5.6), dpi=150, facecolor=C_SURFACE
    )
    for a in (ax, axI):
        a.set_facecolor(C_SURFACE)
        a.set_yscale("log")
        a.set_xticks(m_axis)
        a.set_xlabel("truncation order $M$")
        a.grid(alpha=0.9, which="major", color=C_GRID, lw=0.7)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    series = {
        "chord": (C_PLAIN, r"chord $\Delta O = O - 1$ (non-linearised)"),
        "tangent": (C_TUNED, r"tangent $\Delta O = i\varphi$ (linearised)"),
    }
    # tuned curves stay in the CSV but are not plotted: on this specimen
    # tuning cannot improve the chord (tuned == plain at every M)
    for name, (color, mdl_label) in series.items():
        ax.plot(m_axis, agg[name, "plain"], marker="o", ms=6, lw=2.0,
                color=color, zorder=5, label=mdl_label)
        axI.plot(m_axis, agg[name, "plainI"], marker="o", ms=6, lw=2.0,
                 color=color, zorder=5)
    ax.set_ylim(bottom=1e-16)
    ax.set_ylabel(r"$\|\hat R_M\| \,/\, \|\hat\psi_{\mathrm{MS}}\|$")
    ax.set_title("field remainder", fontsize=11)
    axI.set_ylabel(
        r"$\|\hat I_M - \hat I_{\mathrm{MS}}\|_2 \,/\, \|\hat I_{\mathrm{MS}}\|_2$"
    )
    axI.set_title("detector intensity error", fontsize=11)
    fig.legend(loc="lower center", ncol=2, framealpha=0.9, fontsize=9,
               columnspacing=1.4)
    fig.suptitle(
        f"Truncated Born series vs multislice — sparse strong-phase specimen, "
        f"3 layers of 1.58 mm, averaged over all {n_pos} positions",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(OUT_PNG, facecolor=C_SURFACE)
    print(f"wrote {OUT_PNG} and {OUT_CSV}")
    for name in ("chord", "tangent"):
        print(f"--- {name}")
        for m in range(N_ORDER):
            print(
                f"M = {m + 1}: plain {agg[name, 'plain'][m]:.3e}  "
                f"tuned {agg[name, 'tuned'][m]:.3e}  "
                f"plainI {agg[name, 'plainI'][m]:.3e}  "
                f"tunedI {agg[name, 'tunedI'][m]:.3e}"
            )


if __name__ == "__main__":
    main()

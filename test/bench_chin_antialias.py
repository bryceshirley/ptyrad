"""
Forward bench for the Chin 4A antialias variants on a REAL sharp specimen:
the recovered 21-slice PSO object (chin_results/20260927_lt21_ref_s42_full,
model_iter0100.hdf5), its reconstructed 4-mode probe, dz = 10 A.

Measures, for each variant, the exit-wave relative L2 error against a
converged fine-sub-slice reference of the SAME total stack, and the forward
wall time (CUDA events, best of 5 after warmup, f32 like production):

  a. chin4a_fd      : current Chin 4A, g from raw 4th-order FD canvas  [baseline]
  b. chin4a_g0      : Chin 4A, g term OFF
  c. chin4a_specbl  : g from SPECTRALLY band-limited chi (Gaussian low-pass,
                      sigma_k = k_Nyq/3, spectral gradient)  == chin_g_bandlimit: 0.333
  d. chin4a_fdbl    : band-limit applied to the existing FD g itself (cheap)
  e1. chin4a_aa_fd  : antialias aperture (hard 2/3-Nyquist mask on H_half) + raw FD g
  e2. chin4a_aa_bl  : antialias aperture + band-limited g (c)
  e3. chin4a_aa_g0  : antialias aperture + g OFF (cheapest)
  lt1               : Lie-Trotter at the native slab count (context)
  lt2               : Lie-Trotter at 2 sub-slices/slab (equal transform budget:
                      22 slices for the 11-slab model, 42 for the 21-slab one)

Models are evaluated at the NATIVE 21 slabs and at an 11-slab merge (exact
z-reintegration of the chi density onto 11 equal slabs; the benchmark
configs reconstruct 11 slices from scratch, so no canonical merge exists —
both slab counts are reported).

Ground truth: the 21-slab stack forward-propagated with fine sub-slices
(T^(1/n) per sub-slice with H(dz/n)); n = 16 vs 32 convergence is reported,
plus a symmetric (Strang) n = 32 cross-check. All error math in float64;
the f32 column repeats the computation in production dtype.

Also runs a global order-of-convergence fit (fixed total thickness, z-uniform
sharp specimen built from the same object) for raw-FD g vs band-limited g.

Run (from the repo root, one free GPU):
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=<UUID> \
      /home/dnz75396/ptyrad/.venv/bin/python test/bench_chin_antialias.py
"""

import glob
import json
import os
import sys

import h5py
import numpy as np
import torch
from torch.fft import fft2, ifft2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ptyrad.forward_models.multislice import _multislice_forward_chin4a_impl  # noqa: E402
from ptyrad.utils import fd_gradient4, fftshift2  # noqa: E402

REF_RUN = (
    "/home/dnz75396/ptyrad/demo/output/high_order_benchmarks/chin_results/"
    "20260927_lt21_ref_s42_full*"
)
OUT_JSON = os.path.join(os.path.dirname(__file__), "bench_chin_antialias.json")

N_POS = 6            # scan positions for the error statistics
NPIX = 256
GT_SUBS = (16, 32)   # ground-truth sub-slice counts (task spec)
GT_STRANG_SUB = 32   # symmetric cross-check


# ---------------------------------------------------------------------------
# grids and kernels (same conventions as ptyrad.utils.fresnel_evolution:
# +0.5 half-bin offset, ifftshifted)
# ---------------------------------------------------------------------------


def k_grids(n, dx, dtype, device):
    grid = (torch.arange(-(n // 2), n // 2, dtype=dtype, device=device) + 0.5) / n
    k1d = torch.fft.ifftshift(2 * torch.pi * grid / dx)
    return torch.meshgrid(k1d, k1d, indexing="ij")  # Ky, Kx


def fresnel_kernel(n, dx, dz, k0, dtype, device):
    Ky, Kx = k_grids(n, dx, dtype, device)
    return torch.exp(-1j * dz * (Kx**2 + Ky**2).to(_cplx(dtype)) / (2 * k0))


def antialias_mask(n, dx, frac, dtype, device):
    Ky, Kx = k_grids(n, dx, dtype, device)
    k_cut = frac * torch.pi / dx
    return ((Kx**2 + Ky**2) <= k_cut**2).to(dtype)


def _cplx(dtype):
    return torch.complex128 if dtype == torch.float64 else torch.complex64


# ---------------------------------------------------------------------------
# g variants on the full canvas (chi = phase - i*logamp per slab)
# ---------------------------------------------------------------------------


def g_from_grads(gyp, gxp, gyl, gxl):
    g_re = gyp**2 - gyl**2 + gxp**2 - gxl**2
    g_im = -2.0 * (gyp * gyl + gxp * gxl)
    return torch.complex(g_re, g_im)


def g_fd(phase, logamp, dx):
    """Production path: raw 4th-order central differences, replicate padded."""
    gyp, gxp = fd_gradient4(phase, dx)
    gyl, gxl = fd_gradient4(logamp, dx)
    return g_from_grads(gyp, gxp, gyl, gxl)


def _pad_sizes(n):
    pad = (-(n + 64)) % 16 + 64
    return pad // 2, pad - pad // 2


def gauss_filter_grid(ny, nx, dx, k_cut, dtype, device, shape="gauss"):
    ky = 2 * torch.pi * torch.fft.fftfreq(ny, d=dx, dtype=dtype, device=device)
    kx = 2 * torch.pi * torch.fft.fftfreq(nx, d=dx, dtype=dtype, device=device)
    Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
    K2 = Ky**2 + Kx**2
    if shape == "hard":
        filt = (K2 <= k_cut**2).to(dtype)
    else:
        filt = torch.exp(-0.5 * K2 / k_cut**2)
    return filt, Ky, Kx


def g_spectral_bandlimited(phase, logamp, dx, frac=1.0 / 3.0, shape_k="gauss"):
    """Variant (c): band-limited chi (Gaussian sigma_k = frac*k_Nyq, or HARD
    cutoff at frac*k_Nyq for shape_k='hard' — the production
    chin_g_bandlimit path) on the replicate-padded canvas, SPECTRAL gradient.
    Mirrors PtychoAD._bandlimited_chi_gradients (same padding rule)."""
    k_cut = frac * torch.pi / dx
    shape = phase.shape
    ny, nx = shape[-2], shape[-1]
    py0, py1 = _pad_sizes(ny)
    px0, px1 = _pad_sizes(nx)
    npy, npx = ny + py0 + py1, nx + px0 + px1
    filt, Ky, Kx = gauss_filter_grid(npy, npx, dx, k_cut, phase.dtype, phase.device, shape_k)

    f = torch.stack([phase, logamp], 0).reshape(-1, 1, ny, nx)
    fp = torch.nn.functional.pad(f, (px0, px1, py0, py1), mode="replicate")
    F = fft2(fp) * filt
    gy = ifft2(torch.complex(torch.zeros_like(Ky), Ky) * F).real
    gx = ifft2(torch.complex(torch.zeros_like(Kx), Kx) * F).real
    gy = gy[..., py0 : py0 + ny, px0 : px0 + nx].reshape(2, *shape)
    gx = gx[..., py0 : py0 + ny, px0 : px0 + nx].reshape(2, *shape)
    return g_from_grads(gy[0], gx[0], gy[1], gx[1])


def g_fd_bandlimited(phase, logamp, dx, frac=1.0 / 3.0):
    """Variant (d): raw FD g, then Gaussian band-limit the g field itself."""
    g = g_fd(phase, logamp, dx)
    k_cut = frac * torch.pi / dx
    shape = g.shape
    ny, nx = shape[-2], shape[-1]
    py0, py1 = _pad_sizes(ny)
    px0, px1 = _pad_sizes(nx)
    npy, npx = ny + py0 + py1, nx + px0 + px1
    filt, _, _ = gauss_filter_grid(npy, npx, dx, k_cut, g.real.dtype, g.device)
    f = torch.stack([g.real, g.imag], 0).reshape(-1, 1, ny, nx)
    fp = torch.nn.functional.pad(f, (px0, px1, py0, py1), mode="replicate")
    out = ifft2(fft2(fp) * filt).real[..., py0 : py0 + ny, px0 : px0 + nx]
    out = out.reshape(2, *shape)
    return torch.complex(out[0], out[1])


# ---------------------------------------------------------------------------
# propagation schemes (exit waves; psi enters at the FRONT face of slab 1 and
# leaves at the BACK face of the last slab; total drift n_slabs*dz)
# ---------------------------------------------------------------------------


def lt_sub_exit(chi, psi, dx, dz, k0, n_sub, mask=None):
    """[T^(1/n_sub) then H(dz/n_sub)] per sub-slice. chi: (S, Ny, Nx) complex;
    psi: (..., Ny, Nx) complex. mask: optional aperture on every kernel
    (aperture-consistent reference)."""
    n = chi.shape[-1]
    H = fresnel_kernel(n, dx, dz / n_sub, k0, chi.real.dtype, chi.device)
    if mask is not None:
        H = H * mask
    for s in range(chi.shape[0]):
        T = torch.exp(1j * chi[s] / n_sub)
        for _ in range(n_sub):
            psi = ifft2(H * fft2(psi * T))
    return psi


def strang_sub_exit(chi, psi, dx, dz, k0, n_sub, mask=None):
    """Symmetric cross-check: [H(d/2) T^(1/n) H(d/2)] per sub-slice with the
    half-drifts merged inside each slab."""
    n = chi.shape[-1]
    H = fresnel_kernel(n, dx, dz / n_sub, k0, chi.real.dtype, chi.device)
    H_half = fresnel_kernel(n, dx, 0.5 * dz / n_sub, k0, chi.real.dtype, chi.device)
    if mask is not None:
        H = H * mask
        H_half = H_half * mask
    for s in range(chi.shape[0]):
        T = torch.exp(1j * chi[s] / n_sub)
        psi = ifft2(H_half * fft2(psi))
        for m in range(n_sub):
            psi = psi * T
            psi = ifft2((H if m < n_sub - 1 else H_half) * fft2(psi))
    return psi


def band_project(psi, mask):
    """Project a wave onto the aperture band (for in-band error metrics)."""
    return ifft2(mask * fft2(psi))


def chin4a_exit(chi, g, psi, dx, dz, k0, mask=None):
    """Chin 4A slab sweep (same structure as _multislice_forward_chin4a_impl,
    returning the exit wave). chi, g: (S, Ny, Nx) complex (g may be None);
    mask: optional hard aperture multiplied into H_half."""
    n = chi.shape[-1]
    H_half = fresnel_kernel(n, dx, 0.5 * dz, k0, chi.real.dtype, chi.device)
    if mask is not None:
        H_half = H_half * mask
    W = (2.0 / 3.0) * chi
    if g is not None:
        W = W + (dz / (72.0 * k0)) * g
    T = torch.exp(1j * W)
    B = torch.exp(1j * chi / 6.0)
    S = chi.shape[0]
    psi = psi * B[0]
    for s in range(S):
        psi = ifft2(H_half * fft2(psi))
        psi = psi * T[s]
        psi = ifft2(H_half * fft2(psi))
        psi = psi * (B[s] * B[s + 1] if s < S - 1 else B[s])
    return psi


# ---------------------------------------------------------------------------
# specimen handling
# ---------------------------------------------------------------------------


def merge_slabs(chi, n_out):
    """Exact z-reintegration of the per-slab chi (uniform-density slabs of
    equal thickness) onto n_out equal slabs covering the same total depth."""
    n_in = chi.shape[0]
    if n_out == n_in:
        return chi
    t_in, t_out = 1.0 / n_in, 1.0 / n_out
    out = torch.zeros((n_out, *chi.shape[1:]), dtype=chi.dtype, device=chi.device)
    for j in range(n_out):
        z0, z1 = j * t_out, (j + 1) * t_out
        for k in range(n_in):
            a0, a1 = k * t_in, (k + 1) * t_in
            ov = max(0.0, min(z1, a1) - max(z0, a0))
            if ov > 0:
                out[j] += (ov / t_in) * chi[k]
    return out


def rel_l2(psi, psi_ref):
    d = (psi - psi_ref).abs().square().sum()
    return float(torch.sqrt(d / psi_ref.abs().square().sum()))


def main():
    assert torch.cuda.is_available(), "GPU required"
    device = "cuda"
    run_dir = sorted(glob.glob(REF_RUN))[0]
    with h5py.File(os.path.join(run_dir, "model_iter0100.hdf5"), "r") as f:
        obja = f["optimizable_tensors/obja"][0]      # (21, 639, 639)
        objp = f["optimizable_tensors/objp"][0]
        probe = f["optimizable_tensors/probe"][()]   # (4, 256, 256) complex64
        crop_pos = f["model_attributes/crop_pos"][()]
        dx = float(f["model_attributes/dx"][()])
        dz21 = float(f["optimizable_tensors/slice_thickness"][()])
        kv = float(f["params/init_params/probe_kv"][()])
    lam = 12.398 / np.sqrt((2 * 511.0 + kv) * kv)  # Angstrom (physics.py convention)
    k0 = 2 * np.pi / lam
    S21 = objp.shape[0]
    L = S21 * dz21
    print(f"specimen: {run_dir}")
    print(f"  {S21} slabs x dz={dz21} A, dx={dx:.6f} A, lambda={lam:.6f} A, k0={k0:.2f} 1/A")

    # positions: spread across the scan
    sel = np.linspace(0, len(crop_pos) - 1, N_POS).astype(int)
    pos = crop_pos[sel]
    print(f"  positions (corner px): {pos.tolist()}")

    results = {"meta": {"run_dir": run_dir, "dx": dx, "dz21": dz21, "lambda": lam,
                        "positions": pos.tolist(), "n_pos": N_POS}}

    for dtype in (torch.float64, torch.float32):
        tag = "f64" if dtype == torch.float64 else "f32"
        cdtype = _cplx(dtype)
        phase = torch.tensor(objp, dtype=dtype, device=device)
        logamp = torch.log(torch.tensor(obja, dtype=dtype, device=device).clamp_min(1e-10))
        chi21 = torch.complex(phase, -logamp)  # chi = phase - i*log(amp)
        probe_t = torch.tensor(probe, dtype=torch.complex64).to(cdtype).to(device)

        # canvas-level g variants, per slab count
        stacks, gsets = {}, {}
        for S in (21, 11):
            chiS = merge_slabs(chi21, S)
            phS, laS = chiS.real, -chiS.imag
            stacks[S] = chiS
            gsets[S] = {
                "fd": g_fd(phS, laS, dx),
                "specbl": g_spectral_bandlimited(phS, laS, dx),
                "specblh": g_spectral_bandlimited(phS, laS, dx, shape_k="hard"),
                "fdbl": g_fd_bandlimited(phS, laS, dx),
            }
        mask = antialias_mask(NPIX, dx, 2.0 / 3.0, dtype, device)

        # ground truth per position (21-slab native stack, fine sub-slices).
        # Three metrics per variant:
        #   full   : rel L2 vs the full-band reference
        #   inband : rel L2 of the 2/3-Nyquist band-projected waves (the
        #            physics the antialias model class claims to keep)
        #   cons   : rel L2 vs the APERTURE-CONSISTENT reference (fine
        #            sub-slices with the same masked kernels); equals 'full'
        #            for the unmasked variants
        err, err_band, err_cons = {}, {}, {}
        gt_checks = []
        for p, (cy, cx) in enumerate(pos):
            chi_p21 = stacks[21][:, cy : cy + NPIX, cx : cx + NPIX]
            psi0 = probe_t.clone()
            gts = {n: lt_sub_exit(chi_p21, psi0, dx, dz21, k0, n) for n in GT_SUBS}
            gt_strang = strang_sub_exit(chi_p21, psi0, dx, dz21, k0, GT_STRANG_SUB)
            gt_aa = strang_sub_exit(chi_p21, psi0, dx, dz21, k0, GT_STRANG_SUB, mask=mask)
            gt_checks.append({
                "lt16_vs_lt32": rel_l2(gts[16], gts[32]),
                "lt32_vs_strang32": rel_l2(gts[32], gt_strang),
                "gt_vs_gt_aa": rel_l2(gt_aa, gt_strang),
            })
            gt = gt_strang  # most converged (symmetric); checks quantify the LT refs
            gt_band = band_project(gt, mask)

            for S in (21, 11):
                dzS = L / S
                chi_p = stacks[S][:, cy : cy + NPIX, cx : cx + NPIX]
                gp = {k: v[:, cy : cy + NPIX, cx : cx + NPIX] for k, v in gsets[S].items()}
                variants = {
                    "chin4a_fd": (gp["fd"], None),
                    "chin4a_g0": (None, None),
                    "chin4a_specbl": (gp["specbl"], None),
                    "chin4a_specblh": (gp["specblh"], None),
                    "chin4a_fdbl": (gp["fdbl"], None),
                    "chin4a_aa_fd": (gp["fd"], mask),
                    "chin4a_aa_bl": (gp["specblh"], mask),
                    "chin4a_aa_g0": (None, mask),
                }
                waves = {name: chin4a_exit(chi_p, g, psi0, dx, dzS, k0, mask=m)
                         for name, (g, m) in variants.items()}
                waves["lt1"] = lt_sub_exit(chi_p, psi0, dx, dzS, k0, 1)
                waves["lt2"] = lt_sub_exit(chi_p, psi0, dx, dzS, k0, 2)
                for name, psi in waves.items():
                    err.setdefault((S, name), []).append(rel_l2(psi, gt))
                    err_band.setdefault((S, name), []).append(
                        rel_l2(band_project(psi, mask), gt_band))
                    ref_c = gt_aa if "_aa_" in name else gt
                    err_cons.setdefault((S, name), []).append(rel_l2(psi, ref_c))

        results[f"gt_convergence_{tag}"] = gt_checks
        for key, d in (("errors", err), ("errors_inband", err_band),
                       ("errors_consistent", err_cons)):
            results[f"{key}_{tag}"] = {
                f"{S}sl/{name}": float(np.mean(v)) for (S, name), v in d.items()
            }
        print(f"\n[{tag}] GT convergence per position:")
        for c in gt_checks:
            print(f"  lt16 vs lt32: {c['lt16_vs_lt32']:.3e}   "
                  f"lt32 vs strang32: {c['lt32_vs_strang32']:.3e}   "
                  f"gt vs gt_aa: {c['gt_vs_gt_aa']:.3e}")
        print(f"[{tag}] mean exit-wave rel L2 error (full / in-band / consistent):")
        for (S, name) in sorted(err):
            print(f"  {S:2d}sl  {name:16s} {np.mean(err[(S, name)]):.4e}  "
                  f"{np.mean(err_band[(S, name)]):.4e}  "
                  f"{np.mean(err_cons[(S, name)]):.4e}")

    # -----------------------------------------------------------------------
    # parity check: bench chin4a_exit reproduces the production impl (dp level)
    # -----------------------------------------------------------------------
    dtype, cdtype = torch.float64, torch.complex128
    phase = torch.tensor(objp, dtype=dtype, device=device)
    logamp = torch.log(torch.tensor(obja, dtype=dtype, device=device).clamp_min(1e-10))
    chi21 = torch.complex(phase, -logamp)
    probe_t = torch.tensor(probe, dtype=torch.complex64).to(cdtype).to(device)
    cy, cx = pos[0]
    chi_p = chi21[:, cy : cy + NPIX, cx : cx + NPIX]
    gp = g_fd(chi_p.real, -chi_p.imag, dx)
    obj_patches = torch.stack([torch.exp(-chi_p.imag), chi_p.real], -1)[None, None]
    g_patches = torch.stack([gp.real, gp.imag], -1)[None, None]
    H_half = fresnel_kernel(NPIX, dx, 0.5 * dz21, k0, dtype, device)[None]
    dp_impl = _multislice_forward_chin4a_impl(
        obj_patches, probe_t[None], H_half,
        g_patches, torch.tensor(dz21, dtype=dtype, device=device),
        torch.tensor(k0, dtype=dtype, device=device), eps=0.0,
    )[0]
    psi = chin4a_exit(chi_p, gp, probe_t, dx, dz21, k0)
    dp_bench = fftshift2(fft2(psi[:, None], norm="ortho")).abs().square().mean(dim=(0, 1))
    # impl averages omode with occupancy and sums pmode weighted 1/omode; here
    # omode=1 so impl dp = sum over pmode of |.|^2 / 1; bench mean -> rescale
    dp_bench = dp_bench * probe_t.shape[0]
    parity = rel_l2(dp_bench, dp_impl)
    print(f"\nparity bench-vs-impl (dp rel L2, f64): {parity:.3e}")
    results["parity_dp_f64"] = parity

    # -----------------------------------------------------------------------
    # timing, f32 production dtype: forward dp (batch 1, pmode 4, omode 1)
    # + canvas g-prep per variant
    # -----------------------------------------------------------------------
    def cuda_time(fn, n_warm=3, n_rep=5):
        for _ in range(n_warm):
            fn()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(n_rep):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
        return best  # ms

    timing = {}
    dtype, cdtype = torch.float32, torch.complex64
    phase = torch.tensor(objp, dtype=dtype, device=device)
    logamp = torch.log(torch.tensor(obja, dtype=dtype, device=device).clamp_min(1e-10))
    chi21 = torch.complex(phase, -logamp)
    probe_f32 = torch.tensor(probe, dtype=torch.complex64, device=device)[None]
    mask = antialias_mask(NPIX, dx, 2.0 / 3.0, dtype, device)
    for S in (21, 11):
        dzS = L / S
        chiS = merge_slabs(chi21, S)
        phS, laS = chiS.real, -chiS.imag
        cy, cx = pos[0]
        chi_p = chiS[:, cy : cy + NPIX, cx : cx + NPIX]
        obj_patches = torch.stack([torch.exp(-chi_p.imag), chi_p.real], -1)[None, None].contiguous()
        gp = g_fd(chi_p.real, -chi_p.imag, dx)
        g_patches = torch.stack([gp.real, gp.imag], -1)[None, None].contiguous()
        H_half = fresnel_kernel(NPIX, dx, 0.5 * dzS, k0, dtype, device)[None]
        H_half_aa = H_half * mask
        dz_t = torch.tensor(dzS, dtype=dtype, device=device)
        k0_t = torch.tensor(k0, dtype=dtype, device=device)

        timing[f"{S}sl/forward_chin4a_g"] = cuda_time(lambda: _multislice_forward_chin4a_impl(
            obj_patches, probe_f32, H_half, g_patches, dz_t, k0_t))
        timing[f"{S}sl/forward_chin4a_g0"] = cuda_time(lambda: _multislice_forward_chin4a_impl(
            obj_patches, probe_f32, H_half, None, dz_t, k0_t))
        timing[f"{S}sl/forward_chin4a_aa_g"] = cuda_time(lambda: _multislice_forward_chin4a_impl(
            obj_patches, probe_f32, H_half_aa, g_patches, dz_t, k0_t))

        # equal-budget LT at 2 sub-slices/slab (eager loop + detector transform)
        def lt2_forward():
            psi = lt_sub_exit(chi_p, probe_f32[0], dx, dzS, k0, 2)
            return fftshift2(fft2(psi, norm="ortho")).abs().square().sum(0)

        def lt1_forward():
            psi = lt_sub_exit(chi_p, probe_f32[0], dx, dzS, k0, 1)
            return fftshift2(fft2(psi, norm="ortho")).abs().square().sum(0)

        timing[f"{S}sl/forward_lt2"] = cuda_time(lt2_forward)
        timing[f"{S}sl/forward_lt1"] = cuda_time(lt1_forward)

        # canvas g preparation (once per batch in production, batch=1)
        timing[f"{S}sl/gprep_fd"] = cuda_time(lambda: g_fd(phS, laS, dx))
        timing[f"{S}sl/gprep_specbl"] = cuda_time(lambda: g_spectral_bandlimited(phS, laS, dx))
        timing[f"{S}sl/gprep_specblh"] = cuda_time(
            lambda: g_spectral_bandlimited(phS, laS, dx, shape_k="hard"))
        timing[f"{S}sl/gprep_fdbl"] = cuda_time(lambda: g_fd_bandlimited(phS, laS, dx))

    print("\ntiming (ms, best of 5, f32, batch 1, pmode 4):")
    for k, v in timing.items():
        print(f"  {k:28s} {v:8.3f}")
    results["timing_ms"] = timing

    # -----------------------------------------------------------------------
    # order of convergence on this specimen (z-uniform sharp slab):
    # raw FD g vs band-limited spectral g (and antialias variants)
    # -----------------------------------------------------------------------
    dtype, cdtype = torch.float64, torch.complex128
    phase = torch.tensor(objp, dtype=dtype, device=device)
    logamp = torch.log(torch.tensor(obja, dtype=dtype, device=device).clamp_min(1e-10))
    chi21 = torch.complex(phase, -logamp)
    probe_t = torch.tensor(probe, dtype=torch.complex64).to(cdtype).to(device)
    rho = chi21.sum(0) / L  # chi density per Angstrom, z-uniform, sharp in-plane
    cy, cx = pos[0]
    rho_p = rho[cy : cy + NPIX, cx : cx + NPIX]
    # canvas-level g densities (g scales as dz^2 * g_density)
    gd_fd = g_fd(rho.real, -rho.imag, dx)[cy : cy + NPIX, cx : cx + NPIX]
    gd_bl = g_spectral_bandlimited(rho.real, -rho.imag, dx)[cy : cy + NPIX, cx : cx + NPIX]
    gd_blh = g_spectral_bandlimited(rho.real, -rho.imag, dx, shape_k="hard")[cy : cy + NPIX, cx : cx + NPIX]
    mask64 = antialias_mask(NPIX, dx, 2.0 / 3.0, dtype, device)

    ref = strang_sub_exit(rho_p[None] * L, probe_t, dx, L, k0, 4096)
    ref_chk = strang_sub_exit(rho_p[None] * L, probe_t, dx, L, k0, 2048)
    ord_ref_conv = rel_l2(ref_chk, ref)
    print(f"\norder test reference convergence (strang 2048 vs 4096): {ord_ref_conv:.3e}")

    n_list = (4, 8, 16, 32, 64, 128)
    order_errs = {k: [] for k in ("fd", "specbl", "specblh", "g0", "aa_fd", "aa_bl")}
    for n_sl in n_list:
        dzn = L / n_sl
        chi_n = (rho_p * dzn).expand(n_sl, NPIX, NPIX)
        # with chi_k = rho*dzn, g = grad(chi_k).grad(chi_k) = dzn^2 * g_density
        cases = {
            "fd": (gd_fd, None),
            "specbl": (gd_bl, None),
            "specblh": (gd_blh, None),
            "g0": (None, None),
            "aa_fd": (gd_fd, mask64),
            "aa_bl": (gd_blh, mask64),
        }
        for name in order_errs:
            gd, m = cases[name]
            g = None if gd is None else (gd * dzn**2).expand(n_sl, NPIX, NPIX)
            psi = chin4a_exit(chi_n, g, probe_t, dx, dzn, k0, mask=m)
            order_errs[name].append(rel_l2(psi, ref))

    dzs = [L / n for n in n_list]
    slopes = {}
    print("\nglobal order on the z-uniform sharp specimen (rel L2 exit-wave error):")
    print("dz        " + "".join(f"{s:>12s}" for s in order_errs))
    for i, dzv in enumerate(dzs):
        print(f"{dzv:8.3f}  " + "".join(f"{order_errs[s][i]:12.3e}" for s in order_errs))
    tail = {}
    for name, es in order_errs.items():
        slopes[name] = float(np.polyfit(np.log(dzs), np.log(es), 1)[0])
        tail[name] = float(np.polyfit(np.log(dzs[-3:]), np.log(es[-3:]), 1)[0])
    print("fitted slopes (all):", {k: round(v, 2) for k, v in slopes.items()})
    print("fitted slopes (tail 3):", {k: round(v, 2) for k, v in tail.items()})
    results["order_test"] = {"dzs": dzs, "errors": order_errs, "slopes": slopes,
                             "tail_slopes": tail, "ref_convergence": ord_ref_conv}

    # control: the SAME specimen with chi HARD band-limited at k_Nyq/3.
    # If the order loss above comes from the sharp potential (band-limit
    # condition k_c <= k_Nyq/3 for the cubic products), the fourth order must
    # reappear here for every g variant.
    py0, py1 = _pad_sizes(rho.shape[-2])
    px0, px1 = _pad_sizes(rho.shape[-1])
    npy, npx = rho.shape[-2] + py0 + py1, rho.shape[-1] + px0 + px1
    filt, _, _ = gauss_filter_grid(npy, npx, dx, (1.0 / 3.0) * torch.pi / dx,
                                   dtype, device, shape="hard")
    rf = torch.stack([rho.real, -rho.imag], 0)[:, None]
    rf = torch.nn.functional.pad(rf, (px0, px1, py0, py1), mode="replicate")
    rf = ifft2(fft2(rf) * filt).real[..., py0 : py0 + rho.shape[-2], px0 : px0 + rho.shape[-1]]
    rho_bl = torch.complex(rf[0, 0], -rf[1, 0])
    rho_bl_p = rho_bl[cy : cy + NPIX, cx : cx + NPIX]
    gd_fd_bl = g_fd(rho_bl.real, -rho_bl.imag, dx)[cy : cy + NPIX, cx : cx + NPIX]
    gd_sp_bl = g_spectral_bandlimited(rho_bl.real, -rho_bl.imag, dx)[cy : cy + NPIX, cx : cx + NPIX]
    gd_sp_blh = g_spectral_bandlimited(rho_bl.real, -rho_bl.imag, dx, shape_k="hard")[cy : cy + NPIX, cx : cx + NPIX]

    ref_bl = strang_sub_exit(rho_bl_p[None] * L, probe_t, dx, L, k0, 4096)
    order_errs_bl = {k: [] for k in ("fd", "specbl", "specblh", "g0")}
    for n_sl in n_list:
        dzn = L / n_sl
        chi_n = (rho_bl_p * dzn).expand(n_sl, NPIX, NPIX)
        for name, gd in (("fd", gd_fd_bl), ("specbl", gd_sp_bl),
                         ("specblh", gd_sp_blh), ("g0", None)):
            g = None if gd is None else (gd * dzn**2).expand(n_sl, NPIX, NPIX)
            psi = chin4a_exit(chi_n, g, probe_t, dx, dzn, k0)
            order_errs_bl[name].append(rel_l2(psi, ref_bl))
    slopes_bl = {k: float(np.polyfit(np.log(dzs), np.log(v), 1)[0])
                 for k, v in order_errs_bl.items()}
    tail_bl = {k: float(np.polyfit(np.log(dzs[-3:]), np.log(v[-3:]), 1)[0])
               for k, v in order_errs_bl.items()}
    print("\ncontrol: same specimen, chi band-limited at k_Nyq/3:")
    print("dz        " + "".join(f"{s:>12s}" for s in order_errs_bl))
    for i, dzv in enumerate(dzs):
        print(f"{dzv:8.3f}  " + "".join(f"{order_errs_bl[s][i]:12.3e}" for s in order_errs_bl))
    print("fitted slopes (all):", {k: round(v, 2) for k, v in slopes_bl.items()})
    print("fitted slopes (tail 3):", {k: round(v, 2) for k, v in tail_bl.items()})
    results["order_test_bandlimited_chi"] = {
        "dzs": dzs, "errors": order_errs_bl, "slopes": slopes_bl,
        "tail_slopes": tail_bl}

    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()

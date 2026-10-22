#!/usr/bin/env python
"""
Forward validation of the Chebyshev exact-generator slab model
(src/ptyrad/forward_models/chebyshev.py) on the archived 21-slice PSO
object (chin_results lt21_ref_s42, iteration 100).

Ground truth: fine sub-sliced Lie-Trotter (each 10 A slice split into 16
sub-slices, T^(1/16) then H(dz/16), angular-spectrum kernel, full drift
= 210 A), float64; convergence verified against 32 sub-slices.

Compared exit waves (central 256x256 patch, 4 probe modes):
  (a) Chebyshev, 3 slabs (7 slices each, dz_slab = 70 A)
  (b) Chebyshev, 1 slab (whole 210 A stack)
  (c) Lie-Trotter, 21 slices (full drift, angular spectrum)
  (d) Chin 4A, 11 slices (g = None, Fresnel kernel; object fractionally
      rebinned 21 x 10 A -> 11 x 19.09 A, phase/log-amp conserving)
  (e) Chebyshev, 3 slabs, UNSHIFTED recurrence on 1 + mu + eps
      (demonstrates the float32 cancellation the shifted form avoids)

Metric: relative L2 error after optimal global-phase alignment (the
carrier/drift conventions differ by a constant phase only; alignment
cannot hide spatial errors). Wall times: CUDA events, 3 warmups, best of
5, float32. FFT-pair counts exclude the final detector FFT.

Also runs torch.autograd.gradcheck (float64, tiny grid) through the
Chebyshev forward.

Run: CUDA_VISIBLE_DEVICES=<UUID> PYTHONPATH=src \
     ~/ptyrad/.venv/bin/python scripts/validate_chebyshev_forward.py
"""

import glob
import json
import os

import h5py
import numpy as np
import torch
from torch.fft import fft2, ifft2

from ptyrad.forward_models.chebyshev import (
    ChebyshevSlabKernel,
    chebyshev_coeffs,
    chebyshev_exit_wave,
    multislice_forward_chebyshev,
)
from ptyrad.utils.physics import get_EM_constants

ARCHIVE = sorted(
    glob.glob(
        "/home/dnz75396/ptyrad/demo/output/high_order_benchmarks/chin_results/"
        "20260927_lt21_ref_s42_full*/model_iter0100.hdf5"
    )
)[0]
OUT_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "demo/output/chebyshev/forward_validation.json",
)
NPATCH = 256
KV = 300.0

dev = torch.device("cuda")


def load_specimen():
    with h5py.File(ARCHIVE, "r") as f:
        obja = f["optimizable_tensors/obja"][0]  # (21, 639, 639)
        objp = f["optimizable_tensors/objp"][0]
        probe = f["optimizable_tensors/probe"][...]  # (4, 256, 256) c64
        dx = float(f["model_attributes/dx"][...])
        dz = float(f["optimizable_tensors/slice_thickness"][...])
    y0 = (obja.shape[-2] - NPATCH) // 2
    x0 = (obja.shape[-1] - NPATCH) // 2
    obja = obja[:, y0 : y0 + NPATCH, x0 : x0 + NPATCH]
    objp = objp[:, y0 : y0 + NPATCH, x0 : x0 + NPATCH]
    return obja, objp, probe, dx, dz


def rebin_z(amp, phase, n_new):
    """Fractional z-rebin of a piecewise-constant stack (phase and log-amp
    conserving: column sums are preserved exactly)."""
    n_old = phase.shape[0]
    la = np.log(np.clip(amp, 1e-10, None))
    W = np.zeros((n_new, n_old))
    for i in range(n_new):
        z0, z1 = i * n_old / n_new, (i + 1) * n_old / n_new
        for j in range(n_old):
            W[i, j] = max(0.0, min(z1, j + 1) - max(z0, j))
    ph_new = np.einsum("ij,jyx->iyx", W, phase)
    la_new = np.einsum("ij,jyx->iyx", W, la)
    return np.exp(la_new), ph_new


def kgrids(n, dx, dtype=np.float64):
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx).astype(dtype)
    return k[:, None] ** 2 + k[None, :] ** 2


def lt_exit(obj_cplx, psi, H):
    """T then H per slice, FULL drift (H after the last slice too)."""
    for n in range(obj_cplx.shape[0]):
        psi = ifft2(H * fft2(psi * obj_cplx[n]))
    return psi


def chin4a_exit(amp, phase, psi, H_half):
    """Chin 4A sweep (g = None), as _multislice_forward_chin4a_impl, exit wave."""
    T = torch.polar(amp ** (2.0 / 3.0), phase * (2.0 / 3.0))
    amp6 = amp.clamp_min(1e-10) ** (1.0 / 6.0)
    ph6 = phase / 6.0
    n_sl = amp.shape[0]
    psi = psi * torch.polar(amp6[0], ph6[0])
    for n in range(n_sl):
        psi = ifft2(H_half * fft2(psi))
        psi = psi * T[n]
        psi = ifft2(H_half * fft2(psi))
        if n < n_sl - 1:
            psi = psi * torch.polar(amp6[n + 1] * amp6[n], ph6[n + 1] + ph6[n])
    return psi * torch.polar(amp6[-1], ph6[-1])


def cheb_unshifted_exit(patches, psi, km):
    """Chebyshev recurrence on 1 + mu + eps with g(y) = exp(i*theta*(sqrt(y)-1))
    over [1+lo, 1+hi] — the numerically naive form, for the f32 comparison."""
    lo1, hi1 = 1.0 + km.lo, 1.0 + km.hi
    n = 4096
    j = np.arange(n)
    ym = 0.5 * (hi1 + lo1) + 0.5 * (hi1 - lo1) * np.cos(np.pi * (j + 0.5) / n)
    fj = np.exp(1j * km.theta_half * (np.sqrt(ym) - 1.0))
    C = np.cos(np.pi * np.outer(np.arange(n), j + 0.5) / n)
    c = (2.0 / n) * (C @ fj)
    c[0] *= 0.5
    keep = np.nonzero(np.abs(c) >= km.tol)[0]
    c = c[: int(keep[-1]) + 1]
    coeffs = torch.tensor(c, dtype=km.cdtype, device=dev)
    a = 2.0 / (hi1 - lo1)
    b = -(hi1 + lo1) / (hi1 - lo1)
    mu_mul = km.mu_mul
    mask = km.band_mask

    def bp(v):
        return ifft2(mask * fft2(v))

    def apply_poly(v, eps_r):
        def S_op(w):
            Bpw = w + ifft2(mu_mul * fft2(w)) + eps_r * w  # (1 + B) w
            return a * Bpw + b * w

        t0 = v
        acc = coeffs[0] * t0
        t1 = S_op(v)
        acc = acc + coeffs[1] * t1
        for k in range(2, coeffs.shape[0]):
            t2 = 2.0 * S_op(t1) - t0
            acc = acc + coeffs[k] * t2
            t0, t1 = t1, t2
        return acc

    amp = patches[..., 0]
    phase = patches[..., 1]
    for s0, s1 in km.groups:
        nm1 = phase[:, :, s0:s1].sum(dim=2) / km.k0_dz_slab
        eps_r = (nm1 * (2.0 + nm1))[:, None]
        amp_slab = amp[:, :, s0:s1].prod(dim=2)[:, None]
        psi = apply_poly(bp(psi), eps_r)
        psi = psi * amp_slab
        psi = apply_poly(bp(psi), eps_r)
    return bp(psi), coeffs.shape[0] - 1


def rel_err(psi, ref):
    """Relative L2 after optimal global phase alignment."""
    z = torch.sum(torch.conj(ref) * psi)
    phase = z / z.abs().clamp_min(1e-30)
    return float(torch.linalg.vector_norm(psi * torch.conj(phase) - ref) / torch.linalg.vector_norm(ref))


def rel_err_inband(psi, ref, mask):
    """rel_err with both fields hard-projected onto the band (same mask the
    solver applies); the comparison then ignores out-of-band content the
    band-cropped model removes by construction."""
    return rel_err(ifft2(mask * fft2(psi)), ifft2(mask * fft2(ref)))


def cuda_time(fn, warmup=3, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = np.inf
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


def main():
    obja, objp, probe, dx, dz = load_specimen()
    lambd = get_EM_constants(KV, "wavelength")
    k0 = 2.0 * np.pi / lambd
    nz = objp.shape[0]
    Z = nz * dz
    print(f"Specimen: {nz} slices x {dz:.3f} A = {Z:.1f} A, dx = {dx:.5f} A, "
          f"lambda = {lambd:.6f} A, k0 = {k0:.3f} 1/A")

    results = {"specimen": ARCHIVE, "dx": dx, "dz": dz, "lambda": lambd}
    K2 = kgrids(NPATCH, dx)
    Kz_AS = np.sqrt(np.clip(k0**2 - K2, 0.0, None))

    for prec, cdt, rdt in (("f64", torch.complex128, torch.float64),
                           ("f32", torch.complex64, torch.float32)):
        print(f"\n=== {prec} ===")
        amp_t = torch.tensor(obja, dtype=rdt, device=dev)
        ph_t = torch.tensor(objp, dtype=rdt, device=dev)
        probe_t = torch.tensor(probe, device=dev).to(cdt)[None]  # (1,4,Ny,Nx)
        psi0 = probe_t[:, :, None]  # (1,4,1,Ny,Nx)
        obj_cplx = torch.polar(amp_t, ph_t).to(cdt)
        patches = torch.stack([amp_t, ph_t], dim=-1)[None, None]  # (1,1,21,Ny,Nx,2)

        # --- ground truth: fine sub-sliced LT in f64 always ---
        if prec == "f64":
            gt = {}
            for nsub in (16, 32):
                H_sub = torch.tensor(np.exp(1j * (dz / nsub) * Kz_AS), dtype=torch.complex128, device=dev)
                T_sub = torch.polar(amp_t ** (1.0 / nsub), ph_t / nsub).to(torch.complex128)
                psi = psi0[:, :, 0].clone()
                for n in range(nz):
                    for _ in range(nsub):
                        psi = ifft2(H_sub * fft2(psi * T_sub[n]))
                gt[nsub] = psi
            conv = rel_err(gt[16], gt[32])
            print(f"GT convergence: |GT16 - GT32| / |GT32| = {conv:.3e}")
            results["gt_convergence_16_vs_32"] = conv
            ref64 = gt[32]

            # exactness isolation: fine-LT ground truth on the HOMOGENISED
            # 3-slab medium (each slab replaced by its z-mean phase/log-amp).
            # Chebyshev-3 vs this reference removes the homogenisation error;
            # what remains is the sqrt-generator cross term (the Chebyshev
            # generator sqrt(1+mu+eps) differs from the fine-LT limit
            # sqrt(1+mu) + (n-1) at order mu*eps) plus the 1e-6 truncation.
            la = np.log(np.clip(obja, 1e-10, None))
            ph_h = objp.copy()
            la_h = la.copy()
            for i in range(3):
                s = slice(7 * i, 7 * (i + 1))
                ph_h[s] = objp[s].mean(axis=0, keepdims=True)
                la_h[s] = la[s].mean(axis=0, keepdims=True)
            amp_h = torch.tensor(np.exp(la_h), dtype=rdt, device=dev)
            ph_ht = torch.tensor(ph_h, dtype=rdt, device=dev)
            nsub = 32
            H_sub = torch.tensor(np.exp(1j * (dz / nsub) * Kz_AS), dtype=cdt, device=dev)
            T_sub = torch.polar(amp_h ** (1.0 / nsub), ph_ht / nsub).to(cdt)
            psi = psi0[:, :, 0].clone()
            for n in range(nz):
                for _ in range(nsub):
                    psi = ifft2(H_sub * fft2(psi * T_sub[n]))
            gt_homog = psi
        ref = ref64.to(cdt)

        rows = {}
        fields = {}
        # (c) LT 21, full drift
        H1 = torch.tensor(np.exp(1j * dz * Kz_AS), dtype=cdt, device=dev)
        psi_lt = lt_exit(obj_cplx, psi0[:, :, 0], H1)
        rows["lt21"] = (rel_err(psi_lt, ref), nz)

        # (d) Chin 4A, 11 slices, Fresnel kernel, g=None
        n_c = 11
        dz_c = Z / n_c
        amp11, ph11 = rebin_z(obja, objp, n_c)
        amp11_t = torch.tensor(amp11, dtype=rdt, device=dev)
        ph11_t = torch.tensor(ph11, dtype=rdt, device=dev)
        H_half = torch.tensor(np.exp(-1j * (dz_c / 2.0) * K2 / (2.0 * k0)), dtype=cdt, device=dev)
        psi_chin = chin4a_exit(amp11_t, ph11_t, psi0[:, :, 0], H_half)
        rows["chin4a11"] = (rel_err(psi_chin, ref), 2 * n_c)

        # (a, b) Chebyshev 3 slabs and 1 slab
        kms = {}
        for name, n_slab in (("cheb3", 3), ("cheb1", 1)):
            gs = nz // n_slab
            groups = tuple((i * gs, (i + 1) * gs) for i in range(n_slab))
            km = ChebyshevSlabKernel((NPATCH, NPATCH), dx, lambd, gs * dz, groups,
                                     device=dev, dtype=cdt, verbose=True)
            km.refresh(ph_t)
            kms[name] = km
            psi_cheb = chebyshev_exit_wave(patches, psi0, *km.kernels())[:, :, 0]
            deg_half = km.coeffs.shape[0] - 1
            rows[name] = (rel_err(psi_cheb, ref), 2 * deg_half * n_slab)
            fields[name] = psi_cheb
            if prec == "f64" and name == "cheb3":
                e_h = rel_err(psi_cheb, gt_homog)
                print(f"cheb3 vs homogenised-slab GT32 (in-slab exactness + "
                      f"cross term only): {e_h:.3e}")
                results["cheb3_vs_homogenised_gt"] = e_h

        # (e) unshifted form, 3 slabs
        psi_uns, deg_uns = cheb_unshifted_exit(patches, psi0, kms["cheb3"])
        rows["cheb3_unshifted"] = (rel_err(psi_uns[:, :, 0], ref), 2 * deg_uns * 3)
        fields["cheb3_unshifted"] = psi_uns[:, :, 0]

        # in-band comparison: both fields hard-projected onto the solver's
        # antialias band before the metric (the band crop removes
        # out-of-band GT content by construction; in-band is the
        # apples-to-apples accuracy number)
        mask = kms["cheb3"].band_mask
        F_ref = fft2(ref)
        pw = F_ref.abs().square()
        oob_frac = float((pw * (1.0 - mask)).sum() / pw.sum())
        inband = {
            "lt21": rel_err_inband(psi_lt, ref, mask),
            "chin4a11": rel_err_inband(psi_chin, ref, mask),
            "cheb3": rel_err_inband(fields["cheb3"], ref, mask),
            "cheb1": rel_err_inband(fields["cheb1"], ref, mask),
        }
        print(f"GT power beyond the {kms['cheb3'].band_frac:.4f}*K_Nyq band: {oob_frac:.3e}")
        print(f"{'model':<18}{'in-band rel L2':>16}")
        for k, e in inband.items():
            print(f"{k:<18}{e:>16.3e}")

        # roundoff isolation: each precision's field vs the f64 shifted field
        if prec == "f64":
            cheb3_f64_field = fields["cheb3"]
        else:
            ro_sh = rel_err(fields["cheb3"].to(torch.complex128), cheb3_f64_field)
            ro_un = rel_err(fields["cheb3_unshifted"].to(torch.complex128), cheb3_f64_field)
            print(f"f32 numerical error vs f64 shifted field: shifted {ro_sh:.3e}, "
                  f"unshifted {ro_un:.3e}")
            results["f32_roundoff"] = {"shifted": ro_sh, "unshifted": ro_un}

        print(f"{'model':<18}{'rel L2 vs GT':>14}{'FFT pairs':>11}")
        for k, (e, nfft) in rows.items():
            print(f"{k:<18}{e:>14.3e}{nfft:>11d}")
        results[prec] = {k: {"rel_l2": e, "fft_pairs": nfft} for k, (e, nfft) in rows.items()}
        results[prec]["inband_rel_l2"] = inband
        results[prec]["gt_power_beyond_band"] = oob_frac

        if prec == "f32":
            # wall times (exit-wave computation, CUDA events, best of 5)
            times = {}
            times["lt21"] = cuda_time(lambda: lt_exit(obj_cplx, psi0[:, :, 0], H1))
            times["chin4a11"] = cuda_time(lambda: chin4a_exit(amp11_t, ph11_t, psi0[:, :, 0], H_half))
            for name in ("cheb3", "cheb1"):
                km = kms[name]
                times[name] = cuda_time(lambda km=km: chebyshev_exit_wave(patches, psi0, *km.kernels()))
            cheb3_c = torch.compile(chebyshev_exit_wave, dynamic=False)
            km = kms["cheb3"]
            times["cheb3_compiled"] = cuda_time(lambda: cheb3_c(patches, psi0, *km.kernels()))
            print("wall times (ms, f32, best of 5):",
                  {k: round(v, 3) for k, v in times.items()})
            results["times_ms_f32"] = times

    # --- gradcheck, float64, tiny grid ---
    print("\n=== gradcheck (f64, 16x16, Nz=2, 1 slab) ===")
    ng = 16
    km = ChebyshevSlabKernel((ng, ng), dx, lambd, 2 * dz, ((0, 2),), device=dev,
                             dtype=torch.complex128, verbose=False)
    torch.manual_seed(0)
    ph_s = 0.05 * torch.rand(1, 2, ng, ng, dtype=torch.float64, device=dev)
    km.refresh(ph_s)
    mu_mul, band_mask, coeffs, lo, hi, k0dz, groups = km.kernels()
    print(f"gradcheck half-slab degree: {coeffs.shape[0] - 1}")

    def fwd(obja_p, objp_p, probe_r, probe_i):
        patches = torch.stack([obja_p, objp_p], dim=-1)[None]  # (1,1,2,ng,ng,2)
        pr = torch.complex(probe_r, probe_i)[None]  # (1,1,ng,ng)
        return multislice_forward_chebyshev(
            patches, pr, mu_mul, band_mask, coeffs, lo, hi, k0dz, groups
        )

    inputs = (
        (1.0 - 0.02 * torch.rand(1, 2, ng, ng, dtype=torch.float64, device=dev)).requires_grad_(),
        ph_s[0][None].clone().requires_grad_(),
        torch.randn(1, ng, ng, dtype=torch.float64, device=dev).requires_grad_(),
        torch.randn(1, ng, ng, dtype=torch.float64, device=dev).requires_grad_(),
    )
    ok = torch.autograd.gradcheck(fwd, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print(f"gradcheck passed: {ok}")
    results["gradcheck"] = bool(ok)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()

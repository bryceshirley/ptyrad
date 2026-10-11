"""Forward-consistency cross-check: PtyRAD's own multislice forward model must
reproduce abtem's noise-free patterns from the exported ground truth.

Run (ptyrad venv, from the repo root):
  .venv/bin/python -m generate_data.check_ptyrad_forward demo/data/generated/smoke_2slab.h5

For each test position: object patch from gt_phase (amplitude 1), exported
probe, PtyRAD Fresnel propagator (near_field_evolution) for dz at dx,
multislice_forward -> compare with dp_noiseless.  All 8 dihedral orientations
of the measured pattern are scored; the winner determines meas_flipT
([flipud, fliplr, transpose], applied to the measured data).
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import torch  # ty: ignore[unresolved-import]

from ptyrad.forward_models import multislice_forward
from ptyrad.utils.physics import near_field_evolution


def electron_wavelength_A(kv: float) -> float:
    return 12.3984244 / np.sqrt(kv * (2 * 510.99895 + kv))


def asm_propagator(
    npix: int, dx: float, dz: float, lam: float, half_offset: bool
) -> np.ndarray:
    """Angular-spectrum propagator.  half_offset=True reproduces PtyRAD's
    near_field_evolution (k-grid at (m+0.5)/N - fold_slice heritage);
    False is the standard fftfreq grid (abtem's convention)."""
    off = 0.5 if half_offset else 0.0
    grid = (np.arange(-npix // 2, npix // 2) + off) / npix
    k = 2 * np.pi / lam
    kk = 2 * np.pi * grid / dx
    Ky, Kx = np.meshgrid(kk, kk, indexing="ij")
    return np.fft.ifftshift(np.exp(1j * dz * np.sqrt(k**2 - Kx**2 - Ky**2)))


def fresnel_propagator(npix: int, dx: float, dz: float, lam: float) -> np.ndarray:
    """abtem's order-1 (parabolic) Fresnel propagator: exp(-i pi lam dz k^2)."""
    k = np.fft.fftfreq(npix, d=dx)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    return np.exp(-1j * np.pi * lam * dz * k2)


def dihedral(pat: np.ndarray, flipud: int, fliplr: int, transpose: int) -> np.ndarray:
    out = pat
    if flipud:
        out = out[..., ::-1, :]
    if fliplr:
        out = out[..., :, ::-1]
    if transpose:
        out = np.swapaxes(out, -2, -1)
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float(np.corrcoef(a, b)[0, 1])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    path = Path(args[0] if args else "out/smoke_2slab.h5")
    n_test = int(args[1]) if len(args) > 1 else 5
    use_fine = "--fine" in flags
    standard_grid = "--standard-grid" in flags
    window256 = "--window256" in flags

    with h5py.File(path, "r") as f:
        dp = f["dp_noiseless"][...]
        if use_fine:
            gt = f["gt_phase_fine"][...]
            dz = float(f["dz_fine_A"][()])
        else:
            gt = f["gt_phase"][...]
            dz = float(f.attrs["dz_A"])
        probe = f["probe_win256"][...] if window256 else f["probe"][...]
        pos = f["scan_pos_yx_A"][...]
        dx = float(f.attrs["dx_A"])
        kv = float(f.attrs["kv"])

    n_scans = dp.shape[0]
    npix = probe.shape[-1]  # model window (128, or 256 with --window256)
    det_npix = dp.shape[-1]
    nz = gt.shape[0]
    lam = electron_wavelength_A(kv)
    print(
        f"{path.name}: {n_scans} patterns, {nz} gt slices, dx={dx:.5f} A, dz={dz:.3f} A, "
        f"lambda={lam:.6f} A"
    )

    idx = np.linspace(0, n_scans - 1, n_test).round().astype(int)
    half = npix // 2

    ngt = gt.shape[-1]
    patches = []
    for i in idx:
        # gt pixel j has effective center (j + 0.25) * dx (2x2 mean of the
        # simulation grid), so position y maps to row y/dx - 0.25
        cy = round(float(pos[i, 0]) / dx - 0.25)
        cx = round(float(pos[i, 1]) / dx - 0.25)
        ry = (np.arange(cy - half, cy + half)) % ngt  # periodic supercell
        rx = (np.arange(cx - half, cx + half)) % ngt
        patches.append(gt[np.ix_(np.arange(nz), ry, rx)])
    patches = np.stack(patches)  # (n, nz, npix, npix)

    obj = torch.stack(
        [torch.ones(patches.shape), torch.as_tensor(patches, dtype=torch.float32)],
        dim=-1,
    )[:, None]  # (n, omode=1, nz, ny, nx, 2)
    probe_t = torch.as_tensor(probe, dtype=torch.complex64)[None, None]
    if "--fresnel" in flags:
        H_np = fresnel_propagator(npix, dx, dz, lam)
        print("propagator: parabolic Fresnel (abtem order 1), fftfreq grid")
    elif standard_grid:
        H_np = asm_propagator(npix, dx, dz, lam, half_offset=False)
        print("propagator: standard fftfreq grid")
    else:
        H_np = near_field_evolution((npix, npix), dx, dz, lam)
        ref = asm_propagator(npix, dx, dz, lam, half_offset=True)
        assert np.allclose(H_np, ref), "near_field_evolution changed convention?"
        print("propagator: PtyRAD near_field_evolution (half-pixel-offset grid)")
    H = torch.as_tensor(H_np.astype(np.complex64))[None]

    with torch.no_grad():
        fwd = multislice_forward(obj.float(), probe_t, H).numpy()
    if window256:
        c = npix // 2
        sel = c + 2 * (np.arange(det_npix) - det_npix // 2)
        if "--kernelbin" in flags:
            # approximate the data's 4-native-px window average with a
            # (0.5, 1, 0.5)/2 kernel on the da/2 grid before sampling
            k1 = np.array([0.5, 1.0, 0.5]) / 2.0
            fwd = sum(w * np.roll(fwd, -j, axis=-2) for j, w in zip((-1, 0, 1), k1))
            fwd = sum(w * np.roll(fwd, -j, axis=-1) for j, w in zip((-1, 0, 1), k1))
        fwd = fwd[:, sel[:, None], sel[None, :]]
    fwd /= fwd.sum(axis=(-2, -1), keepdims=True)

    meas = dp[idx]
    meas = meas / meas.sum(axis=(-2, -1), keepdims=True)

    print(f"\n{'flipT':<12} {'mean Pearson':>13} {'mean relL2':>11}")
    results = {}
    for fu in (0, 1):
        for fl in (0, 1):
            for tr in (0, 1):
                m = dihedral(meas, fu, fl, tr)
                rs = [pearson(m[k], fwd[k]) for k in range(len(idx))]
                l2 = [
                    float(np.linalg.norm(m[k] - fwd[k]) / np.linalg.norm(fwd[k]))
                    for k in range(len(idx))
                ]
                results[(fu, fl, tr)] = (np.mean(rs), np.mean(l2), rs, l2)
                print(f"[{fu}, {fl}, {tr}]    {np.mean(rs):13.5f} {np.mean(l2):11.4f}")

    best = max(results, key=lambda k: results[k][0])
    mean_r, mean_l2, rs, l2 = results[best]
    print(
        f"\nbest orientation flipT={list(best)}: per-position Pearson "
        f"{[f'{r:.5f}' for r in rs]}, relL2 {[f'{v:.4f}' for v in l2]}"
    )
    ok = mean_r >= 0.99 and mean_l2 <= 0.10
    print(f"GATE (Pearson >= 0.99, relL2 <= ~0.1): {'PASS' if ok else 'FAIL'}")
    if best != (0, 0, 0):
        print(
            "NOTE: non-identity orientation needed -> set meas_flipT accordingly in configs"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

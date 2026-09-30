"""Build a deep-stack 4D-STEM phantom from a REAL ptychographic reconstruction
instead of an ASE atomic model.

Motivation: the ASE + abtem parametrized-potential path leaves a forward-model
mismatch (infinite-projection potential vs the reconstructor's transmission
slices) that shows up as depth artifacts. Here the ground-truth object IS a
reconstructed phase stack (tBL_WSe2, 80 kV, 24.9 mrad), so the phantom's
per-slice transmission is exactly what a multislice reconstructor parametrizes.

Pipeline:
  1. crop the reconstructed phase to the well-illuminated window,
  2. rebin the 12 recon slices -> 6 (the WSe2 bilayer's 6 atomic planes;
     summing pairs is the correct projected-potential merge), dz -> 2 A,
  3. tile the 6-slice unit to n_slices (default 128 -> 256 A),
  4. feed phase/sigma as an abtem PotentialArray, scan a single reconstructed
     probe mode across it (custom Waves ensemble -> multislice),
  5. crop+bin the far field to the detector, scale to dose, Poisson.
"""

from dataclasses import dataclass

import h5py
import numpy as np
from abtem import PotentialArray, Waves
from abtem.core.axes import PositionsAxis
from abtem.core.energy import energy2sigma, energy2wavelength

SRC = "/home/dnz75396/ptyrad/demo/data/tBL_WSe2/Panel_g-h_Themis/model_iter0200.hdf5"


@dataclass(frozen=True)
class ReconPhantom:
    phase: np.ndarray      # (n_slices, N, N) per-slice transmission phase (GT)
    probe: np.ndarray      # (N_probe, N_probe) complex, unit-power
    dx_A: float
    dz_A: float
    kv: float
    unit_slices: int       # slices in the repeating unit (6)


def load_phantom(
    src: str = SRC,
    crop_px: int = 384,
    unit_slices: int = 6,
    n_slices: int = 128,
    probe_mode: int = 0,
    kv: float = 80.0,
) -> ReconPhantom:
    with h5py.File(src, "r") as f:
        objp = f["optimizable_tensors/objp"][0].astype(np.float64)  # (12,583,583)
        probe = f["optimizable_tensors/probe"][probe_mode].astype(np.complex64)
        dx = float(f["model_attributes/dx"][()])
    nz0, ny, _ = objp.shape
    o = (ny - crop_px) // 2
    crop = objp[:, o : o + crop_px, o : o + crop_px]
    # rebin nz0 -> unit_slices by summing consecutive groups (projected pot.)
    assert nz0 % unit_slices == 0, f"{nz0} not divisible by {unit_slices}"
    g = nz0 // unit_slices
    unit = crop.reshape(unit_slices, g, crop_px, crop_px).sum(axis=1)  # (6,N,N)
    dz = 1.0 * g  # original slice_thickness (1.0) times grouping
    # tile the unit to n_slices (cycle)
    reps = int(np.ceil(n_slices / unit_slices))
    phase = np.concatenate([unit] * reps, axis=0)[:n_slices]
    probe = probe / np.sqrt((np.abs(probe) ** 2).sum())
    return ReconPhantom(phase.astype(np.float32), probe, dx, float(dz), kv, unit_slices)


def _place(probe: np.ndarray, N: int, cy: int, cx: int) -> np.ndarray:
    canvas = np.zeros((N, N), np.complex64)
    n = probe.shape[0]
    canvas[cy - n // 2 : cy + n // 2, cx - n // 2 : cx + n // 2] = probe
    return canvas


def scan_positions_px(N: int, n_slow: int, n_fast: int, step_px: float):
    """Centered raster of probe-CENTER pixel positions on the N-grid."""
    span_s = (n_slow - 1) * step_px
    span_f = (n_fast - 1) * step_px
    y0 = (N - span_s) / 2
    x0 = (N - span_f) / 2
    ys = y0 + np.arange(n_slow) * step_px
    xs = x0 + np.arange(n_fast) * step_px
    # (n_slow, n_fast) grid, slow=y outer
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([YY.ravel(), XX.ravel()], axis=1)  # (npos, 2) [y,x] in px


def simulate(
    ph: ReconPhantom,
    positions_px: np.ndarray,
    device: str = "gpu",
    batch: int = 128,
) -> np.ndarray:
    """Custom-probe multislice scan -> full far-field intensities
    (npos, N, N), fftshifted (DC centered)."""
    N = ph.phase.shape[-1]
    sigma = energy2sigma(ph.kv * 1e3)
    pot = PotentialArray(
        (ph.phase / sigma).astype(np.float32),
        slice_thickness=ph.dz_A,
        sampling=(ph.dx_A, ph.dx_A),
    )
    if device == "gpu":
        pot = pot.to_gpu()
    out = []
    for i0 in range(0, len(positions_px), batch):
        chunk = positions_px[i0 : i0 + batch]
        arr = np.stack(
            [_place(ph.probe, N, int(round(y)), int(round(x))) for y, x in chunk]
        )
        w = Waves(
            arr,
            energy=ph.kv * 1e3,
            sampling=(ph.dx_A, ph.dx_A),
            ensemble_axes_metadata=[PositionsAxis(values=tuple(map(tuple, chunk)))],
        )
        if device == "gpu":
            w = w.to_gpu()
        ew = w.multislice(pot)
        dp = ew.diffraction_patterns(max_angle=None)
        d = dp.array
        d = d.get() if hasattr(d, "get") else np.asarray(d)
        out.append(d.astype(np.float64))
    return np.concatenate(out, axis=0)


def crop_bin_center(dp: np.ndarray, det_npix: int) -> np.ndarray:
    """Centered flux-conserving b x b bin of a DC-centered far field down to
    det_npix, with DC (N//2) mapped to det_npix//2 (PtyRAD fftshift
    convention). N must be an integer multiple of det_npix."""
    N = dp.shape[-1]
    n = det_npix
    b = N // n
    assert b * n == N, f"{N} not divisible by {n}"
    lead = dp.shape[:-2]
    start = (N // 2) - b * (n // 2)  # crop so DC lands at output n//2
    arr = dp[..., start : start + b * n, start : start + b * n]
    arr = arr.reshape(*lead, n, b, n, b).sum(axis=(-3, -1))
    return arr


def scale_and_poisson(fractional, electrons_per_pattern, seed=0):
    flat = fractional.reshape(-1, *fractional.shape[-2:])
    cap = float(flat.sum(axis=(-2, -1)).mean())
    scale = electrons_per_pattern / cap
    noiseless = (fractional * scale).astype(np.float64)
    rng = np.random.default_rng(seed)
    noisy = rng.poisson(noiseless).astype(np.float64)
    return noisy, noiseless, cap


def produce(
    out_h5: str,
    n_slices: int = 128,
    crop_px: int = 384,
    det_npix: int = 128,
    n_slow: int = 64,
    n_fast: int = 64,
    step_px: float = 3.0,
    dose: float = 1e4,
    device: str = "gpu",
    seed: int = 0,
    batch: int = 128,
):
    ph = load_phantom(crop_px=crop_px, n_slices=n_slices)
    pos = scan_positions_px(crop_px, n_slow, n_fast, step_px)
    dp_full = simulate(ph, pos, device=device, batch=batch)
    # fraction of incident flux -> divide by probe total (probe is unit-power in
    # real space; the far-field DC-sum equals the same total, so normalize by it)
    det = crop_bin_center(dp_full, det_npix)
    det = det / det.sum(axis=(-2, -1), keepdims=True).mean()  # -> mean sum 1
    noisy, noiseless, cap = scale_and_poisson(det, dose, seed=seed)
    dx = ph.dx_A
    lam = energy2wavelength(ph.kv * 1e3)
    dk = 1.0 / (det_npix * dx)                 # 1/A per detector pixel
    da_mrad = dk * lam * 1e3
    recon_dx = 1.0 / (det_npix * dk)           # = dx (self-consistent)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("dp", data=noisy.astype(np.float32), compression="gzip")
        f.create_dataset("dp_noiseless", data=noiseless.astype(np.float32),
                         compression="gzip")
        f.create_dataset("gt_phase", data=ph.phase)          # (n_slices,N,N)
        f.create_dataset("probe", data=ph.probe)
        f.create_dataset("scan_pos_yx_A", data=(pos * dx).astype(np.float32))
        f.attrs.update(dict(
            dx_A=dx, dz_A=ph.dz_A, kv=ph.kv, semiangle_mrad=24.9,
            da_mrad=da_mrad, dk=dk, recon_dx_A=recon_dx,
            electrons_per_pattern=float(dose), capture_fraction=cap,
            n_slices=n_slices, unit_slices=ph.unit_slices, crop_px=crop_px,
            det_npix=det_npix, n_slow=n_slow, n_fast=n_fast, step_A=step_px * dx,
            source="tBL_WSe2 Panel_g-h_Themis model_iter0200",
        ))
    return dict(dp=noisy.shape, thickness_A=n_slices * ph.dz_A, dx_A=dx,
                dz_A=ph.dz_A, da_mrad=da_mrad, step_A=step_px * dx,
                capture_fraction=cap)

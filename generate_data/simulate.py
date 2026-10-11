"""abtem simulation drivers: scan, detector crop/bin, noise, ground truth."""

import numpy as np
from abtem import GridScan, PixelatedDetector, Potential, Probe
from abtem.core.energy import energy2sigma

from generate_data.sampling import SamplingPlan


def make_probe(plan: SamplingPlan, device: str = "cpu") -> Probe:
    return Probe(
        energy=plan.kv * 1e3,
        semiangle_cutoff=plan.semiangle_mrad,
        extent=plan.extent_A,
        gpts=plan.gpts,
        aberrations=plan.aberrations,
        device=device,
    )


def centered_scan_axes(
    plan: SamplingPlan, n_slow: int, n_fast: int, step_A: float
) -> tuple[np.ndarray, np.ndarray]:
    """(xs, ys) coordinates in A of the raster scan centered in the cell."""
    x0 = (plan.extent_A - (n_fast - 1) * step_A) / 2
    y0 = (plan.extent_A - (n_slow - 1) * step_A) / 2
    return x0 + np.arange(n_fast) * step_A, y0 + np.arange(n_slow) * step_A


def make_centered_scan(
    plan: SamplingPlan, n_slow: int, n_fast: int, step_A: float
) -> GridScan:
    """Raster scan centered in the lateral cell.

    abtem GridScan's first axis is x; we use (n_fast, n_slow) so that axis 0
    of the resulting measurement is x (fast) and axis 1 is y (slow), and
    handle the PtyRAD ordering (slow=y first) at flatten time.
    """
    xs, ys = centered_scan_axes(plan, n_slow, n_fast, step_A)
    return GridScan(
        start=(xs[0], ys[0]),
        end=(xs[0] + n_fast * step_A, ys[0] + n_slow * step_A),
        gpts=(n_fast, n_slow),
        endpoint=False,
    )


def simulate_scan(
    atoms,
    plan: SamplingPlan,
    scan,  # GridScan or a sequence of (x, y) positions
    slice_thickness_A: float = 1.0,
    device: str = "cpu",
    max_batch: int | str = "auto",
):
    """Run the multislice STEM simulation; returns the (lazy) abtem
    DiffractionPatterns on the native grid up to the antialias cutoff.

    max_batch: positions per wave batch; abtem's 'auto' over-allocates on GPU
    for large scans (OOM on an 80 GB A100 at 4096 positions x 1024^2) - pass
    an explicit value (e.g. 32) for production scans.
    """
    potential = Potential(
        atoms,
        gpts=plan.gpts,
        slice_thickness=slice_thickness_A,
        projection="infinite",
        device=device,
    )
    probe = make_probe(plan, device=device)
    detector = PixelatedDetector(max_angle="cutoff", to_cpu=True)
    measurement = probe.scan(
        potential, scan=scan, detectors=detector, max_batch=max_batch
    )
    return measurement


def crop_and_bin(dp_measurement, plan: SamplingPlan) -> np.ndarray:
    """Crop to the detector window and bin 4x to the detector grid, with the
    bin windows CENTERED on the target pixels so the zero-frequency native
    pixel lands exactly at det_npix/2 (PtyRAD's fftshift convention).

    Binned pixel m samples native angles 4*(m - det_npix/2)*da_native with a
    symmetric width-4 window (weights 0.5,1,1,1,0.5) - flux-conserving 4x
    binning on a grid aligned to DC.  Returns float64
    (..., det_npix, det_npix) in incident-flux fractions.
    """
    if plan.bin_factor == 4:
        weights = (0.5, 1.0, 1.0, 1.0, 0.5)
    elif plan.bin_factor == 2:
        # same construction one octave down: symmetric width-2 window
        # centered on the target pixel, flux-conserving, DC at det_npix/2
        weights = (0.5, 1.0, 0.5)
    else:
        raise NotImplementedError("centered binning implemented for bin_factor in (2, 4)")
    b = plan.bin_factor
    # crop b px wider than the detector span so the edge windows fit
    cropped = dp_measurement.crop(gpts=(plan.crop_px + b, plan.crop_px + b))
    raw = cropped.array
    arr = raw.get() if hasattr(raw, "get") else np.asarray(raw)  # cupy -> numpy
    n = plan.det_npix
    for axis_from_end in (2, 1):
        shape = list(arr.shape)
        shape[-axis_from_end] = n
        parts = np.zeros(shape, dtype=arr.dtype)
        for j, w in enumerate(weights):
            sl = [slice(None)] * arr.ndim
            sl[-axis_from_end] = slice(j, j + b * n, b)
            parts += w * arr[tuple(sl)]
        arr = parts
    return arr


def scale_and_poisson(
    fractional: np.ndarray, electrons_per_pattern: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale so the MEAN total per pattern equals `electrons_per_pattern`,
    then apply Poisson noise.

    Returns (noisy, noiseless, capture_fraction) where capture_fraction is
    the mean fraction of incident flux surviving the detector crop.
    """
    flat = fractional.reshape(-1, *fractional.shape[-2:])
    capture_fraction = float(flat.sum(axis=(-2, -1)).mean())
    scale = electrons_per_pattern / capture_fraction
    noiseless = (fractional * scale).astype(np.float64)
    rng = np.random.default_rng(seed)
    noisy = rng.poisson(noiseless).astype(np.float64)
    return noisy, noiseless, capture_fraction


def export_ground_truth_phase(
    atoms, plan: SamplingPlan, slice_thicknesses, device: str = "cpu"
) -> np.ndarray:
    """Per-slice transmission phase phi_j = sigma * V_j on the reconstruction
    grid (dx = extent/512 = lambda/(det_npix * da)).

    slice_thicknesses: scalar or tuple summing to the cell height; slices
    follow abtem's entrance convention (atoms in [z_j, z_j + dz_j] are
    projected onto the slice, transmitted at its entrance).
    """
    potential = Potential(
        atoms,
        gpts=plan.gpts,
        slice_thickness=slice_thicknesses,
        projection="infinite",
        device=device,
    )
    arr = potential.build(lazy=False).array
    arr = np.asarray(arr)
    phase = energy2sigma(plan.kv * 1e3) * arr
    # resample simulation grid -> reconstruction grid (exact 2x block mean
    # for gpts=1024 -> 512)
    factor = plan.gpts // plan.crop_px
    nz = phase.shape[0]
    n = plan.crop_px
    phase = phase.reshape(nz, n, factor, n, factor).mean(axis=(-3, -1))
    return phase.astype(np.float32)


def make_recon_probe(plan: SamplingPlan, npix: int | None = None) -> np.ndarray:
    """Complex probe on the PtyRAD grid: npix x npix (default det_npix) at
    recon_dx_A, centered, real-space intensity normalized to 1.

    Rebuilt analytically on the window grid.  (A probe k-sampled from the
    simulation grid was tested and measured slightly WORSE in the
    forward-consistency check - the rebuilt probe's softer rim partially
    mimics the detector's window-averaged binning - so the simple rebuild is
    kept.)
    """
    n = npix or plan.det_npix
    extent = n * plan.recon_dx_A
    probe = Probe(
        energy=plan.kv * 1e3,
        semiangle_cutoff=plan.semiangle_mrad,
        extent=extent,
        gpts=n,
        aberrations=plan.aberrations,
        device="cpu",
    )
    waves = probe.build(scan=(extent / 2, extent / 2), lazy=False)
    arr = np.asarray(waves.array).squeeze()
    arr = arr / np.sqrt((np.abs(arr) ** 2).sum())
    return arr.astype(np.complex64)

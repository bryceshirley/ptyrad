"""Single-position CBED pre-scan check.

Simulates ONE probe position at the cell center, applies the crop+bin
detector recipe, saves a PNG, and asserts the BF-disk radius matches
semiangle/da within 1 px and that the pattern is centered. Run this before
committing GPU-hours to a full scan:

    uv run generate-data cbed [smoke|a] [--device cpu|gpu]
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from generate_data.datasets import OUT_DIR
from generate_data.sampling import PRODUCTION_PLAN, SMOKE_PLAN
from generate_data.simulate import crop_and_bin, make_probe, simulate_scan
from generate_data.structures import build_sample_a_diluted, build_smoke_sample


def measure_bf_disk(pattern: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Radius (px, from thresholded area) and center of mass of the BF disk."""
    n = pattern.shape[-1]
    c = n // 2
    plateau = pattern[c - 2 : c + 3, c - 2 : c + 3].mean()
    mask = pattern > 0.5 * plateau
    radius = float(np.sqrt(mask.sum() / np.pi))
    iy, ix = np.nonzero(mask)
    w = pattern[iy, ix]
    com = (float((iy * w).sum() / w.sum()), float((ix * w).sum() / w.sum()))
    return radius, com


def run_check(sample: str = "smoke", device: str = "cpu") -> None:
    OUT_DIR.mkdir(exist_ok=True)
    if sample == "smoke":
        info, plan = build_smoke_sample(), SMOKE_PLAN
    else:
        info, plan = build_sample_a_diluted(), PRODUCTION_PLAN
    assert abs(info.lateral_extent_A - plan.extent_A) < 1e-6

    center = (plan.extent_A / 2, plan.extent_A / 2)
    print(
        f"sample={sample} device={device} L={plan.extent_A:.2f} A, "
        f"da={plan.da_mrad:.4f} mrad, predicted BF radius {plan.bf_radius_px:.2f} px"
    )

    t0 = time.time()
    meas = simulate_scan(info.atoms, plan, scan=[center], device=device).compute()
    print(f"simulated in {time.time() - t0:.1f} s; raw shape {meas.array.shape}")

    binned = crop_and_bin(meas, plan)
    pattern = binned.reshape(plan.det_npix, plan.det_npix)
    total = pattern.sum()
    radius, com = measure_bf_disk(pattern)
    print(f"captured flux fraction: {total:.4f}")
    print(f"measured BF radius {radius:.2f} px (target {plan.bf_radius_px:.2f} +- 1)")
    print(f"disk COM {com} (target {(plan.det_npix / 2,) * 2} +- 1)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pattern, cmap="magma")
    axes[0].set_title(f"{sample}: CBED, linear")
    axes[1].imshow(np.log10(pattern + pattern.max() * 1e-7), cmap="magma")
    axes[1].set_title("log10")
    png = OUT_DIR / f"single_cbed_{sample}.png"
    fig.savefig(png, dpi=150)
    print(f"saved {png}")

    # BF-disk radius is measured on the VACUUM (probe-only) pattern: through a
    # thick crystal, dynamical scattering destroys the flat-disk plateau the
    # threshold recipe needs, but the calibration (da) is a vacuum property.
    probe = make_probe(plan, device=device)
    vac = probe.build(scan=[center], lazy=False).diffraction_patterns(
        max_angle="cutoff"
    )
    vac_pattern = crop_and_bin(vac, plan).reshape(plan.det_npix, plan.det_npix)
    vac_radius, vac_com = measure_bf_disk(vac_pattern)
    print(
        f"vacuum BF radius {vac_radius:.2f} px (target {plan.bf_radius_px:.2f} +- 1), "
        f"COM {vac_com}"
    )

    assert abs(vac_radius - plan.bf_radius_px) <= 1.0, "BF radius out of tolerance"
    assert abs(com[0] - plan.det_npix / 2) <= 1.0, "pattern not centered (y)"
    assert abs(com[1] - plan.det_npix / 2) <= 1.0, "pattern not centered (x)"
    assert total >= 0.9, "large flux loss at the crop"
    print("single-CBED pre-scan check PASSED")

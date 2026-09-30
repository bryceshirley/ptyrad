"""Production dataset pipeline: simulate a scan, package to the PtyRAD h5."""

import time
from pathlib import Path

import numpy as np

from generate_data.h5io import write_dataset
from generate_data.sampling import SamplingPlan
from generate_data.simulate import (
    centered_scan_axes,
    crop_and_bin,
    export_ground_truth_phase,
    make_recon_probe,
    scale_and_poisson,
    simulate_scan,
)
from generate_data.structures import StructureInfo


def produce_dataset(
    info: StructureInfo,
    plan: SamplingPlan,
    out_path: Path,
    n_slow: int,
    n_fast: int,
    step_A: float,
    dose: float = 1e4,
    sim_slice_A: float = 1.0,
    device: str = "gpu",
    seed: int = 0,
    max_batch: int | str = 32,
    rows_per_block: int = 8,
) -> dict:
    """Run the full pipeline and write the PtyRAD-contract h5.  Returns attrs.

    The scan is split into slow-axis (y) blocks computed sequentially - abtem's
    full-scan dask graph over-allocates GPU memory at 4096 positions x 1024^2
    (OOM on an 80 GB A100 even at max_batch 32); block-wise compute with a pool
    flush between blocks keeps the footprint bounded.
    """
    xs, ys = centered_scan_axes(plan, n_slow, n_fast, step_A)
    t0 = time.time()
    blocks = []
    for y0 in range(0, n_slow, rows_per_block):
        rows = ys[y0 : y0 + rows_per_block]
        positions = [(x, y) for x in xs for y in rows]  # abtem (x, y) order
        meas = simulate_scan(
            info.atoms,
            plan,
            positions,
            slice_thickness_A=sim_slice_A,
            device=device,
            max_batch=max_batch,
        ).compute()
        # positions axis is flat in the order given: index = ix * n_rows + iy
        binned = crop_and_bin(meas, plan)  # (n_fast * n_rows, kx, ky)
        binned = binned.reshape(n_fast, len(rows), plan.det_npix, plan.det_npix)
        blocks.append(binned.transpose(1, 0, 3, 2))  # -> (n_rows, n_fast, ky, kx)
        if device == "gpu":
            import cupy

            cupy.get_default_memory_pool().free_all_blocks()
        print(
            f"  rows {y0}..{y0 + len(rows) - 1} done ({time.time() - t0:.0f} s elapsed)"
        )
    t_sim = time.time() - t0
    print(f"scan simulated in {t_sim:.1f} s")

    arr = np.concatenate(blocks, axis=0).reshape(
        n_slow * n_fast, plan.det_npix, plan.det_npix
    )
    noisy, noiseless, capture = scale_and_poisson(arr, dose, seed=seed)
    print(f"capture fraction {capture:.4f}")

    gt = export_ground_truth_phase(
        info.atoms, plan, info.gt_slice_thicknesses, device="cpu"
    )
    gt = gt.transpose(0, 2, 1)  # (slice, x, y) -> (slice, y, x)
    probe = make_recon_probe(plan).T

    xs, ys = centered_scan_axes(plan, n_slow, n_fast, step_A)
    pos = np.array([(y, x) for y in ys for x in xs], dtype=np.float64)

    attrs = {
        "kv": plan.kv,
        "semiangle_mrad": plan.semiangle_mrad,
        "da_mrad": plan.da_mrad,
        "step_A": step_A,
        "dz_A": info.recon_dz_A,
        "thickness_A": info.thickness_A,
        "electrons_per_pattern": dose,
        "dx_A": plan.recon_dx_A,
        "capture_fraction": capture,
        "sim_slice_A": sim_slice_A,
        "t_sim_s": t_sim,
    }
    write_dataset(
        out_path,
        noisy,
        noiseless,
        gt,
        probe,
        attrs,
        extra={"scan_pos_yx_A": pos},
    )
    print(f"wrote {out_path}")
    return attrs

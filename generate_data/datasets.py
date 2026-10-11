"""Named dataset recipes: every dataset this repo can produce, in one registry.

Each recipe pairs a structure builder with a sampling plan and scan
parameters and writes a PtyRAD-contract h5 to out/<name>.h5 (see h5io.py
for the contract). Produce one with:

    uv run generate-data produce <name> [--device cpu|gpu]
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from generate_data.produce import produce_dataset
from generate_data.sampling import (
    PRODUCTION_PLAN,
    SMOKE_PLAN,
    T256_PLAN,
    TWIST_PLAN,
    SamplingPlan,
)
from generate_data.structures import (
    build_multi_element_stack,
    build_plane_stack,
    build_sample_a_diluted,
    build_smoke_sample,
    build_twisted_stack,
)

# repo-root/demo/data/generated — gitignored, next to ptyrad's other demo data
OUT_DIR = Path(__file__).resolve().parents[2] / "demo" / "data" / "generated"


@dataclass(frozen=True)
class Recipe:
    """One producible dataset: structure + plan + scan geometry + seed."""

    builder: Callable
    plan: SamplingPlan
    n_slow: int
    n_fast: int
    step_A: float
    seed: int
    dose: float = 1e4
    description: str = ""
    builder_kwargs: dict = field(default_factory=dict)


RECIPES: dict[str, Recipe] = {
    "sample_a": Recipe(
        builder=build_sample_a_diluted,
        plan=PRODUCTION_PLAN,
        n_slow=64,
        n_fast=64,
        step_A=0.85,
        seed=7,
        description="Si(001) 92x92x347 A with three Ge delta-layers (60/170/290 A)",
    ),
    "multi5": Recipe(
        builder=build_multi_element_stack,
        plan=PRODUCTION_PLAN,
        n_slow=64,
        n_fast=64,
        step_A=0.85,
        seed=11,
        description="five-material stack: C / Si / SrTiO3 / Ge / Au slabs",
    ),
    "planes128": Recipe(
        builder=build_plane_stack,
        plan=PRODUCTION_PLAN,
        n_slow=64,
        n_fast=64,
        step_A=0.85,
        seed=23,
        description="128-slice stack, 16 single atomic planes (13 Si + 3 Ge), 24 A gaps",
    ),
    "dense128b": Recipe(
        builder=build_plane_stack,
        builder_kwargs=dict(
            plane_every=4, first_plane_slice=2, ge_plane_indices=(8, 16, 24), plane_layers=4
        ),
        plan=PRODUCTION_PLAN,
        n_slow=64,
        n_fast=64,
        step_A=0.85,
        seed=31,
        description="32 x 1-uc Si slabs every 4th slice (12 A pitch), 3 Ge markers",
    ),
    "dense128t": Recipe(
        builder=build_twisted_stack,
        plan=TWIST_PLAN,
        n_slow=64,
        n_fast=64,
        step_A=0.85,
        seed=47,
        description="dense128b skeleton with a 2.5 deg/slab twist, structured probe",
    ),
    "dense256t": Recipe(
        builder=build_twisted_stack,
        builder_kwargs=dict(n_slices=256, ge_plane_indices=(16, 32, 48), ramp_slabs=(21, 42)),
        plan=T256_PLAN,
        n_slow=32,
        n_fast=32,
        step_A=1.7,
        seed=53,
        description="256-slice (768 A) twisted stack, 256^2 detector, 32x32 scan",
    ),
}

# datasets with bespoke pipelines, handled explicitly in produce_named()
SPECIAL = {
    "smoke_2slab": "2-slab Si/vacuum/Ge smoke test, 16x16 scan, 128^2 detector",
    "tbl128": "tBL_WSe2 reconstruction-derived phantom tiled to 128 slices (256 A)",
}


def list_datasets() -> dict[str, str]:
    out = {name: r.description for name, r in RECIPES.items()}
    out.update(SPECIAL)
    return out


def produce_named(name: str, device: str = "gpu") -> dict:
    """Produce dataset `name` into out/<name>.h5 and return its h5 attrs."""
    OUT_DIR.mkdir(exist_ok=True)
    if name == "smoke_2slab":
        return _produce_smoke(device)
    if name == "tbl128":
        from generate_data.simulate_from_recon import produce

        return produce(
            str(OUT_DIR / "tbl128.h5"),
            n_slices=128,
            crop_px=384,
            det_npix=128,
            n_slow=64,
            n_fast=64,
            step_px=3.0,
            dose=1e4,
            device=device,
            seed=71,
        )
    r = RECIPES[name]
    info = r.builder(**r.builder_kwargs)
    print(f"{name}: {len(info.atoms)} atoms, thickness {info.thickness_A:.1f} A")
    return produce_dataset(
        info,
        r.plan,
        OUT_DIR / f"{name}.h5",
        n_slow=r.n_slow,
        n_fast=r.n_fast,
        step_A=r.step_A,
        dose=r.dose,
        device=device,
        seed=r.seed,
    )


def _produce_smoke(device: str) -> dict:
    """Smoke dataset keeps its bespoke path: it also stores scan positions and
    a fine-sliced (2 A) ground truth so check_ptyrad_forward.py can separate
    convention errors from coarse-slicing error."""
    from generate_data.h5io import write_dataset
    from generate_data.simulate import (
        centered_scan_axes,
        crop_and_bin,
        export_ground_truth_phase,
        make_centered_scan,
        make_recon_probe,
        scale_and_poisson,
        simulate_scan,
    )

    n_slow = n_fast = 16
    step_A, dose = 1.2, 1e4
    info, plan = build_smoke_sample(), SMOKE_PLAN

    scan = make_centered_scan(plan, n_slow, n_fast, step_A)
    t0 = time.time()
    meas = simulate_scan(info.atoms, plan, scan, device=device).compute()
    print(f"scan simulated in {time.time() - t0:.1f} s; shape {meas.array.shape}")
    labels = [a.label for a in meas.axes_metadata]
    assert labels == ["x", "y", "kx", "ky"], f"unexpected axis order {labels}"

    binned = crop_and_bin(meas, plan)  # (x, y, kx, ky)
    arr = binned.transpose(1, 0, 3, 2)  # PtyRAD raster: slow=y outer, (ky, kx)
    arr = arr.reshape(n_slow * n_fast, plan.det_npix, plan.det_npix)
    noisy, noiseless, capture = scale_and_poisson(arr, dose, seed=42)
    print(f"capture fraction {capture:.4f}")

    gt = export_ground_truth_phase(
        info.atoms, plan, info.gt_slice_thicknesses, device="cpu"
    ).transpose(0, 2, 1)
    dz_fine = 2.0
    gt_fine = export_ground_truth_phase(info.atoms, plan, dz_fine, device="cpu")
    gt_fine = gt_fine.transpose(0, 2, 1)
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
    }
    path = OUT_DIR / "smoke_2slab.h5"
    write_dataset(
        path,
        noisy,
        noiseless,
        gt,
        probe,
        attrs,
        extra={
            "scan_pos_yx_A": pos,
            "gt_slice_thicknesses_A": np.array(info.gt_slice_thicknesses),
            "gt_phase_fine": gt_fine,
            "dz_fine_A": np.array(dz_fine),
        },
    )
    print(f"wrote {path}")
    return attrs


def export_gate(
    name: str,
    slab_cells: int = 2,
    marker_cells: int = 2,
    si_depths: tuple[float, ...] = (),
) -> Path:
    """Export a GT-only gate file (gt_phase + probe + attrs, no scan) for
    cheap design iteration on candidate sample_a variants."""
    import h5py

    from generate_data.simulate import export_ground_truth_phase, make_recon_probe

    OUT_DIR.mkdir(exist_ok=True)
    plan = PRODUCTION_PLAN
    info = build_sample_a_diluted(
        si_slab_depths_A=si_depths, slab_cells=slab_cells, marker_cells=marker_cells
    )
    print(f"{name}: {len(info.atoms)} atoms, slabs {info.slab_z_ranges}")
    gt = export_ground_truth_phase(
        info.atoms, plan, info.gt_slice_thicknesses, device="cpu"
    ).transpose(0, 2, 1)
    probe = make_recon_probe(plan).T
    path = OUT_DIR / f"gate_{name}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("gt_phase", data=gt)
        f.create_dataset("probe", data=probe)
        f.attrs.update(
            {
                "kv": plan.kv,
                "dz_A": info.recon_dz_A,
                "dx_A": plan.recon_dx_A,
                "electrons_per_pattern": 1e4,
            }
        )
    print(f"wrote {path}")
    return path

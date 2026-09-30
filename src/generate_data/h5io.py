"""HDF5 writer obeying the PtyRAD data contract."""

from pathlib import Path

import h5py
import numpy as np

REQUIRED_ATTRS = (
    "kv",
    "semiangle_mrad",
    "da_mrad",
    "step_A",
    "dz_A",
    "thickness_A",
    "electrons_per_pattern",
    "dx_A",
)


def write_dataset(
    path,
    dp: np.ndarray,
    dp_noiseless: np.ndarray,
    gt_phase: np.ndarray,
    probe: np.ndarray,
    attrs: dict,
    extra: dict | None = None,
):
    """Write the simulation products.

    dp / dp_noiseless: (N_scans, ky, kx), stored float32; flat scan order is
    PtyRAD's raster (slow=y outer, fast=x inner).
    gt_phase: (Nlayer, Ny, Nx) float32 per-slice phase on the recon grid.
    probe: (Ny, Nx) complex64 on the recon grid.
    """
    missing = [k for k in REQUIRED_ATTRS if k not in attrs]
    if missing:
        raise ValueError(f"missing required attrs: {missing}")
    if dp.ndim != 3 or dp.shape != dp_noiseless.shape:
        raise ValueError(f"dp must be 3D and match dp_noiseless, got {dp.shape}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("dp", data=dp.astype(np.float32))
        f.create_dataset("dp_noiseless", data=dp_noiseless.astype(np.float32))
        f.create_dataset("gt_phase", data=gt_phase.astype(np.float32))
        f.create_dataset("probe", data=probe.astype(np.complex64))
        for key, val in attrs.items():
            f.attrs[key] = val
        if extra:
            for key, val in extra.items():
                f.create_dataset(key, data=val)

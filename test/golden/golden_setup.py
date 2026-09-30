"""
Deterministic tiny PtychoAD setup shared by the golden generator and the
bit-for-bit default-behaviour regression test.

Everything here must stay byte-stable: fixed seeds, fixed shapes, float32.
"""

import numpy as np
import torch

from ptyrad.models import PtychoAD
from ptyrad.utils import near_field_evolution

# Small but non-trivial: batched positions, 2 probe modes, 2 object modes,
# 3 slices, object canvas larger than the probe so crop positions differ.
N_POS = 4
NPIX = 32
CANVAS = 48
NZ = 3
OMODE = 2
PMODE = 2
DX = 0.15  # Ang
DZ = 2.0  # Ang
LAMBD = 0.025  # Ang (~200 kV)


def golden_indices():
    return torch.arange(N_POS)


def build_golden_model(tilt=False, opt_dz=False, model_params_extra=None, device="cpu", nz=NZ):
    rng = np.random.default_rng(1234)

    obj = (
        0.9
        + 0.1 * rng.random((OMODE, nz, CANVAS, CANVAS))
        + 1j * rng.random((OMODE, nz, CANVAS, CANVAS))
    ).astype(np.complex64)
    # store as amp*exp(i phase) directly: PtychoAD takes abs/angle itself
    probe = (
        rng.random((PMODE, NPIX, NPIX)) - 0.5 + 1j * (rng.random((PMODE, NPIX, NPIX)) - 0.5)
    ).astype(np.complex64)
    measurements = rng.random((N_POS, NPIX, NPIX)).astype(np.float32)
    crop_pos = np.stack(
        [rng.integers(0, CANVAS - NPIX, N_POS), rng.integers(0, CANVAS - NPIX, N_POS)], axis=1
    ).astype(np.int32)
    obj_tilts = (
        np.array([[1.5, -2.0]], dtype=np.float32) if tilt else np.zeros((1, 2), dtype=np.float32)
    )

    H = near_field_evolution((NPIX, NPIX), DX, DZ, LAMBD).astype(np.complex64)

    init_variables = {
        "obj": obj,
        "obj_tilts": obj_tilts,
        "slice_thickness": DZ,
        "probe": probe,
        "probe_pos_shifts": np.zeros((N_POS, 2), dtype=np.float32),
        "omode_occu": (np.ones(OMODE) / OMODE).astype(np.float32),
        "H": H,
        "measurements": measurements,
        "N_scan_slow": 2,
        "N_scan_fast": 2,
        "crop_pos": crop_pos,
        "dx": DX,
        "dk": 1.0 / (NPIX * DX),
        "lambd": LAMBD,
        "random_seed": 42,
        "length_unit": "Ang",
        "scan_affine": None,
    }

    model_params = {
        "detector_blur_std": None,
        "obj_preblur_std": None,
        "update_params": {
            "obja": {"lr": 1e-3, "start_iter": 1},
            "objp": {"lr": 1e-3, "start_iter": 1},
            "obj_tilts": {"lr": 1e-4 if tilt else 0, "start_iter": 1 if tilt else None},
            "slice_thickness": {"lr": 1e-4 if opt_dz else 0, "start_iter": 1 if opt_dz else None},
            "probe": {"lr": 1e-4, "start_iter": 1},
            "probe_pos_shifts": {"lr": 0, "start_iter": None},
        },
        "optimizer_params": {"name": "Adam", "configs": {}, "load_state": None},
    }
    if model_params_extra:
        model_params.update(model_params_extra)

    return PtychoAD(init_variables, model_params, device=device, verbose=False)

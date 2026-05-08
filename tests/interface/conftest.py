import os
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture
def cli_env():
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SRC_ROOT) if not existing else f"{SRC_ROOT}{os.pathsep}{existing}"
    )
    return env


@pytest.fixture
def minimal_params_dict(tmp_path):
    meas_path = tmp_path / "synthetic_meas.raw"
    meas_path.write_bytes(b"placeholder")
    return {
        "init_params": {
            "probe_kv": 80,
            "probe_conv_angle": 24.9,
            "meas_Npix": 8,
            "pos_N_scan_slow": 2,
            "pos_N_scan_fast": 2,
            "pos_scan_step_size": 0.5,
            "obj_Nlayer": 1,
            "obj_slice_thickness": 2.0,
            "meas_calibration": {"mode": "dx", "value": 0.2},
            "meas_params": {
                "path": str(meas_path),
                "key": None,
                "shape": [4, 8, 8],
                "gap": 0,
            },
        },
        "recon_params": {
            "NITER": 1,
            "SAVE_ITERS": 1,
            "output_dir": str(tmp_path / "output"),
        },
    }


@pytest.fixture
def synthetic_array():
    return np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

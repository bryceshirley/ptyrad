"""
Generate golden outputs of the default forward path (solver_type='multislice',
propagator_kernel='angular_spectrum', splitting='lie_trotter').

Run this ONLY from a commit whose defaults are known-good; the saved arrays are
compared bit-for-bit by test/test_chin_splitting.py::test_defaults_bit_for_bit.

Usage: PYTHONPATH=src python test/golden/gen_golden_defaults.py
"""

import os

import numpy as np
import torch

from test.golden.golden_setup import build_golden_model, golden_indices


def main():
    out_path = os.path.join(os.path.dirname(__file__), "golden_defaults.npz")
    arrays = {}

    for tag, kwargs in (
        ("plain", {}),
        ("tilt", {"tilt": True}),
        ("tilt_dz", {"tilt": True, "opt_dz": True}),
        ("dz", {"opt_dz": True}),
    ):
        model = build_golden_model(**kwargs)
        idx = golden_indices()
        with torch.no_grad():
            arrays[f"H_{tag}"] = model.get_propagators(idx).cpu().numpy()
            arrays[f"dp_{tag}"] = model(idx).cpu().numpy()

    np.savez(out_path, **arrays)
    print(f"Saved {sorted(arrays)} to {out_path}")


if __name__ == "__main__":
    main()

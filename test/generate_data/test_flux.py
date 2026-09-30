"""Flux sanity on the production file.

- Crop-loss: mean captured fraction of incident flux after the ~70 mrad crop
  must be >= 97% of the incident (measured: 0.9917; a single on-column probe
  position can dip to ~0.95, so the gate is on the ensemble mean).
- Poisson sanity: on the noisy ensemble, residual variance ~ mean intensity.
"""

import itertools

import h5py
import numpy as np
import pytest

from generate_data.datasets import OUT_DIR as OUT

FILE = OUT / "si_ge_markers.h5"

pytestmark = pytest.mark.skipif(
    not FILE.exists(), reason="production file not simulated yet"
)


def test_capture_fraction():
    with h5py.File(FILE, "r") as f:
        capture = float(f.attrs["capture_fraction"])
    assert capture >= 0.97, f"capture fraction {capture:.4f} - crop misconfigured?"


def test_poisson_variance_matches_mean():
    with h5py.File(FILE, "r") as f:
        noisy = f["dp"][::16].astype(np.float64)
        clean = f["dp_noiseless"][::16].astype(np.float64)
    resid2 = (noisy - clean) ** 2
    # global Fano factor: sum of squared residuals / sum of means -> 1
    fano = resid2.sum() / clean.sum()
    assert abs(fano - 1.0) < 0.05, f"Fano factor {fano:.3f}"
    # and per-intensity-decade check
    bins = np.array([1.0, 10.0, 100.0, 1000.0])
    for lo, hi in itertools.pairwise(bins):
        mask = (clean >= lo) & (clean < hi)
        if mask.sum() > 1000:
            f_bin = resid2[mask].sum() / clean[mask].sum()
            assert abs(f_bin - 1.0) < 0.1, f"Fano {f_bin:.3f} in bin [{lo},{hi})"

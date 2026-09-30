"""The written HDF5 files obey the PtyRAD data contract.

Runs on every simulated file present in out/ (skips if none exist yet).
"""


import h5py
import numpy as np
import pytest

from generate_data.datasets import OUT_DIR as OUT
from generate_data.h5io import REQUIRED_ATTRS

CANDIDATES = [
    "smoke_2slab.h5",
    "si_ge_markers.h5",
    "multi5.h5",
    "planes128.h5",
    "dense128b.h5",
]
FILES = [OUT / name for name in CANDIDATES if (OUT / name).exists()]


@pytest.fixture(params=FILES, ids=lambda p: p.name)
def h5file(request):
    with h5py.File(request.param, "r") as f:
        yield f


@pytest.mark.skipif(not FILES, reason="no simulated files yet")
class TestContract:
    def test_dp_layout(self, h5file):
        dp = h5file["dp"]
        assert dp.ndim == 3
        assert dp.shape[1:] == (128, 128)
        assert dp.dtype == np.float32

    def test_dp_nonnegative_and_dose(self, h5file):
        dp = h5file["dp"][...]
        assert (dp >= 0).all()
        mean_total = dp.sum(axis=(1, 2)).mean()
        target = h5file.attrs["electrons_per_pattern"]
        assert abs(mean_total - target) / target < 0.02

    def test_noiseless_matches(self, h5file):
        assert h5file["dp_noiseless"].shape == h5file["dp"].shape
        dpn = h5file["dp_noiseless"][...]
        assert (dpn >= 0).all()
        mean_total = dpn.sum(axis=(1, 2)).mean()
        target = h5file.attrs["electrons_per_pattern"]
        assert abs(mean_total - target) / target < 0.02

    def test_gt_phase_layout(self, h5file):
        gt = h5file["gt_phase"]
        assert gt.ndim == 3
        assert gt.dtype == np.float32
        assert gt.shape[1] == gt.shape[2]

    def test_probe(self, h5file):
        probe = h5file["probe"]
        assert probe.dtype == np.complex64
        assert probe.shape == (128, 128)

    def test_required_attrs(self, h5file):
        for key in REQUIRED_ATTRS:
            assert key in h5file.attrs, f"missing attr {key}"

    def test_scan_count_consistent(self, h5file):
        n = h5file["dp"].shape[0]
        assert n in (256, 4096)

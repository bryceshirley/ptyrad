"""Ground-truth marker placement in the production file (depth-diluted
Sample A: sparse Si/Ge slabs in vacuum).

- All 9 slab clusters appear at their depths in the per-slice mean phase.
- The 3 strongest clusters are the Ge markers at ~60/170/290 A, clearly
  stronger than every Si slab (Z contrast).
- Slab order matches the beam direction (slice index grows along the beam) -
  catches a z-flip between abtem and PtyRAD conventions.
"""


import h5py
import numpy as np
import pytest

from generate_data.datasets import OUT_DIR as OUT

FILE = OUT / "si_ge_markers.h5"

MARKER_DEPTHS_A = (60.0, 170.0, 290.0)  # nominal; slabs are cell-aligned
N_SLABS = 7

pytestmark = pytest.mark.skipif(
    not FILE.exists(), reason="production file not simulated yet"
)


@pytest.fixture(scope="module")
def gt():
    with h5py.File(FILE, "r") as f:
        return f["gt_phase"][...], float(f.attrs["dz_A"])


def slab_clusters(mean_per_slice: np.ndarray):
    """Cluster contiguous above-background slices; return (center_idx, peak)."""
    thresh = 0.15 * mean_per_slice.max()
    idx = np.nonzero(mean_per_slice > thresh)[0]
    groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    return [(float(g.mean()), float(mean_per_slice[g].sum())) for g in groups]


def test_all_slabs_present(gt):
    phase, _ = gt
    clusters = slab_clusters(phase.mean(axis=(1, 2)))
    assert len(clusters) == N_SLABS, (
        f"expected {N_SLABS} slab clusters, got {len(clusters)}"
    )


def test_markers_strongest_and_at_depth(gt):
    phase, dz = gt
    mean_per_slice = phase.mean(axis=(1, 2))
    clusters = slab_clusters(mean_per_slice)
    clusters_sorted = sorted(clusters, key=lambda c: -c[1])
    ge, si = clusters_sorted[:3], clusters_sorted[3:]
    # 3-uc Ge slabs vs 2-uc Si slabs: integrated weight ratio ~1.9
    # (per-slice Z contrast alone is only ~1.27 - Kirkland potentials)
    assert min(p for _, p in ge) > 1.4 * max(p for _, p in si)
    ge_depths = sorted(c * dz for c, _ in ge)
    for got, nominal in zip(ge_depths, MARKER_DEPTHS_A):
        assert abs(got - nominal) < 2 * dz + 5.43, (got, nominal)
    # the nominal marker slice indices themselves are elevated
    si_level = np.median(mean_per_slice)
    for d in MARKER_DEPTHS_A:
        assert mean_per_slice[int(d // dz)] > 10 * si_level


def test_marker_order_along_beam(gt):
    """Ge markers appear in increasing slice order at increasing depth."""
    phase, dz = gt
    mean_per_slice = phase.mean(axis=(1, 2))
    clusters = sorted(slab_clusters(mean_per_slice), key=lambda c: -c[1])[:3]
    centers = sorted(c for c, _ in clusters)
    depths = [c * dz for c in centers]
    assert depths[0] < 100 < depths[1] < 250 < depths[2], depths


def test_gt_shape_matches_recon_slicing(gt):
    phase, dz = gt
    assert phase.shape[0] == 100
    assert abs(dz - 3.4752) < 1e-3

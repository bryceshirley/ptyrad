"""Scan tiling (utils.tile_scan): index map correctness and the HDF5
round-trip that builds the paper-scale benchmark dataset."""

import h5py
import numpy as np
import pytest

from ptyrad.load import load_array_from_file
from ptyrad.utils.tile_scan import tile_scan_file, tile_scan_indices


def test_indices_identity_tile():
    idx = tile_scan_indices(3, 4, 1, 1)
    assert np.array_equal(idx, np.arange(12))


def test_indices_2x2_periodicity():
    s = 3
    idx = tile_scan_indices(s, s, 2, 2).reshape(2 * s, 2 * s)
    base = np.arange(s * s).reshape(s, s)
    assert np.array_equal(idx, np.tile(base, (2, 2)))


def test_indices_rectangular():
    idx = tile_scan_indices(2, 3, 3, 1).reshape(6, 3)
    base = np.arange(6).reshape(2, 3)
    assert np.array_equal(idx, np.tile(base, (3, 1)))


def test_tile_scan_file_roundtrip(tmp_path):
    s, npix = 4, 8
    rng = np.random.default_rng(0)
    dp = rng.random((s * s, npix, npix)).astype(np.float64)
    src = tmp_path / "src.hdf5"
    with h5py.File(src, "w") as f:
        f.create_dataset("dp", data=dp)
        f.create_dataset("voltage", data=np.array([[300.0]]))
    dst = tmp_path / "tiled.hdf5"
    n_out, so, fo = tile_scan_file(str(src), str(dst), 2, 2, chunk_rows=5)
    assert (n_out, so, fo) == (4 * s * s, 2 * s, 2 * s)
    with h5py.File(dst, "r") as f:
        out = f["dp"][()]
        assert out.dtype == np.float32
        assert f["voltage"][()] == 300.0
    grid = out.reshape(2 * s, 2 * s, npix, npix)
    src_grid = dp.reshape(s, s, npix, npix).astype(np.float32)
    for qy in range(2):
        for qx in range(2):
            quad = grid[qy * s : (qy + 1) * s, qx * s : (qx + 1) * s]
            assert np.array_equal(quad, src_grid)
    # the produced file must load through the stock measurement loader
    meas = load_array_from_file(str(dst), key="dp")
    assert meas.shape == (4 * s * s, npix, npix)


def test_tile_scan_file_nonsquare_requires_dims(tmp_path):
    src = tmp_path / "src.hdf5"
    with h5py.File(src, "w") as f:
        f.create_dataset("dp", data=np.zeros((6, 4, 4)))
    with pytest.raises(ValueError, match="not square"):
        tile_scan_file(str(src), str(tmp_path / "d.hdf5"), 2, 2)
    n_out, so, fo = tile_scan_file(str(src), str(tmp_path / "d2.hdf5"), 2, 2, n_slow=2, n_fast=3)
    assert (n_out, so, fo) == (24, 4, 6)

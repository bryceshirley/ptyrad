import h5py
import numpy as np
import pytest
import scipy.io as sio

from ptyrad.io.handlers import load_array_from_file
from ptyrad.io.hierarchy import load_hdf5


def test_load_npy_preserves_shape_and_dtype(tmp_path, synthetic_array):
    path = tmp_path / "meas.npy"
    np.save(path, synthetic_array)

    loaded = load_array_from_file(str(path))

    np.testing.assert_array_equal(loaded, synthetic_array)
    assert loaded.dtype == synthetic_array.dtype


def test_load_hdf5_by_key_and_selection(tmp_path, synthetic_array):
    path = tmp_path / "meas.hdf5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("entry")
        group.create_dataset("data", data=synthetic_array)

    loaded = load_array_from_file(str(path), key="entry.data")
    selected = load_array_from_file(
        str(path), key="entry.data", selection=[[0, 1], None, [1, 4, 2]]
    )

    np.testing.assert_array_equal(loaded, synthetic_array)
    np.testing.assert_array_equal(selected, synthetic_array[0:1, :, 1:4:2])


def test_load_hdf5_autodiscovers_single_nd_dataset(tmp_path, synthetic_array):
    path = tmp_path / "single.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("the_only_cube", data=synthetic_array)

    loaded = load_array_from_file(str(path), ndims=[3])

    np.testing.assert_array_equal(loaded, synthetic_array)


def test_load_hdf5_rejects_selection_without_explicit_key(tmp_path, synthetic_array):
    path = tmp_path / "full-tree.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", data=synthetic_array)

    with pytest.raises(ValueError, match="key=None"):
        load_hdf5(str(path), key=None, selection=[None, None, None])


def test_load_mat_by_key(tmp_path, synthetic_array):
    path = tmp_path / "meas.mat"
    sio.savemat(path, {"dp": synthetic_array})

    loaded = load_array_from_file(str(path), key="dp")

    np.testing.assert_array_equal(loaded, synthetic_array)


def test_load_raw_with_offset_and_gap(tmp_path):
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / "meas.raw"
    gap = 2
    offset = 4

    with path.open("wb") as f:
        f.write(b"ABCD")
        for frame in data:
            frame.tofile(f)
            f.write(b"xy")

    loaded = load_array_from_file(
        str(path), shape=data.shape, offset=offset, gap=gap
    )

    np.testing.assert_array_equal(loaded, data)


def test_load_array_from_file_reports_unsupported_extension(tmp_path):
    path = tmp_path / "meas.txt"
    path.write_text("not an array", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_array_from_file(str(path))

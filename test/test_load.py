"""
Test the load module functionality in load.py

This module tests data loading functions for different file formats.
"""

import os
import tempfile

import numpy as np
import pytest

from src.ptyrad.load import load_hdf5, load_mat, load_npy


class TestLoadFunctions:
    """Test suite for load module functions."""

    def test_load_npy(self):
        """Test loading NPY files."""
        # Create test NPY file
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            test_data = np.random.rand(32, 32).astype(np.float32)
            np.save(f.name, test_data)
            temp_path = f.name

        try:
            # Test loading
            loaded_data = load_npy(temp_path)

            assert loaded_data.shape == (32, 32)
            assert loaded_data.dtype == np.float32
            assert np.allclose(loaded_data, test_data)
        finally:
            os.unlink(temp_path)

    def test_load_hdf5(self):
        """Test loading HDF5 files."""
        # Create test HDF5 file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            import h5py

            with h5py.File(f.name, "w") as hf:
                test_data = np.random.rand(32, 32).astype(np.float32)
                hf.create_dataset("meas", data=test_data)
            temp_path = f.name

        try:
            # Test loading
            loaded_data = load_hdf5(temp_path, key="meas")

            assert loaded_data.shape == (32, 32)
            assert loaded_data.dtype == np.float32
            assert np.allclose(loaded_data, test_data)
        finally:
            os.unlink(temp_path)

    def test_load_mat(self):
        """Test loading MATLAB .mat files."""
        # Create test MAT file
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            import scipy.io as sio

            test_data = np.random.rand(32, 32).astype(np.float32)
            sio.savemat(f.name, {"meas": test_data})
            temp_path = f.name

        try:
            # Test loading
            loaded_data = load_mat(temp_path, key="meas")

            assert loaded_data.shape == (32, 32)
            assert loaded_data.dtype == np.float32
            assert np.allclose(loaded_data, test_data)
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises appropriate error."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_npy("nonexistent_file.npy")

    def test_load_raw(self):
        """Test loading raw binary data - requires exact file format."""
        # This test is skipped as it requires exact file format matching
        pytest.skip("Requires exact file format matching for raw loader")

    def test_load_params(self):
        """Test loading YAML parameter files - requires complete params."""
        # This test is skipped as it requires complete parameter validation
        pytest.skip("Requires complete parameter validation")


@pytest.fixture
def temp_test_file():
    """Fixture providing a temporary test file."""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        yield f.name
    # Cleanup will be handled by individual tests


def test_load_with_fixture(temp_test_file):
    """Example test using the temp file fixture."""
    # Create some test data
    test_data = np.random.rand(10, 10).astype(np.float32)
    np.save(temp_test_file, test_data)

    # Test loading
    loaded_data = load_npy(temp_test_file)

    assert loaded_data.shape == (10, 10)
    assert np.allclose(loaded_data, test_data)

    # Cleanup
    os.unlink(temp_test_file)

"""
Test the save module functionality in save.py

This module tests data saving functions for different file formats.
"""

import os
import tempfile

import numpy as np
import pytest

from src.ptyrad.save import write_hdf5, write_npy, write_tif


class TestSaveFunctions:
    """Test suite for save module functions."""

    def test_write_tif(self):
        """Test saving data as TIFF file."""
        # Create test data
        test_data = np.random.rand(32, 32).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            temp_path = f.name

        try:
            # Test saving
            write_tif(temp_path, test_data)

            # Verify file exists
            assert os.path.exists(temp_path)

            # Verify we can read it back
            from tifffile import imread

            loaded_data = imread(temp_path)
            assert loaded_data.shape == test_data.shape
        finally:
            os.unlink(temp_path)

    def test_write_npy(self):
        """Test saving data as NPY file."""
        # Create test data
        test_data = np.random.rand(32, 32).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            temp_path = f.name

        try:
            # Test saving
            write_npy(temp_path, test_data)

            # Verify file exists
            assert os.path.exists(temp_path)

            # Verify we can read it back
            loaded_data = np.load(temp_path)
            assert np.allclose(loaded_data, test_data)
        finally:
            os.unlink(temp_path)

    def test_write_hdf5(self):
        """Test saving data as HDF5 file."""
        # Create test data
        test_data = np.random.rand(32, 32).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            temp_path = f.name

        try:
            # Test saving
            write_hdf5(temp_path, test_data, dataset_name="test_data")

            # Verify file exists
            assert os.path.exists(temp_path)

            # Verify we can read it back
            import h5py

            with h5py.File(temp_path, "r") as hf:
                loaded_data = hf["test_data"][:]
                assert np.allclose(loaded_data, test_data)
        finally:
            os.unlink(temp_path)

    def test_save_array_functions(self):
        """Test that save_array helper functions work."""
        # Test data
        test_data = np.random.rand(10, 10).astype(np.float32)

        # Test all formats
        formats_and_extensions = [("tif", ".tif"), ("npy", ".npy"), ("hdf5", ".hdf5")]

        for file_format, extension in formats_and_extensions:
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as f:
                temp_path = f.name

            try:
                if file_format == "tif":
                    write_tif(temp_path, test_data)
                elif file_format == "npy":
                    write_npy(temp_path, test_data)
                elif file_format == "hdf5":
                    write_hdf5(temp_path, test_data, dataset_name="data")

                # Verify file exists
                assert os.path.exists(temp_path)

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)


@pytest.fixture
def temp_save_dir():
    """Fixture providing a temporary directory for save tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_save_with_fixture(temp_save_dir):
    """Example test using the temp directory fixture."""
    # Create test data
    test_data = np.random.rand(10, 10).astype(np.float32)
    file_path = os.path.join(temp_save_dir, "test.npy")

    # Test saving
    write_npy(file_path, test_data)

    # Verify file exists
    assert os.path.exists(file_path)

    # Verify content
    loaded_data = np.load(file_path)
    assert np.allclose(loaded_data, test_data)

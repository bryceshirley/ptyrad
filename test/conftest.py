# Pytest configuration and fixtures
"""Configuration and fixtures for ptyrad tests."""

from test.utils import (
    create_minimal_init_variables_2d,
    create_minimal_init_variables_3d,
    generate_minimal_params_2d,
    generate_minimal_params_3d,
    generate_test_diffraction_patterns,
    generate_test_fresnel_propagator,
    generate_test_object,
    generate_test_probe,
    set_random_seed,
)

import pytest
import torch


@pytest.fixture
def sample_data():
    """Example fixture for sample data."""
    return {"test": "data"}


@pytest.fixture
def random_seed():
    """Fixture to set random seed for reproducibility."""
    set_random_seed(42)


@pytest.fixture
def basic_forward_setup():
    """Fixture providing basic forward model setup."""
    obj_patches = generate_test_object(size=(32, 32), batch_size=2)
    probe = generate_test_probe(size=(32, 32), batch_size=2)
    H = generate_test_fresnel_propagator(size=(32, 32), batch_size=2)
    return obj_patches, probe, H


@pytest.fixture
def minimal_params_2d():
    """Fixture providing minimal 2D parameters for testing."""
    return generate_minimal_params_2d()


@pytest.fixture
def minimal_params_3d():
    """Fixture providing minimal 3D parameters for testing."""
    return generate_minimal_params_3d()


@pytest.fixture
def test_probe():
    """Fixture providing a test probe."""
    return generate_test_probe(size=(32, 32), batch_size=2)


@pytest.fixture
def test_object():
    """Fixture providing a test object."""
    return generate_test_object(size=(32, 32), batch_size=2)


@pytest.fixture
def test_diffraction_patterns():
    """Fixture providing test diffraction patterns."""
    return generate_test_diffraction_patterns(num_patterns=16, size=(32, 32))


@pytest.fixture
def test_propagator():
    """Fixture providing a test Fresnel propagator."""
    return generate_test_fresnel_propagator(size=(32, 32), batch_size=2)


@pytest.fixture
def available_devices():
    """Fixture providing list of available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.fixture
def minimal_init_variables_2d():
    """Fixture providing minimal 2D init_variables for model testing."""
    return create_minimal_init_variables_2d()


@pytest.fixture
def minimal_init_variables_3d():
    """Fixture providing minimal 3D init_variables for model testing."""
    return create_minimal_init_variables_3d()

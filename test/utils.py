"""
Test utilities and helper functions for PtyRAD tests.

This module provides helper functions for generating test data,
creating test parameters, and other utilities needed for testing.
"""

import numpy as np
import torch


def generate_test_probe(size=(32, 32), num_modes=1, batch_size=1, requires_grad=False):
    """
    Generate a test probe tensor.

    Args:
        size: Spatial dimensions (Ny, Nx)
        num_modes: Number of probe modes
        batch_size: Batch size
        requires_grad: Whether tensor should track gradients

    Returns:
        Complex tensor of shape (batch_size, num_modes, Ny, Nx)
    """
    probe_real = torch.rand(batch_size, num_modes, size[0], size[1], requires_grad=requires_grad)
    probe_imag = torch.rand(batch_size, num_modes, size[0], size[1], requires_grad=requires_grad)
    return probe_real + 1j * probe_imag


def generate_test_object(
    size=(32, 32), num_slices=1, num_modes=1, batch_size=1, requires_grad=False
):
    """
    Generate a test object tensor.

    Args:
        size: Spatial dimensions (Ny, Nx)
        num_slices: Number of object slices (Nz)
        num_modes: Number of object modes
        batch_size: Batch size
        requires_grad: Whether tensor should track gradients

    Returns:
        Float tensor of shape (batch_size, num_modes, num_slices, Ny, Nx, 2)
        where last dimension represents [amplitude, phase]
    """
    obj = torch.rand(
        batch_size, num_modes, num_slices, size[0], size[1], 2, requires_grad=requires_grad
    )
    # Ensure amplitude is positive
    obj[..., 0] = torch.abs(obj[..., 0])
    return obj


def generate_test_diffraction_patterns(num_patterns=10, size=(32, 32), requires_grad=False):
    """
    Generate test diffraction patterns.

    Args:
        num_patterns: Number of diffraction patterns
        size: Detector dimensions (Ky, Kx)
        requires_grad: Whether tensor should track gradients

    Returns:
        Float tensor of shape (num_patterns, Ky, Kx)
    """
    return torch.rand(num_patterns, size[0], size[1], requires_grad=requires_grad)


def generate_test_fresnel_propagator(size=(32, 32), batch_size=1):
    """
    Generate a test Fresnel propagator.

    Args:
        size: Spatial dimensions (Ky, Kx)
        batch_size: Batch size

    Returns:
        Complex tensor of shape (batch_size, Ky, Kx)
    """
    H_real = torch.ones(batch_size, size[0], size[1])
    H_imag = torch.ones(batch_size, size[0], size[1])
    return H_real + 1j * H_imag


def generate_minimal_params_2d():
    """
    Generate minimal parameters for 2D testing.

    Returns:
        Dictionary with minimal parameters for testing
    """
    return {
        "init_params": {
            "probe_kv": 80,
            "probe_conv_angle": 24.9,
            "meas_Npix": 32,
            "pos_N_scan_slow": 4,
            "pos_N_scan_fast": 4,
            "pos_scan_step_size": 0.4290,
            "probe_pmode_max": 1,
            "obj_Nlayer": 1,
            "obj_slice_thickness": 2,
            "meas_params": generate_test_diffraction_patterns(16, (32, 32)),
        },
        "recon_params": {
            "NITER": 2,
            "SAVE_ITERS": None,
            "BATCH_SIZE": {"size": 4, "grad_accumulation": 1},
        },
    }


def generate_minimal_params_3d():
    """
    Generate minimal parameters for 3D testing.

    Returns:
        Dictionary with minimal parameters for 3D testing
    """
    params = generate_minimal_params_2d()
    params["init_params"]["obj_Nlayer"] = 3
    params["init_params"]["obj_slice_thickness"] = 2
    return params


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_available_devices():
    """
    Get list of available devices for testing.

    Returns:
        List of device strings (e.g., ['cpu'], ['cpu', 'cuda'])
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def create_test_object_patches(batch_size=2, num_modes=1, num_slices=1, size=(32, 32)):
    """
    Create test object patches for forward model testing.

    Args:
        batch_size: Number of samples in batch
        num_modes: Number of object modes
        num_slices: Number of object slices
        size: Spatial dimensions (Ny, Nx)

    Returns:
        Object patches tensor
    """
    return generate_test_object(
        size=size, num_slices=num_slices, num_modes=num_modes, batch_size=batch_size
    )


def create_test_probe(batch_size=2, num_modes=1, size=(32, 32)):
    """
    Create test probe for forward model testing.

    Args:
        batch_size: Number of samples in batch
        num_modes: Number of probe modes
        size: Spatial dimensions (Ny, Nx)

    Returns:
        Probe tensor
    """
    return generate_test_probe(size=size, num_modes=num_modes, batch_size=batch_size)


def create_test_propagator(batch_size=2, size=(32, 32)):
    """
    Create test Fresnel propagator for forward model testing.

    Args:
        batch_size: Number of samples in batch
        size: Spatial dimensions (Ky, Kx)

    Returns:
        Fresnel propagator tensor
    """
    return generate_test_fresnel_propagator(size=size, batch_size=batch_size)


def create_minimal_init_variables_2d():
    """
    Create minimal init_variables for 2D model testing.

    Returns:
        Dictionary with init_variables for testing
    """
    import numpy as np

    # Generate test data
    measurements = generate_test_diffraction_patterns(16, (32, 32)).numpy()

    # Create object (1 slice, 1 mode, 32x32, complex)
    obj_real = np.random.rand(1, 32, 32).astype(np.float32)
    obj_imag = np.random.rand(1, 32, 32).astype(np.float32)
    obj = obj_real + 1j * obj_imag

    # Create probe (1 mode, 32x32, complex)
    probe_real = np.random.rand(1, 32, 32).astype(np.float32)
    probe_imag = np.random.rand(1, 32, 32).astype(np.float32)
    probe = probe_real + 1j * probe_imag

    # Create propagator (32x32, complex)
    H_real = np.ones((32, 32), dtype=np.float32)
    H_imag = np.zeros((32, 32), dtype=np.float32)
    H = H_real + 1j * H_imag

    # Create minimal init_variables
    init_variables = {
        "obj": obj,
        "obj_tilts": np.zeros((1, 3), dtype=np.float32),  # (omode, 3) for tilt angles
        "slice_thickness": np.array([2.0], dtype=np.float32),  # (Nlayer,)
        "probe": probe,
        "probe_pos_shifts": np.zeros((16, 2), dtype=np.float32),  # (N_scans, 2) for x,y shifts
        "omode_occu": np.array([1.0], dtype=np.float32),  # (omode,)
        "H": H,
        "measurements": measurements,
        "N_scan_slow": 4,
        "N_scan_fast": 4,
        "crop_pos": np.array([0, 0, 32, 32], dtype=np.int32),  # [y1, x1, y2, x2]
        "dx": np.array([0.1494], dtype=np.float32),  # pixel size
        "dk": np.array([0.01], dtype=np.float32),  # k-space sampling
        "lambd": np.array([0.0025], dtype=np.float32),  # wavelength
        "random_seed": 42,
        "length_unit": "nm",
        "scan_affine": np.eye(2, dtype=np.float32),  # Identity affine matrix
    }

    return init_variables


def create_minimal_init_variables_3d():
    """
    Create minimal init_variables for 3D model testing.

    Returns:
        Dictionary with init_variables for testing
    """
    init_variables = create_minimal_init_variables_2d()
    # Modify for 3D - create 3 slices
    obj_real = np.random.rand(3, 32, 32).astype(np.float32)
    obj_imag = np.random.rand(3, 32, 32).astype(np.float32)
    init_variables["obj"] = obj_real + 1j * obj_imag
    init_variables["slice_thickness"] = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    return init_variables


def minimal_model_params():
    """
    Create minimal model parameters for testing.

    Returns:
        Dictionary with minimal model parameters
    """
    return {
        "detector_blur_std": None,
        "obj_preblur_std": None,
        "update_params": {
            "obja": {"lr": 0.01, "start_iter": 0, "end_iter": None},
            "objp": {"lr": 0.01, "start_iter": 0, "end_iter": None},
            "obj_tilts": {"lr": 0.001, "start_iter": 0, "end_iter": None},
            "slice_thickness": {"lr": 0.001, "start_iter": 0, "end_iter": None},
            "probe": {"lr": 0.01, "start_iter": 0, "end_iter": None},
            "probe_pos_shifts": {"lr": 0.001, "start_iter": 0, "end_iter": None},
        },
        "optimizer_params": {
            "lr": 0.01,
            "weight_decay": 0,
        },
    }

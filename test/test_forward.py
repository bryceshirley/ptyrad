"""
Test the forward model implementation in forward.py

This module tests the multislice forward model and Born approximation forward model
that compute diffraction patterns from object patches and probes using vectorized operations.
"""

import torch
from ptyrad.forward import (
    multislice_forward_model_vec_all,
    multislice_forward_model_vec_all_born,
)


class TestMultisliceForward:
    """Test suite for multislice forward model."""

    def test_2d_forward_model_basic(self):
        """Test basic 2D multislice forward model with single mode."""
        # Setup: 2 samples, 1 object mode, 1 slice, 32x32 patches
        obj_patches = torch.rand(2, 1, 1, 32, 32, 2)  # (N, omode, Nz, Ny, Nx, 2)
        probe = torch.rand(2, 1, 32, 32) + 1j * torch.rand(2, 1, 32, 32)  # (N, pmode, Ny, Nx)
        H = torch.ones(2, 32, 32) + 1j * torch.ones(2, 32, 32)  # (N, Ky, Kx)

        # Execute
        dp = multislice_forward_model_vec_all(obj_patches, probe, H)

        # Verify
        assert dp.shape == (2, 32, 32), f"Expected shape (2, 32, 32), got {dp.shape}"
        assert dp.dtype == torch.float32, f"Expected float32, got {dp.dtype}"
        assert torch.all(dp >= 0), "Diffraction patterns should be non-negative"
        assert torch.all(torch.isfinite(dp)), "Diffraction patterns should be finite"

    def test_3d_forward_model_multiple_slices(self):
        """Test 3D multislice forward model with multiple slices."""
        # Setup: 2 samples, 1 object mode, 3 slices, 32x32 patches
        obj_patches = torch.rand(2, 1, 3, 32, 32, 2)  # (N, omode, Nz, Ny, Nx, 2)
        probe = torch.rand(2, 1, 32, 32) + 1j * torch.rand(2, 1, 32, 32)
        H = torch.ones(2, 32, 32) + 1j * torch.ones(2, 32, 32)

        # Execute
        dp = multislice_forward_model_vec_all(obj_patches, probe, H)

        # Verify
        assert dp.shape == (2, 32, 32), f"Expected shape (2, 32, 32), got {dp.shape}"
        assert torch.all(dp >= 0), "Diffraction patterns should be non-negative"

    def test_multiple_object_modes(self):
        """Test forward model with multiple object modes."""
        # Setup: 2 samples, 3 object modes, 1 slice, 32x32 patches
        obj_patches = torch.rand(2, 3, 1, 32, 32, 2)  # (N, omode, Nz, Ny, Nx, 2)
        probe = torch.rand(2, 1, 32, 32) + 1j * torch.rand(2, 1, 32, 32)
        H = torch.ones(2, 32, 32) + 1j * torch.ones(2, 32, 32)

        # Execute
        dp = multislice_forward_model_vec_all(obj_patches, probe, H)

        # Verify
        assert dp.shape == (2, 32, 32), f"Expected shape (2, 32, 32), got {dp.shape}"
        assert torch.all(dp >= 0), "Diffraction patterns should be non-negative"

    def test_multiple_probe_modes(self):
        """Test forward model with multiple probe modes."""
        # Setup: 2 samples, 1 object mode, 1 slice, 32x32 patches, 2 probe modes
        obj_patches = torch.rand(2, 1, 1, 32, 32, 2)
        probe = torch.rand(2, 2, 32, 32) + 1j * torch.rand(2, 2, 32, 32)  # 2 probe modes
        H = torch.ones(2, 32, 32) + 1j * torch.ones(2, 32, 32)

        # Execute
        dp = multislice_forward_model_vec_all(obj_patches, probe, H)

        # Verify
        assert dp.shape == (2, 32, 32), f"Expected shape (2, 32, 32), got {dp.shape}"
        assert torch.all(dp >= 0), "Diffraction patterns should be non-negative"

    def test_object_mode_occupancy(self):
        """Test forward model with object mode occupancy."""
        # Setup
        obj_patches = torch.rand(2, 3, 1, 32, 32, 2)  # 3 object modes
        probe = torch.rand(2, 1, 32, 32) + 1j * torch.rand(2, 1, 32, 32)
        H = torch.ones(2, 32, 32) + 1j * torch.ones(2, 32, 32)
        omode_occu = torch.tensor([0.2, 0.3, 0.5])  # Sum to 1.0

        # Execute
        dp = multislice_forward_model_vec_all(obj_patches, probe, H, omode_occu=omode_occu)

        # Verify
        assert dp.shape == (2, 32, 32), f"Expected shape (2, 32, 32), got {dp.shape}"
        assert torch.all(dp >= 0), "Diffraction patterns should be non-negative"

    def test_gradient_computation(self):
        """Test that gradients can be computed through the forward model."""
        # Setup with requires_grad=True
        obj_patches = torch.rand(2, 1, 1, 32, 32, 2, requires_grad=True)
        probe_real = torch.rand(2, 1, 32, 32, requires_grad=True)
        probe_imag = torch.rand(2, 1, 32, 32, requires_grad=True)
        probe = probe_real + 1j * probe_imag
        H = torch.ones(2, 32, 32) + 1j * torch.ones(2, 32, 32)

        # Execute
        dp = multislice_forward_model_vec_all(obj_patches, probe, H)

        # Compute gradients
        loss = dp.sum()
        loss.backward()

        # Verify gradients exist
        assert obj_patches.grad is not None, "Object patches should have gradients"
        assert probe_real.grad is not None, "Probe real part should have gradients"
        assert probe_imag.grad is not None, "Probe imaginary part should have gradients"
        assert torch.all(torch.isfinite(obj_patches.grad)), "Object gradients should be finite"
        assert torch.all(torch.isfinite(probe_real.grad)), "Probe real gradients should be finite"
        assert torch.all(torch.isfinite(probe_imag.grad)), "Probe imag gradients should be finite"


class TestBornForward:
    """Test suite for Born approximation forward model."""

    def test_born_first_order(self):
        """Test Born approximation with first-order (n_max=1)."""
        # Setup: 2 samples, 1 object mode, 3 slices, 32x32 patches
        obj_patches = torch.rand(2, 1, 3, 32, 32, 2)
        probe = torch.rand(2, 1, 32, 32) + 1j * torch.rand(2, 1, 32, 32)
        H = torch.ones(2, 32, 32) + 1j * torch.ones(2, 32, 32)

        # Execute with first-order Born approximation
        dp = multislice_forward_model_vec_all_born(obj_patches, probe, H, n_max=1)

        # Verify
        assert dp.shape == (2, 32, 32), f"Expected shape (2, 32, 32), got {dp.shape}"
        assert torch.all(dp >= 0), "Diffraction patterns should be non-negative"
        assert torch.all(torch.isfinite(dp)), "Diffraction patterns should be finite"

    def test_born_higher_order(self):
        """Test Born approximation with higher-order (n_max=2)."""
        # Setup: 2 samples, 1 object mode, 5 slices, 32x32 patches
        obj_patches = torch.rand(2, 1, 5, 32, 32, 2)
        probe = torch.rand(2, 1, 32, 32) + 1j * torch.rand(2, 1, 32, 32)
        H = torch.ones(2, 32, 32) + 1j * torch.ones(2, 32, 32)

        # Execute with higher-order Born approximation
        dp = multislice_forward_model_vec_all_born(obj_patches, probe, H, n_max=2)

        # Verify
        assert dp.shape == (2, 32, 32), f"Expected shape (2, 32, 32), got {dp.shape}"
        assert torch.all(dp >= 0), "Diffraction patterns should be non-negative"
        assert torch.all(torch.isfinite(dp)), "Diffraction patterns should be finite"

    def test_born_full_multislice(self):
        """Test Born approximation with full multislice (n_max=Nz)."""
        # Setup: 2 samples, 1 object mode, 4 slices, 32x32 patches
        obj_patches = torch.rand(2, 1, 4, 32, 32, 2)
        probe = torch.rand(2, 1, 32, 32) + 1j * torch.rand(2, 1, 32, 32)
        H = torch.ones(2, 32, 32) + 1j * torch.ones(2, 32, 32)

        # Execute with full multislice Born approximation
        dp = multislice_forward_model_vec_all_born(obj_patches, probe, H, n_max=4)

        # Verify
        assert dp.shape == (2, 32, 32), f"Expected shape (2, 32, 32), got {dp.shape}"
        assert torch.all(dp >= 0), "Diffraction patterns should be non-negative"
        assert torch.all(torch.isfinite(dp)), "Diffraction patterns should be finite"

    def test_born_multiple_modes(self):
        """Test Born approximation with multiple object and probe modes."""
        # Setup: 2 samples, 2 object modes, 2 probe modes, 3 slices, 32x32 patches
        obj_patches = torch.rand(2, 2, 3, 32, 32, 2)
        probe = torch.rand(2, 2, 32, 32) + 1j * torch.rand(2, 2, 32, 32)
        H = torch.ones(2, 32, 32) + 1j * torch.ones(2, 32, 32)
        omode_occu = torch.tensor([0.4, 0.6])

        # Execute
        dp = multislice_forward_model_vec_all_born(
            obj_patches, probe, H, omode_occu=omode_occu, n_max=2
        )

        # Verify
        assert dp.shape == (2, 32, 32), f"Expected shape (2, 32, 32), got {dp.shape}"
        assert torch.all(dp >= 0), "Diffraction patterns should be non-negative"

    def test_born_gradient_computation(self):
        """Test that gradients can be computed through the Born forward model."""
        # Setup with requires_grad=True
        obj_patches = torch.rand(2, 1, 3, 32, 32, 2, requires_grad=True)
        probe_real = torch.rand(2, 1, 32, 32, requires_grad=True)
        probe_imag = torch.rand(2, 1, 32, 32, requires_grad=True)
        probe = probe_real + 1j * probe_imag
        H = torch.ones(2, 32, 32) + 1j * torch.ones(2, 32, 32)

        # Execute
        dp = multislice_forward_model_vec_all_born(obj_patches, probe, H, n_max=2)

        # Compute gradients
        loss = dp.sum()
        loss.backward()

        # Verify gradients exist
        assert obj_patches.grad is not None, "Object patches should have gradients"
        assert probe_real.grad is not None, "Probe real part should have gradients"
        assert probe_imag.grad is not None, "Probe imaginary part should have gradients"
        assert torch.all(torch.isfinite(obj_patches.grad)), "Object gradients should be finite"
        assert torch.all(torch.isfinite(probe_real.grad)), "Probe real gradients should be finite"
        assert torch.all(torch.isfinite(probe_imag.grad)), "Probe imag gradients should be finite"

    def test_born_numerical_stability(self):
        """Test numerical stability of Born approximation with small values."""
        # Setup with very small values
        obj_patches = torch.rand(2, 1, 3, 32, 32, 2) * 1e-6
        probe = torch.rand(2, 1, 32, 32) * 1e-6 + 1j * torch.rand(2, 1, 32, 32) * 1e-6
        H = torch.ones(2, 32, 32) * 1e-6 + 1j * torch.ones(2, 32, 32) * 1e-6

        # Execute
        dp = multislice_forward_model_vec_all_born(obj_patches, probe, H, n_max=1)

        # Verify
        assert torch.all(torch.isfinite(dp)), "Should handle small values without NaN/inf"
        assert torch.all(dp >= 0), "Diffraction patterns should still be non-negative"

    def test_born_different_batch_sizes(self):
        """Test Born approximation with different batch sizes."""
        for batch_size in [1, 4, 8]:
            # Setup
            obj_patches = torch.rand(batch_size, 1, 2, 32, 32, 2)
            probe = torch.rand(batch_size, 1, 32, 32) + 1j * torch.rand(batch_size, 1, 32, 32)
            H = torch.ones(batch_size, 32, 32) + 1j * torch.ones(batch_size, 32, 32)

            # Execute
            dp = multislice_forward_model_vec_all_born(obj_patches, probe, H, n_max=1)

            # Verify
            assert dp.shape == (
                batch_size,
                32,
                32,
            ), f"Expected shape ({batch_size}, 32, 32), got {dp.shape}"
            assert torch.all(
                dp >= 0
            ), f"Batch size {batch_size}: Diffraction patterns should be non-negative"

    def test_born_different_patch_sizes(self):
        """Test Born approximation with different patch sizes."""
        for size in [16, 32, 64]:
            # Setup
            obj_patches = torch.rand(2, 1, 2, size, size, 2)
            probe = torch.rand(2, 1, size, size) + 1j * torch.rand(2, 1, size, size)
            H = torch.ones(2, size, size) + 1j * torch.ones(2, size, size)

            # Execute
            dp = multislice_forward_model_vec_all_born(obj_patches, probe, H, n_max=1)

            # Verify
            assert dp.shape == (
                2,
                size,
                size,
            ), f"Expected shape (2, {size}, {size}), got {dp.shape}"
            assert torch.all(dp >= 0), f"Size {size}: Diffraction patterns should be non-negative"

    def test_born_device_compatibility(self):
        """Test Born approximation on different devices."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            # Setup
            obj_patches = torch.rand(2, 1, 2, 32, 32, 2, device=device)
            probe = torch.rand(2, 1, 32, 32, device=device) + 1j * torch.rand(
                2, 1, 32, 32, device=device
            )
            H = torch.ones(2, 32, 32, device=device) + 1j * torch.ones(2, 32, 32, device=device)

            # Execute
            dp = multislice_forward_model_vec_all_born(obj_patches, probe, H, n_max=1)

            # Verify
            assert dp.device.type == device, f"Expected device {device}, got {dp.device}"
            assert dp.shape == (
                2,
                32,
                32,
            ), f"Device {device}: Expected shape (2, 32, 32), got {dp.shape}"
            assert torch.all(
                dp >= 0
            ), f"Device {device}: Diffraction patterns should be non-negative"

    def test_numerical_stability(self):
        """Test numerical stability with small values."""
        # Setup with very small values
        obj_patches = torch.rand(2, 1, 1, 32, 32, 2) * 1e-6
        probe = torch.rand(2, 1, 32, 32) * 1e-6 + 1j * torch.rand(2, 1, 32, 32) * 1e-6
        H = torch.ones(2, 32, 32) * 1e-6 + 1j * torch.ones(2, 32, 32) * 1e-6

        # Execute
        dp = multislice_forward_model_vec_all(obj_patches, probe, H)

        # Verify
        assert torch.all(torch.isfinite(dp)), "Should handle small values without NaN/inf"
        assert torch.all(dp >= 0), "Diffraction patterns should still be non-negative"

    def test_different_batch_sizes(self):
        """Test forward model with different batch sizes."""
        for batch_size in [1, 4, 8, 16]:
            # Setup
            obj_patches = torch.rand(batch_size, 1, 1, 32, 32, 2)
            probe = torch.rand(batch_size, 1, 32, 32) + 1j * torch.rand(batch_size, 1, 32, 32)
            H = torch.ones(batch_size, 32, 32) + 1j * torch.ones(batch_size, 32, 32)

            # Execute
            dp = multislice_forward_model_vec_all(obj_patches, probe, H)

            # Verify
            assert dp.shape == (
                batch_size,
                32,
                32,
            ), f"Expected shape ({batch_size}, 32, 32), got {dp.shape}"
            assert torch.all(
                dp >= 0
            ), f"Batch size {batch_size}: Diffraction patterns should be non-negative"

    def test_different_patch_sizes(self):
        """Test forward model with different patch sizes."""
        for size in [16, 32, 64]:
            # Setup
            obj_patches = torch.rand(2, 1, 1, size, size, 2)
            probe = torch.rand(2, 1, size, size) + 1j * torch.rand(2, 1, size, size)
            H = torch.ones(2, size, size) + 1j * torch.ones(2, size, size)

            # Execute
            dp = multislice_forward_model_vec_all(obj_patches, probe, H)

            # Verify
            assert dp.shape == (
                2,
                size,
                size,
            ), f"Expected shape (2, {size}, {size}), got {dp.shape}"
            assert torch.all(dp >= 0), f"Size {size}: Diffraction patterns should be non-negative"

    def test_device_compatibility(self):
        """Test forward model on different devices."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            # Setup
            obj_patches = torch.rand(2, 1, 1, 32, 32, 2, device=device)
            probe = torch.rand(2, 1, 32, 32, device=device) + 1j * torch.rand(
                2, 1, 32, 32, device=device
            )
            H = torch.ones(2, 32, 32, device=device) + 1j * torch.ones(2, 32, 32, device=device)

            # Execute
            dp = multislice_forward_model_vec_all(obj_patches, probe, H)

            # Verify
            assert dp.device.type == device, f"Expected device {device}, got {dp.device}"
            assert dp.shape == (
                2,
                32,
                32,
            ), f"Device {device}: Expected shape (2, 32, 32), got {dp.shape}"
            assert torch.all(
                dp >= 0
            ), f"Device {device}: Diffraction patterns should be non-negative"

"""
Test loss function implementations in losses.py

This module tests the CombinedLoss class which implements various loss functions
used for ptychographic reconstruction, including Gaussian and Poisson statistics.
"""

from test.utils import set_random_seed

import pytest
import torch
from ptyrad.losses import CombinedLoss


class TestCombinedLoss:
    """Test suite for CombinedLoss class."""

    def test_basic_combined_loss(self):
        """Test basic combined loss calculation."""
        set_random_seed(42)

        # Create loss params for Gaussian loss (loss_single)
        loss_params = {"loss_single": {"state": True, "weight": 1.0, "dp_pow": 0.5}}

        loss_fn = CombinedLoss(loss_params, device="cpu")

        # Create sample data
        dp_meas = torch.rand(4, 32, 32) + 1.0
        dp_fwd = torch.rand(4, 32, 32) + 1.0

        # Compute loss
        loss = loss_fn.get_loss_single(dp_fwd, dp_meas)

        # Verify loss properties
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_combined_loss_gradient(self):
        """Test gradient computation for combined loss."""
        set_random_seed(42)

        loss_params = {"loss_single": {"state": True, "weight": 1.0, "dp_pow": 0.5}}

        loss_fn = CombinedLoss(loss_params, device="cpu")

        dp_meas = torch.rand(4, 32, 32) + 1.0
        dp_fwd = torch.rand(4, 32, 32) + 1.0
        dp_fwd.requires_grad = True

        loss = loss_fn.get_loss_single(dp_fwd, dp_meas)
        loss.backward()

        # Verify gradients
        assert dp_fwd.grad is not None, "Should have gradients"
        assert torch.all(torch.isfinite(dp_fwd.grad)), "Gradients should be finite"
        assert dp_fwd.grad.shape == dp_fwd.shape, "Gradient shape should match input shape"

    def test_combined_loss_different_weights(self):
        """Test combined loss with different weights."""
        set_random_seed(42)

        # Test different weight values
        for weight in [0.1, 0.5, 1.0, 2.0]:
            loss_params = {"loss_single": {"state": True, "weight": weight, "dp_pow": 0.5}}

            loss_fn = CombinedLoss(loss_params, device="cpu")

            dp_meas = torch.rand(4, 32, 32) + 1.0
            dp_fwd = torch.rand(4, 32, 32) + 1.0

            loss = loss_fn.get_loss_single(dp_fwd, dp_meas)

            assert loss.dim() == 0, f"Loss should be scalar for weight {weight}"
            assert loss >= 0, f"Loss should be non-negative for weight {weight}"
            assert torch.isfinite(loss), f"Loss should be finite for weight {weight}"

    def test_combined_loss_different_powers(self):
        """Test combined loss with different dp_pow values."""
        set_random_seed(42)

        # Test different power values
        for dp_pow in [0.2, 0.3, 0.5, 0.8]:
            loss_params = {"loss_single": {"state": True, "weight": 1.0, "dp_pow": dp_pow}}

            loss_fn = CombinedLoss(loss_params, device="cpu")

            dp_meas = torch.rand(4, 32, 32) + 1.0
            dp_fwd = torch.rand(4, 32, 32) + 1.0

            loss = loss_fn.get_loss_single(dp_fwd, dp_meas)

            assert loss.dim() == 0, f"Loss should be scalar for dp_pow {dp_pow}"
            assert torch.isfinite(loss), f"Loss should be finite for dp_pow {dp_pow}"

    def test_combined_loss_identical_inputs(self):
        """Test combined loss with identical inputs."""
        set_random_seed(42)

        loss_params = {"loss_single": {"state": True, "weight": 1.0, "dp_pow": 0.5}}

        loss_fn = CombinedLoss(loss_params, device="cpu")

        dp_meas = torch.rand(4, 32, 32) + 1.0
        dp_fwd = dp_meas.clone()

        loss = loss_fn.get_loss_single(dp_fwd, dp_meas)

        # For identical inputs, loss should be very small (but not necessarily zero due to normalization)
        assert loss >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_combined_loss_different_shapes(self):
        """Test combined loss with different input shapes."""
        set_random_seed(42)

        loss_params = {"loss_single": {"state": True, "weight": 1.0, "dp_pow": 0.5}}

        loss_fn = CombinedLoss(loss_params, device="cpu")

        # Test different batch sizes
        for batch_size in [1, 2, 8, 16]:
            dp_meas = torch.rand(batch_size, 32, 32) + 1.0
            dp_fwd = torch.rand(batch_size, 32, 32) + 1.0

            loss = loss_fn.get_loss_single(dp_fwd, dp_meas)

            assert loss.dim() == 0, f"Loss should be scalar for batch size {batch_size}"
            assert torch.isfinite(loss), f"Loss should be finite for batch size {batch_size}"

    def test_combined_loss_different_pattern_sizes(self):
        """Test combined loss with different pattern sizes."""
        set_random_seed(42)

        loss_params = {"loss_single": {"state": True, "weight": 1.0, "dp_pow": 0.5}}

        loss_fn = CombinedLoss(loss_params, device="cpu")

        for size in [16, 32, 64]:
            dp_meas = torch.rand(4, size, size) + 1.0
            dp_fwd = torch.rand(4, size, size) + 1.0

            loss = loss_fn.get_loss_single(dp_fwd, dp_meas)

            assert loss.dim() == 0, f"Loss should be scalar for size {size}"
            assert torch.isfinite(loss), f"Loss should be finite for size {size}"

    def test_combined_loss_with_small_values(self):
        """Test combined loss with small input values."""
        set_random_seed(42)

        loss_params = {"loss_single": {"state": True, "weight": 1.0, "dp_pow": 0.5}}

        loss_fn = CombinedLoss(loss_params, device="cpu")

        # Test with small values
        dp_meas = torch.rand(4, 32, 32) * 0.1 + 0.01
        dp_fwd = torch.rand(4, 32, 32) * 0.1 + 0.01

        loss = loss_fn.get_loss_single(dp_fwd, dp_meas)

        assert torch.isfinite(loss), "Should handle small values"
        assert loss >= 0, "Loss should be non-negative"

    def test_combined_loss_with_large_values(self):
        """Test combined loss with large input values."""
        set_random_seed(42)

        loss_params = {"loss_single": {"state": True, "weight": 1.0, "dp_pow": 0.5}}

        loss_fn = CombinedLoss(loss_params, device="cpu")

        # Test with large values
        dp_meas = torch.rand(4, 32, 32) * 100.0 + 1.0
        dp_fwd = torch.rand(4, 32, 32) * 100.0 + 1.0

        loss = loss_fn.get_loss_single(dp_fwd, dp_meas)

        assert torch.isfinite(loss), "Should handle large values"
        assert loss >= 0, "Loss should be non-negative"

    def test_combined_loss_device_compatibility(self):
        """Test combined loss on different devices."""
        set_random_seed(42)

        loss_params = {"loss_single": {"state": True, "weight": 1.0, "dp_pow": 0.5}}

        # Test CPU
        loss_fn_cpu = CombinedLoss(loss_params, device="cpu")
        dp_meas = torch.rand(4, 32, 32) + 1.0
        dp_fwd = torch.rand(4, 32, 32) + 1.0

        loss = loss_fn_cpu.get_loss_single(dp_fwd, dp_meas)
        assert torch.isfinite(loss), "CPU: Loss should be finite"

        # Test GPU if available
        if torch.cuda.is_available():
            loss_fn_gpu = CombinedLoss(loss_params, device="cuda")
            dp_meas_gpu = dp_meas.cuda()
            dp_fwd_gpu = dp_fwd.cuda()

            loss_gpu = loss_fn_gpu.get_loss_single(dp_fwd_gpu, dp_meas_gpu)
            assert torch.isfinite(loss_gpu), "GPU: Loss should be finite"


class TestLossEdgeCases:
    """Test edge cases for loss functions."""

    def test_loss_with_zeros(self):
        """Test loss handling of zero values."""
        set_random_seed(42)

        loss_params = {"loss_single": {"state": True, "weight": 1.0, "dp_pow": 0.5}}

        loss_fn = CombinedLoss(loss_params, device="cpu")

        # Add some zero values
        dp_meas = torch.rand(4, 32, 32) + 1.0
        dp_fwd = torch.rand(4, 32, 32) + 1.0
        dp_meas[0, 0, 0] = 0.0
        dp_fwd[0, 0, 0] = 0.0

        loss = loss_fn.get_loss_single(dp_fwd, dp_meas)

        # Should handle zeros without crashing
        assert torch.isfinite(loss), "Should handle zero values"

    def test_loss_with_negative_values(self):
        """Test loss handling of negative values."""
        set_random_seed(42)

        loss_params = {"loss_single": {"state": True, "weight": 1.0, "dp_pow": 0.5}}

        loss_fn = CombinedLoss(loss_params, device="cpu")

        # Add some negative values (shouldn't occur in real data but test robustness)
        dp_meas = torch.rand(4, 32, 32) + 1.0
        dp_fwd = torch.rand(4, 32, 32) + 1.0
        dp_meas[0, 0, 0] = -0.1
        dp_fwd[0, 0, 0] = -0.1

        # This might produce NaN due to negative values with power operations
        # but shouldn't crash
        loss = loss_fn.get_loss_single(dp_fwd, dp_meas)

        # Just verify it doesn't crash - result may be NaN
        assert loss.dim() == 0


@pytest.fixture
def sample_loss_data():
    """Fixture providing sample data for loss function tests."""
    set_random_seed(42)
    dp_meas = torch.rand(4, 32, 32) + 1.0
    dp_fwd = torch.rand(4, 32, 32) + 1.0
    return dp_meas, dp_fwd


def test_basic_loss_with_fixture(sample_loss_data):
    """Example test using the fixture."""
    dp_meas, dp_fwd = sample_loss_data

    loss_params = {"loss_single": {"state": True, "weight": 1.0, "dp_pow": 0.5}}

    loss_fn = CombinedLoss(loss_params, device="cpu")
    loss = loss_fn.get_loss_single(dp_fwd, dp_meas)

    assert loss.dim() == 0
    assert loss >= 0
    assert torch.isfinite(loss)

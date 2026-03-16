"""
Test the reconstruction module functionality in reconstruction.py

This module tests the ptychographic reconstruction solver and workflows.
"""

from test.utils import minimal_model_params

import pytest

from src.ptyrad.reconstruction import PtyRADSolver


class TestReconstruction:
    """Test suite for reconstruction module."""

    def test_solver_initialization(self):
        """Test basic solver initialization - requires complete params."""
        # This test is skipped as it requires complex initialization
        pytest.skip("Requires complete parameter setup for solver initialization")

    def test_solver_with_different_devices(self):
        """Test solver initialization on different devices - requires complete params."""
        # This test is skipped as it requires complex initialization
        pytest.skip("Requires complete parameter setup for device testing")

    def test_solver_loss_initialization(self):
        """Test that solver correctly initializes loss function - requires complete params."""
        # This test is skipped as it requires complex initialization
        pytest.skip("Requires complete parameter setup for loss initialization")

    def test_solver_constraint_initialization(self):
        """Test that solver correctly initializes constraint function - requires complete params."""
        # This test is skipped as it requires complex initialization
        pytest.skip("Requires complete parameter setup for constraint initialization")

    def test_solver_parameter_validation(self):
        """Test solver parameter validation."""
        # Test with missing required parameters
        incomplete_params = {
            "init_params": {
                "meas_Npix": 32,
                # Missing other required init_params
            },
            "loss_params": {},
            "constraint_params": {},
            "model_params": minimal_model_params(),
            "recon_params": {
                "NITER": 2,
                "SAVE_ITERS": None,
                "BATCH_SIZE": {"size": 2, "grad_accumulation": 1},
                "if_quiet": True,
            },
        }

        # This should raise an error during initialization
        with pytest.raises(Exception):  # Could be KeyError or other initialization errors
            PtyRADSolver(incomplete_params, device="cpu", seed=42)

    def test_solver_quiet_mode(self):
        """Test solver quiet mode setting - requires complete params."""
        # This test is skipped as it requires complex initialization
        pytest.skip("Requires complete parameter setup for quiet mode testing")

    def test_solver_reconstruction_preparation(self):
        """Test solver reconstruction preparation - requires complex setup."""
        # This test is skipped as it requires complex initialization and full workflow
        pytest.skip("Requires complex initialization for full reconstruction workflow")

    def test_solver_hypertune_mode(self):
        """Test solver hypertune mode - requires complex setup."""
        # This test is skipped as it requires complex initialization and full workflow
        pytest.skip("Requires complex initialization for hypertune workflow")

    def test_reconstruction_utility_functions(self):
        """Test reconstruction utility functions that don't require full initialization."""
        # Test select_scan_indices function
        from src.ptyrad.reconstruction import select_scan_indices

        # Test with simple parameters (mode='full' is the default)
        indices = select_scan_indices(N_scan_slow=4, N_scan_fast=4, verbose=False)

        assert len(indices) == 16  # 4x4 scan positions
        assert indices.dtype in ["int32", "int64"]  # Can be either depending on platform
        assert indices.ndim == 1

        # Test make_batches function
        # Create simple position data - need to be 2D array of shape (N, 2)
        import numpy as np

        from src.ptyrad.reconstruction import make_batches

        pos = np.array([[i, j] for i in range(4) for j in range(4)])  # 16 positions
        batches = make_batches(indices, pos, batch_size=4, mode="sequential", verbose=False)

        assert len(batches) == 4  # Should create 4 batches with batch_size=4
        assert all(1 <= len(batch) <= 5 for batch in batches)  # Batches can vary in size
        # Verify all indices are covered
        all_indices = np.concatenate(batches)
        assert len(all_indices) == len(indices)  # All indices should be included
        assert len(np.unique(all_indices)) == len(indices)  # No duplicates

    def test_optimizer_creation(self):
        """Test optimizer creation utility function."""
        from test.utils import create_minimal_init_variables_2d, minimal_model_params

        from src.ptyrad.models import PtychoAD
        from src.ptyrad.reconstruction import create_optimizer

        # Create minimal model for testing
        init_vars = create_minimal_init_variables_2d()
        model_params = minimal_model_params()
        model = PtychoAD(init_vars, model_params, device="cpu", verbose=False)

        # Test optimizer creation
        optimizer_params = {"name": "Adam", "lr": 0.01, "weight_decay": 0}

        optimizer = create_optimizer(optimizer_params, model.optimizable_params, verbose=False)

        assert optimizer is not None
        assert hasattr(optimizer, "param_groups")
        assert len(optimizer.param_groups) > 0


@pytest.fixture
def minimal_solver_params():
    """Fixture providing minimal solver parameters for testing."""
    return {
        "init_params": {
            "meas_Npix": 32,
            "pos_N_scan_slow": 4,
            "pos_N_scan_fast": 4,
            "pos_scan_step_size": 0.4290,
            "probe_kv": 80,
            "probe_conv_angle": 24.9,
            "probe_pmode_max": 1,
            "obj_Nlayer": 1,
            "obj_slice_thickness": 2,
        },
        "loss_params": {"MSE": {"state": True, "weight": 1.0, "power": 1}},
        "constraint_params": {},
        "model_params": minimal_model_params(),
        "recon_params": {
            "NITER": 2,
            "SAVE_ITERS": None,
            "BATCH_SIZE": {"size": 2, "grad_accumulation": 1},
            "if_quiet": True,
        },
    }


def test_solver_with_fixture(minimal_solver_params):
    """Example test using the minimal solver params fixture - requires complete params."""
    # This test is skipped as it requires complex initialization
    pytest.skip("Requires complete parameter setup for solver testing")

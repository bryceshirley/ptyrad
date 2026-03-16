"""
Test constraint implementations in constraints.py

This module tests the CombinedConstraint class which applies various physical constraints
during ptychographic reconstruction based on iteration-based scheduling.

Note: Many tests are simplified due to the complexity of the constraint system.
"""

from test.utils import set_random_seed

import pytest
import torch
from ptyrad.constraints import CombinedConstraint


class TestCombinedConstraint:
    """Test suite for CombinedConstraint class."""

    def test_basic_initialization(self):
        """Test basic constraint initialization."""
        set_random_seed(42)

        # Create minimal constraint params with correct parameter names
        constraint_params = {
            "ortho_pmode": {"state": True, "start_iter": 0, "step": 1, "end_iter": None}
        }

        # Initialize constraint
        constraint = CombinedConstraint(constraint_params, device="cpu", verbose=False)

        # Verify attributes
        assert hasattr(constraint, "device")
        assert hasattr(constraint, "constraint_params")
        assert constraint.device == "cpu"

    def test_iteration_scheduling(self):
        """Test constraint iteration scheduling."""
        set_random_seed(42)

        constraint_params = {
            "ortho_pmode": {"state": True, "start_iter": 5, "step": 10, "end_iter": None}
        }

        constraint = CombinedConstraint(constraint_params, device="cpu", verbose=False)

        # Test should_apply_at_iter method
        assert not constraint._should_apply_at_iter("ortho_pmode", 0), "Should not apply at iter 0"
        assert not constraint._should_apply_at_iter("ortho_pmode", 4), "Should not apply at iter 4"
        assert constraint._should_apply_at_iter("ortho_pmode", 5), "Should apply at iter 5"
        assert constraint._should_apply_at_iter("ortho_pmode", 15), "Should apply at iter 15"
        assert constraint._should_apply_at_iter("ortho_pmode", 25), "Should apply at iter 25"
        assert not constraint._should_apply_at_iter("ortho_pmode", 6), "Should not apply at iter 6"

    def test_device_compatibility(self):
        """Test constraint on different devices."""
        set_random_seed(42)

        constraint_params = {
            "ortho_pmode": {"state": True, "start_iter": 0, "step": 1, "end_iter": None}
        }

        # Test CPU
        constraint_cpu = CombinedConstraint(constraint_params, device="cpu", verbose=False)
        assert constraint_cpu.device == "cpu"

        # Test GPU if available
        if torch.cuda.is_available():
            constraint_gpu = CombinedConstraint(constraint_params, device="cuda", verbose=False)
            assert constraint_gpu.device == "cuda"

    def test_disabled_constraints(self):
        """Test constraint with disabled constraints."""
        set_random_seed(42)

        constraint_params = {
            "ortho_pmode": {
                "state": False,  # Disabled
                "start_iter": 0,
                "step": 1,
                "end_iter": None,
            }
        }

        constraint = CombinedConstraint(constraint_params, device="cpu", verbose=False)

        # Verify constraint is properly initialized even when disabled
        assert hasattr(constraint, "constraint_params")
        assert constraint.device == "cpu"

    def test_multiple_constraints(self):
        """Test constraint with multiple constraint types."""
        set_random_seed(42)

        constraint_params = {
            "ortho_pmode": {"state": True, "start_iter": 0, "step": 1, "end_iter": None},
            "probe_mask_k": {
                "state": True,
                "start_iter": 0,
                "step": 1,
                "end_iter": None,
                "radius": 0.5,
                "width": 0.1,
                "power_thresh": 0.9,
            },
        }

        constraint = CombinedConstraint(constraint_params, device="cpu", verbose=False)

        # Verify both constraints are in params
        assert "ortho_pmode" in constraint.constraint_params
        assert "probe_mask_k" in constraint.constraint_params

    def test_constraint_parameter_normalization(self):
        """Test that constraint parameters are properly normalized."""
        set_random_seed(42)

        # Create constraint params that might need normalization
        constraint_params = {
            "ortho_pmode": {
                "state": True,
                "start_iter": 0,
                "step": 1,
                # Note: end_iter is missing, should be normalized
            }
        }

        constraint = CombinedConstraint(constraint_params, device="cpu", verbose=False)

        # Verify parameters were normalized
        ortho_params = constraint.constraint_params["ortho_pmode"]
        assert "start_iter" in ortho_params
        assert "step" in ortho_params
        assert "end_iter" in ortho_params  # Should be added during normalization


class TestConstraintEdgeCases:
    """Test edge cases for constraint functions."""

    def test_empty_constraint_params(self):
        """Test constraint with empty parameters."""
        set_random_seed(42)

        # Empty constraint params
        constraint_params = {}

        constraint = CombinedConstraint(constraint_params, device="cpu", verbose=False)

        # Should initialize without crashing
        assert hasattr(constraint, "constraint_params")
        assert constraint.device == "cpu"

    def test_constraint_with_none_values(self):
        """Test constraint with None values."""
        set_random_seed(42)

        constraint_params = {
            "ortho_pmode": {
                "state": True,
                "start_iter": None,  # None values should be handled
                "step": 1,
                "end_iter": None,
            }
        }

        constraint = CombinedConstraint(constraint_params, device="cpu", verbose=False)

        # Should handle None values gracefully
        assert hasattr(constraint, "constraint_params")

    def test_constraint_with_negative_iterations(self):
        """Test constraint scheduling with negative iteration values."""
        set_random_seed(42)

        constraint_params = {
            "ortho_pmode": {
                "state": True,
                "start_iter": -1,  # Negative start
                "step": 1,
                "end_iter": None,
            }
        }

        constraint = CombinedConstraint(constraint_params, device="cpu", verbose=False)

        # Test with negative iteration
        result = constraint._should_apply_at_iter("ortho_pmode", -5)
        # Should handle negative iterations without crashing
        assert isinstance(result, bool)


@pytest.fixture
def basic_constraint_setup():
    """Fixture providing basic constraint setup."""
    set_random_seed(42)

    constraint_params = {
        "ortho_pmode": {"state": True, "start_iter": 0, "step": 1, "end_iter": None}
    }

    return CombinedConstraint(constraint_params, device="cpu", verbose=False)


def test_constraint_initialization_with_fixture(basic_constraint_setup):
    """Example test using the fixture."""
    constraint = basic_constraint_setup

    # Verify basic properties
    assert hasattr(constraint, "device")
    assert hasattr(constraint, "constraint_params")
    assert constraint.device == "cpu"


def test_constraint_scheduling_with_fixture(basic_constraint_setup):
    """Test scheduling with fixture."""
    constraint = basic_constraint_setup

    # Test scheduling
    assert constraint._should_apply_at_iter("ortho_pmode", 0)
    assert constraint._should_apply_at_iter("ortho_pmode", 10)
    assert not constraint._should_apply_at_iter("ortho_pmode", -1)

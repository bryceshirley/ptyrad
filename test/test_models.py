"""
Simplified test the PtychoAD model implementation in models.py

This module tests the main optimization model that holds optimizable tensors
and interacts with loss and constraints for ptychographic reconstruction.
"""

from test.utils import (
    create_minimal_init_variables_2d,
    minimal_model_params,
)

import pytest
import torch

from src.ptyrad.models import PtychoAD


class TestPtychoADModel:
    """Test suite for PtychoAD model."""

    def test_basic_initialization(self):
        """Test basic model initialization."""
        init_vars = create_minimal_init_variables_2d()
        model_params = minimal_model_params()
        model = PtychoAD(init_vars, model_params, device="cpu", verbose=False)

        assert model.device == "cpu"
        assert hasattr(model, "opt_obja")
        assert hasattr(model, "opt_objp")
        assert hasattr(model, "opt_probe")
        assert hasattr(model, "optimizable_tensors")

    def test_device_placement(self):
        """Test device placement."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            init_vars = create_minimal_init_variables_2d()
            model_params = minimal_model_params()
            model = PtychoAD(init_vars, model_params, device=device, verbose=False)

            assert str(model.device) == device
            # Check that all parameters are on the correct device
            for param in model.parameters():
                assert param.device.type == device

    def test_parameter_shapes(self):
        """Test parameter shapes for 2D case."""
        # Test 2D case
        init_vars_2d = create_minimal_init_variables_2d()
        model_params = minimal_model_params()
        model_2d = PtychoAD(init_vars_2d, model_params, device="cpu", verbose=False)

        assert model_2d.opt_obja.shape == (1, 32, 32)  # (omode, Ny, Nx)
        assert model_2d.opt_objp.shape == (1, 32, 32)  # (omode, Ny, Nx)
        assert model_2d.opt_probe.shape[0] == 1  # pmode
        assert model_2d.opt_probe.shape[1:] == (32, 32, 2)  # (Ny, Nx, 2) for real/imag

    def test_model_state_dict(self):
        """Test model state dict save/load."""
        init_vars = create_minimal_init_variables_2d()
        model_params = minimal_model_params()
        model = PtychoAD(init_vars, model_params, device="cpu", verbose=False)

        # Save state dict
        state_dict = model.state_dict()
        assert "opt_obja" in state_dict
        assert "opt_objp" in state_dict
        assert "opt_probe" in state_dict

        # Create new model and load state
        model2 = PtychoAD(init_vars, model_params, device="cpu", verbose=False)
        model2.load_state_dict(state_dict)

        # Check that parameters are the same
        assert torch.allclose(model.opt_obja, model2.opt_obja)
        assert torch.allclose(model.opt_objp, model2.opt_objp)
        assert torch.allclose(model.opt_probe, model2.opt_probe)

    def test_training_eval_modes(self):
        """Test training/eval modes."""
        init_vars = create_minimal_init_variables_2d()
        model_params = minimal_model_params()
        model = PtychoAD(init_vars, model_params, device="cpu", verbose=False)

        # Test training mode
        model.train()
        assert model.training

        # Test eval mode
        model.eval()
        assert not model.training


@pytest.fixture
def basic_model_setup():
    """Fixture providing basic model setup."""
    init_vars = create_minimal_init_variables_2d()
    model_params = minimal_model_params()
    model = PtychoAD(init_vars, model_params, device="cpu", verbose=False)
    return model


def test_with_fixture(basic_model_setup):
    """Example test using the fixture."""
    model = basic_model_setup
    assert hasattr(model, "opt_obja")
    assert hasattr(model, "opt_objp")
    assert hasattr(model, "opt_probe")

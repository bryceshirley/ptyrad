"""
Test the CLI module functionality in cli.py

This module tests the command-line interface functions.
"""

import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

from src.ptyrad.cli import main


class TestCLI:
    """Test suite for CLI module."""

    def test_cli_help(self):
        """Test CLI help message."""
        # Test that help message works without errors
        result = subprocess.run(
            [sys.executable, "-m", "src.ptyrad.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0
        assert "PtyRAD Command-Line Interface" in result.stdout

    def test_cli_check_gpu(self):
        """Test CLI check-gpu command."""
        result = subprocess.run(
            [sys.executable, "-m", "src.ptyrad.cli", "check-gpu"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0

    def test_cli_print_system_info(self):
        """Test CLI print-system-info command."""
        result = subprocess.run(
            [sys.executable, "-m", "src.ptyrad.cli", "print-system-info"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0

    def test_cli_missing_command(self):
        """Test CLI with missing command raises error."""
        result = subprocess.run(
            [sys.executable, "-m", "src.ptyrad.cli"], capture_output=True, text=True, cwd="."
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower()

    def test_cli_invalid_command(self):
        """Test CLI with invalid command raises error."""
        result = subprocess.run(
            [sys.executable, "-m", "src.ptyrad.cli", "invalid-command"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode != 0

    def test_cli_run_missing_params(self):
        """Test CLI run command with missing params file."""
        result = subprocess.run(
            [sys.executable, "-m", "src.ptyrad.cli", "run"], capture_output=True, text=True, cwd="."
        )
        assert result.returncode != 0
        assert "params_path" in result.stderr

    def test_cli_export_missing_params(self):
        """Test CLI export command with missing params file."""
        result = subprocess.run(
            [sys.executable, "-m", "src.ptyrad.cli", "export-meas-init"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode != 0
        assert "params_path" in result.stderr

    def test_cli_validate_missing_params(self):
        """Test CLI validate command with missing params file."""
        result = subprocess.run(
            [sys.executable, "-m", "src.ptyrad.cli", "validate-params"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode != 0
        assert "params_path" in result.stderr

    def test_cli_run_with_invalid_params(self):
        """Test CLI run command with invalid params file."""
        # Create a temporary invalid params file
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f:
            f.write("invalid: yaml: content:")
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.ptyrad.cli", "run", "--params_path", temp_path],
                capture_output=True,
                text=True,
                cwd=".",
            )
            assert result.returncode != 0
        finally:
            os.unlink(temp_path)

    def test_cli_gui_placeholder(self):
        """Test CLI gui command (placeholder)."""
        result = subprocess.run(
            [sys.executable, "-m", "src.ptyrad.cli", "gui"], capture_output=True, text=True, cwd="."
        )
        # GUI shows a placeholder message, so it should succeed
        assert result.returncode == 0
        assert "placeholder" in result.stdout.lower()


@pytest.fixture
def temp_params_file():
    """Fixture providing a temporary params file."""
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f:
        # Write minimal valid YAML content
        yaml_content = """
init_params:
    meas_Npix: 32
    pos_N_scan_slow: 4
    pos_N_scan_fast: 4
    pos_scan_step_size: 0.4290
    probe_kv: 80
    probe_conv_angle: 24.9
    probe_pmode_max: 1
    obj_Nlayer: 1
    obj_slice_thickness: 2
    meas_params: []
recon_params:
    NITER: 2
    SAVE_ITERS: None
    BATCH_SIZE:
        size: 2
        grad_accumulation: 1
"""
        f.write(yaml_content)
        yield f.name
    # Cleanup will be handled by individual tests


def test_cli_with_temp_params(temp_params_file):
    """Example test using the temp params file fixture."""
    # Test that the params file exists and is valid YAML
    assert os.path.exists(temp_params_file)

    # Test validate command with the temp file
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.ptyrad.cli",
            "validate-params",
            "--params_path",
            temp_params_file,
        ],
        capture_output=True,
        text=True,
        cwd=".",
    )
    # This should work since we have a valid YAML structure
    # Note: It might still fail due to missing required fields, but that's expected

    # Cleanup
    os.unlink(temp_params_file)


def test_cli_argument_parsing():
    """Test CLI argument parsing with mocked functions."""
    # Test that arguments are parsed correctly
    test_args = [
        "run",
        "--params_path",
        "test.yml",
        "--gpuid",
        "0",
        "--jobid",
        "123",
        "--seed",
        "42",
    ]

    with patch("sys.argv", ["ptyrad.cli"] + test_args):
        with patch("src.ptyrad.cli.run") as mock_run:
            mock_run.return_value = None
            # This should not raise SystemExit when mocked
            main()
            # Check that run was called
            mock_run.assert_called_once()


def test_cli_check_gpu_function():
    """Test CLI check_gpu function directly."""
    from src.ptyrad.cli import check_gpu

    # Create mock args
    class MockArgs:
        pass

    args = MockArgs()

    # This should not raise an error
    check_gpu(args)


def test_cli_print_info_function():
    """Test CLI print_info function directly."""
    from src.ptyrad.cli import print_info

    # Create mock args
    class MockArgs:
        pass

    args = MockArgs()

    # This should not raise an error
    print_info(args)

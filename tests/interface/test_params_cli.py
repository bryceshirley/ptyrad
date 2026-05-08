import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from ptyrad.params import load_params

from .conftest import PROJECT_ROOT


STARTER_PARAMS_ROOT = PROJECT_ROOT / "src" / "ptyrad" / "starter" / "params"


def _write_yaml(path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _fake_external_file(tmp_path, suffix):
    suffix = suffix or ".dat"
    path = tmp_path / f"external{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")
    return str(path)


def _rewrite_external_paths(data, tmp_path):
    """Keep starter YAML semantics while replacing external demo paths."""
    data = deepcopy(data)
    init_params = data.get("init_params", {})

    meas_params = init_params.get("meas_params")
    if isinstance(meas_params, dict) and meas_params.get("path"):
        suffix = Path(str(meas_params["path"])).suffix
        meas_params["path"] = _fake_external_file(tmp_path, suffix)

    for key in ("probe_params", "pos_params", "obj_params"):
        value = init_params.get(key)
        if isinstance(value, str):
            suffix = Path(value).suffix
            init_params[key] = _fake_external_file(tmp_path, suffix)

    tilt_params = init_params.get("tilt_params")
    if isinstance(tilt_params, dict) and tilt_params.get("path"):
        suffix = Path(str(tilt_params["path"])).suffix
        tilt_params["path"] = _fake_external_file(tmp_path, suffix)
    elif isinstance(tilt_params, str):
        suffix = Path(tilt_params).suffix
        init_params["tilt_params"] = _fake_external_file(tmp_path, suffix)

    return data


def test_all_bundled_starter_params_validate_as_schemas(tmp_path):
    yaml_paths = sorted(STARTER_PARAMS_ROOT.rglob("*.yaml"))
    assert yaml_paths, "No bundled starter params found"

    for source_path in yaml_paths:
        data = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        rewritten = _rewrite_external_paths(data, tmp_path / source_path.stem)
        temp_path = _write_yaml(tmp_path / source_path.name, rewritten)

        params = load_params(str(temp_path), validate=True)

        assert "init_params" in params
        assert "recon_params" in params


def test_load_params_applies_defaults(minimal_params_dict, tmp_path):
    params_path = _write_yaml(tmp_path / "minimal.yaml", minimal_params_dict)

    params = load_params(str(params_path), validate=True)

    assert params["params_path"] == str(params_path)
    assert params["init_params"]["pos_N_scans"] == 4
    assert "model_params" in params
    assert "loss_params" in params
    assert "constraint_params" in params


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda data: data["recon_params"].update({"unexpected_option": True}),
            "extra",
        ),
        (
            lambda data: data["recon_params"].update({"GROUP_MODE": "not-a-mode"}),
            "GROUP_MODE",
        ),
        (
            lambda data: data["init_params"].update(
                {"meas_calibration": {"mode": "dx"}}
            ),
            "value",
        ),
        (
            lambda data: data["init_params"].update(
                {"probe_source": "custom", "probe_params": None}
            ),
            "probe_source='custom'",
        ),
    ],
)
def test_invalid_params_fail_with_useful_errors(
    minimal_params_dict, tmp_path, mutate, expected
):
    params_data = deepcopy(minimal_params_dict)
    mutate(params_data)
    params_path = _write_yaml(tmp_path / "invalid.yaml", params_data)

    with pytest.raises(Exception) as excinfo:
        load_params(str(params_path), validate=True)

    assert expected in str(excinfo.value)


def test_cli_help_and_validate_params(cli_env, minimal_params_dict, tmp_path):
    params_path = _write_yaml(tmp_path / "valid.yaml", minimal_params_dict)

    help_result = subprocess.run(
        [sys.executable, "-m", "ptyrad", "--help"],
        cwd=PROJECT_ROOT,
        env=cli_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "PtyRAD Command-Line Interface" in help_result.stdout

    validate_result = subprocess.run(
        [sys.executable, "-m", "ptyrad", "validate-params", str(params_path)],
        cwd=PROJECT_ROOT,
        env=cli_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate_result.returncode == 0, validate_result.stderr
    assert "Success!" in validate_result.stdout

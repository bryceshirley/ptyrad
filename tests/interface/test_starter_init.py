import subprocess
import sys
from pathlib import Path

import ptyrad.starter

from .conftest import PROJECT_ROOT


def _relative_files(directory):
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def test_ptyrad_init_exports_packaged_starter_files(cli_env, tmp_path):
    out_dir = tmp_path / "starter"

    result = subprocess.run(
        [sys.executable, "-m", "ptyrad", "init", str(out_dir)],
        cwd=PROJECT_ROOT,
        env=cli_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert out_dir.is_dir()

    source_dir = PROJECT_ROOT / "src" / "ptyrad" / "starter"
    package_dir = Path(ptyrad.starter.__file__).parent

    source_files = _relative_files(source_dir)
    package_files = _relative_files(package_dir)
    output_files = _relative_files(out_dir)

    assert source_files <= package_files
    expected_output = {path for path in source_files if not path.endswith("__init__.py")}
    assert expected_output <= output_files
    assert "params/templates/minimal.yaml" in output_files
    assert "scripts/download_demo_data.py" in output_files


def test_ptyrad_get_templates_exports_template_params(cli_env, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "ptyrad", "get-templates", str(tmp_path)],
        cwd=PROJECT_ROOT,
        env=cli_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    exported = tmp_path / "templates"
    assert (exported / "minimal.yaml").is_file()
    assert (exported / "standard.yaml").is_file()

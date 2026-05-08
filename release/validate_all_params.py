#!/usr/bin/env python3
"""Run the starter parameter validation pytest gate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate bundled starter parameter files through the interface test suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    args, pytest_args = parser.parse_known_args()

    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/interface/test_params_cli.py::test_all_bundled_starter_params_validate_as_schemas",
        *pytest_args,
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

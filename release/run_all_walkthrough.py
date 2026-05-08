"""
Run every walkthrough PtyRAD params file.
"""

import argparse
import logging
import re
from pathlib import Path

from ptyrad.params import load_params
from ptyrad.runtime.device import set_accelerator, set_gpu_device
from ptyrad.runtime.diagnostics import print_system_info
from ptyrad.runtime.logging import LoggingManager
from ptyrad.solver import PtyRADSolver


DEFAULT_PARAMS_PATH = Path("src/ptyrad/starter/params/walkthrough")
RESUME_PARAM_PREFIXES = {"04", "05", "06"}
RESUME_PARAM_KEYS = ("probe_params", "pos_params", "obj_params")


def positive_int(value: str) -> int:
    n_iter = int(value)
    if n_iter < 1:
        raise argparse.ArgumentTypeError("--n_iter must be >= 1")
    return n_iter


def get_yaml_paths(params_path: Path) -> list[Path]:
    if params_path.is_file():
        return [params_path]
    if not params_path.is_dir():
        raise FileNotFoundError(f"Params path does not exist: {params_path}")

    yaml_paths = sorted(
        path for pattern in ("*.yaml", "*.yml") for path in params_path.glob(pattern)
    )
    if not yaml_paths:
        raise FileNotFoundError(f"No YAML params files found under: {params_path}")
    return yaml_paths


def replace_yaml_value(text: str, key: str, value: int) -> str:
    pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*:\s*)[^#\n]*(#.*)?$")

    def replace(match: re.Match) -> str:
        comment = match.group(2)
        suffix = f" {comment}" if comment else ""
        return f"{match.group(1)}{value}{suffix}"

    return pattern.sub(replace, text, count=1)


def replace_resume_model_paths(text: str, n_iter: int) -> str:
    for key in RESUME_PARAM_KEYS:
        pattern = re.compile(rf"(?m)^(\s*{key}\s*:\s*.*?model_iter)\d+(\.hdf5.*)$")
        text = pattern.sub(rf"\g<1>{n_iter:04d}\2", text, count=1)
    return text


def apply_yaml_n_iter_overrides(text: str, params_path: Path, n_iter: int) -> str:
    text = replace_yaml_value(text, "NITER", n_iter)
    text = replace_yaml_value(text, "SAVE_ITERS", n_iter)

    if params_path.name[:2] in RESUME_PARAM_PREFIXES:
        text = replace_resume_model_paths(text, n_iter)
    return text


def run_walkthrough_params(
    yaml_paths: list[Path],
    *,
    n_iter: int | None,
    gpuid: str,
    skip_validate: bool,
) -> None:
    accelerator = set_accelerator()
    device = set_gpu_device(gpuid)
    logger = logging.getLogger("ptyrad")

    print_system_info()

    for index, params_path in enumerate(yaml_paths, start=1):
        logger.info("")
        logger.info(
            "### Walkthrough params %d/%d: %s ###",
            index,
            len(yaml_paths),
            params_path,
        )

        original_text = None
        if n_iter is not None:
            original_text = params_path.read_text(encoding="utf-8")
            updated_text = apply_yaml_n_iter_overrides(original_text, params_path, n_iter)
            params_path.write_text(updated_text, encoding="utf-8")

        try:
            params = load_params(str(params_path), validate=not skip_validate)
            solver = PtyRADSolver(params, device=device, seed=42, acc=accelerator)
            solver.run()
        finally:
            if original_text is not None:
                params_path.write_text(original_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all PtyRAD walkthrough params files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--params_path",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
        help="Directory containing walkthrough YAML files, or a single YAML file.",
    )
    parser.add_argument(
        "--n_iter",
        type=positive_int,
        required=False,
        default=None,
        help="Override NITER, SAVE_ITERS, and resume model_iterXXXX.hdf5 references.",
    )
    parser.add_argument(
        "--gpuid",
        type=str,
        required=False,
        default="0",
        help="GPU ID to use ('acc', 'cpu', or an integer).",
    )
    parser.add_argument(
        "--skip_validate",
        action="store_true",
        help="Skip parameter validation and default filling.",
    )
    args = parser.parse_args()

    LoggingManager(
        log_file="ptyrad_log.txt",
        log_dir="auto",
        prefix_time="datetime",
        append_to_file=True,
        show_timestamp=True,
        verbosity="INFO",
    )

    yaml_paths = get_yaml_paths(args.params_path)
    run_walkthrough_params(
        yaml_paths,
        n_iter=args.n_iter,
        gpuid=args.gpuid,
        skip_validate=args.skip_validate,
    )


if __name__ == "__main__":
    main()

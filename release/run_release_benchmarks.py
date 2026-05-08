#!/usr/bin/env python3
"""Run local PtyRAD release benchmarks against the demo reconstructions.

This script is intentionally local-only: it expects demo datasets and suitable
hardware to be available, then records correctness, quality, and timing metrics
from the generated ``model_iterXXXX.hdf5`` files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ptyrad import __version__ as ptyrad_version  # noqa: E402
from ptyrad.io.load import load_ptyrad  # noqa: E402


DEMO_PARAMS = {
    "tBL_WSe2": PROJECT_ROOT
    / "src"
    / "ptyrad"
    / "starter"
    / "params"
    / "examples"
    / "tBL_WSe2.yaml",
    "PSO": PROJECT_ROOT
    / "src"
    / "ptyrad"
    / "starter"
    / "params"
    / "examples"
    / "PSO.yaml",
}

DEFAULT_BASELINE = PROJECT_ROOT / "release" / "baselines" / "v1_0_0_demo_benchmarks.json"
CRITICAL_TENSORS = ("objp", "obja", "probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local v1.0.0 release benchmarks for PtyRAD demo datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(DEMO_PARAMS),
        default=sorted(DEMO_PARAMS),
        help="Demo benchmark(s) to run.",
    )
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--gpuid", default="0", help="GPU ID, 'cpu', or 'acc'.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "output" / "release_benchmarks",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Refresh stored baseline metrics from successful benchmark runs.",
    )
    parser.add_argument(
        "--strict-baseline",
        action="store_true",
        help="Treat missing baselines as failures instead of warnings.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only inspect the newest matching model files under output-root.",
    )
    return parser.parse_args()


def run_text(command: list[str], *, cwd: Path) -> str | None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_sha() -> str | None:
    return run_text(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)


def device_info() -> dict[str, Any]:
    try:
        import torch

        info: dict[str, Any] = {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        return info
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        return {"error": str(exc)}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "policy": default_policy(), "benchmarks": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def default_policy() -> dict[str, float]:
    return {
        "quality_loss_rel_fail": 0.15,
        "tensor_stats_rel_fail": 0.20,
        "time_warn_rel": 0.25,
        "time_fail_rel": 1.00,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_benchmark(demo: str, source_path: Path, args: argparse.Namespace) -> tuple[int, float]:
    output_dir = args.output_root / demo
    command = [
        sys.executable,
        str(PROJECT_ROOT / "release" / "run_ptyrad_override.py"),
        "--params_path", str(source_path),
        "--n_iter", str(args.n_iter),
        "--output_path", str(output_dir),
        "--gpuid", args.gpuid,
    ]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing else f"{SRC_ROOT}{os.pathsep}{existing}"

    start = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    wall_s = time.perf_counter() - start
    return result.returncode, wall_s


def newest_model_file(demo: str, args: argparse.Namespace) -> Path | None:
    expected = f"model_iter{args.n_iter:04d}.hdf5"
    root = args.output_root / demo
    candidates = sorted(root.rglob(expected), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def finite_stats(value: Any) -> tuple[dict[str, Any], str | None]:
    arr = np.asarray(value)
    if arr.size == 0:
        return {"shape": list(arr.shape), "dtype": str(arr.dtype), "size": 0}, "empty array"
    if not np.isfinite(arr).all():
        return {"shape": list(arr.shape), "dtype": str(arr.dtype)}, "contains NaN or inf"

    magnitudes = np.abs(arr) if np.iscomplexobj(arr) else arr
    return (
        {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "mean": float(np.mean(magnitudes)),
            "std": float(np.std(magnitudes)),
            "min": float(np.min(magnitudes)),
            "max": float(np.max(magnitudes)),
            "abs_mean": float(np.mean(np.abs(arr))),
        },
        None,
    )


def final_loss(loss_iters: Any) -> float | None:
    arr = np.asarray(loss_iters)
    if arr.ndim == 2 and arr.shape[0] and arr.shape[1] >= 2:
        return float(arr[-1, 1])
    return None


def normalize_avg_losses(avg_losses: Any) -> dict[str, float]:
    if not isinstance(avg_losses, dict):
        return {}
    return {str(key): float(value) for key, value in avg_losses.items()}


def collect_metrics(model_path: Path, wall_s: float) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    ckpt = load_ptyrad(str(model_path))

    required = (
        "optimizable_tensors",
        "model_attributes",
        "params",
        "loss_iters",
        "avg_losses",
        "avg_iter_t",
        "niter",
    )
    for key in required:
        if key not in ckpt:
            problems.append(f"missing required key: {key}")

    tensor_stats: dict[str, Any] = {}
    tensors = ckpt.get("optimizable_tensors", {})
    for name in CRITICAL_TENSORS:
        if name not in tensors:
            problems.append(f"missing tensor: {name}")
            continue
        stats, problem = finite_stats(tensors[name])
        tensor_stats[name] = stats
        if problem:
            problems.append(f"{name}: {problem}")

    metrics = {
        "model_path": str(model_path),
        "niter": int(ckpt.get("niter", -1)),
        "final_total_loss": final_loss(ckpt.get("loss_iters")),
        "avg_losses": normalize_avg_losses(ckpt.get("avg_losses")),
        "avg_iter_t": float(ckpt.get("avg_iter_t", math.nan)),
        "total_wall_s": float(wall_s),
        "tensor_stats": tensor_stats,
    }
    return metrics, problems


def rel_change(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    if not math.isfinite(value) or not math.isfinite(baseline):
        return None
    if baseline == 0:
        return 0.0 if value == 0 else math.inf
    return (value - baseline) / abs(baseline)


def compare_to_baseline(
    demo: str,
    metrics: dict[str, Any],
    baseline: dict[str, Any] | None,
    policy: dict[str, float],
    *,
    strict_missing: bool,
) -> tuple[str, list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    if baseline is None:
        message = "missing baseline; run with --update-baseline after reviewing results"
        if strict_missing:
            failures.append(message)
        else:
            warnings.append(message)
        return ("fail" if failures else "warn"), warnings, failures

    loss_delta = rel_change(metrics.get("final_total_loss"), baseline.get("final_total_loss"))
    if loss_delta is not None and loss_delta > policy["quality_loss_rel_fail"]:
        failures.append(
            f"final_total_loss regressed by {loss_delta:.1%} for {demo}"
        )

    for tensor_name in CRITICAL_TENSORS:
        current_stats = metrics.get("tensor_stats", {}).get(tensor_name, {})
        baseline_stats = baseline.get("tensor_stats", {}).get(tensor_name, {})
        for stat_name in ("abs_mean", "std"):
            delta = rel_change(current_stats.get(stat_name), baseline_stats.get(stat_name))
            if delta is not None and abs(delta) > policy["tensor_stats_rel_fail"]:
                failures.append(
                    f"{tensor_name}.{stat_name} drifted by {delta:.1%} for {demo}"
                )

    time_delta = rel_change(metrics.get("avg_iter_t"), baseline.get("avg_iter_t"))
    if time_delta is not None:
        if time_delta > policy["time_fail_rel"]:
            failures.append(f"avg_iter_t regressed by {time_delta:.1%} for {demo}")
        elif time_delta > policy["time_warn_rel"]:
            warnings.append(f"avg_iter_t regressed by {time_delta:.1%} for {demo}")

    if failures:
        return "fail", warnings, failures
    if warnings:
        return "warn", warnings, failures
    return "pass", warnings, failures


def baseline_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_total_loss": metrics.get("final_total_loss"),
        "avg_iter_t": metrics.get("avg_iter_t"),
        "tensor_stats": metrics.get("tensor_stats", {}),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PtyRAD Release Benchmark Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- PtyRAD: {report['ptyrad_version']}",
        f"- Git SHA: {report.get('git_sha')}",
        f"- Overall status: **{report['status'].upper()}**",
        "",
        "| Benchmark | Status | Final loss | Avg iter time | Wall time | Model |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for name, result in report["benchmarks"].items():
        metrics = result.get("metrics", {})
        lines.append(
            "| {name} | {status} | {loss} | {avg_t} | {wall} | {model} |".format(
                name=name,
                status=result["status"],
                loss=_fmt(metrics.get("final_total_loss")),
                avg_t=_fmt(metrics.get("avg_iter_t")),
                wall=_fmt(metrics.get("total_wall_s")),
                model=metrics.get("model_path", ""),
            )
        )
    lines.append("")
    for name, result in report["benchmarks"].items():
        messages = result.get("warnings", []) + result.get("failures", [])
        if messages:
            lines.append(f"## {name}")
            lines.extend(f"- {message}" for message in messages)
            lines.append("")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.6g}"
    return ""


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    baseline_doc = load_json(args.baseline)
    policy = {**default_policy(), **baseline_doc.get("policy", {})}
    stored_benchmarks = baseline_doc.setdefault("benchmarks", {})

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ptyrad_version": ptyrad_version,
        "git_sha": git_sha(),
        "device": device_info(),
        "policy": policy,
        "benchmarks": {},
        "status": "pass",
    }

    status_rank = {"pass": 0, "warn": 1, "fail": 2}

    for demo in args.benchmarks:
        source_path = DEMO_PARAMS[demo]
        wall_s = math.nan
        run_returncode = 0

        if not args.skip_run:
            run_returncode, wall_s = run_benchmark(demo, source_path, args)

        model_path = newest_model_file(demo, args)
        failures: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {
            "params_path": str(source_path),
            "total_wall_s": float(wall_s),
        }

        if run_returncode != 0:
            failures.append(f"benchmark command failed with exit code {run_returncode}")
        if model_path is None:
            failures.append(f"missing model_iter{args.n_iter:04d}.hdf5 under {args.output_root / demo}")
        else:
            try:
                metrics, structural_problems = collect_metrics(model_path, wall_s)
                metrics["params_path"] = str(source_path)
                failures.extend(structural_problems)
            except Exception as exc:
                failures.append(f"failed to read model file: {exc}")

        if failures:
            status = "fail"
        else:
            status, warnings, baseline_failures = compare_to_baseline(
                demo,
                metrics,
                stored_benchmarks.get(demo),
                policy,
                strict_missing=args.strict_baseline,
            )
            failures.extend(baseline_failures)

        if args.update_baseline and status != "fail":
            stored_benchmarks[demo] = baseline_payload(metrics)

        report["benchmarks"][demo] = {
            "status": status,
            "warnings": warnings,
            "failures": failures,
            "metrics": metrics,
        }
        if status_rank[status] > status_rank[report["status"]]:
            report["status"] = status

    if args.update_baseline:
        baseline_doc["policy"] = policy
        write_json(args.baseline, baseline_doc)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json = args.output_root / f"release_benchmark_report_{timestamp}.json"
    report_md = args.output_root / f"release_benchmark_report_{timestamp}.md"
    write_json(report_json, report)
    report_md.write_text(render_markdown(report), encoding="utf-8")
    write_json(args.output_root / "latest_release_benchmark_report.json", report)
    (args.output_root / "latest_release_benchmark_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )

    print(f"Report written to: {report_json}")
    print(f"Overall status: {report['status'].upper()}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())

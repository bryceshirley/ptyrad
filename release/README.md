# PtyRAD Release Directory

## Start here

Two files cover the full release process:

| File | Purpose |
|------|---------|
| **`README.md`** (this file) | Quick orientation: what each script does and the two most common commands |
| **[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md)** | Step-by-step runbook: work through this top-to-bottom before and during every release |

If you are about to cut a release, open `RELEASE_CHECKLIST.md`. If you just want to know what a script does or run a one-off check, stay here.

## Scripts at a glance

| Script | What it does |
|--------|-------------|
| `validate_all_params.py` | Validates every starter params file against the Pydantic schema |
| `verify_ptyrad_init.py` | Smoke-tests `ptyrad init` output — confirms package structure and version string |
| `sync_notebooks.py` | Syncs source notebooks → packaged starter copies (cleared outputs) and docs copies |
| `sync_params.py` | Syncs example, walkthrough, and template params → docs |
| `sync_all.py` | Runs both sync scripts in one go |
| `run_all_walkthrough.py` | Runs every walkthrough params file at a configurable iteration count |
| `run_integration_matrix.sh` | Full local integration matrix across dataset/config combinations |
| `run_release_benchmarks.py` | Runs demo benchmarks and compares against stored baselines |
| `test_pypi_packaging.sh` | Builds wheel + sdist, installs in fresh env, smoke-tests CLI and init |
| `test_conda_packaging.sh` | Local conda build smoke test (fixture only — real release is handled by conda-forge bot) |

## Most common commands

Fast CI gate (run before any PR merge):

```bash
python -m pytest tests/interface
python release/validate_all_params.py
```

Local release certification (requires demo data and GPU):

```bash
python release/run_release_benchmarks.py --benchmarks tBL_WSe2 PSO --gpuid 0
```

Refresh baselines after a known-good run:

```bash
python release/run_release_benchmarks.py --benchmarks tBL_WSe2 PSO --gpuid 0 --update-baseline
```

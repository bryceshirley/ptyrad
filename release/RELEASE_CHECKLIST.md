# PtyRAD Release Checklist

This is the repo-local release runbook. It is synthesized from the current
`tests/interface` smoke tests, `release/` validation scripts, packaging checks,
and the older Obsidian release checklist.

Use `X.Y.Z` below for the target version, for example `1.0.0` or
`0.1.0b13.post4`.

## 0. Release Setup

- [ ] Start from the intended release branch, usually `dev` before merging to `main`.
- [ ] Confirm the worktree and staged files are intentional:

```bash
git status --short --branch
```

- [ ] Activate the project Conda environment:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ptyrad
```

- [ ] Install the package locally with test and packaging helpers:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

- [ ] Confirm the CLI resolves to this checkout:

```bash
python -m ptyrad --help
ptyrad --help
```

Automation to consider:

- [ ] Add a single `release/preflight.sh` or `python -m release.preflight` command that checks branch, clean/staged state, Conda env, import path, and CLI path before running expensive checks.

## 1. Version And Metadata

- [ ] Decide the exact PEP 440 version string.
- [ ] Update `src/ptyrad/__init__.py` (**only file that needs a version change** — `pyproject.toml` reads it dynamically via `dynamic = ["version"]`):

```python
__version__ = "X.Y.Z" # YYYY.MM.DD
```

- [ ] Update `recipe/meta.yaml` to the same version **for the local conda build smoke test only** (the real conda-forge release is handled by the bot after PyPI is published).
- [ ] Verify `src/ptyrad/__init__.py`, `recipe/meta.yaml`, and the planned Git tag all match.
- [ ] For `1.0.0`, audit `pyproject.toml` classifiers, especially `Development Status :: 4 - Beta`.
- [ ] Recheck `pyproject.toml`:
  - [ ] `requires-python = ">=3.10"` is still correct.
  - [ ] `dependencies` includes all runtime dependencies and no stale ones.
  - [ ] `[project.scripts]` still has `ptyrad = "ptyrad.cli.entry:main"`.
  - [ ] `tool.setuptools.package-data` still packages `ptyrad.starter`.

Automation to consider:

- [ ] Add a version consistency check that fails if `__version__`, `recipe/meta.yaml`, and the latest intended tag disagree.
- [ ] Make `recipe/meta.yaml` read the package version automatically, or generate it from one source of truth.

## 2. Changelog And User-Facing Docs

- [ ] Add a new section to `CHANGELOG.md`:

```markdown
## [X.Y.Z] - YYYY-MM-DD
```

- [ ] Fill in Added / Changed / Fixed / Removed items as appropriate.
- [ ] Mirror the same changelog update to `docs/changelog.md`.
- [ ] Verify the Changelog URL in `pyproject.toml` (`https://github.com/chiahao3/ptyrad/blob/main/CHANGELOG.md`) still resolves.
- [ ] Review `README.md` for stale installation, dependency, CLI, badge, paper, Zenodo, and docs links.
- [ ] Review docs affected by the release:
  - [ ] `docs/installation.md`
  - [ ] `docs/quickstart.md`
  - [ ] `docs/params_overview.md` if parameters changed.
  - [ ] `docs/index.md` if pages were added or removed.

Automation to consider:

- [ ] Generate `docs/changelog.md` from `CHANGELOG.md` instead of manually mirroring it.
- [ ] Add a lightweight link checker for README and docs release URLs.

## 3. Sync Generated Release Assets

Run this section whenever notebooks, starter params, examples, templates, or docs copies changed.

- [ ] Sync notebooks from `notebooks/` to docs and package copies:

```bash
python release/sync_notebooks.py
```

- [ ] Sync example, walkthrough, and template params to docs:

```bash
python release/sync_params.py
```

- [ ] Or run both:

```bash
python release/sync_all.py
```

- [ ] Inspect the diff after sync:

```bash
git diff --stat
git diff -- src/ptyrad/starter docs notebooks
```

- [ ] Confirm packaged notebooks under `src/ptyrad/starter/notebooks/` have cleared outputs.

Automation to consider:

- [ ] Add tests for `release/sync_notebooks.py`, `release/sync_params.py`, and `release/sync_all.py`.
- [ ] Add a CI check that fails when generated docs/package notebook or params copies are out of sync.

## 4. Fast Interface Test Gate

These are the CI-facing checks in `.github/workflows/interface_tests.yml`.

- [ ] Run all interface tests:

```bash
python -m pytest tests/interface
```

- [ ] Run the starter parameter schema gate directly:

```bash
python release/validate_all_params.py
```

- [ ] Confirm CLI validation succeeds on at least one starter params file:

```bash
python -m ptyrad validate-params src/ptyrad/starter/params/templates/minimal.yaml
```

- [ ] Confirm starter export commands work from the installed package (`init` takes a folder **name**, `get-templates` takes a destination **directory**):

```bash
python -m ptyrad init release_init_smoke
python -m ptyrad get-templates release_template_smoke
```

- [ ] Remove any temporary smoke output after inspection:

```bash
rm -rf release_init_smoke release_template_smoke
```

Automation to consider:

- [ ] Put the direct CLI smoke commands into `tests/interface` so CI covers them without a manual checklist step.
- [ ] Add a matrix for Python 3.10, 3.11, and 3.12 if release support needs explicit coverage.

## 5. Docs Build

- [ ] Build the docs locally:

```bash
cd docs
make clean html
cd ..
```

- [ ] Review warnings. Treat broken autosummary, missing files, broken references, and import errors as release blockers.
- [ ] Open `docs/_build/html/index.html` and spot-check:
  - [ ] Installation
  - [ ] Quickstart
  - [ ] Workflow notebooks
  - [ ] Examples
  - [ ] Walkthrough
  - [ ] API pages
  - [ ] Changelog

Automation to consider:

- [ ] Add a docs-build GitHub Action on pull requests and release branches.
- [ ] Add a docs warning budget of zero for release builds.

## 6. Notebook Validation

- [ ] Run the six source notebooks top-to-bottom in the `ptyrad` Conda env:
  - [ ] `notebooks/run_ptyrad.ipynb`
  - [ ] `notebooks/read_ptyrad_output_hdf5.ipynb`
  - [ ] `notebooks/get_error_distribution.ipynb`
  - [ ] `notebooks/get_local_obj_tilts.ipynb`
  - [ ] `notebooks/get_affine_from_image.ipynb`
  - [ ] `notebooks/get_reconstruction_provenance.ipynb`
- [ ] Re-run `python release/sync_notebooks.py` after notebook edits.
- [ ] Confirm package copies in `src/ptyrad/starter/notebooks/` are output-free.

Automation to consider:

- [ ] Add an optional `nbmake` or `papermill` notebook test job that can run locally or on scheduled CI with demo data.
- [ ] Add a notebook output-stripping check for packaged starter notebooks.

## 7. Local Walkthrough And Integration Checks

Quick walkthrough smoke:

- [ ] Run all walkthrough params at a short iteration count:

```bash
python release/run_all_walkthrough.py --n_iter 1 --gpuid 0
```

- [ ] If GPU is unavailable, run the same command with CPU:

```bash
python release/run_all_walkthrough.py --n_iter 1 --gpuid cpu
```

Full local integration matrix:

- [ ] Run the legacy matrix only on hardware with demo data and suitable GPUs:

```bash
bash release/run_integration_matrix.sh
```

- [ ] Review logs under `output/test_logs_*`.
- [ ] If failures occur, preserve the failing log path in release notes or the issue/PR.

Automation to consider:

- [ ] Add named quick/full modes to `release/run_integration_matrix.sh` so short smoke checks do not require editing script arrays.
- [ ] Emit a machine-readable JSON summary from the integration matrix.
- [ ] Add scheduled GPU CI for a short single-GPU reconstruction smoke.

## 8. Release Benchmarks

These are local certification checks using real demo datasets and available hardware.

- [ ] Run both demo benchmarks:

```bash
python release/run_release_benchmarks.py --benchmarks tBL_WSe2 PSO --gpuid 0
```

- [ ] Review:

```bash
output/release_benchmarks/latest_release_benchmark_report.md
output/release_benchmarks/latest_release_benchmark_report.json
```

- [ ] Confirm the report status is `PASS` or an understood `WARN`.
- [ ] Confirm critical tensors `objp`, `obja`, and `probe` are present and finite.
- [ ] Confirm final loss, tensor stats, and average iteration time are acceptable relative to `release/baselines/v1_0_0_demo_benchmarks.json`.
- [ ] After reviewing a known-good release run, refresh baselines explicitly:

```bash
python release/run_release_benchmarks.py --benchmarks tBL_WSe2 PSO --gpuid 0 --update-baseline
```

- [ ] Commit baseline changes only when the drift is intentional and explained.

Automation to consider:

- [ ] Make benchmark reports publishable as GitHub Action artifacts for GPU runners.
- [ ] Store hardware/device metadata in a comparable history file so timing changes are easier to interpret.
- [ ] Split quality regression thresholds from timing regression thresholds per benchmark.

## 9. PyPI Packaging Preflight

- [ ] Remove old artifacts or let the script do it.
- [ ] Run the PyPI packaging smoke test:

```bash
bash release/test_pypi_packaging.sh
```

- [ ] Confirm the script:
  - [ ] Builds wheel and sdist with `python -m build`.
  - [ ] Installs the wheel in fresh `test_env_pypi`.
  - [ ] Runs `ptyrad --help`.
  - [ ] Runs `python -m ptyrad --help`.
  - [ ] Runs `python release/verify_ptyrad_init.py test_init`.
  - [ ] Removes `test_env_pypi`.

- [ ] Run `twine check dist/*` separately (the script does not run it):

```bash
pip install twine  # if not installed
twine check dist/*
```

- [ ] If the script exits early, clean up manually if needed:

```bash
conda env remove -n test_env_pypi
```

- [ ] Confirm `.github/workflows/publish_pypi.yml` is triggered by the GitHub `release` event (not by tag push alone).

Automation to consider:

- [ ] Make `release/test_pypi_packaging.sh` use a unique temp env name and a cleanup trap.
- [ ] Add `twine check dist/*` output capture to a release artifact.

## 10. Conda Packaging Preflight

This is a local conda-build smoke test. The real conda-forge release should be handled by the conda-forge bot after PyPI is published.

> **Note:** `recipe/meta.yaml` is a **local testing fixture only** — it points at the local source tree (`path: ..`), not PyPI. The actual conda-forge release is handled automatically: after the PyPI tag lands, the conda-forge bot opens a PR on the remote feedstock to update the version and SHA256.

- [ ] Confirm `recipe/meta.yaml` version matches `src/ptyrad/__init__.py`.
- [ ] Confirm the recipe entry point matches `pyproject.toml`:

```yaml
ptyrad = ptyrad.cli.entry:main
```

- [ ] Run the local conda build and init smoke test:

```bash
bash release/test_conda_packaging.sh
```

- [ ] Confirm the script:
  - [ ] Runs `conda build recipe/`.
  - [ ] Installs local `ptyrad` into fresh `test_env_conda`.
  - [ ] Runs `python release/verify_ptyrad_init.py test_init`.
  - [ ] Removes `test_env_conda`.

- [ ] If the script exits early, clean up manually if needed:

```bash
conda env remove -n test_env_conda
```

Automation to consider:

- [ ] Make `release/test_conda_packaging.sh` use a unique temp env name and a cleanup trap.
- [ ] Add a generated dependency diff between `pyproject.toml` and `recipe/meta.yaml`.

## 11. Cross-Platform Checks

- [ ] Linux with CUDA: run interface tests, walkthrough smoke, benchmarks, PyPI packaging, and conda packaging.
- [ ] Linux CPU-only: run interface tests and at least one short reconstruction smoke with `--gpuid cpu`.
- [ ] macOS Apple Silicon / MPS: run interface tests, CLI smoke, `ptyrad init`, and a short reconstruction smoke.
- [ ] Windows: run interface tests, CLI smoke, `ptyrad init`, and a short reconstruction smoke.
- [ ] For multi-GPU Linux, run the integration matrix path that uses `accelerate launch --multi_gpu`.

Automation to consider:

- [ ] Add OS matrix CI for interface tests.
- [ ] Add a manual GPU workflow for short Linux CUDA smoke tests.
- [ ] Add a documented self-hosted runner path for multi-GPU checks.

## 12. Final Diff Review

- [ ] Review the final diff:

```bash
git diff --stat
git diff
```

- [ ] Confirm generated output is not accidentally staged:
  - [ ] `dist/`
  - [ ] `build/`
  - [ ] `*.egg-info`
  - [ ] `output/`
  - [ ] docs build output unless intentionally tracked.
- [ ] Stage only intended release files.
- [ ] Re-run `git status --short --branch`.

Automation to consider:

- [ ] Add a release artifact ignore audit that fails on accidental `dist/`, `build/`, `output/`, or temp env files.

## 13. Release Commit, Tag, And GitHub Release

- [ ] Commit release changes:

```bash
git add src/ptyrad/__init__.py recipe/meta.yaml CHANGELOG.md docs/changelog.md
git add README.md docs/installation.md docs/quickstart.md docs/params_overview.md docs/index.md  # if changed
git add release/baselines/v1_0_0_demo_benchmarks.json  # only if intentionally refreshed
git commit -m "Version to X.Y.Z"
```

- [ ] Merge or fast-forward to `main` according to the project flow.
- [ ] Create an annotated tag:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
```

- [ ] Push the release commit and tag:

```bash
git push origin main
git push origin vX.Y.Z
```

- [ ] Create or publish the GitHub Release for `vX.Y.Z`.
- [ ] Confirm publishing the GitHub Release triggers `.github/workflows/publish_pypi.yml`.

Automation to consider:

- [ ] Add a release script that refuses to tag unless version, changelog, benchmark report, and packaging checks are present.
- [ ] Generate GitHub Release notes from the changelog section.

## 14. Post-Release Verification

- [ ] Watch the PyPI publish workflow and confirm it succeeds.
- [ ] Verify the PyPI project page: `https://pypi.org/project/ptyrad/`.
- [ ] Install from PyPI in a fresh env:

```bash
conda create -n ptyrad_release_verify python=3.12 -y
conda activate ptyrad_release_verify
pip install ptyrad==X.Y.Z
ptyrad --help
python -m ptyrad --help
python -c "import ptyrad; print(ptyrad.__version__)"
ptyrad init output/release_verify_init
```

- [ ] Clean up:

```bash
conda deactivate
conda env remove -n ptyrad_release_verify
```

- [ ] Watch for the conda-forge bot PR on the feedstock.
- [ ] Review and merge the feedstock PR after checks pass.
- [ ] After conda-forge publishes, test:

```bash
conda create -n ptyrad_conda_verify python=3.12 ptyrad -c conda-forge -y
conda activate ptyrad_conda_verify
ptyrad --help
python -m ptyrad --help
python -c "import ptyrad; print(ptyrad.__version__)"
```

- [ ] Clean up:

```bash
conda deactivate
conda env remove -n ptyrad_conda_verify
```

- [ ] Confirm Read the Docs or docs hosting reflects the release if applicable.
- [ ] Confirm Zenodo archival if configured.

Automation to consider:

- [ ] Add a post-release verification script that installs from PyPI and conda-forge in disposable envs.
- [ ] Add a release dashboard issue template with PyPI, conda-forge, docs, and Zenodo checkboxes.

## 15. Hotfix Protocol

- [ ] Branch from the released tag or current stable branch.
- [ ] Make the smallest targeted fix.
- [ ] Add or update a regression test.
- [ ] Bump version:
  - [ ] Beta hotfix: `X.Y.ZbN.postM`
  - [ ] Stable bugfix: `X.Y.(Z+1)`
  - [ ] Packaging-only stable fix, if appropriate: `X.Y.Z.postM`
- [ ] Update `CHANGELOG.md` and `docs/changelog.md`.
- [ ] Run the relevant subset:
  - [ ] `python -m pytest tests/interface`
  - [ ] `python release/validate_all_params.py`
  - [ ] `bash release/test_pypi_packaging.sh` for packaging fixes.
  - [ ] `bash release/test_conda_packaging.sh` for conda recipe fixes.
  - [ ] `python release/run_release_benchmarks.py --benchmarks tBL_WSe2 PSO --gpuid 0` if reconstruction behavior changed.
- [ ] Commit, tag, publish a GitHub Release, and verify PyPI/conda-forge as above.

Automation to consider:

- [ ] Add a hotfix checklist template that asks whether the fix affects packaging, params, docs, or reconstruction numerics.

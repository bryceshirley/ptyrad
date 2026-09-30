# generate_data

Simulates synthetic 4D-STEM datasets with [abtem](https://abtem.readthedocs.io)
and packages them as HDF5 files that PtyRAD reconstructs directly. Every
dataset ships with its per-slice ground-truth phase and the exact
reconstruction probe, so reconstructions can be scored against known truth.

Fully part of ptyrad: this package (`src/generate_data`), tests in
`test/generate_data`, PtyRAD config templates in `demo/params/`, and
generated data in `demo/data/generated/` (gitignored).

## Install

The simulation dependencies are the `datagen` extra of ptyrad:

```bash
cd ~/ptyrad
uv pip install -e ".[datagen]" --python .venv/bin/python   # adds abtem + cupy
```

## Producing a dataset

```bash
cd ~/ptyrad
.venv/bin/generate-data list                      # all recipes + descriptions
.venv/bin/generate-data produce smoke_2slab       # ~minutes, good first run
.venv/bin/generate-data produce dense128b --device gpu
```

Output lands in `demo/data/generated/<name>.h5`. Recipes are defined in
`datasets.py` — a new dataset is a new `Recipe` entry (structure builder +
sampling plan + scan geometry + seed); seeds make the Poisson noise
reproducible.

Datasets: `smoke_2slab` (2-slab walkthrough), `sample_a` (Si + 3 Ge
delta-layers), `multi5` (C/Si/SrTiO3/Ge/Au), `planes128` (16 atomic planes /
128 slices), `dense128b` (32 slabs, 12 A pitch), `dense128t` / `dense256t`
(progressively twisted stacks, structured probe), `tbl128`
(reconstruction-derived phantom).

Pre-flight checks:

```bash
.venv/bin/generate-data cbed smoke     # BF-disk radius/centering vs plan (cheap)
.venv/bin/generate-data gate d3 2 2 15 115 225   # GT-only export for design iteration
```

GPU note (this box): abtem/cupy sees only two CUDA devices — pin by UUID,
e.g. `CUDA_VISIBLE_DEVICES=GPU-b0d04bcb-6a95-8ece-f114-ff510679a5a8`.
GPUs 0/3 are MIG-disabled; an out-of-range integer silently falls back to
CPU (~40x slower).

## Data contract (what PtyRAD reads)

`generate_data.h5io.write_dataset` writes:

| key            | shape / type                    | meaning                                   |
|----------------|---------------------------------|-------------------------------------------|
| `dp`           | `(N, ky, kx)` float32           | Poisson-noisy patterns, PtyRAD raster order (slow=y outer, fast=x inner) |
| `dp_noiseless` | same                            | pre-noise patterns (forward checks)        |
| `gt_phase`     | `(Nlayer, Ny, Nx)` float32      | per-slice ground-truth phase on the recon grid |
| `probe`        | `(Ny, Nx)` complex64            | the exact probe on the recon grid          |

Required attrs: `kv`, `semiangle_mrad`, `da_mrad`, `step_A`, `dz_A`,
`thickness_A`, `electrons_per_pattern`, `dx_A`. The smoke dataset adds
`scan_pos_yx_A`, `gt_phase_fine` (2 A slicing) and `gt_slice_thicknesses_A`
for the forward-consistency check.

## Reconstructing in PtyRAD

The dedicated smoke/dense128b config templates were removed in the
2026-09-30 params cleanup (git history, commit 03e0acc, has them). Adapt
one of the kept configs instead — `demo/params/pso_born_grow.yml` (Born +
adaptive order growth, the right starting point for generated deep
stacks) or `demo/params/pso_ms_b1_n100.yml` (multislice) — using the
attrs-to-field mapping below.

Launch (from `~/ptyrad/demo`):

```bash
PTYRAD_DISABLE_ISS_PRECOND=1 \
CUDA_VISIBLE_DEVICES=GPU-b0d04bcb-6a95-8ece-f114-ff510679a5a8 \
../.venv/bin/python -m ptyrad run --params_path params/<your_config>.yml
```

Adapting a config to another dataset — every needed number is in the h5
attrs:

| params field                      | comes from                              |
|-----------------------------------|-----------------------------------------|
| `meas_params.path` (+`key: 'dp'`)  | the `demo/data/generated/<name>.h5` you produced |
| `probe_kv`, `probe_conv_angle`    | attrs `kv`, `semiangle_mrad`             |
| `meas_calibration {mode: da}`     | attr `da_mrad` — use the EXACT value     |
| `pos_N_scan_slow/fast`, `pos_scan_step_size` | scan geometry / attr `step_A` |
| `meas_Npix`                       | pattern size (`dp.shape[-1]`)            |
| `obj_Nlayer`, `obj_slice_thickness` | attrs `thickness_A` / `dz_A` (entrance-transmit convention) |
| `meas_flipT: null`                | verified: identity orientation wins      |

Box/engine gotchas (hard-won — respect these):

- **`PTYRAD_DISABLE_ISS_PRECOND=1` is mandatory for every batch-1 Born run**
  (the ISS illumination preconditioner destroys entrance slices at batch 1).
  `demo/params/ptyrad_iss_precond_env_gate.patch` records the env gate.
- Pin the GPU by UUID (same MIG issue as generation).
- Reruns of an unchanged config resolve to the **same output folder and
  silently overwrite checkpoints** — give reruns a distinct `prefix`.
- `GROUP_MODE: 'sparse'` crashes at batch size 1 — use `'random'`.
- Do not add `hypertune_params` to configs on this box.

## Validating the round trip

After producing `smoke_2slab`, confirm PtyRAD's own forward model reproduces
the abtem patterns from the exported ground truth:

```bash
cd ~/ptyrad
.venv/bin/python -m generate_data.check_ptyrad_forward demo/data/generated/smoke_2slab.h5
```

It scores all 8 dihedral orientations and reports the residual; identity must
win. abtem (Fresnel + antialias) vs PtyRAD (ASM) is expected to match
imperfectly but closely; a large residual means a convention broke.

## Tests

```bash
cd ~/ptyrad
.venv/bin/pytest test/generate_data
```

The h5-contract and flux tests validate any datasets present in
`demo/data/generated/`; they skip when no datasets have been produced.

# PtyRAD Release Validation

Fast CI-facing checks live under `tests/interface`:

```bash
python -m pytest tests/interface
```

Local release certification uses the real demo datasets and available hardware:

```bash
python release/run_release_benchmarks.py --benchmarks tBL_WSe2 PSO --gpuid 0
```

After reviewing a known-good run, refresh the stored baselines explicitly:

```bash
python release/run_release_benchmarks.py --benchmarks tBL_WSe2 PSO --gpuid 0 --update-baseline
```

Packaging, sync helpers, and long-running pre-release utilities are kept here;
`tools/` contains compatibility wrappers for the previous paths.

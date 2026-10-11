# PSO Fig-S5 NVLink campaign — operator prompt (current as of 2026-10-11)

**Mission:** benchmark the PSO reconstruction at the paper's TRUE Fig-S5
workload on rented NVLink A100 SXM GPUs — fixed Born order M=6 and M=8 at
1 / 2 / 4 GPUs versus multislice on the SAME hardware — and characterise
2-vs-4 GPU scaling. Paper context (not the comparison): ~45 min/iter on a
Quadro P5000 (Chen et al. Science 2021 SM, p.~230). The comparison that
matters is Born-multiGPU vs MS on the same A100s.

The workload (verified against the SM, `demo/data/PSO/abg2533_chen_sm.pdf`):
256×256 detector, 128×128 = 16384 diffraction patterns, 12 probe modes,
**42 slices of dz = 5 Å** (sample is ~21 nm / 210 Å; an earlier "dz 10 Å"
note was arithmetically wrong). Fixed M, grow OFF (constant per-iter cost
for clean scaling; grow confounds timing and §K.4 showed its M=1 first
epoch imprints weak contrast).

## State: PREP COMPLETE — blocked ONLY on RunPod balance

Everything is committed and pushed on branch `nvlink-bench` (through
`adbf1fb`). The RunPod account balance was **−$0.03** on 2026-10-11; the
deploy script refuses to launch below budget (default $35). Once the
account is topped up, the whole campaign is turnkey.

Already done (do NOT redo):
- Configs `demo/params/nvl42_{m6,m8,ms}.yml` — full workload, NITER 2,
  validated via load_params. `redo_smoke_m6.yml` = 1-iter 256-position
  smoke.
- The 16384-DP dataset: the distributed .mat holds only the 64×64 scan01
  subset (4096 DPs); the full-geometry file is built by 2×2 raster tiling
  (`python -m ptyrad.utils.tile_scan`, tested in `test/test_tile_scan.py`).
  It is built ON THE POD (4.3 GB, ~20 s); never upload or commit it.
  Workload-exact for timing, not physics (object = 2×2-periodic PSO).
- All four E2E arm types smoke-validated locally against the worktree src:
  baseline / slicedist P=2 / P=4 / modesplit P=4 all produce IDENTICAL
  loss 0.6232 on the smoke config.
- src additions (tested, CPU suites green, 149 passed):
  `born_forward_dist(carry="chain"|"direct")`, `make_modesplit_forward`
  (P-way, `MODE_P` env in the driver), `depth_bounds`,
  `ptyrad.utils.tile_scan`.
- Local PCIe results already recorded in `demo/bench_results.md` §M:
  chain ≥ direct at P=4 on no-P2P PCIe; 12 probe modes rescue slice-split
  (slice-4 f+b at Nz=42: 1.96× M=6 / 1.72× M=8 / 1.48× M=1).

## Run sequence (local side: `~/ptyrad-nvlink/demo/nvlink/redo/`)

```bash
cd ~/ptyrad-nvlink/demo/nvlink/redo
bash deploy_redo.sh                # 4x A100 SXM; ABORTS if balance < $35
bash watchdog_redo.sh 2 6 &        # kill-switch FIRST: min $2 bal, 6 h deadline
bash sync_back.sh &                # incremental copy-back every 120 s
bash push_and_run.sh               # clone+install on pod, upload .mat (287 MB
                                   # @ ~0.5 MB/s ≈ 10 min), build tiled file,
                                   # launch campaign detached
```

RunPod API key: `~/.runpod_api_key` (REST rest.runpod.io/v1 + GraphQL;
helpers in `redo/runpod_api.sh`). Pod state lands in `redo/pod_state.env`.
Tail progress: `ssh -p $POD_PORT root@$POD_IP tail -f
/workspace/ptyrad/demo/nvlink/campaign.log`, or watch
`runpod_results/redo/` fill locally.

The pod-side campaign (`redo/run_redo_bench.sh`, launched by push_and_run)
runs, cheap-first, each arm under `timeout`:
1. Micro benches at the Fig-S5 shape (PMODE=12): `bench_carry_direct.py`
   (chain-vs-direct ON NVSwitch, M=6/8 × P=2/4), `bench_slice_nzsweep.py`,
   `bench_ms_vs_born_b1.py`.
2. E2E s/iter arms (NITER 2, per-arm param copies with distinct prefixes):
   ms_1gpu, m6_slice4, m6_mode4, m6_1gpu, m6_slice2, m8_slice4, m8_1gpu,
   m8_mode4.

Budget: one 4×A100 SXM pod (~$6.4/h) covers 1/2/4-GPU scaling; expect
~4–5 h ≈ $26–32 total. Rough per-iter forecasts from local A100 per-batch
timings (53.9/71.0 ms f+b per batch at M=6/M=8, 12 modes, Nz=42; MS ~23 ms
pmode-insensitive): born 1-GPU ~16/20 min/iter, MS ~6 min/iter — so MS
beats 1-GPU born at 12 modes and the multi-GPU split IS the story.

## Deliverable

Append **section O** to `demo/bench_results.md`: 42-slice / 12-mode /
16384-DP numbers for fixed M=6 and M=8 — forward, fwd+bw (micro) and
end-to-end s/iter — at 1/2/4 GPUs plus MS on the same hardware; the
`nvidia-smi topo -m` excerpt; the winning parallel axis; an honest verdict
on 2-vs-4 scaling vs same-hardware MS; the NVSwitch chain-vs-direct table
next to §M's PCIe one. Report losses/failures honestly. Commit on
`nvlink-bench` only — single-line message, no body, no attribution — and
NEVER commit .mat/.hdf5 data.

## Interpretation notes

- hyp-4 is REFUTED (§L.6): slice-split's exposed cost is per-hop
  launch/sync latency, not bandwidth — expect diminishing 2→4 slice-split
  returns; 4-GPU gains should come from mode-split (12-way capable) and
  §L.7's inverted ranking (pure mode > combo > pure slice at 4-GPU NVLink).
  Report which axis actually delivers.
- chain-vs-direct caveat: the `direct` carry issues all copies from a
  source on ONE stream, so on NVSwitch it may understate direct; note this
  if direct loses narrowly there.
- Carry codecs are dead on NVLink (sync `.to()` == pinned-half, §L); do
  not re-test.

## Post-mortem rules (both prior rentals had losses — do not repeat)

- The watchdog is NOT optional: round 2 lost the 30-iter artifacts when a
  hung arm (silent 2-GPU distsmoke hang) idled the pod to $0 and RunPod's
  auto-terminate wiped the disk. Independent balance+deadline kill-switch
  + incremental copy-back, both running BEFORE the campaign starts.
- A pod stuck "initializing" > ~12 min never recovers: terminate and
  redeploy (deploy_redo.sh does this itself). Round 1 wasted 32 min.
- stop/start WIPES the container disk when volumeInGb=0 (re-setup ~15 min
  incl. the 287 MB upload).

## Gotchas (standing)

- `PTYRAD_DISABLE_ISS_PRECOND=1` is MANDATORY for every batch-1 Born run
  (runner exports it).
- Everything eager; no torch.compile (multi-device dynamo backward bug;
  eager==compiled for this born anyway).
- Reconstructions seed 42 (the driver sets it) — loss-identity checks
  across arms are meaningless otherwise.
- PtyRAD output folders are date+config named: identical configs SILENTLY
  OVERWRITE. The runner's per-arm `redo_<arm>` prefixes handle this; keep
  the pattern for any extra arm.
- RunPod PyTorch image is PEP-668: `pip install --break-system-packages`
  (push_and_run.sh does).
- `GROUP_MODE: 'sparse'` crashes at batch 1; keep `'random'`.
- Local box (4×A100 PCIe, all-PHB, NO P2P) is for smokes only — the user
  has said benchmarks must NOT run on the local GPUs.

## History

Full record in `demo/bench_results.md`: §L (round 1, 2×SXM NV12),
§L.6–L.7 (round 2, 4×NVSwitch, post-mortem), §M (local chain-vs-direct +
12-mode slice-split rescue), §N (this campaign's prep record). Memory:
`multigpu-born-parallelism-outcome.md`. The pre-redo round-1 prompt this
file replaced is in git history (nvlink-bench, before adbf1fb).

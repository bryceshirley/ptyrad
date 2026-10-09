# NVLink validation of the multi-GPU Born benchmarks (agent/operator prompt)

You are on a rented NVLink GPU box (target: 2-4x A100 SXM4 or H100 SXM, e.g.
RunPod 4x A100 SXM pod, ~$6.4/hr). Your job: re-run the multi-GPU Born-series
benchmarks from `demo/bench_results.md` on the real PSO born6 sample and
measure what NVLink changes versus the original all-PHB PCIe box (KVM guest,
no P2P, 4x A100 80GB PCIe). Work economically -- the full campaign fits in
~2 hours of GPU time; do not hypertune or rerun endlessly.

## What the PCIe box measured (the numbers to beat)

Forward speedup vs 1 GPU (real checkpoint, batch N; `bench_results.md` SSD-SSF):

| N  | slice4 real | slice4 FLOOR (free carry) | mode2 | mode4 | batch4 |
|----|-------------|---------------------------|-------|-------|--------|
| 1  | 0.33-0.35x  | 0.58x                     | 1.78x | 2.08x | N/A    |
| 8  | 0.56x       | 2.97x                     | 1.86x | 3.10x | 2.93x  |
| 32 | 0.56x       | 3.29x                     | 1.92x | 3.40x | 3.65x  |

End-to-end 20-iter reconstruction: baseline 64.5 s/iter; mode-split 2-GPU
67.7 s/iter (0.95x -- SLOWER). Object all-reduce 68 MB: 3.5 ms (2 GPU) /
14 ms (4 GPU), host-staged.

## Hypotheses to test (in priority order)

1. **Slice-split at N>=8 approaches its compute floor** (0.56x -> ~3x at N=32)
   because the carry becomes a direct GPU->GPU copy. Read the `4-GPU sync-c64`
   column of `bench_slice_real.py` -- on NVLink the sync `.to()` path should
   now BEAT `pinned-half` (no host staging, no codec needed).
2. **Batch-1 stays lost** (granularity wall): slice4 N=1 should stay near its
   0.58x free-carry ceiling, NOT reach 1x. If it beats 1x, that is a finding.
3. **Data-parallel all-reduce cost collapses** (14 ms -> ~1-2 ms on 4 GPUs),
   pushing batch4 fwd+bw from 2.67x toward ~3.5x (`bench_axes_real.py`).
4. **Deep stacks (the multislice-comparison regime): batch-1 slice-split wins
   from Nz~48-64 even on PCIe** (`bench_slice_nzsweep.py`, SSG: f+b 1.59x at
   Nz=64 -> 2.35x at Nz=512; floor 3.3x). On NVLink the crossover should move
   down toward Nz~32 and the realized x toward the floor. This is the only
   multi-GPU axis multislice cannot copy (sequential slices), so it is the
   headline number for the Born-vs-multislice benchmark. Run the sweep; time
   4-GPU f+b with PRE-PLACED object blocks only. **Repeat at BORN_M=2 and
   BORN_M=1 (SSG.2): on PCIe the win shrinks with M (f+b @Nz=512: 2.35x M=6,
   1.22x M=2, 1.03x M=1 -- hop latency has no cross-order overlap to hide
   behind at small M), while the floor is M-independent (~3.3x). The NVLink
   prediction to verify: ISS/double jump from ~1x to ~3x.** That would make
   the P-GPU flat-curve extension hold for the cheap headline orders too.
5. **Multislice head-to-head (SSH)**: `bench_ms_vs_born_b1.py` -- verify the
   NVLink box reproduces: born6 4-GPU f+b beats as-implemented MS from
   Nz~24-32, 6.4x at Nz=512. Also test the COMBINED split (SSH.2): P_m mode
   groups x P_z slice blocks (e.g. 2x4 on 8 GPUs) -- mode-split is a second
   DIFFERENTIAL axis vs MS at batch 1 (MS is pmode-insensitive/launch-bound,
   Born is pmode-proportional/compute-bound).
6. **Mode-split end-to-end** moves from 0.95x toward its 1.13x microbench step
   gain (per-batch 10.5 MB patch transfers become cheap). Secondary.

## Setup

```bash
git clone -b nvlink-bench <REPO_URL> ptyrad && cd ptyrad
uv venv .venv && uv pip install -e . torchvision h5py   # or pip
```

Copy the two data files from the home box (NOT in git; verify checksums):

```
scp <home>:~/ptyrad/demo/data/PSO/sample_data_PrScO3.mat demo/data/PSO/   # 287 MB
scp <home>:"~/ptyrad/demo/output/PSO/20260929_pso_born6_qr_msT_noreg_full_N4096_dp256_random1_p4_1obj_21slice_dz10_plr1e-4_oalr5e-4_oplr5e-4_dpblur1.0_orblur0.4_ozblur1.0_mamp0.03_4.0_oathr0.96_oposc_sng1.0_spr0.1/model_iter0100.hdf5" \
    "demo/output/PSO/<same folder name>/"                                  # 69 MB
sha256sum:
  e5c470797e90a21381b9fdb0a353493f3ead96fcafdd4b9a163bb110059e472a  sample_data_PrScO3.mat
  44e8cd4b9991f43680138ee87ad0eb64c2a96b83bff98ca5f28dca6cc86c235b  model_iter0100.hdf5
```

(`PSO_CKPT=<path>` overrides the checkpoint location for the bench scripts.)

## Run

```bash
bash demo/nvlink/run_nvlink_bench.sh            # full, ~2 h incl. recons
bash demo/nvlink/run_nvlink_bench.sh --skip-recon   # benches only, ~20 min
```

The script aborts if `nvidia-smi topo -m` shows no NV* links, if GPUs are
busy, or if the checkpoint is missing. Results land in
`demo/nvlink/results_<timestamp>/`.

## Success criteria / what to record

- Correctness FIRST: forward rel err ~1e-6 (sync/c64) and object-grad rel err
  ~2e-4 must reproduce before any timing is trusted.
- Append a "SSG. NVLink box" section to `demo/bench_results.md` mirroring the
  SSD/SSE tables, with the box spec (`nvidia-smi topo -m` excerpt) and the
  four hypothesis verdicts. Report losses honestly -- if NVLink does NOT
  rescue a case, that is the result.
- Commit on this branch only (single-line commit message, no attribution
  lines, no bodies). Do NOT push to any public remote without the owner's
  explicit OK, and never commit the .mat/.hdf5 data files.

## Gotchas (learned on the home box, they still apply)

- `PTYRAD_DISABLE_ISS_PRECOND=1` is MANDATORY for every batch-1 Born run
  (the runner exports it).
- Everything runs eager; do not add torch.compile (compiled backward has a
  multi-device dynamo bug, and eager==compiled for this born anyway).
- Reconstructions use seed 42 -- both arms must keep it or the loss-identity
  check (identical to 4 digits every iter) is meaningless.
- PtyRAD output folders are named date+config: rerunning an unchanged config
  SILENTLY OVERWRITES its checkpoints. Give reruns a distinct `prefix`.
- The `quarter` (fp8) carry codec is KNOWN-BROKEN on the real object (single
  global scale, rel err up to 1.9). It is in the sweep for completeness; do
  not "fix" it unless testing per-block scaling explicitly.
- `GROUP_MODE: 'sparse'` crashes at batch 1; keep `'random'`.

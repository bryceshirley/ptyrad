# NVLink campaign context (rewritten 2026-10-11 — supersedes the round-1 plan)

Single source of truth for a fresh session: **read NVLINK_PROMPT.md** (same
directory) — it holds the mission, the current state, the exact run
sequence, the deliverable spec, and the post-mortem rules. This file only
carries the orientation facts that don't fit there.

## Where things live

- **Branch/worktree:** all campaign work is on `nvlink-bench`, checked out
  at `~/ptyrad-nvlink` (pushed through `adbf1fb`; pod clones
  `git clone -b nvlink-bench https://github.com/bryceshirley/ptyrad`).
  The main checkout `~/ptyrad` is on `linesearch-born` — do NOT commit
  campaign work there. Run everything with the worktree's src
  (`PYTHONPATH=~/ptyrad-nvlink/src` when using the main venv
  `~/ptyrad/.venv`; verify `ptyrad.__file__` resolves to the worktree —
  a smoke once silently imported the main checkout instead).
- **Local-side scripts:** `~/ptyrad-nvlink/demo/nvlink/redo/`
  (deploy / watchdog / sync_back / push_and_run + runpod_api.sh helpers).
- **Pod-side campaign:** `demo/nvlink/redo/run_redo_bench.sh` (in the repo,
  so the pod gets it via clone).
- **Data:** `~/ptyrad/demo/data/PSO/sample_data_PrScO3.mat` (287 MB,
  sha256 e5c47079…; worktree symlinks to it). The 4.3 GB tiled
  16384-DP file is REBUILT on the pod — never uploaded, never committed.
- **Results so far:** `demo/bench_results.md` §A–§N;
  prior rental artifacts in `demo/nvlink/runpod_results/` (main checkout).
- **Local GPUs:** 4×A100-80GB PCIe, all-PHB, NO NVLink/P2P — smokes only,
  no benchmarks (owner instruction). Pin by UUID (see the `ptyrad-box`
  skill; integer CUDA_VISIBLE_DEVICES can silently fall back to CPU).

## Decisions already made (don't relitigate)

- Fixed M=6 and M=8, grow OFF, for constant per-iter cost (§K.4/§L.6
  background: cap-8 beat cap-6; grow's M=1 first epoch hurts contrast).
- dz = 5 Å × 42 slices (= the paper's ~21 nm sample). The "dz 10 Å" note
  in older memos was wrong; legacy configs were fixed accordingly.
- One 4×A100 SXM pod serves 1/2/4-GPU scaling (NVSwitch all-to-all);
  no separate 2× rental.
- Chain carry is the default; `direct` is re-tested on NVSwitch only
  (campaign step 1a) with the one-stream caveat noted in the prompt.
- MS baseline runs through the same driver (`run_cmp_modesplit.py
  baseline` with the nvl42_ms-derived config), same seed 42, same host.

## Open items for the rental session

1. Top up RunPod (deploy gate wants ≥ $35; est. spend $26–32).
2. Run the sequence in NVLINK_PROMPT.md; babysit via sync_back +
   campaign.log; the watchdog owns termination.
3. Write up §O in bench_results.md (spec in the prompt), commit on
   nvlink-bench (single-line, no attribution), push.
4. Update memory `multigpu-born-parallelism-outcome.md` with the verdicts
   (it currently records "prep complete, blocked on balance").
5. If credit allows after the main table: optional NVSwitch rerun of
   bench_carry_direct with per-pair side streams if direct lost narrowly.

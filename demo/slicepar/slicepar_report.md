# Carry schemes and launch order for the slice-parallel Born forward model

Branch `slicepar-carry` (worktree `~/ptyrad-slicepar`, forked from
`linesearch-born` @ 7282053). All new behaviour is behind
`carry_scheme=` / `carry_transport=` / `skip_zero_planes=` kwargs of
`born_forward_dist`; defaults reproduce the stock path bitwise (verified by
digest). Date: 2026-10-10.

**Constraint:** only 2 of the 4 GPUs were available (the other two were
reserved for another session), so every measurement is P ∈ {1, 2} on GPUs
0–1. P = 4 statements are model extrapolations or logical-placement
correctness runs (4 blocks mapped onto 2 devices). The NVLink machine (the
RunPod rental of the earlier campaign) is gone; this is the PCIe box only.

## TL;DR

* **H1 refuted.** The engine already enqueues all of phase 1 before the
  first carry; `chain_reordered` ≡ `chain` (bitwise and in time). There is
  no synchronous `.to()` in the engine.
* **H2 confirmed at P=2** (direct ≡ chain, bitwise); at P=4 the model says
  the hypothesis' premise is wrong *on this box*: a hop costs α ≈ 2.05 ms
  (driver-staged no-P2P peer copy), so even `direct` would break even only
  at Nz ≈ 300 (n=256). The dominant fix is the **transport**, not the carry
  topology: a pinned-bounce hop costs α ≈ 0.034 ms (60× less).
* **H3 refuted everywhere measurable**: `transpose` is 1.3–2.9× slower than
  `chain` in every cell (its window N ≲ P²/2 = 2 is empty at P = 2, and the
  256² sub-claim "never wins" holds).
* The real findings: (a) the **blocking probe-replica copy** in the
  `slicedist` caller was costing ~2.5 ms × 2 per step — routing it through
  the non-blocking hop cut the P=2 step 3.4× at n=256, Nz=128 in the
  engine harness (at PSO scale, Nz=21/M=6, the real solver stayed at its
  usual ~15 ms/batch — there the tax hides inside the pipeline); (b) with
  that fixed, **P=2 slice-split wins from Nz ≈ 8–16 at n ≥ 1024 and from
  Nz ≈ 128 (pmode 1) / Nz ≈ 32 (pmode 4) at n = 256**, plateauing at
  1.7–1.9×; (c) P=2 also lifts the single-GPU OOM ceiling (n=2048, Nz=256
  step fits at P=2, OOMs at P=1).

## Step 0 — Environment

| item | value |
|---|---|
| GPUs | 4 × NVIDIA A100 80GB PCIe (driver 580.178.04); **2 visible/used** (UUIDs `GPU-03a8aae1…`, `GPU-5c1cb7cd…`) |
| Topology | all pairs PHB (PCIe through host bridge), KVM guest; `nvidia-smi topo -m` below |
| Peer access | `torch.cuda.can_device_access_peer` = **False** for the pair (no P2P anywhere on this box) |
| NVLink | none on any pair |
| Software | torch 2.10.0+cu128, CUDA 12.8, NCCL 2.27.5, Python 3.12 (`~/ptyrad/.venv`) |
| CPU | 2 × AMD EPYC 7702 (124 cores visible) |
| Profiler | `nsys` not installed → `torch.profiler` + CUDA events |

```
      GPU0  GPU1  GPU2  GPU3   CPU Affinity  NUMA
GPU0   X    PHB   PHB   PHB    0-123         0-1
GPU1  PHB    X    PHB   PHB    0-123         0-1
GPU2  PHB   PHB    X    PHB    0-123         0-1
GPU3  PHB   PHB   PHB    X     0-123         0-1
```

Every hop pays the no-P2P driver-staged path; there are no NVLinked pairs
to compare against on this machine.

## Step 1 — Current execution order

References are to the stock files at fork point 7282053:
`src/ptyrad/forward_models/born_dist.py` (engine) and
`demo/run_cmp_modesplit.py` (the `slicedist` role; untracked in git, now
committed on this branch).

Program order per `born_forward_dist` call (engine):

1. psi init, per block in a loop over GPUs — `born_dist.py:76-80`
   (fft2 of probe replica, ifft2(H·pk); kernels enqueued device by device).
2. coeffs → `torch.complex(...).to(last_dev)` once — `born_dist.py:82-84`.
3. Per order n: **phase A on every block** (fft2·mult, cumsum, last-row
   view) — `born_dist.py:90-95`; **then all carries**
   `total.to(next_dev, non_blocking=True)` — `born_dist.py:96-102`;
   **then phase C on every block** — `born_dist.py:103-112`.
4. Detector reduce on the last device — `born_dist.py:115-116`.

Host-synchronising calls: **none inside the engine** (the only `.to`s are
`non_blocking=True`; no `.item()`, `.cpu()`, data-dependent shapes).
In the `slicedist` caller:

* `probes.to(dev)` per block per call, **blocking** — `run_cmp_modesplit.py:122`;
* `idx.to(dev)` (small pageable H2D, blocking) — `run_cmp_modesplit.py:114`;
* `dp.to("cuda:0")` at return — `run_cmp_modesplit.py:130`;
* H-block cache rebuild `.detach().to(dev)` on H change (once per H) — `:105`;
* solver-side per-batch loss `.item()`/logging (outside the model).

Answers:

1. **Is all phase-1 work enqueued on every GPU before the first carry
   copy?** Yes — the loops are phase-blocked (A on all blocks, then B, then
   C). The interleaved form survives only in the legacy `slicesplit` role
   (`run_cmp_modesplit.py:238-257`, synchronous `.to` at `:255`), which is
   not the current path.
2. **Adjoint:** autograd through the cross-device copies — no custom
   backward anywhere. The chrome trace of forward+backward+step
   (`trace_chain_Nz128_M4.json.gz`) shows both devices busy during the
   backward window (5.10 / 5.75 ms busy, 1.16 ms mutual overlap): the
   backward is concurrent (one autograd worker thread per device).
3. **gather → constrain → scatter:** `DistConstraint.__call__`
   (`run_cmp_modesplit.py:166-180`) — leaves → master `opt_obja/objp` on
   cuda:0, stock constraints, scatter back, once per iteration. Loss
   reduction: `DistLoss` (`:146-158`) — per-block L1 terms reduced to
   cuda:0 scalars; the data-term loss on the cuda:0 copy of dp.

**Prediction for H1 from the launch order alone:** refuted — phase 1 is
already launched on all GPUs before any carry, and the carries are
non-blocking.

## Step 2 — Profile of the current implementation (`chain`, peer transport)

CUDA-event instrumentation of the order loop (per-(order, block, phase)
events on each device's stream, origins aligned after a full sync;
instrumented output verified identical to the library, rel err 0.0).
Files: `timeline_chain_Nz{32,128}_M{1,4}.json`, Gantt:
`fig_timeline_current.png` (Nz=128, M=4, n=256, forward).

Key numbers (n=256, P=2, forward):

| case | span | phase A/order/GPU | phase C/order/GPU | carry hop | GPU1 idle/order |
|---|---|---|---|---|---|
| Nz=128, M=4 | 13.0 ms | 0.34 ms | 0.19–0.30 ms | **2.5–2.7 ms on BOTH streams** | ~2.5 ms |
| Nz=32, M=4 | 5.4 ms | 0.06–0.20 ms | 0.04–0.07 ms | 0.1–2.3 ms (bimodal) | 0.1–2.6 ms |

* **Phase-1 overlap fraction ≈ 0.0** in steady state (every order): GPU 1's
  A_n depends on C_{n-1}, which waits on the carry; the ~2.5 ms staged
  peer copy dominates the ~0.65 ms/order compute, so the GPUs almost never
  compute simultaneously. The first order's non-overlap is launch-rate
  bound instead (the host needs ~0.4 ms to enqueue a 64-slice phase, about
  as long as the GPU takes to run it at n=256).
* The carry copy occupies **both** devices' streams for ~2.5 ms each
  (driver-staged no-P2P `cudaMemcpyPeer` path) — it is latency, not
  bandwidth: the same cost from 0.5 MB up to 33 MB (Step 3).
* With the pinned transport the same loop has phase-1 overlap 0.54–0.61
  and span 3.9 ms instead of 13.0 ms (`fig_timeline_best.png`) — see
  Step 5.

## Step 3 — Communication microbenchmarks

`bench_micro.py` → `micro_results.csv`. complex64 planes; median over 30
reps (IQR in file). Selected rows (ms):

| n (plane) | MB | sync `.to` | non-blocking `.to` | bidir exchange | chalf `.to` | pinned staged | pinned 4-chunk pipe |
|---|---|---|---|---|---|---|---|
| 256 | 0.52 | 2.445 | 2.476 | 0.152 | 0.074 | **0.084** | 0.143 |
| 512 | 2.10 | 2.454 | 2.508 | 0.377 | 0.135 | 0.194 | 0.190 |
| 1024 | 8.39 | 2.563 | 2.568 | 1.037 | 0.268 | 0.663 | 0.448 |
| 2048 | 33.55 | 2.538 | 2.535 | 3.760 | 0.884 | 2.594 | **1.654** |
| 4096 | 134.22 | 8.835 | 8.781 | 14.287 | 3.310 | 10.278 | **6.459** |

Volume all-to-all (slabs → half-column shards, P²=4 chunks on side
streams; cross-device bytes listed): 2.53 ms @ 8.4 MB (N=32, n=256) up to
91.2 ms @ 2.15 GB (N=128, n=2048).

Fitted link model (ms, per message α + per-byte β):

| transport | α (ms) | β (ms/MB) | effective BW |
|---|---|---|---|
| peer `.to` (driver staged, no P2P) | **2.053** | 0.048 | ~21 GB/s asymptotic |
| pinned bounce (D2H+H2D, event-ordered) | **0.034** | 0.076 | ~13 GB/s |
| volume all-to-all (peer chunks) | 2.524 | 0.031 | ~32 GB/s aggregate |

Notes:

* `sync .to` ≡ `non_blocking .to` on this box — the cost is inside the
  driver's staged peer path, not in host blocking. The distribution is
  multimodal ({~0.1, ~0.55, ~2.5} ms at 0.5 MB; the 2.5 ms mode dominates
  from idle, the fast modes appear under sustained traffic). Preallocating
  the destination (`copy_` into a buffer) does **not** help — it is not an
  allocation effect.
* **Fan-in degenerates at 2 GPUs**; the closest analog (bidirectional
  exchange) shows the two directions do run concurrently (0.15 ms for
  2×0.5 MB).
* Half-precision (complex32) carries work at the tensor level and halve
  the bytes (0.074 ms @ 0.26 MB); not wired into the engine (would break
  the 1e-6 gate), measured for the record.
* NCCL `all_to_all_single` under torchrun was not run — with the fitted
  α_a2a already 2.5 ms and the transpose losing by ≥ 10 ms/order at every
  size (Step 5), a best-case NCCL transpose could not change the verdict;
  skipped under the time-box.

## Step 4 — Schemes implemented and correctness

`born_forward_dist(..., carry_scheme=)`: `chain` (default, stock),
`chain_reordered` (same code path — the stock engine is already
phase-ordered), `direct` (one concurrent round, fixed-order local sums),
`logscan` (Hillis–Steele, ⌈log₂P⌉ rounds, exclusive prefix recovered
locally as S_p − T_p), `transpose` (slabs → Ny-column shards, full-depth
cumsum, shards → slabs). Plus two orthogonal options:
`carry_transport='peer'|'pinned'` (all cross-device hops; `pinned` is a
custom `autograd.Function` staging through cached pinned bounce buffers
with event ordering, never syncing the host) and `skip_zero_planes=`
(continues the nilpotent window across slab boundaries; the skipped rows
are float-exact zeros).

* **Adjoint:** autograd throughout (`_PinnedHop.backward` is the mirrored
  hop); backward concurrency confirmed from the chrome trace (Step 1).
* **Non-blocking copies:** peer hops are `non_blocking=True` (PyTorch's
  cross-device `copy_` inserts the src/dst stream events and
  `record_stream` internally); the pinned path orders D2H/H2D with
  explicit events on both streams and protects bounce-buffer reuse with a
  consumer event — the host is never synced.
* A pre-existing crash was fixed: at tiny head blocks (e.g. P=4, N=7,
  M=4) the shrinking window emptied block 0 and the stock code indexed a
  size-0 dim / ran cuFFT on an empty batch. The guard is byte-identical
  whenever the window is non-empty (digest-verified).

**Gates** (fp32, n=256; P ∈ {2, 4} × N ∈ {7, 32, 129} × M ∈ {1, 4} ×
pmode ∈ {1, 4}; P=4 logically on 2 devices — same scheme code paths, real
cross-device hops; `gates12*.csv`):

| gate | result |
|---|---|
| forward detector intensity vs single-device `born_forward` | **pass**, worst rel err 2.2e-7 (< 1e-6) over all 120 scheme×config combos; same for pinned transport and skip_zero_planes |
| object (amp+phase) and probe grads vs single-device autograd | **pass**, worst rel err 4.1e-7 (< 1e-5) |
| 50 repeated forwards bitwise identical | **pass** (unique=1 for all 10 scheme×P combos, peer and pinned) |
| `CUDA_LAUNCH_BLOCKING=1` digest match | **pass** (both transports) |
| 20-iteration `slicedist` loss curve vs `chain` baseline | see Step 4b below |

Bitwise structure (digests): `chain` ≡ `chain_reordered` ≡ `direct`
(identical summation order); `logscan` and `transpose` differ at
reassociation level only; `transpose` gives the same digest at P=2 and
P=4 (the full-depth cumsum is partition-independent and in fact matches
the single-device reference to 0.0 when the window is inactive).

**Note on the exit wave:** the engine only exposes the detector intensity;
the 1e-6 gate is on |Ψ|² summed over modes. The bitwise transpose/
single-device agreement implies field-level agreement.

### Step 4b — 20-iteration reconstruction gate (real PSO data)

Real PSO born6 config (Nz=21, 4096 scans, batch 1, 4 pmodes, dz=10 Å;
`params_gate/gate_*.yml`, seed 42, `PTYRAD_DISABLE_ISS_PRECOND=1`,
`SLICE_P=2`, Adam `foreach=False` — torch 2.10's multi-tensor Adam
crashes on multi-device parameter sets; single-tensor path used for all
arms equally). Arms run: `chain`+peer (baseline), `chain`+pinned,
`transpose`+peer. `chain_reordered`/`direct`/`logscan` were not re-run:
at P=2 their forwards are **bitwise identical** to `chain` (digest gate),
so their 20-iteration trajectories are identical by construction.

Result (`fig_gate_loss.png`): the loss trajectories are **identical to the
logged precision (4 decimals) at every common iteration**:

| arm | iterations completed | final logged loss | max \|Δloss\| vs chain |
|---|---|---|---|
| chain + peer (baseline) | 20 / 20 | 0.2865 | — |
| chain + pinned | 10 (time-boxed) | 0.3038 (@10; chain @10 = 0.3038) | 0.0e+00 |
| transpose + peer | 15 (time-boxed) | 0.2936 (@15; chain @15 = 0.2936) | 0.0e+00 |

The pinned and transpose arms hit the 35-min per-experiment box before 20
iterations (pinned ran ~63 ms/batch — see the transport caveat in Step 5 —
and transpose ~36 ms/batch vs chain's 15 ms/batch); the curves match
exactly over every completed iteration, which together with the bitwise
digest gate makes the equivalence conclusion safe. **Gate: pass.**

An operational finding from this gate: in the real batch-1 solver the
pinned transport was ~4× slower per batch (63 vs 15 ms) — the opposite of
the engine-level benchmarks. Under back-to-back batch-1 traffic the
driver's peer path stays in its fast (~0.1 ms) mode, while the pinned path
pays host staging on every hop at pmode 4 (2 MB carries, M=6 orders, fwd +
bwd). The transport choice is therefore regime-dependent: `peer` for hot
single-recon batch-1 loops, `pinned` where hops are latency-exposed
(cold/idle links, large per-step synthetic workloads). Root cause of the
driver path's idle-mode 2.5 ms tax remains unattributed (KVM/vfio staging
behaviour).

## Step 5 — Benchmark grid

`bench_grid.py` → `slicepar_results.csv` (728 rows; every parameter
recorded). One Adam step (and separately one forward) per cell; CUDA-sync
bracketed wall time, 5 warm-ups, median/IQR over 20 repeats, batch 1,
omode 1. P=1 = single-device `born_forward` (eager; `@torch.compile`
max-autotune arm at n=256). 19 cells skipped by the 30-min budget
(n=2048: all of Nz=256 M=4 and three Nz=256 M=1 step arms) and one OOM
(P=1 step n=2048 Nz=256 M=1 — **P=2 fits**, 370.5 ms).

Headline speed-ups vs P=1 eager (step, pmode 1; full tables in
`analysis.md`, figures `fig_crossover.png`, `fig_best_scheme.png`):

| n | M | Nz=8 | 16 | 32 | 64 | 128 | 256 | N* |
|---|---|---|---|---|---|---|---|---|
| 256 (chain) | 4 | 0.5 | 0.5 | 0.5 | 0.3 | **1.63** | **1.77** | 128 |
| 1024 (chain) | 4 | 1.00 | **1.32** | 1.48 | 1.61 | 1.68 | 1.75 | 16 |
| 2048 (chain+pinned) | 4 | 1.04 | **1.40** | 1.56 | 1.65 | 1.82 | — | 8–16 |

* **All carry-family schemes are equal at P=2** (within IQR, usually
  <2%): chain ≡ chain_reordered ≡ direct ≡ logscan. H1/H2 as predicted.
* **transpose loses everywhere**: 0.35–0.77× (M=1) and 0.06–0.44× (M=4).
* **Transport**: `pinned` ≈ `peer` in the sustained-load cells (under
  back-to-back traffic the driver path hits its fast mode) and far better
  in latency-exposed ones — in the isolated-forward timeline it cuts the
  span 13.0 → 3.9 ms (n=256, Nz=128, M=4). BUT in the real batch-1 solver
  it ran 4× slower per batch (Step 4b). Regime-dependent: keep `peer` as
  default for hot batch-1 loops; `pinned` is a per-machine knob for
  latency-exposed hops.
* **Probe modes** (secondary run, n=256): pmode 4 multiplies phase-A work
  by 4 and pulls the crossover in: step M=4 Nz=32 → 1.37×, Nz=128 →
  1.69× (P1 42.0 ms vs 24.9 ms). With pmode 4 the win regime starts at
  **Nz ≈ 32** at n=256.
* **P=1 eager vs compiled** (n=256): compiled forward ≈ eager (4.63 vs
  4.53 ms @ Nz=128 M=4), compiled step ~10% *slower* (15.57 vs 14.15 ms);
  speed-ups vs compiled are therefore equal or slightly better than vs
  eager; no separate table needed.
* **Half-precision carries**: engine does not support them (gate would
  fail); microbench numbers in Step 3.
* **skip_zero_planes** (n=256, Nz ∈ {4..32}, M=4): differences are inside
  the link-noise (±10%, non-systematic, e.g. −18% at one Nz=8 cell, +19%
  at one Nz=16 cell). Bitwise-identical output, free in principle; the
  M ~ Nz/P regime where it should pay (PSO-like) is exactly where the
  whole split is latency-bound on this box. Keep it off by default,
  revisit on a fast-hop machine.

## Step 6 — Analysis

* **Crossover** N* (step mode, first Nz with speed-up > 1): table above
  and `analysis.md`; map of the fastest arm per (n, Nz) in
  `fig_best_scheme.png`. The apparent winner among the carry family in any
  one cell is noise; the real boundaries are: P=1 below N*, carry family
  above it, transpose never.
* **Model** T(2) = W/2 + c_scheme with the Step-3 α/β
  (`fig_model.png`): matches chain+pinned well at n ≥ 1024 (within
  ~5–10%); for chain+peer it *over*-predicts in sustained-load cells
  because the 2.05 ms α is the idle-link mode — under back-to-back
  traffic the driver path drops to its ~0.1–0.55 ms modes (the model's
  single-α assumption fails; the link has state). At n=256 the model
  misses the launch-overhead floor (~1–2 ms/step of enqueue time that
  does not shrink with P) and under-predicts P=2 times at small Nz.
  The transpose model tracks the measured shape (bandwidth-dominated,
  ∝ Nz·n²) but the pack/unpack term (contiguous() on strided shards +
  cat) adds ~20–30%.
* **On NVLink, is the transpose ever best?** Untestable here (no NVLink).
  From the model: the transpose moves ~Nz/P planes per GPU per order vs 1
  plane per hop for the carry schemes, so it needs the carry's α to
  dominate Nz/P planes of bandwidth. With NVLink α ≈ 10–20 µs and
  ~10–25 µs/plane @256², the window N ≲ P²/2 (= 8 at P=4) is below any
  depth where slice-splitting is worth it — and the earlier NVSwitch
  campaign found hop *latency*, not bandwidth, is what survives there
  (bench_results §L). Expected answer: **no**.
* **P=4 extrapolation (H2)**: win condition for the chain over P=1 is
  W(3/4) > M·3·(α+bβ); at n=256/M=4 peer that is Nz ≳ 900 (chain) and
  Nz ≳ 310 (direct) — the H2 "break-even 17–30" premise assumed ~0.1 ms
  hops. With the pinned transport (α+bβ ≈ 0.07 ms) the same formulas give
  Nz* ≈ 33 (chain) and ≈ 11 (direct): **the fix is the transport first;
  only after that does the carry topology matter, in the direction H2
  predicts.**

## Step 7 — Overlap across orders

Not implemented as a separate scheme, with justification: at P=2 the
wavefront overlap already happens by construction — GPU 0's stream has no
cross-device dependency in the forward, so it runs orders ahead (visible
in `fig_timeline_current.png`: cuda:0 finishes A3 at t≈8.4 ms while
cuda:1 is at A1), and GPU 1's A_{n+1} genuinely depends on carry_n via
C_n, which no launch reordering can remove. With the pinned transport the
carry latency is 0.03 ms against 0.65 ms/order of compute — there is
nothing left to hide at P=2 (phase-1 overlap 0.54–0.61, span 3.9 ms vs
13.0 ms). The remaining idea worth recording: the carry enters the
recursion *linearly*, so GPU 1 could compute its carry-independent part
of every order and add the carry response later (an ISS-like
restructuring); that would double some compute to remove the remaining
serial chain, and only pays at P ≥ 4 on a slow-hop link — out of scope
here.

Nilpotent-slack additions from the review discussion: (1) cross-slab
window skipping is implemented (`skip_zero_planes`, gated, bitwise-exact;
unresolvable at n=256 under link noise); (2) remainder-slice placement —
the harness and the caller both already give the remainder to block 0
(the forward-light end); for training the mirror argument favours the
last block, but at one plane per remainder slice the effect is below the
noise here; (3) micro-batch staggering: no batch in the PSO regime,
not built.

## Recommendations by regime (this box: PCIe, no P2P, 2 GPUs)

| regime | recommendation |
|---|---|
| n=256, Nz ≲ 64, pmode 1 (PSO-like today) | **1 GPU.** Nothing beats it; the split is launch- and latency-bound. |
| n=256, Nz ≳ 128, or pmode ≥ 4 and Nz ≳ 32 | 2-GPU `chain` (peer default; try `pinned` per machine); expect 1.3–1.8×. |
| n ≥ 1024, Nz ≳ 8–16 | 2-GPU `chain`; 1.3–1.8×, and it lifts the P=1 OOM ceiling (n=2048, Nz=256). |
| any size | **never `transpose`**; `direct`/`logscan` only become relevant at P ≥ 3 *after* the transport is fast. |
| caller fix regardless | replace the blocking `probes.to(dev)` with the non-blocking/pinned hop (3.4× step at n=256, Nz=128) — one line, now in the branch. |

## Reproduction

All from `~/ptyrad-slicepar/demo/slicepar`, env
`CUDA_VISIBLE_DEVICES="GPU-03a8aae1-354c-2d7f-d513-2ace87ccacc8,GPU-5c1cb7cd-b1a6-01b3-ef6e-ba1d766f59cb"`,
`PYTHONPATH=~/ptyrad-slicepar/src:.`, python `~/ptyrad/.venv/bin/python`:

```
python check_correctness.py                      # gates 1-2, peer
python check_correctness.py pinned               # gates 1-2, pinned
python check_correctness.py skipz                # gates 1-2, skip_zero_planes
python check_correctness.py hash                 # 50x digest
CUDA_LAUNCH_BLOCKING=1 python check_correctness.py hash   # race check
python bench_micro.py                            # Step 3
python profile_timeline.py chain 128 4 256 chain_Nz128_M4       # Step 2
SLICEPAR_TRANSPORT=pinned python profile_timeline.py chain 128 4 256 pinned_Nz128_M4
python bench_grid.py --n 256  --arms main        # Step 5 (repeat 1024/2048,
python bench_grid.py --n 256  --arms skipz       #  and skipz/pmode4/compiled)
python bench_grid.py --n 256  --arms pmode4
python bench_grid.py --n 256  --arms compiled
python analyze.py all && python model_fit.py     # Step 6 figures/tables
# recon gate (from ~/ptyrad/demo, data lives there):
#   see run_gates.sh — PTYRAD_DISABLE_ISS_PRECOND=1 SLICE_P=2
#   CARRY_SCHEME=<scheme> CARRY_TRANSPORT=<tr> python
#   ~/ptyrad-slicepar/demo/run_cmp_modesplit.py slicedist params_gate/gate_<arm>.yml
```

## Changed files (branch `slicepar-carry`)

* `src/ptyrad/forward_models/born_dist.py` — carry schemes, transports,
  skip_zero_planes, empty-window guards (default path digest-identical).
* `src/ptyrad/forward_models/__init__.py` — export `born_forward_dist`,
  `make_blocks` (mirrors the uncommitted line on `linesearch-born`).
* `demo/run_cmp_modesplit.py` — committed (was untracked);
  `CARRY_SCHEME`/`CARRY_TRANSPORT` env plumbing; probe replica goes through the
  transport hop.
* `demo/slicepar/` — this study: harness (`common.py`,
  `check_correctness.py`, `bench_micro.py`, `bench_grid.py`,
  `profile_timeline.py`, `analyze.py`, `model_fit.py`, `run_*.sh`,
  `params_gate/`), results (`slicepar_results.csv`, `micro_results.csv`,
  `gates12*.csv`, `hash_*.txt`, `timeline_*.json`, `trace_*.json.gz`,
  `analysis.md`), figures (`fig_timeline_current.png`,
  `fig_timeline_best.png`, `fig_best_scheme.png`, `fig_crossover.png`,
  `fig_model.png`), and this report.

Nothing was pushed to any remote; no issues/PRs; the paper is untouched.

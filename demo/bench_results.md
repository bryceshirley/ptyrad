# PSO born6 1-vs-2-GPU benchmark (2026-10-09)

Config: `params_recovered.yml` (PrScO3 PSO), 256² dp, 21 slices dz=10 Å,
4 probe modes, born6 + per-iter detector refit (32 views). 4096 scan
positions. Box: 2× A100 80 GB, PCIe/PHB (no P2P). ISS-precond disabled.

## A. Real full-iteration wall time (`ptyrad run`, NITER=1)

| config | batch | wall / iter | notes |
|---|---|---|---|
| 1 GPU | 1  | **71.7 s** | incl. 2.3 s refit; ~17.5 ms/position |
| 1 GPU | 32 | **57.8 s** | incl. 2.4 s refit; ~14.1 ms/position |
| 2 GPU data-parallel (accelerate) | 32 | **crashes** | DDP buffer-broadcast bug (expanded buffer); pre-existing |
| 2 GPU data-parallel (accelerate) | 1  | **N/A** | global batch 1 can't split across 2 procs (split_batches=True) |

Batch 1 → 32 is only 1.24× faster: **this workload is compute-bound, not
launch-bound** (born6 × 4 modes × 21 slices with autograd backward). That is
the opposite of the small-frame multislice regime, and it is exactly why
splitting the compute across GPUs can pay off here.

## B. Probe-mode parallelism (the batch-1-friendly option)

Probe modes sum incoherently only at the detector (born.py:125), so splitting
the 4 modes 2+2 across GPUs is **numerically exact** (rel err 4.2e-08) and
needs no change to `born_forward`. Autograd handles the cross-device object
gradient; the two partial intensity maps are added on cuda:0.

Isolated born6 forward+backward (`bench_mode_parallel.py`):

| batch | 1 GPU (4 modes) | 2 GPU (2+2) | speedup |
|---|---|---|---|
| 1  | 10.75 ms | 7.03 ms | **1.53×** |
| 32 | 312.1 ms | 198.5 ms | **1.57×** |

So mode-parallelism gives ~1.5× on the born compute itself — at batch 1 and
batch 32 alike — because the model is compute-bound. This works at batch 1,
where data-parallel cannot.

### Projected real-iteration effect (batch 1)
Born f+b is ~10.75 ms of the ~17.5 ms/position; the rest (refit, constraints,
optimizer, Python) does not parallelize over modes. Saving ~3.7 ms/position
→ ~71.7 s projected to ~57 s/iter, i.e. ~1.25× at the iteration level (the
2nd GPU helps only the part it covers). The full 1.5× is only realizable if
more of the per-position work (constraints/optimizer) is also split or the
non-born overhead is reduced.

## C. Slice (depth) parallelism — the plane split

Splits the SLICE axis across GPUs: planes [0,12) on dev0, [12,21) on dev1.
This is the paper's distributed-Σ regime (paper-1.tex, "Scattering Operator
Decomposition", ~line 514): the cumulative sum Σ that assembles the
downstream field **couples the planes**, so a slice split must carry the
running total across the device boundary *every Born order*. We use the cheap
two-level scan — one carry plane dev0→dev1 per order (host-staged, no P2P) —
not a full all-to-all transpose. Numerically exact (rel err 3.6e-08 vs
`born_forward`).

Isolated born6 forward+backward (`bench_slice_parallel.py`):

| batch | 1-GPU compiled | 1-GPU eager | 2-GPU slice-split (pipelined) | vs compiled | vs eager |
|---|---|---|---|---|---|
| 1  | 10.78 ms | 9.95 ms  | 9.13 ms  | **1.18×** | 1.09× |
| 32 | 311.4 ms | 278.2 ms | 241.6 ms | **1.29×** | 1.15× |

### Cost decomposition — the loss is 100% data movement
Two diagnostics pin down where the gap from the ideal 2× goes:

| experiment (N=32) | time | vs eager | reading |
|---|---|---|---|
| no-transfer floor (carry forced to 0) | 139 ms | **2.00×** | compute+pipeline alone hit the ideal 2× |
| real carry, complex64 | 225 ms | 1.24× | the 86 ms gap is **all the carry transfer** |
| real carry, complex32 (half) | 201 ms | 1.39× | halving carry bytes recovers ~⅓ of the gap |

With the carry transfer removed, slice-split hits the **ideal 2.00×** at
N=32 (1.53× at N=1, launch-overhead-limited). So the pipeline overlap and
serialization are NOT the bottleneck — **the entire loss is moving the carry
plane across the device boundary** over host-staged PCIe (this box is PHB, no
P2P). At batch 32 the carry is one (32,4,256,256) complex64 plane ≈ 268 MB,
five times, host-staged at ~16 GB/s ≈ 84 ms — matching the measured gap.
This **empirically confirms the paper's claim**: with planes distributed, the
bottleneck is data movement, not the Fourier transforms.

### Carry compression is training-safe (correction of an earlier claim)
An earlier note here REJECTED the half-precision carry as "gradient-unsafe
(75% error, biased)". **That was wrong — an artefact of an unrealistic test**
(a random, data-inconsistent target with the estimate sitting at the true
object, which puts the optimiser in a pathological regime). Re-tested properly
against a self-consistent data target at a mid-reconstruction operating point,
and — the decisive yardstick — against the gradient's own **photon shot-noise
floor** (`bench_slice_compress.py`):

| carry scheme | fwd err | object-grad err | bias | xfer bytes |
|---|---|---|---|---|
| half c32 | 1.3e-5 | 4.5e-4 | 0.00 | 2× |
| bf16 (RN or SR) | 1.1e-4 | 3.6e-3 | 0.00 | 2× |
| fp8 e4m3 (**quarter**) | 1.7e-3 | 5.7e-2 | 0.00 | 4× |
| fp8 e5m2 | 3.4e-3 | 1.1e-1 | 0.01 | 4× |
| int8 (RN or SR) | 7e-4 | 2.5e-2 | 0.00 | ~4× |
| low-rank r=64 | 4.0e-2 | **1.4** | 0.08 | 2× |
| zfp tol=1e-3 | 3.6e-7 | 1.2e-5 | 0.00 | **188×** |

**Shot-noise gradient floor** (how much photon noise alone moves the gradient,
relative, weak-phase PSO object): **97× at dose 1e4, 7.8× at 1e6, 0.75× even at
1e8.** Every scheme except low-rank sits *far* below this floor — i.e. the
compression perturbs the gradient orders of magnitude less than photon noise
already does. End-to-end 50-step reconstruction drift vs full-carry:
half **1.3e-6**, fp8/quarter **1.9e-4**, versus **2.9e-2 drift from photon
noise alone** (dose 1e6) — half is ~22000× below the noise floor, quarter
~150× below.

Takeaways:
- **Half (c32) carry is training-safe.** Use it. 2× byte reduction, gradient
  effect 4 orders below the noise floor.
- **Quarter precision (fp8) also works** (accuracy-wise): 4× bytes, still ~150×
  below the noise floor.
- **Stochastic rounding is unnecessary** — the deterministic rounding is already
  unbiased (bias ~0) here; SR only adds variance.
- **zfp is the compression champion** (13–188× at negligible gradient cost,
  because the carry is spatially smooth) — but it is CPU-side here; realising it
  in the transfer path needs a GPU codec (nvcomp/cuZFP).
- **Low-rank is the only loser** (grad err > noise floor; random data is its
  worst case, but it is also the weakest performer — skip).

### Improving the slice split (what works, what doesn't)
1. **Carry compression — WORKS and is training-safe** (see table above). Half
   (c32) is the practical choice.
2. **Offset wavefront pipeline — WORKS (this is the right structure).**
   Information flows strictly downstream (Σ is lower-triangular), so the
   upstream block never needs the downstream one: block b runs flat-out and
   block b+1 trails it by one order. Confirmed by the no-transfer floor
   hitting the ideal scaling (2.00× on 2 GPUs, **3.30× on 4**, N=32).
3. **Pinned-async carry overlap — WORKS (+20% at batch).** A plain
   `.to(dev)` from non-pinned memory is synchronous, so it sits on the
   critical path. Staging through a pinned host buffer with D2H on a dev0
   copy-stream ∥ H2D on a dev1 copy-stream (double-buffered) truly overlaps
   the carry with compute: 2-GPU forward **1.24× → 1.49×** at N=32, rel err
   0.00e+00 (exact). Regresses at N=1 (0.96×) where the carry is tiny and the
   event overhead dominates — a batch-regime win. (`bench_slice_opt.py` Part 2.)
4. **Split plane S=12 optimal** (2-way): sweep S=9→1.09×, 11→1.18×,
   **12→1.24×**, 14→1.11×, 16→1.06× (vs eager, N=32, c64).
5. **Fast interconnect is the real unlock.** This box is all-PHB — no NVLink
   (confirmed: `nvidia-smi topo -m` all-PHB, `nvlink -s` inactive). The carry
   is host-staged at ~16–25 GB/s. NVLink (~300 GB/s) would let the realized
   speedup approach the compute floor (2× on 2 GPUs, 3.3× on 4).

### Scaling to 4 GPUs (`bench_slice_4way.py`, forward-only, N=32)
MIG disabled on GPUs 0&3 → four full A100s. P-way = carry CHAIN (block b's
running prefix passed to block b+1 each order; grand total at the last block
= D_n). Exact to 4e-8 at P=2 and P=4.

| pinned-async carry codec (N=32) | P=1 | P=2 | P=4 |
|---|---|---|---|
| compute floor (no transfer) | 96 ms | 52.8 ms (**1.82×**) | 29.1 ms (**3.30×**) |
| sync `.to`, c64 | 96 ms | 84 ms (1.14×) | 98 ms (0.98×) |
| pinned-async, c64 (exact) | 96 ms | 70 ms (1.37×) | 74 ms (1.31×) |
| pinned-async, **half** (2×, training-safe) | 96 ms | 67 ms (1.44×) | 56 ms (**1.71×**) |
| pinned-async, **quarter** fp8 (4×, safe) | 96 ms | 67 ms (1.44×) | 55 ms (**1.74×**) |

All codecs are training-safe (gradient effect ≪ shot-noise floor, above). c64 is
bit-exact (P=2 3.7e-8, P=4 3.98e-8); half 2.2e-5 fwd; quarter 2.9e-3 fwd — all
negligible vs the Born truncation (3.6e-8 only *looks* smaller; it is below the
noise floor too). The no-transfer row zeroes the carry (compute-only, not a
correct output).

Key points:
- **Compression flips the P=2 vs P=4 ordering.** At c64 the extra hop makes
  4-way (1.31×) ≤ 2-way (1.37×); compressed, 4-way's compute edge wins
  (1.71–1.74× > 1.44×). With a training-safe compressed carry, **4 GPUs finally
  beat 2** on this box.
- **Compression saturates at half.** Going half→quarter (another 2× on bytes)
  buys almost nothing (1.71→1.74×): the fp8 encode/decode compute offsets the
  byte saving, and the *remaining* gap to 3.30× is no longer bytes but **per-hop
  latency** (3 serial host-staged hops + event sync on PHB). zfp's 188× ratio
  wouldn't help wall-clock for the same reason (latency-bound, and CPU-side).
- **Best realized 4-GPU: ~1.74× (training-safe)**, vs 3.30× compute floor. The
  residual ~1.9× gap is interconnect *latency* — only NVLink/P2P closes it.

**The compute scales near-linearly to 4 GPUs (3.30×); the PCIe carry chain is
the ceiling.** With a plain synchronous `.to()` the P=4 chain (3 hops × 5
orders of host-staged copies) cancels the entire compute gain → 0.98× net.
Overlapping every hop with a pinned double-buffer (D2H on a src copy-stream ∥
H2D on a dst copy-stream, `forward_pway_pinned`) rescues P=4 to **1.31×** and
P=2 to 1.37×, exact. That is the realized ceiling on this PHB box: the carry is
still host-staged at ~16–25 GB/s and three serial hops can't be hidden behind
29 ms of compute. Both P-way rows **regress at N=1** (2-way 0.96×, 4-way
0.41×) — the carry is tiny and stream/event overhead dominates; P-way slice
split is a batch-regime technique.

**Bottom line for 4-GPU slice split:** compute floor 3.30×, best realized
**1.31×** (N=32, pinned-async). The ~2.5× gap to the floor is pure interconnect
— NVLink/P2P (~300 GB/s) is required to approach 3.3×. On PHB, 4-way slice
split is worth it only in the memory-bound regime (the volume won't fit on 1–2
devices), not for speed.

### Slice-split vs mode-split — corrected
- **Mode-split (1.53–1.57×, exact, batch-1 OK)** remains the best 2-GPU choice
  for this compute-bound config: two independent *compiled* `born_forward`
  calls, no `born.py` change, no cross-device dependency. But it **replicates**
  the object on every device and caps at #probe-modes (4).
- **Slice-split** needs the carry chain and is latency-bound on PHB (best
  realized, training-safe: 2-way ~1.44×; 4-way **1.74×** with a compressed
  carry). Its compute scales to 3.3× on 4 GPUs and it **partitions** the volume
  (~1/P per-device
  memory — the paper's "doubles feasible size"). It wins when the volume won't
  fit on one device, when #GPUs > #modes, or on NVLink hardware.

Net for the 256²/21-slice PSO config: **use mode-split.** Slice-split is a
memory-/hardware-driven choice, not a speed win on this box.

## D. Slice-split on the REAL born6 PSO checkpoint (`bench_slice_real.py`)
Not a synthetic microbench: loads the actual `model_iter0100.hdf5` from
`output/PSO/20260929_pso_born6_qr_msT_noreg_full_...21slice_dz10_...` — real
obja/objp (1×21×639×639), probe (4×256²), H.pow(z), born_coeffs (6×2, values
up to |c|≈3), crop_pos — crops patches exactly as `models.get_obj_ROI`, runs
the real compiled `born_forward` + detector blur (σ=1) as 1-GPU ground truth
vs the 4-GPU depth split.

**Correctness (real object): PASS.** Forward sync/c64 rel err 1.8e-6 (P2) /
1.8e-6 (P4); object-gradient rel err **1.9e-4** (P4, amplitude loss, autograd
carry) — far below the shot-noise floor, confirming the distributed autograd
is faithful on real data.

**Timing — slice-split LOSES at the real operating point:**

| N | 1-GPU fwd | 4-GPU fwd (real, half) | real× | 4-GPU fwd (free carry) | compute floor× |
|---|-----------|------------------------|-------|------------------------|-----------------|
| 1 (real BATCH_SIZE) | 3.58 ms | 10.20 ms | **0.35×** | 6.16 ms | **0.58×** |
| 8 | 23.37 ms | 41.80 ms | **0.56×** | 7.86 ms | 2.97× |
| 32 | 91.43 ms | 162.87 ms | **0.56×** | 27.83 ms | **3.29×** |

(fwd+bw tracks fwd: N=1 0.66×, N=8 0.80×, N=32 0.82×.) One GPU is faster than
the 4-GPU slice-split at **every** batch size on this real config.

### Is it the implementation? The NOTRANSFER probe says no.
`NOTRANSFER=1` replaces every carry with a cached zero (breaks correctness) to
isolate the pure compute-split ceiling from the PHB host-staged carry cost.
Two regimes fall out cleanly:

- **Batch 1 is a fundamental granularity wall, not an implementation bug.**
  With *free* transfer the 4-GPU split is still **0.58×** — slower. born6 at
  batch 1 is only 3.6 ms on one A100; split 4 ways the per-device compute
  (~0.9 ms) is swamped by fixed per-kernel launch + the short 6-order
  pipeline's fill/drain. No implementation (and no NVLink) rescues batch 1 —
  the work is too small to parallelize. (eager 1-GPU = compiled 1-GPU = 3.6 ms,
  so torch.compile is not the baseline advantage either.)
- **Batch ≥8 parallelizes near-perfectly in compute (floor 2.97×→3.29× at
  N=32) but the PHB carry transfer, not my loop, eats it.** The gap from 3.29×
  (free) to 0.56× (real) at N=32 is pure host-staged carry cost — and it does
  **not** amortize with batch: the carry prefix is (N, pmode, omode, Ny, Nx),
  so its volume scales with N exactly like the compute (~67 MB/hop × 18 hops =
  ~2.4 GB c64 / ~1.2 GB half, round-tripped through pinned host memory on a
  no-P2P/no-NVLink PCIe fabric). The real× therefore **plateaus at 0.56×** for
  every batch — there is no batch size that fixes it on this box. **Only
  NVLink/P2P (direct GPU↔GPU carry) would let the real number approach the
  3.3× compute floor.**

So: the slowdown is (i) batch-1 granularity and (ii) PHB carry bandwidth —
both intrinsic to this config/hardware, not the eager Python implementation,
which splits the compute to within 1% of ideal (3.29× of 3.3×) at N=32.

Two corrections to the §C microbench optimism:

1. **The 1.71× was a baseline artefact.** §C compared the eager carry-chain
   P=4 against an eager 1-GPU that measured ~10 ms. On the real checkpoint the
   1-GPU forward is **3.6 ms** (compiled and eager measure identically here —
   `forward_pway` P=1 = 3.56 ms vs compiled `born_forward` 3.58 ms), so there
   is no 10 ms baseline to beat. Per-block compute at batch 1 is ~0.9 ms; the
   18 serialized host-staged carry hops (3 hops × 6 orders, wavefront) dominate.
   The 3.3× compute floor is unreachable because latency, not compute, is the
   wall — exactly as predicted, but the real 1-GPU baseline is fast enough that
   the split never crosses into a win.
2. **quarter/fp8 is NOT training-safe on real data.** The §C "quarter safe"
   result was an artefact of *smooth synthetic* carry. On the real object the
   carry's dynamic range defeats fp8-e4m3's single global scale: forward rel
   err **0.15 (P2) / 1.9 (P4)** — unusable. `half` holds (5.9e-4 P2 / 2.2e-3
   P4). Moot here since the split loses regardless, but the general claim is
   narrowed: **half is the only safe compressed carry; quarter needs
   per-slice/block scaling to be viable.**

Net for this test case: slice-split is the wrong axis — see §E for the
head-to-head against mode- and batch-split on the same real tensors.

## E. All three axes on the real checkpoint (`bench_axes_real.py`)
Same real `model_iter0100.hdf5` tensors; 1-GPU compiled `born_forward`+blur is
the baseline. slice = depth carry-chain (half codec, pinned); mode = split the
4 probe modes (incoherent detector sum → per-group born outputs ADD, exact rel
err 3.9e-8, keeps compiled born, batch-1 capable, caps at 4 modes); batch =
data-parallel over scan positions (needs N≥G, so **N/A at the real
BATCH_SIZE=1**). fwd+bw measured eager (compiled backward has a dynamo
multi-device metrics bug; eager==compiled fwd here so ratios hold).

**Forward ×:**

| N | slice2 | slice4 | mode2 | mode4 | batch2 | batch4 |
|---|--------|--------|-------|-------|--------|--------|
| 1 (real) | 0.72× | 0.33× | **1.78×** | **2.08×** | N/A | N/A |
| 8 | 0.82× | 0.56× | 1.86× | 3.10× | 1.85× | 2.93× |
| 32 | 0.81× | 0.56× | 1.92× | 3.40× | 1.96× | 3.65× |

**Forward+backward × (the real training step):**

| N | mode2 | mode4 | batch2 | batch4 |
|---|-------|-------|--------|--------|
| 1 (real) | **1.13×** | 0.58× | N/A | N/A |
| 8 | 1.54× | 1.48× | 1.54× | 1.39× |
| 32 | 1.61× | 1.64× | 1.86× | 2.67× |

Takeaways:
- **Batch 1 (the real config): mode-split 2-GPU is the only win** — 1.78× fwd,
  1.13× full step. **mode4 over-splits** (2.08× fwd but **0.58× fwd+bw**): one
  mode per GPU makes the per-device backward tiny, so the 4-way gather +
  object-grad all-reduce + launch overhead swamps it. batch- and slice-split
  can't help batch 1 at all.
- **The forward parallelizes far better than the full step.** born6 backward is
  ~2× the forward (1-GPU fwd 3.6 ms, fwd+bw 9.5 ms); mode-split's backward
  carries the object-grad reduction, so fwd gains (1.8–3.4×) collapse to
  ~1.1–1.6× on fwd+bw. Mode-split fwd+bw plateaus at ~1.6× regardless of batch
  (4-mode cap + reduction cost).
- **Batch ≥8: data-parallel wins the full step** (batch4 2.67× at N=32) — its
  only comm is ONE object all-reduce/step (68 MB: 3.5 ms G2 / 14 ms G4,
  host-staged), vs mode-split's per-step mode-gather. Mode-split still leads on
  pure forward (3.40× at N=32).
- **Slice-split loses on every axis/batch** (≤0.82×) — §D.

Practical recommendation: **batch 1 → mode-split across 2 GPUs** (1.8× fwd,
~1.1× step, exact, keeps compiled born, no `born.py` change). **Batch ≥8 →
data-parallel.** Slice-split only if the volume won't fit on one device.

## F. END-TO-END: 20-iteration reconstruction, stock vs mode-split (2026-10-09)
The definitive test — not a microbench. Full `ptyrad run`-equivalent
reconstructions (`run_cmp_modesplit.py`) of the real PSO born6 config
(`params/cmp_baseline.yml` / `cmp_modesplit.yml`, regenerated from the
schema-fixed bench_b1), fresh simu init, seed 42 (identical batch order
verified), NITER 20, SAVE_ITERS 5, eager both arms, batch 1, refit on. The
mode-split arm patches `ptyrad.forward_models.born_forward` (modes 0–1 on
cuda:0, 2–3 on cuda:1, exact incoherent sum, autograd through the cross-device
hops); both GPUs confirmed active (57%/42%). Outputs in
`output/PSO/20261009_cmp_{baseline,modesplit}_20it_*/`.

**Reconstruction: EQUIVALENT.** Total loss identical to all 4 printed digits at
every iter (0.4239 → 0.3265 → 0.3038 → 0.2936 → 0.2865; original 20260929 run,
different seed: 0.2864 at iter 20). Checkpoint tensors at iters 5/10/15/20:
obja rel diff ~6e-6, objp 1.0–1.5e-3, probe 2.0–2.6e-3, born_coeffs ~1e-4·5 —
the compounding of the per-step 3.9e-8 float reassociation + nondeterministic
backward atomics over 81,920 batch steps; the diffs plateau rather than grow.

**Wall time: mode-split is 5% SLOWER in the real loop.**

| arm | s/iter | total (20 it) |
|-----|--------|----------------|
| baseline 1-GPU | **64.5 ± 0.5** | 22 min 17 s |
| mode-split 2-GPU | 67.7 ± 1.5 | 23 min 21 s (**0.95×**) |

The microbench 1.13× (fwd+bw, pre-placed replicas) does not survive the real
iteration: born fwd+bw is only ~9.5 ms of the ~15.7 ms real per-batch step
(optimizer, loss, constraints, cropping, logging), the split saves ~1 ms of
that, and the per-batch transfers it adds — object patches (10.5 MB), probe
group, DP output back, and the backward's reverse hops, all host-staged PCIe —
cost more than the saving (net +0.8 ms/batch).

**FINAL verdict for this config (batch 1, 256², 21 slices, 4 modes, no
NVLink): no multi-GPU strategy beats one GPU in the real training loop.**
Mode-split (microbench 1.78×/2.08× fwd) loses to transfer+overhead dilution;
slice-split loses outright (§D); batch-split can't run at batch 1. Multi-GPU
on this box = independent reconstructions, or memory capacity (slice-split's
~1/P partitioning), or larger batches with data-parallel (§E: 2.67× at N=32).

## Files
- `run_cmp_modesplit.py` — end-to-end 20-iter reconstruction driver
  (baseline|modesplit), mode-split via born_forward patch; `params/cmp_*.yml`.
- `bench_axes_real.py` — slice vs mode vs batch (data-parallel) on the REAL
  checkpoint; forward + fwd+bw speedups at N=1/8/32, mode-split correctness,
  object all-reduce cost.
- `bench_slice_real.py` — slice split vs the REAL born6 PSO checkpoint on 4
  GPUs: real-object forward + gradient correctness and 1-vs-4-GPU timing.
- `params/bench_b1.yml`, `params/bench_b32.yml` — NITER=1 benchmark params
  (recovered params updated to current schema: removed ridge/weight*, mu1/mu2,
  kick_mode, transmission_correction; compiler_configs.disable→default).
- `bench_mode_parallel.py` — mode-parallel micro-benchmark + correctness check.
- `bench_slice_parallel.py` — slice-parallel (plane split) micro-benchmark +
  correctness check; naïve vs pipelined carry transfer.
- `bench_slice_opt.py` — Part 1: half-carry forward-vs-gradient accuracy
  (shows half-carry corrupts the gradient). Part 2: pinned-async carry overlap
  (2-way forward, 1.49×, exact).
- `bench_slice_4way.py` — P-way (1/2/4) slice split via carry chain; compute
  floor vs real-transfer scaling; pinned-async overlap. `NOTRANSFER=1` exposes
  the compute ceiling; `CODEC=half|quarter` compresses the carry.
- `bench_slice_compress.py` — carry-compression accuracy + gradient-safety
  sweep (half/bf16/fp8/int8/low-rank/zfp), the shot-noise gradient floor, and
  end-to-end reconstruction drift. Shows compression is training-safe.

## Recommendation (both batch sizes, this config)
Mode-parallelism over the 4 probe modes (2+2) is the best 2-GPU option:
exact, works at batch 1, ~1.5× on the born compute, keeps cudagraphs, needs
no change to `born_forward`. Slice-parallelism only pays off when the volume
exceeds one device's memory. Data-parallel is blocked at batch 1 and crashes
at batch 32 (DDP bug below).

## Open items
- 2-GPU data-parallel (accelerate) crashes in DDP `_sync_module_states`: a
  model buffer is an expanded/broadcast view (overlapping storage). Fix =
  make that buffer `.contiguous()` before `accelerator.prepare`. Not yet done.
- Neither mode- nor slice-parallelism is wired into the real `ptyrad run`
  training loop yet; numbers above are the isolated born6 forward+backward.

## G. Deep-stack Nz sweep — the batch-1 regime where slice-split DOES win (2026-10-09)
Motivated by the multislice comparison (A100_results cost plots): Born/ISS
beats multislice at batch 1 because its parallel-over-slices cumsum soaks up
idle GPU capacity — and batch/mode parallelism are model-agnostic (multislice
can do both equally), so **slice-split is the only multi-GPU axis multislice
cannot copy** (its slices are sequentially dependent; Born's are a scan).
`bench_slice_nzsweep.py`, batch 1, 256², 4 pmodes, M=6, synthetic volumes:

| Nz | 1-GPU fwd | 4G fwd floor | 4G fwd real (PHB) | 1-GPU f+b | 4G f+b real (pre-placed) |
|----|-----------|--------------|--------------------|-----------|---------------------------|
| 21 | 3.6 ms | 5.3 (0.67×) | 6.6 (0.54×) | 9.7 ms | — (0.66×, §D) |
| 32 | 5.6 | 5.1 (1.11×) | 6.9 (0.81×) | 15.2 | |
| 64 | 11.2 | 5.3 (**2.12×**) | 7.9 (**1.42×**) | 30.2 | 19.0 (**1.59×**) |
| 128 | 22.0 | 7.7 (2.88×) | 12.9 (1.71×) | 59.8 | 34.1 (1.75×) |
| 256 | 43.8 | 13.7 (3.20×) | 22.4 (1.96×) | 118.6 | 57.0 (2.08×) |
| 512 | 87.5 | 26.5 (3.30×) | 41.7 (2.10×) | 236.6 | 100.9 (**2.35×**) |

Findings:
- At this config the 1-GPU curve is linear in Nz from ~32 on (the A100
  saturates); the 4-GPU floor stays FLAT to Nz≈64 — slice-split extends the
  flat region by ~P, exactly the capability multislice lacks.
- **Batch-1 crossover: Nz ≈ 48–64 — even on this no-P2P PCIe box.** The carry
  per hop is one 256² field, Nz-independent, while compute grows ∝ Nz, so the
  transfer tax amortizes with depth. 2.35× f+b at Nz=512 real; NVLink headroom
  to the 3.3× floor (crossover would move down toward Nz≈32).
- This CONFIRMS the long-flagged untested "deep stacks" win regime, and
  resolves the §D verdict's scope: "no multi-GPU win at batch 1" is true at
  Nz=21 — NOT at Nz≥64. 4-GPU f+b must be measured with PRE-PLACED object
  blocks (the object lives distributed for the whole solve); moving blocks
  per step at Nz=512 costs ~0.5 GB/step and masks ~0.9× of speedup.

### §G.2 Order (M) dependence — why NVLink matters MOST for ISS/double
Same sweep at M=2 (double scattering) and M=1 (ISS), batch 1 (`BORN_M` env):

| metric @ Nz=512 | M=6 | M=2 | M=1 (ISS) |
|---|---|---|---|
| 4G compute floor (fwd) | 3.30× | 3.30× | 3.26× |
| 4G real fwd (PHB) | 2.10× | 1.32× | 1.20× |
| 4G real f+b (PHB, pre-placed) | **2.35×** | **1.22×** | **1.03×** |
| f+b crossover Nz (PHB) | ~48–64 | ~64–128 | ~512 (barely) |

- The **floor is M-independent** (crossover Nz≈32, 3.3× at 512 for all M):
  the per-order pipeline fill/drain fear doesn't materialize.
- The **real (PHB) win shrinks as M drops**: transfer *volume* per compute is
  M-independent (hops ∝ M, compute ∝ M·Nz), but the hop *latency* only hides
  behind cross-order overlap — at M=1 the 3-hop chain is strictly sequential
  with nothing to overlap, so the ~2.5 ms/hop host-staging is fully exposed
  (vs ~0.8 ms/hop effective at M=6).
- **Consequence for the multislice benchmark: NVLink is what unlocks
  slice-split for the cheap orders.** On PCIe, deep-stack slice-split pays
  only for born6 (2.35×); for ISS it's a wash (1.03×). With free carries all
  M sit at ~3.3× — so the P-GPU flat-curve-extension claim for ISS/double
  (the headline methods) is an NVLink-conditional claim, while for born6 it
  already holds on PCIe.

## H. Multislice vs born6 at BATCH 1 — can M=6 + NVLink + P GPUs beat MS?
`bench_ms_vs_born_b1.py` (+ compiled follow-up), batch 1, 256², 4 pmodes,
best-of-eager/compiled for each method. "f+b" = forward+adjoint, the recon
cost and the quantity in the A100_results cost plots (measured MS f+b eager
@Nz=64 = 21.5 ms ≈ the plot's ~21 ms — same quantity).

| Nz | MS fwd (cmp) | MS f+b (best=eager) | born6 1G f+b | born6 4G floor f+b | floor vs MS |
|----|--------------|----------------------|--------------|---------------------|-------------|
| 21 | 0.7 | 7.2 | 9.8 | 8.4 | 0.86× |
| 32 | 1.1 | 11.1 | 15.1 | 8.6 | 1.3× |
| 64 | 2.0 | 21.5 | 30.2 | 11.4 | 1.9× |
| 128 | 3.9 | 44.2 | 59.6 | 18.1 | 2.4× |
| 256 | 7.7 | 127.1 | 118.6 | 33.6 | 3.8× |
| 512 | 15.3 | 413.9 | 237.8 | 65.1 | **6.4×** |

- **MS's adjoint is its Achilles heel**: autograd through the sequential
  slice loop is superlinear (f+b/fwd = 27× at Nz=512 — the plots' superlinear
  MS curve). torch.compile makes it WORSE (789 ms vs 414 eager at 512);
  compiled born ≈ eager born (262 vs 238). Structurally: Born's cumsum
  parallelizes the adjoint too (cumsum backward = reversed cumsum); MS's
  adjoint is as sequential as its forward with worse constants.
- **Verdict: YES on f+b.** born6 beats as-implemented MS on ONE GPU from
  Nz≈256, and with 4 GPUs at the NVLink floor from **Nz≈24–32**, reaching
  6.4× at Nz=512 (PCIe real: 4.1×). Forward-only: NO at P=4 (26.8 vs 15.3 ms
  at 512; P=8 ≈ tie) — but recon pays f+b, not fwd.
- Caveat: a hand-rolled optimal MS adjoint (~3× compiled fwd ≈ 46 ms @512)
  would beat the P=4 floor (65 ms); born then needs ~8 GPUs (slice×mode,
  below) to win again. Against ptyrad's actual MS, 4 GPUs suffice.

### H.2 Probe modes: the SECOND differential axis (corrects §E's claim)
User observation, confirmed: MS batch-1 wall-time is pmode-INSENSITIVE
(plots: MS@N=64 ≈ 19.6 ms at p1 vs 20.9 ms at p6 — launch-bound), while
Born is pmode-proportional (compute-bound; p6 curves bend at N≈32-64 where
p1 is flat). So **mode-split divides Born's time by ~P_m and does ~nothing
for MS** — NOT model-agnostic at batch 1 as §E assumed (that claim holds
only for batch-split). Born's two differential axes COMBINE: P = P_m × P_z
GPUs (e.g. 8 = 2 mode groups × 4 slice blocks) → floor ≈ 65/2 ≈ 33 ms at
Nz=512 p4, beating even an ideal MS adjoint. With enough NVLinked GPUs,
born6 wins any depth at batch 1; MS has no parallel axis to answer with
(modes don't change its time, slices can't split).

## I. Combined mode x slice split (`bench_combo_modeslice.py`) — the 8-GPU plan
PM mode-groups x PZ slice-blocks on PM*PZ GPUs (env PM/PZ/BORN_M/NOTRANSFER).
Validated 2x2 on this box (exact: 7e-8). Batch 1, M=6, 256², p4, Nz=512,
f+b INCLUDING each config's required comms (slice: none — object grads live
distributed; mode: cross-replica grad-reduce ∝ Nz):

| 4-GPU config | f+b real (PHB) | f+b floor (NVLink-attainable) |
|---|---|---|
| combo 2m×2z | **93.9 ms (2.52×)** | 79.0 (3.00×) |
| pure slice 1m×4z | 100.9 (2.35×) | **64.8 (3.65×)** |
| pure mode 4m×1z | 128.5 (1.84×) | grad-reduce-bound |

- **On PCIe, combo 2×2 is the best 4-GPU config at deep Nz** (fewer chain
  hops per group than 4z; only a 2-way grad-reduce ~11 ms vs pure mode's
  4-way full-object reduce ~57 ms).
- **On NVLink, pure slice wins at equal GPUs** (best floor): it PARTITIONS
  the object, so object gradients need zero communication; every mode group
  added costs a replica grad-reduce ∝ Nz.
- **8-GPU NVLink extrapolation** (slice floor eff. 91% at P=4): 1m×8z f+b
  ≈ 34–40 ms at Nz=512 → beats even an idealized hand-rolled MS adjoint
  (~46 ms), closing §H's caveat. 4-GPU NVLink (65 ms) already beats
  as-implemented MS (414 ms) by 6.4×.
- GOTCHA (cost a 2× artefact before fixing): a CPU-resident `omode_occu`
  forces a pageable H2D at each chain's end, which **blocks the host until
  that GPU chain completes and serializes the groups** (pure mode measured
  0.9×!). Pre-place every small tensor, including scalars/occupancies.

## J. Slice-split in the REAL 20-iter loop at Nz=21 — final verdict (2026-10-09)
User-requested end-to-end comparison vs the baseline/modesplit runs (same
seed 42, fresh simu init). Three slice variants were built and validated
(losses match baseline EXACTLY, e.g. iter1 0.4239, iter2 0.3746):

| arm (4 GPUs) | engine | s/iter | per batch |
|---|---|---|---|
| baseline 1-GPU | stock | **64.5** | **15.7 ms** |
| slicesplit | ripple carry, patches via cuda:0 | 102.8 | ~25 ms |
| slicedist | per-GPU object LEAVES (grads+Adam stay on-device), ripple | ~106 | ~26 ms |
| slicedist | + `born_forward_dist` two-level scan (new SOURCE engine) | ~107 (5-iter smoke) | ~25 ms |

**Why 15.7 ms/batch is unbeatable at Nz=21 on this box:** born f+b is 9.7 ms
of it; the 4-GPU ZERO-TRANSFER floor is 8.4 ms → max possible saving ≈1.3
ms/batch (~8%/iter) even with perfect NVLink; the real no-P2P chain ADDS ~6
ms of hop latency (18 serial hops/step, interconnect-bound, restructure-
immune) and the distributed housekeeping (4-device cropping, probe replicas,
single-tensor Adam over 9 leaves) ~3 ms. Full 20-iter run skipped — the
5-iter smoke (exact losses, save path exercised) is the record.

**Source improvements landed (useful at Nz≥64 / on NVLink, not at 21):**
- `forward_models/born_dist.py`: `born_forward_dist` — proper TWO-LEVEL SCAN
  (phase A: local scatter+scan all GPUs in parallel; B: carries as pure
  transfer+add; C: psi updates in parallel) replacing the ripple carry that
  serialized compute behind each hop. Exact (4e-8). Nz=64 f+b: 1.60→**1.90×**,
  fwd 1.41→1.77×; at Nz=21 unchanged (hop-latency-bound).
- `create_optimizer`: param groups spanning multiple CUDA devices now auto-
  fall back to the single-tensor path (foreach/fused Adam errors on
  multi-device groups); explicit configs take precedence.
- LBFGS support REMOVED (closure stepping, no per-param lr, incompatible
  with multi-device groups); clear ValueError.
- `run_cmp_modesplit.py` roles slicesplit/slicedist: the distributed-object
  pattern (per-device leaves, per-block sparse loss, gather-constrain-scatter
  once per iter) — the template for an NVLink deep-stack solver.

FINAL for this sample (21 slices, batch 1): 1 GPU per reconstruction remains
optimal; mode-split ~parity; slice-split 0.6×. Slice-split's domain is
Nz≥64 (PCIe) / Nz≥32 (NVLink floor) and volumes exceeding one GPU's memory.

## K. 32-slice head-to-head: born6 slice-2 (2 GPU) vs multislice (1 GPU)
User-requested, real 5-iter reconstructions (dz 10->6.5625 A, same 210 A
thickness, seed 42). Concurrent runs contended (~89.5/89.9 s/iter, std +-8 s
— host PCIe/CPU shared between MS data loads and carry hops); SOLO reruns
give the clean numbers:

| arm (Nz=32, solo) | s/iter | per batch | iter-2 loss |
|---|---|---|---|
| multislice 1-GPU | **74.6** | 18.2 ms | 0.3625 |
| born6 slice-split 2-GPU (slicedist, SLICE_P=2) | 78.1 | 19.1 ms | 0.3625 |

- **Dead heat, slight MS edge (0.96x) on this PCIe box**; convergence
  equivalent (0.3158 vs 0.3161 at iter 5). Decomposition matches the
  microbenches: MS 11.1 f+b + ~7 ms housekeeping; born6 11.7 + ~7 + refit.
- The split DID rescue born6: 1-GPU born6 @32 would be ~22 ms/batch
  (~92 s/iter) -> 2-GPU parity with MS.
- **NVLink projection**: born f+b 11.7 -> 8.9 (measured floor) => ~68.5
  s/iter vs MS 74.6 = **~1.09x end-to-end** at Nz=32. The 1.25x born-only
  ratio dilutes through the fixed ~7 ms/batch both methods share. Deeper
  stacks grow it (~1.5x at Nz=64).
- 2-GPU slicedist at Nz=21 (smoke_slicedist_2gpu): 67.4 s/iter vs 64.5
  baseline — parity; the P=2 chain (6 hops) + halved housekeeping recovered
  the 4-GPU version's losses (105.6).

### K.2 Nz=42: the PCIe crossover (2026-10-09)
Same head-to-head at 42 slices (dz=5 A, 210 A), 2 iters, arms CONCURRENT on
disjoint GPUs (symmetric contention, std +-0.7 s): MS 1-GPU **92.9** s/iter
(22.7 ms/batch) vs born6 slice-2 2-GPU **89.7** (21.9 ms/batch) — **born6
WINS 1.04x**, convergence equal (0.3560 vs 0.3562 at iter 2). Depth trend:
0.86x @21 -> 0.96x @32 -> **1.04x @42** -> ~1.5x @64 (SSH projection).
**Practical crossover for born6+2GPU over multislice: Nz ~ 35-40 on plain
PCIe; NVLink moves it toward ~30 and adds ~10% at 42.**

### K.3 Adaptive order growth + slice split (Nz=42, 2 GPU, 5 iters)
born_iterations 1 + grow_tol 1e-2 (start at ISS, refit grows the order) runs
unmodified through the DISTRIBUTED solver (forward_dist reads
model.born_iterations per call). Trajectory: M=1 (42.3 s/iter) -> 2 (51.1)
-> 3 (61.0) -> 4 (70.8) -> settles M=5 (79.2; residual 6.7e-3 < tol).
**Avg 60.9 s/iter = 1.47x vs fixed-M6 slice-2 (89.7) and 1.53x vs MS 1-GPU
(92.9)**; steady state M=5 beats MS 1.17x (better than fixed-M6's 1.04x —
the refit finds dz=5 A needs only M=5). No convergence penalty: iter-5 loss
0.3161 on the fixed-order trajectory. Best deep-stack config on this PCIe
box: **born-grow + 2-GPU slice split.**

# Chin 4A anti-aliasing (propagator aperture + band-limited g)

Branch `chin-antialias` (worktree `~/ptyrad-chin-aa`). Two config flags, defaults
unchanged (off):

- `propagator_antialias: f` — circular aperture at `f * k_Nyquist` multiplied
  into every Chin drift kernel (`H_half` and the composed kernels). The edge is
  cosine-tapered over 2 k-pixels (abTEM-style); a hard edge rings in real space
  and NaNs the batch-1 PSO run within one iteration (obja crosses zero near
  step ~1300/4096).
- `chin_g_bandlimit: s` — the gradient-correction term g is built from
  spectrally band-limited chi (Gaussian low-pass, sigma_k = `s * pi/dx`,
  spectral gradients on the replicate-padded canvas) instead of raw 4th-order
  finite differences.

Forward bench: `test/bench_chin_antialias.py` -> `test/bench_chin_antialias.json`.
Specimen = recovered 21-slice PSO object (`20260927_lt21_ref_s42_full`,
iter 100), its 4-mode probe, dz = 10 A, 6 probe positions; errors are relative
exit-wave L2 vs a fine-sub-slice reference (n = 16 vs 32 convergence ~3e-3,
Strang cross-check agrees), error math in f64 (f32 repeats match to ~1e-5).

## 1. Forward bench (f64, mean behaviour across 6 positions)

Variants: `fd` = production Chin 4A (raw FD g); `g0` = g off; `specbl` =
band-limited spectral g (`chin_g_bandlimit: 0.333`); `specblh` = same with
harder cutoff; `fdbl` = band-limit applied to the FD g; `aa_*` = with
`propagator_antialias: 0.667` (tapered, production mask); `lt1`/`lt2` =
Lie-Trotter at 1 / 2 sub-slices per slab.

| arm | 21sl full | 21sl in-band | 21sl consistent | 11sl full | 11sl in-band | 11sl consistent |
|---|---|---|---|---|---|---|
| chin4a_fd | 0.0345 | 0.0191 | 0.0345 | 0.1182 | 0.1036 | 0.1182 |
| chin4a_g0 | 0.0365 | 0.0227 | 0.0365 | 0.1258 | 0.1127 | 0.1258 |
| chin4a_specbl | 0.0349 | 0.0200 | 0.0349 | 0.1194 | 0.1052 | 0.1194 |
| chin4a_specblh | 0.0348 | 0.0197 | 0.0348 | 0.1177 | 0.1033 | 0.1177 |
| chin4a_fdbl | 0.0346 | 0.0194 | 0.0346 | 0.1185 | 0.1040 | 0.1185 |
| chin4a_aa_fd | 0.1437 | 0.0260 | 0.0187 | 0.1736 | 0.1018 | 0.1007 |
| chin4a_aa_bl | 0.1438 | 0.0264 | 0.0193 | 0.1735 | 0.1015 | 0.1005 |
| chin4a_aa_g0 | 0.1442 | 0.0287 | 0.0222 | 0.1792 | 0.1111 | 0.1100 |
| lt1 | 0.1357 | 0.1254 | 0.1357 | 0.3565 | 0.3446 | 0.3565 |
| lt2 | 0.0571 | 0.0517 | 0.0571 | 0.1283 | 0.1183 | 0.1283 |

Honest reading:

- The "full" column is a trap for the aa arms: comparisons against the
  un-apertured ground truth are dominated by out-of-band content. Even the
  ground truth with the aperture applied differs from the raw ground truth by
  **gt_vs_gt_aa ≈ 0.143** (0.1429–0.1433 across positions) — exactly the level
  the aa arms sit at. Quote the in-band and self-consistent numbers instead.
- In-band (project both waves onto the aperture band): at 21sl the aa arms are
  slightly WORSE than non-aa (0.0260 vs 0.0191) — at dz = 10 A the aperture
  removes band-edge content the un-apertured model propagates usefully. At the
  11-slab slicing they are marginally better (0.1018 vs 0.1036).
- Self-consistent (aperture applied to the reference propagation too, i.e. the
  physics the antialias model class claims to keep): the aperture nearly halves
  the 21sl error (0.0187 vs 0.0345) and improves 11sl (0.1007 vs 0.1182) — the
  apertured model is more faithful to its own band-limited physics.
- Parity check: the tapered (production) aperture vs the earlier hard-edged
  bench mask shifts these numbers only marginally — 21sl consistent
  0.0183 -> 0.0187, 11sl in-band 0.1019 -> 0.1018, gt_vs_gt_aa 0.1395 -> 0.1429.
  No conclusion changes; all JSON numbers above are the tapered (production
  parity) ones.

## 2. g-variant null at production slicings

At both production slab counts, every g-ON variant is within ~1.5% relative of
raw FD g (21sl: fd 0.03447, fdbl 0.03463, specblh 0.03476, specbl 0.03493):
**band-limiting g buys nothing at production dz** — the raw FD g is not
aliasing-limited at dz = 10–19 A. Turning g off entirely costs a modest ~6%
(21sl 0.0365, 11sl 0.1258), and more with the aperture on (aa_g0 in-band
0.0287 vs aa_fd 0.0260 at 21sl).

The dz-sweep order test tells the same story at the asymptote: raw-FD g and
band-limited g both keep the high-order tail (tail slopes 3.8 / 3.3–3.5),
g0 degrades it (2.8), and the aa arms plateau at 0.143 against the un-apertured
reference (out-of-band floor, as expected). Against a band-limited-chi
reference the fd and specblh tails are identical (3.47 both).

## 3. Timing

Forward pass, f32, best-of-5 CUDA events (21sl / 11sl, ms):
chin4a with g 4.38 / 2.63; g0 4.66 / 2.58; aa+g 4.41 / 2.44; lt2 4.15 / 2.39;
lt1 2.76 / 1.62. The aperture is free (one precomputed mask multiply); the
band-limited g preparation costs ~1.7x the FD g prep (4.26 vs 2.49 ms at 21sl)
for zero accuracy gain — no reason to use it in production.

Reconstruction wall time (100 iters, batch 1, one A100 each, s/iter):
archived chin4a_matched baseline 50.6; arm (i) aa + raw g 52.0 (the aperture
is ~free, +3%); arm (ii) aa + band-limited g 58.5 (+16%); arm (iii) aa + g off
29.4 (g is ~40% of the per-iteration cost; LT11 runs at 24.0).

## 4. Reconstruction arms (PSO matched protocol, 11 slices, dz 19.1 A, batch 1, 100 iters)

Three arms, each an exact copy of the archived `pso_chin4a_matched` protocol
with only the new flags added (`demo/params/pso_chin4a_aa_n100.yml`,
`..._aa_blg_n100.yml`, `..._aa_g0_n100.yml`; outputs under
`demo/output/high_order_benchmarks/chin_aa/`):

- (i) `propagator_antialias: 0.667`, raw FD g
- (ii) + `chin_g_bandlimit: 0.333`
- (iii) `propagator_antialias: 0.667`, `splitting_gradient_term: false` (g off)

Fine-band metric (`scripts/chin_aa_fine_band.py`, identical to
`~/papers/paper_1/figs-p2/paper2_pso_fine_band.py`): depth-summed objp phase,
common centred crop, annulus 0.3–0.6 Nyquist, power ratio vs the
`20260927_lt21_ref_s42_full` benchmark at iter 100. Known ladder reproduced:
1 : 0.876 (chin4a 11sl baseline) : 0.136 (LT 11sl).

All six panels are each run's FINAL model = `model_iter0100.hdf5` (every arm
ran the full 100 iterations, so the budget is equal; no 50-vs-100 asymmetry).
Results (`scripts/chin_aa_fine_band.json`, runs of 2026-10-22):

| run | fine-band ratio | s/iter | final loss |
|---|---|---|---|
| LT 21 (benchmark) | 1.000 | — | — |
| Chin 4A 11sl baseline (archived) | 0.876 | 50.6 | — |
| LT 11 (archived) | 0.136 | 24.0 | — |
| (i) aa + raw FD g | 0.845 | 52.0 | 0.2931 |
| (ii) aa + band-limited g | 28.15 (DIVERGED) | 58.5 | 0.2906 |
| (iii) aa + g off | 0.777 | 29.4 | 0.2926 |

Verdicts vs the 0.876 baseline:

- **Arm (i): the antialias aperture does not help reconstruction** at the
  production protocol — 0.845 vs 0.876, a mild fine-band cost, consistent with
  the forward bench's in-band numbers being a wash-to-slightly-worse at
  production dz. (Single runs, no seed replicates: a ~0.03 difference should
  be read as "no improvement, possibly slightly worse", not as a precise gap.)
- **Arm (ii): band-limited g is UNSTABLE in the reconstruction loop.** The
  28.15 "ratio" is divergence, not resolution: the fine-band annulus power
  grows monotonically (3.9e10 at iter 20 -> 1.2e11 at 50 -> 9.1e11 at 100)
  and the noise accumulates in the exit-side slices (per-slice objp std rises
  0.21 entrance -> 1.43 exit; obja hits -450 before the mirrored-amplitude
  constraint clips it). The data loss is meanwhile the LOWEST of the three
  arms (0.2906) — the runaway power lives in data-null high-k channels, so
  the loss never sees it. Do not use `chin_g_bandlimit` in-loop.
- **Arm (iii): dropping g is stable and 43% cheaper but costs fine band**
  (0.777, below arm (i)'s 0.845) — the gradient-correction term is doing real
  work in the reconstruction, not just in the forward error metric.

## 5. Absorption-theorem check

Hypothesis: g-only changes modify the per-slab transmission correction, which
the reconstructed object can absorb — so arms (i), (ii), (iii) should land at
the same fine-band level; only the kernel (aperture) change can move the
reconstruction relative to the 0.876 baseline.

Outcome: **the absorption expectation fails in both g directions.**

- Changing the FORM of g (arm ii) does not merely re-gauge the object — it
  destabilizes the solver (progressive exit-slice noise, above). The
  band-limited g changes the per-step gradient geometry, and batch-1 SBCD
  amplifies the difference instead of absorbing it.
- REMOVING g (arm iii) moves the converged-at-100-iters result down
  0.845 -> 0.777 even though the run is perfectly stable. At dz = 19.1 A the
  object does not (within this iteration budget) re-absorb the lost
  correction; some of the gap may be slower convergence rather than a
  different fixed point, but at equal budget the g term earns its 1.8x
  per-iteration cost in fine-band terms.
- The kernel change moved the reconstruction (0.876 -> 0.845), which the
  theorem permits — here it moved it the wrong way.

Bottom line: no antialias variant beats the existing Chin 4A production
configuration. The aperture is physically defensible but buys nothing at
production sampling (and mildly hurts fine band); the band-limited g is
forward-only tooling (keep for the bench, never in-loop); g stays ON.

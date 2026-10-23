# Chebyshev exact-generator slab forward model

Branch `cheb-forward`; code in `src/ptyrad/forward_models/chebyshev.py`
(`solver_type: chebyshev`, slab count `cheb_n_slab`, hooks in
`models.py`/`reconstruction.py`), validation in
`scripts/validate_chebyshev_forward.py`, metric in `scripts/cheb_fine_band.py`,
recon yml `demo/params/pso_cheb3_matched.yml`, numbers in
`demo/output/chebyshev/forward_validation.json` and `fine_band_ratios.json`.

## Idea

Evaluate the one-way slab exponential

    U(dz) = exp( i k0 dz (sqrt(1 + mu + eps) - 1) ),
    mu  = -K_perp^2/k0^2  (band-projected Fourier multiplier),
    eps = 2(n-1) + (n-1)^2,  (n-1) = phase_slab/(k0 dz_slab),

EXACTLY (to a 1e-6 truncation tolerance) per object slab with a Chebyshev
operator polynomial in B = mu + eps, so that few, thick slabs carry zero
splitting error inside each slab. The question: with the PSO depth
information supporting only ~2.4 independent slabs (below), does a 3-slab
exact-propagator reconstruction keep the fine-band quality of the 21-slice
Lie-Trotter benchmark at 1/7 the slice count?

## Implementation notes

- **Shift-stabilised recurrence.** The three-term recurrence runs on
  B = mu + eps over the shifted interval [a-1, b-1], with the scalar
  function written as f(x) = exp(i theta x / (1 + sqrt(1+x))) so
  sqrt(1+x)-1 is never formed by subtraction. The naive recurrence on
  1 + mu + eps expands over a sliver around y = 1 and loses an order of
  magnitude to cancellation in float32 (measured below); in float64 the
  two forms agree to ~1e-15 (rel_l2 0.13902868729669524 vs ...363).
- **Padded-interval detached coefficients.** With the band projection,
  P mu P has spectrum in [mu_min, 0] and eps is a pointwise multiplier in
  [eps_min, eps_max], so by Weyl the spectrum of B lies in
  [mu_min + eps_min, eps_max]. The interval is computed from the current
  object, padded 30% per side, and coefficients (host-side numpy f64,
  first-kind-node interpolation, tail-truncated at |c| >= 1e-6, degree cap
  512) are rebuilt only when the object's eps range escapes the padded
  interval — a cheap host check once per iteration
  (`ChebyshevSlabKernel.refresh`). Coefficients are detached constants;
  gradients flow only through the B applications, so the fixed linear
  recurrence is cleanly autodiff-able (`gradcheck` passes in f64; this is
  the point of Chebyshev over Lanczos). Band projection: mu is zeroed for
  k_perp >= 0.9 * min(k0, sqrt(2) K_Nyq) = 42.72 1/A here, i.e.
  mu_min = -1.79e-2.
- **Hybrid absorption screen.** B must stay Hermitian (real interval), so
  the amplitude channel does not enter it. The slab amplitude screen
  A = prod(slice amps) (computed as exp(sum(log)) — prod's backward
  host-syncs and breaks CUDA-graph capture) is applied pointwise at the
  slab centre: one slab is U(dz/2) * A * U(dz/2), two half-slab Chebyshev
  applications around a Strang-style absorption split (exact for amp = 1).
  Stored coefficients are therefore those of the HALF-slab exponential.
- **Fused + CUDA-graphed recurrence.** Fast path pre-folds the interval
  map into the multipliers (mu_t in k-space, eps_t in real space) and
  fuses the pointwise tail of each recurrence step (t2 = G + eps_t*t1 -
  t0; acc += c_k*t2) into one compiled real-view kernel
  (`PTYRAD_CHEB_EAGER=1` forces the reference path). On top, the whole
  per-position forward+backward is captured as a CUDA graph
  (`PTYRAD_CHEB_GRAPH=1`, `make_graphed_callables`, re-captured on
  coefficient rebuild or shape change; verified bit-identical outputs and
  gradients) — essential because a degree-~100 recurrence is hundreds of
  tiny kernel launches per position at batch 1.

## Forward validation (archived 21-slice PSO object, 210 A)

Specimen: `chin_results/20260927_lt21_ref_s42` iter-100 model, 21 x 10 A,
central 256^2 patch, 4 probe modes. Ground truth: 32-sub-slice (0.3 A)
Lie-Trotter, angular-spectrum kernel, f64; 16-vs-32 sub-slice convergence
3.05e-3, i.e. the GT is converged ~40x below the smallest model error
measured against it. Metric: relative L2 after optimal global-phase
alignment. FFT pairs exclude the detector FFT.

| model (f64)        | rel L2 vs GT | FFT pairs |
|--------------------|-------------:|----------:|
| LT, 21 slices      |       0.1295 |        21 |
| Chin 4A, 11 slices |       0.1665 |        22 |
| Chebyshev, 3 slabs |       0.1390 |       642 |
| Chebyshev, 1 slab  |       0.2209 |       566 |

- Chebyshev-3 (3 slabs of 70 A) lands between the 21-slice LT reference
  and the 11-slice Chin 4A model — with 3 object degrees of freedom in z
  it is MORE accurate than Chin 4A at 11.
- f32 roundoff isolation (f32 field vs the f64 shifted field): shifted
  8.3e-6, unshifted 1.1e-4 — the shift stabilisation buys 13x and keeps
  f32 roundoff four orders below model error.
- `cheb3_vs_homogenised_gt` = 0.0441. This diagnostic recomputes the fine
  sub-sliced LT ground truth on the z-HOMOGENISED 3-slab medium (each
  7-slice slab replaced by its z-mean phase/log-amp) and compares
  Chebyshev-3 against THAT. It removes the homogenisation error — the
  information lost by representing 21 slices as 3 slabs, which no 3-slab
  model can avoid — so what remains is the model discrepancy proper: the
  Chebyshev generator sqrt(1+mu+eps) differs from the fine-LT limit
  sqrt(1+mu) + (n-1) at order mu*eps (plus 1e-6 truncation). Reading:
  of the total 0.1390 error, ~0.132 (in quadrature) is homogenisation and
  only 0.044 is the generator cross term; the polynomial evaluation
  itself is exact. Fewer-slice error is dominated by the representation,
  not the propagator — which is exactly what an exact-generator slab
  model is supposed to achieve.

## Degree / cost accounting

- Half-slab degree ~= k0*dz_slab*(sqrt(1+hi)-sqrt(1+lo))/2 + cushion
  ~= dz_slab*k_band^2/(4 k0) for the mu-dominated interval: estimate
  100.5, actual 103 (live recon) / 107 (archived object) at dz_slab = 70 A.
  One FFT pair per recurrence step, two half-slab applications per slab:
  3 x 2 x 107 = 642 pairs (vs 21 for LT21).
- Degree is linear in dz_slab, so total FFT pairs are nearly independent
  of slab count (642 at 3 slabs vs 566 at 1 slab): the band limit k_band
  sets the cost, not the slab partition. Fewer slabs save memory/DoF, not
  propagation FLOPs.
- Exit-wave wall (f32, 256^2 x 4 modes, best of 5): LT21 1.59 ms, Chin4A-11
  2.35 ms, cheb3 165.5 ms eager / 95.3 ms torch.compiled — the 30x FFT-pair
  ratio shows up as ~60x eager because the recurrence is launch/bandwidth
  bound, recovered partly by fusion and (in recon) CUDA graphs.

## Depth-information framing

The PSO stack is 210 A. Measured PSO depth information is rank ~3 with
dz_min ~= 86 A axial sampling limit (chin-splitting / depth-diagnosis
results), i.e. 210/86 ~= 2.4 independent slabs are recoverable. A 3-slab
model is therefore information-matched: the 21-slice benchmark oversamples
z ~7x relative to what the data can determine (held together by z-blur
regularisation), while 3 slabs assign one optimizable layer per
independently-resolvable depth channel and let the exact slab propagator
carry the intra-slab physics.

## 3-slab reconstruction vs the ladder

Run: `pso_cheb3_matched.yml` — 3 slices x 70 A, chin4a_matched b1-n100
protocol (batch 1, full 4096 positions, Adam, oalr/oplr 1e-3-class with
the per-slice-count rescalings of the matched protocol, sparse 0.0524,
zblur 0.5238), 50 iterations, PTYRAD_CHEB_GRAPH=1, A100 80GB PCIe.
Metric: fine-band (0.3-0.6 Nyquist) power of the depth-summed objp,
ratio to the LT21 benchmark, computed exactly as
`figs-p2/paper2_pso_fine_band.py` (common centre crop 639^2). Ladder:
LT21 = 1, Chin4A-11 = 0.876, LT11 = 0.136.

Run `20261022_cheb3_matched_n50` completed 50/50 iterations (exit 0,
2 h 58 min 56 s total). ONE coefficient build (half-slab degree 103,
interval [-2.329e-2, 5.374e-3]) and ONE CUDA-graph capture served the
whole run — the 30%-padded interval was never escaped by the object.

| arm (fine band, ratio to LT21)      | iters | ratio | data loss | s/iter |
|-------------------------------------|------:|------:|----------:|-------:|
| LT, 21 slices (benchmark)           |   100 | 1.000 |    0.2642 |   31.4 |
| Chin 4A, 11 slices (matched proto)  |   100 | 0.876 |    0.2647 |   50.4 |
| LT, 11 slices                       |   100 | 0.136 |    0.2712 |   23.9 |
| **Chebyshev, 3 slabs**              |**50** |**0.530**| **0.2785**|**213.4**|

- Chebyshev final: total loss 0.3021 = data (loss_single) 0.2785 +
  sparse 0.0236; measured 213.38 +- 1.23 s/iter (A100 80GB PCIe, batch 1,
  4096 positions, PTYRAD_CHEB_GRAPH=1) = 52.1 ms/position fwd+bwd, vs
  8.3 ms (LT21, 31.4 s/iter at N3769) and 12.3 ms (Chin4A-11).
- Fine-band trajectory (iters 5..50, step 5): 0.011, 0.047, 0.106,
  0.168, 0.220, 0.268, 0.327, 0.412, 0.483, 0.530 — still rising
  ~+0.01/iter at termination, no plateau (data loss meanwhile nearly
  flat: 0.2789 -> 0.2785 over the last three iterations; the fine band
  sharpens long after the loss saturates, exactly as in the baselines).

**50-vs-100-iteration caveat, stated plainly:** this is a 50-iteration
run scored against 100-iteration baselines. The fine-band metric grows
monotonically through training and had not plateaued at iter 50, so
0.530 is a LOWER BOUND on the protocol-matched number and the comparison
is biased against the Chebyshev arm. The only safe claims are
directional: at half the iteration budget, 3-slab Chebyshev already
sits 3.9x above the fully-converged 11-slice Lie-Trotter arm (0.136 at
100 iters) and at 60% of the fully-converged Chin4A-11 arm (0.876);
whether it reaches Chin4A/LT21 parity requires a matched 100-iteration
run, which was outside this budget.

## Verdict

- **Quality at fewer slices: yes, with the iteration caveat.** The
  3-slab exact-generator model clearly does NOT collapse the way
  Lie-Trotter does when slices are removed (LT11 = 0.136 with 100
  iterations; plain 3-slice LT would be far worse): at 50 of 100
  iterations it already recovers 0.530 of the benchmark fine-band power
  and is still climbing without plateau, consistent with the forward
  validation (cheb3 field error 0.139 vs LT21's 0.129, and only 0.044
  of it is propagator error) and with the information argument (~2.4
  independent slabs in the data). Parity with the 21-slice benchmark is
  plausible but not demonstrated at this budget.
- **Wall cost: expensive, and not reduced by using fewer slabs.**
  213 s/iter vs 31-50 s/iter for the Trotter/Chin arms (6.3x LT21,
  4.2x Chin4A-11 per position), because the Chebyshev degree is set by
  k0*dz_slab*k_band^2-type scaling, not by the slab count: total FFT
  pairs are ~constant in the partition (642 at 3 slabs, 566 at 1). The
  payoff of 3 slabs is information-matched degrees of freedom (3
  optimizable layers, no z-blur crutch), memory, and zero intra-slab
  splitting error — not propagation FLOPs. As a drop-in speed play it
  loses; as an exact few-DoF depth model it works.

## Band-crop optimisation + 100-iteration runs (2026-10-23)

### Band convention found, and what was changed

The kernel shipped with the WIDE paper-script band: `k_band =
0.9 * min(k0, sqrt(2)*K_Nyq)` = 42.72 1/A here = **1.273 K_Nyq** (the
0.9-of-corner convention), with the mask baked into the mu multiplier
only — off-band field components were kept and propagated with mu = 0.
Since the degree scales as ~ dz*k_band^2/(4 k0), the mu-span was paying a
(1.273/0.667)^2 ~ 3.6x degree penalty for spectral content the data does
not contain (the measured 120^2 dp occupies < 0.47 Nyquist of the padded
256^2 grid; 2/3 Nyquist is the antialias standard).

Changes (solver numerics only; shift-stabilised recurrence and detached
coefficients untouched):

1. **2/3-Nyquist hard band** (`cheb_band_frac` now a fraction of K_Nyq,
   default 2/3, safety-clamped below k0): k_band 42.72 -> 22.38 1/A,
   mu_min -1.791e-2 -> -4.916e-3. The SAME hard mask is baked into mu and
   applied to the field at the ENTRANCE of every half-slab polynomial
   application and at the exit (the real-space eps/amplitude multiplies
   scatter O(eps) content off-band inside an application; the projection
   kills it before it accumulates or reaches the detector mis-propagated).
   Cost: 2*n_slab+1 extra FFT pairs per forward (~3% at 3 slabs).
2. **Tail tolerance 1e-6 -> 1e-4** (`cheb_tol` default; logarithmic
   saving, free — truncation sits 3 orders below the 0.139 model error).
3. Lanczos/power interval probe: SKIPPED — after the crop the band edge
   is occupied by design (scattering fills the antialias band), so Ritz
   extremes would reproduce the mask bound; the eps span is host-checked
   already. Nothing left for it to win.

### Degree / cost, before -> after

| quantity | wide band, tol 1e-6 | 2/3 Nyq, tol 1e-4 |
|---|---:|---:|
| half-slab degree (recon init, dz 70 A) | 103 | **33** |
| half-slab degree (archived object)     | 107 | **37** |
| FFT pairs per 3-slab forward           | 642 | 222 (+7 proj) |
| exit wave f32 256^2 x4 modes, fused    | 165.5 ms | 51.3 ms |
| exit wave, torch.compiled              | 95.3 ms | 34.6 ms |
| recon s/iter (b1, N4096, CUDA graphs)  | 213.4 | **82.8 +- 2.2** |
| recon ms/position fwd+bwd              | 52.1 | **20.2** |

One coefficient build (degree 33) and ONE CUDA-graph capture served the
whole 100-iteration run; the 30%-padded interval was never escaped.

### Validation gate (rerun of the committed machinery)

- **In-band cheb3 rel L2 = 0.1387 vs the committed 0.1390** — within the
  0.002 acceptance; the crop is cost, not accuracy. In-band = both fields
  hard-projected onto the solver band before the metric (the committed
  number was effectively in-band already: the wide band kept everything).
- Full-band cheb3 = 0.1956. The shift is EXACTLY the removed out-of-band
  GT content: the fine-LT ground truth holds 1.937e-2 of its power beyond
  2/3 Nyquist (probe tails + thermal/high-angle scattering of the
  21-slice model object), and sqrt(0.1387^2*(1-0.0194) + 0.0194) =
  0.1955. A model that zeroes that band by construction pays it as a
  constant; it carries no object information at the detector (the
  measured dp ends at 0.47 Nyquist).
- In-band ladder for context: LT21 0.1181, Chin4A-11 0.1150, cheb1
  0.2214 (f64 = f32 to 4 digits).
- f32 shift-stabilisation roundoff: shifted 3.5e-6 (was 8.3e-6 at degree
  107), unshifted 1.16e-4 — the stabilisation is still load-bearing and
  the margin improved with the shorter recurrence. gradcheck passes.

### 100-iteration 3-slab reconstruction (regularised protocol arm)

`pso_cheb3_opt_n100.yml` = `pso_cheb3_matched.yml` with ONLY `NITER: 50
-> 100` and `prefix` changed; PTYRAD_DISABLE_ISS_PRECOND=1,
PTYRAD_CHEB_GRAPH=1, A100 80GB. Completed 100/100 in 2 h 19 min 34 s
(82.78 +- 2.20 s/iter).

| arm (fine band, ratio to LT21) | iters | ratio | band corr | s/iter |
|---|---:|---:|---:|---:|
| LT, 21 slices (benchmark)   | 100 | 1.000 | (1)    | 31.4 |
| Chin 4A, 11 slices          | 100 | 0.876 | +0.490 | 50.4 |
| LT, 11 slices               | 100 | 0.136 | +0.321 | 23.9 |
| Chebyshev 3 slabs, wide band| 50  | 0.530 | +0.423 | 213.4 |
| **Chebyshev 3 slabs, optimised** | **100** | **0.592** | **+0.429** | **82.8** |

("band corr" = Pearson correlation of the 0.3-0.6 Nyq band-filtered
depth-summed phase against the benchmark, `scripts/cheb_band_corr.py` —
a fidelity check the power ratio alone cannot give.)

- Fine-band trajectory (iters 5..100, step 5): 0.007, 0.039, 0.094,
  0.160, 0.219, 0.263, 0.298, 0.331, 0.379, 0.441, 0.486, 0.515, 0.537,
  0.554, 0.566, 0.575, 0.581, 0.586, 0.589, **0.592** — a clean plateau
  (+0.003 over the last 10 iterations). The 50-iteration caveat of the
  previous section is now resolved: **3-slab Chebyshev converges to
  ~0.59 of the benchmark fine-band power**, 4.4x above LT11, but it does
  NOT reach Chin4A-11 (0.876) or LT21 parity. Band corr +0.429 lands
  between LT11 (+0.321) and Chin4A (+0.490), consistent with the power
  verdict.
- Matched-iteration note, stated honestly: at iter 50 the optimised run
  reads 0.441 vs the wide-band run's 0.530 (corr +0.351 vs +0.423) —
  the band-cropped arm converges somewhat more slowly per iteration
  (single runs with random batch order; the in-band forward physics is
  validated identical to 3e-4, so this is trajectory noise and/or the
  loss-floor change, not model error). By iteration 100 it is past
  everything the 50-iteration wide-band run reached, at 39% of the
  wall per iteration.
- Loss: final total 0.4457 = data 0.4243 + sparse 0.0214, flat from
  ~iter 97. NOT comparable to the wide-band run's 0.2785: the crop
  zeroes the model beyond 2/3 Nyquist while the measured 120^2 dp is
  "power"-padded out to 256^2, so the synthetic padded ring contributes
  a constant, GRADIENT-FREE loss floor (~0.15 in loss_single units).
  Object-domain metrics above are the comparable quantities.

### Unregularised twin (does the information-matched slab model need the regularisation?)

`pso_cheb3_optnoreg_n100.yml` = `pso_cheb3_opt_n100.yml` with exactly six
switches (plus prefix): `loss_sparse.state: false`, and `freq: 1 -> null`
for `obj_rblur`, `obj_zblur`, `mirrored_amp`, `obja_thresh`,
`objp_postiv`. Kept: data loss, dp blur (measurement model), probe-side
constraints (ortho_pmode, fix_probe_int), optimizer/lrs, NITER 100.
Ran concurrently on a second A100 (same 82 s/iter); terminated externally
at ~iter 54 to free the GPU — checkpoints through iter 50, and the trend
is unambiguous:

| iter | noreg fine-band ratio | noreg band corr | reg ratio | reg corr |
|---:|---:|---:|---:|---:|
| 10 | 0.331 | +0.060 | 0.039 | — |
| 20 | 0.930 | +0.189 | 0.160 | — |
| 30 | 1.719 | +0.249 | 0.263 | +0.167 |
| 40 | 2.511 | +0.246 | 0.331 | — |
| 50 | 3.317 | +0.231 | 0.441 | +0.351 |

- No loss divergence (data loss decreased monotonically, 0.4187 at iter
  41 vs reg's 0.4283 at 50 — the extra DoF fit the data better, as they
  must), i.e. it does NOT blow up the way the chin blg arm did.
- But the fine-band POWER grows linearly without bound (+~0.08/iter,
  3.3x the benchmark at iter 50 and still rising) while the band
  CORRELATION peaks at +0.251 (iter 35) and then declines. Past ~iter
  25 the added band power is uncorrelated with the benchmark structure:
  noise pumping, not resolution. The power "ratio" of an unregularised
  run is meaningless on its own — which is exactly why the corr column
  exists.
- Verdict: the information-matched 3-slab parameterisation does NOT make
  the object regularisation unnecessary. Early on the unconstrained run
  actually learns structure FASTER (corr +0.249 vs +0.167 at iter 30),
  but it saturates at corr ~0.25 and then degrades, while the
  regularised arm climbs to +0.429. The regularisation is what keeps
  the fine band signal-dominated; 3 slabs reduce the null space in z,
  not in the lateral fine band.

### Batching micro-bench (NOT the protocol; eager fused path, no CUDA graphs)

`scripts/cheb_batch_microbench.py`, replicated positions, archived
object, f32: fwd+bwd ms/position 162 (b1), 63 (b2), 56 (b4), 44 (b8),
61 (b16). Batching amortises the launch overhead of the length-37
recurrence by ~3.7x at b4-b8 — relevant ONLY if CUDA graphs are
unavailable; the graphed b1 protocol path already runs at 20 ms/position,
so batching buys nothing for the protocol (and b1 is mandated).

### Honest verdict on wall cost

82.8 s/iter vs LT21's 31.4 s/iter = **2.64x** (per-position, correcting
for N4096 vs N3769: 20.2 vs 8.3 ms = 2.4x), down from 6.8x. It is now
CHEAPER per iteration than the wide-band arm was at 50 iterations total
budget, and 1.6x the Chin4A-11 arm (50.4 s/iter) rather than 4.2x. But
the converged quality is 0.59 of benchmark fine-band power vs Chin4A's
0.876 at 1.6x the cost — so on this (thin, 210 A, depth-rank ~2.4)
specimen the exact-generator slab model still does not beat the
splitting ladder on a quality-per-second basis. Its case remains
architectural: exact intra-slab physics with information-matched z-DoF,
now at a wall cost that is merely ~2.6x LT21 instead of ~7x.

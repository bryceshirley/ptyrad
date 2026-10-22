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

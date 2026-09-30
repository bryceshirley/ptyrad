# High-order splittings — what was tried, what survived

Condensed 2026-09-30 from `CHIN_SPLITTING_REPORT.md` (2026-09-25),
`fe_v2_results.md`/`fe_v3_results.md` (forward-error studies v2/v3,
2026-09-26/28), `claims_ledger.md` (C1–C24), `DEPTH_DIAGNOSIS.md`,
`CRYO_RESULTS.md`, and the deep-stack Born benchmark (`PROMPT.md`/
`WORKLOG.md`); the full reports were deleted in the 2026-09-30 cleanup —
this condensation is the surviving record.

## What survives in the code (the paper set)

- `splitting: 'chin_4a' | 'chin_4b'` (tied kicks), requiring
  `propagator_kernel: 'fresnel'` — with the angular-spectrum kernel the
  double commutator [V,[T,V]] is nonlocal and the schemes drop to second
  order. `chin4b_drop_end_props` (far-field only: probe plane shifts by
  a1·dz ≈ 0.2113·dz, pure defocus). `splitting_gradient_term: false` = g0
  is the production setting (C18: hold-out 0.2880 g0 vs 0.2882 g-on,
  runtime 21.7 vs 36.1 s/iter — the g-gather fusion work is moot).
- `strang_forward` — kept as the equivalence demonstration (C4): with a
  reconstructed probe and far-field detection, conjugate Lie–Trotter =
  Strang to ≤1e-14 far-field relative L1 (the half drifts are absorbed by
  the probe and the unit-modulus far-field phase). Corollary: fixed-probe
  LT baselines overstate LT error 2.2–4.8× at N = 11–21.
- Tests: `test/test_chin_splitting.py` (order-of-convergence vs dense
  expm, unitarity, gradcheck, absorption, wrong-sign negative controls,
  bit-for-bit golden defaults in `test/golden/` — do NOT regenerate the
  golden files casually).

### Verified math (the paper claims)

Chin PLA 226,344 (1997); Chin & Chen JCP 117,1409 (2002). With paraxial
generators, [B,[A,B]] = i(∇η)²/k₀ is LOCAL, so fourth order costs zero
extra FFTs: 4A kick T = exp(i(2/3)χ + i·dz·g/72k₀) (Simpson 1/6,2/3,1/6);
4B Gauss–Legendre a1 = (1−1/√3)/2, a2 = 1/√3, correction (2−√3)/48;
correction sign s = +1 with kernel exp(−i dz K²/2k₀) and O = exp(+iχ)
(wrong sign → third order, 2.96 — the built negative control).

Measured orders (float64, dense-expm reference): local slopes LT 1.89 /
Strang 3.02 / Strang+grad 3.28 / **Chin 4A 5.09 / 4B 5.08**; global
(fixed thickness) 1.04 / 2.00 / — / 3.99 / 3.99. Orders survive a
realistic PSO potential (sharp Gaussian columns, 300 kV): local 5.03/5.21,
global 4.6–4.7. Forward error at the arm slicings: chin@11 (dz 19.1) =
2.3e-3 — 30× BELOW LT@21 (7.5e-2); at dz = 70 every scheme saturates O(1).

Cost: two FFT pairs/slice = LT at twice the slices. A100 micro-benchmark
(batch 8, 256², fwd+bwd): LT11 7.4 ms, chin4B@11 14.8, chin4A@11 15.3,
LT21 18.4, LT22 19.7. Full recon at batch 8: chin@11 13.5 s/iter ≈ LT21
14.2. At batch 1 the chin arms are SLOWER (50 vs 32 s/iter) unless g0.

### The half-slices result (batch-1 PSO, 100 iters, vs converged LT21 benchmark)

| arm | slices (dz Å) | loss_single@100 | z-sum phase corr | s/iter |
|---|---|---|---|---|
| lt21 ASM (benchmark) | 21 (10) | 0.2638 | 1.000 (self) | 32.1 |
| **chin4a matched** | 11 (19.1) | **0.2647** | **0.964** | 50.5 |
| lt11 | 11 (19.1) | 0.2712 | 0.951 | 23.9 |
| grad11 | 11 (19.1) | 0.2718 | 0.949 | 45.9 |
| chin4a 6sl | 6 (35) | 0.2784 | 0.933 | 39.0 |
| lt6 | 6 (35) | 0.2920 | 0.908 | 19.6 |

Δloss to benchmark 0.0009 = run-to-run stochasticity. The earlier
chin-vs-benchmark gap was a REGULARISATION artefact: matching the physical
regularisation in Å (z-blur std 0.524 slices, oalr/oplr ×1.909, sparse
×11/21) recovered it. chin4b 11sl (unmatched) hit corr 0.9655 and held
0.794 of the fine band. Fine-band power (0.3–0.6 Nyquist, where the Pr–O
dumbbell splitting lives), ratio to benchmark: chin4a matched 11sl
**0.65** vs lt11 **0.08** — same-reg same-iteration pair proves it is the
scheme; visually the dumbbells resolve under chin and merge under lt11.
At 6 slices no scheme keeps the fine band and LT additionally loses half
the MAIN band (0.53).

## What was tried and removed (2026-09-30)

**saba3** (3-pt Gauss–Legendre, 3 FFT pairs/slice; local order 7 with g),
**lt_x2** (two half-phase transmissions per slab — the cost-matched
control), **kick_mode='untied'** (free per-slab moment maps μ₁/μ₂,
initialised at tied values), **transmission_correction='gradient'**
(corrected LT t_k = O_k·exp(i dz g_k/12k₀), the symmetric-BCH [Y,[Y,X]]
cancellation): exact-order math verified, ~50× error-constant gain on
SMOOTH χ, but **neutral on sharp atomic potentials** (the null grad11
row above — with atomic columns the nonlocal [X,[X,Y]] kinetic commutator
dominates) at +50–55%/iter for the canvas-g gather. Removed with their
tests (test_saba3_ltx2, test_gradient_correction) and 17+ arm configs.

## The forward-error studies (why the removal is justified)

**Dose/detector facts (C1–C3, load-bearing for every S number):** the PSO
dp array is in ELECTRON COUNTS (isolated single-electron events integrate
to median 0.990 stored units; PSF-summed noise covariance G = 1.048 ±
0.032 u/e⁻); dose 2.053e5 e/pattern; each electron smears over 1–5 px so
single-pixel Poisson estimators understate dose ~2×; metric ΔD = Poisson
deviance per pattern on the real 120² grid (0.823 mrad/px, signal to the
69.8 mrad corner = FOLZ ring), S = ΔD/√(2M).

**Forward error at true projections (v2 tables recomputed in ΔD, Pr
worst case):** every tied scheme is statistically detectable at every
N ≤ 42 (tied depth-structure floor S ≈ 11–40 at N = 21–42 — hundreds of
σ over 4096 patterns); ONLY moment-matched fourth-order arms reach S < 1
(SABA₃-mm from N=21: S 0.67; Chin4A-mm N=42: S 0.12). All tied schemes
share a common depth-structure floor set by the unresolved first moment
μ₁; moment matching removes it at zero FFT cost (C6/C7). The tied-vs-mm
gap GROWS 28% on the true non-separable potential (C12 gate tripped).
Other findings: absorption sign moves absolute errors −34%/+8%, no
ordering changes (C8); the apparent ε²dz³ residual was aliasing of the
discrete double commutator (C14: clean dz⁴ on a 2× oversampled grid);
diffraction dominates and successive-slab contributions do NOT cancel
(stack/Σ singles = 1.42, refuting the v2 self-cancellation conjecture,
C11); **open C22**: the insertion test (E_D+E_R additive → correction
should help) contradicts the 3D λ-scan (minimum at λ=0, any correction
hurts) — the role of refraction is unresolved; g0 rests on the direct
real-data control instead.

**Fit test (Part 2, L-BFGS on simulated non-separable data):** **F1
CONFIRMED — the fit absorbs 49–99% of the forward error** (tied 49–53%,
LT 69–82%, untied 84–85%, lt21 99.5%). The untied object-error advantage
(e.g. saba3_u4 103 vs t4 665 mrad) exists ONLY from a truth start: from
flat starts the untied models converge into pure-overfit minima
(hold-out 10× worse, object error ~3 rad) — F2/F3 truth-start artefacts.
F4: the refraction commutator field is the right SIZE (peak 1.71 rad,
rms 0.21–0.23 near Pr, same order as the 0.33–0.86 rad fitted bias) but
the WRONG SHAPE (spatial corr ≤ +0.19) — the bias comes from depth
discretisation, with refraction absorbed by the fit. F5 untestable
(harness freed per-position probes — fix before reuse).

**Part 3 verdicts (real PSO, batch-1, 100 it, dz-keyed matched reg,
fixed 8% hold-out via PTYRAD_HOLDOUT_PATH, seeds 42/1337):**

| arm | hold-out @100 (s42/s1337) | fine band /LT21 | s/iter |
|---|---|---|---|
| lt12 | 0.28779 / 0.28790 | 0.146 | 22.4 |
| lt6x2 | 0.28820 / 0.28766 | 0.147 | 21.1 |
| chin4b_t6 g0 | 0.28797 / 0.28852 | 0.161 | 21.7 |
| saba3_t4 g0 | 0.29210 / 0.29253 | 0.209 | 21.3 |
| chin4b_u6 (untied) | 0.333 / 0.338 | 0.149 | 25.4 |
| saba3_u4 (untied) | 0.835 / 0.907 | 0.075 | 27.5 |
| lt21 (ref) | 0.27988 | 1 | 31.4 |
| chin4b_t11 g0 (21 pairs) | 0.28183 | 0.915 | 27.0 |

- **(a) NO** fourth-order arm at ≈11 FFT pairs beats LT12/LT6×2 beyond
  seed spread (≤5e-4) — at the rule's lr AND per-arm tuned lr (best:
  lt12 0.28236 @0.25×, chin4b_t6 0.28451 @0.5×, saba3_t4 0.28895; C23).
- **(b) NO** — nothing at ≈11 pairs approaches LT21 (0.27672 tuned; gap
  5–40× spread); even chin4b_t11 at 21 pairs falls short — cost parity
  ≠ loss parity.
- **(c)** fine-band retention tracks the FFT-plane count/cost tier, not
  the splitting order.
- **Untied fails on real data** (C16); C20 (regularise μ maps) foreclosed
  by the code removal. **Ground-truth bridge (C17)**: same protocol on
  noisy simulated data ranks lt21 < lt12 < lt6x2 < chin4b_t6 < saba3_t4
  on BOTH hold-out and true-object RMS (836→1034 mrad) — the real-data
  ranking is not an unknown-truth artefact.
- Operational (C19/C23): three protocol-lr NaN divergences traced to
  amplitude pixels ≤ 0 in fractional-power/log transmissions →
  negative-amplitude clamp guards (kept in the code); the dz-keyed lr
  rule oalr/oplr×(dz/10) OVERSHOOTS at batch 1 — every arm incl. LT21
  gains at 0.25–0.5×; re-tune lr before comparing arms.
- Flagged, unchanged: every propagator grid uses (m+0.5)/N bins → a
  uniform ~0.92 px shear of the frame across the PSO stack, inherited
  from fold_slice, consistent across kernels.

## The ceiling that explains it (related closed studies)

- **Depth information (measured on the converged PSO benchmark):**
  dz_min ≈ 2λ/α² = **86 Å** at 300 kV / 21.4 mrad → the 210 Å stack
  supports ~2.4 independent slabs. Measured: adjacent-slice corr 0.992 at
  10 Å with correlation length ≥100 Å; SVD of the slice stack needs
  **3 components for 95% variance** (iter 20 and 100 alike). The fine
  slices exist to control first-order splitting error, not to resolve
  depth — exactly the regime where a fourth-order scheme at ~dz_min/2
  substitutes for fine LT slicing.
- **Depth diagnosis (deep-stack phantoms):** per-slice localisation
  failure is this axial-sampling limit, not solver/gauge/regularisation/
  L1; projected phase recovers fine. The dense128t twist result is real;
  multislice twist remeasured 23.2°, not 33°.
- **Cryo ferritin (JOB B):** depth localisation resolution-limited
  (dz_min), not dose-limited (high-dose 5.2× control changed nothing
  axial) and not model-limited (grow ≈ ms ≈ bornM2); lateral partly
  dose-limited.
- **Deep-stack Born benchmark:** Born n=3 + detector-space coefficient
  refit beats multislice **1.85×** (100 slices) to **3.7×** (256,
  twisted) wall-clock at matched iteration with better loss AND GT
  correlation (paper-2 §speed); dense128b forces n=3; design-gate M*
  ledger: dense128a M*=10 decoupling. Outstanding for the paper at close:
  time-to-object-error, compiled ms baseline, controlled thickness sweep,
  200-iter parity runs.

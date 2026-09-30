# Exact line-search / maximum-likelihood engines — what was tried, what survived

Condensed 2026-09-30 from `LINESEARCH_BORN_SPEC.md` (the port spec; last
tracked before commit 999a03d) and the ML-ISS campaign records. Surviving
code: `src/ptyrad/mliss.py` + `test/test_mliss.py` / `test_mliss_m2.py`
(committed e619471). Retired: `linesearch.py` (exact quartic line search;
survives at aa212b7 and on branch `ml-strategies` in the sep13 worktree
lineage) and `mlms.py` (ML-multislice; same branch). The shared helpers
and `brent_min` moved verbatim INTO mliss.py at retirement.

## The structural fact everything rests on

In the ISS/first-Born model the detector field is
ψ = P + Σ_j IFFT[H_{−z_j}·FFT[g_j·φ_j]], g_j = O_j−1, φ_j the unscattered
illumination. **F is exactly affine in {g_j}** (slices enter as a sum, not
an ordered product) and exactly linear in P at fixed object. Hence along
ANY direction — including a JOINT step over all N slices — the far field
moves on a straight line and the Gaussian intensity cost is an exact
QUARTIC in the scalar step: four coefficients from ONE extra forward
evaluation (no new adjoint; PtyRAD's autograd supplies gradients, keeping
the ptypy/PtyRAD cross-implementation check independent), then a
host-side cubic solve. Multislice is quartic only along single-slice
directions — that is why ML-MS costs (N²+6N−3)|B| while ML-ISS is linear.

## The engine family (Thibault & Guizar-Sicairos, NJP 14, 063004)

- **LISS/BLISS** — per-view / batched amplitude-objective exact line
  search. Recovered as configs of the one engine:
  BLISS = `MLISSConfig(objective='amplitude', step_sigma_floor=1.0,
  damp=0.5)` via `mliss_model_update_batched`; LISS = same per-view.
- **ML-ISS** — `mliss_model_update(_batched)`; objectives
  gaussian/poisson/amplitude; `step_mode` 'alternating' (scalar) or
  'joint' (needs `probe_dir_weight` ≈ N — pdw=1 under-steps the probe
  ~40×); `damp` default 1.0. **`counts_per_unit` is REQUIRED**
  (σ² = I + 1/c): tBL-WSe2 scan_x128_y128.raw is in ADU, gain ~120 ADU/e⁻,
  norm const 67097.3 → **c = 558**; 'auto' deliberately refuses.
  `objective='poisson'` needs `poisson_floor` (0.01 counts) or the
  (1 − I/u) weight diverges from a vacuum start.
- **ML-ISS2** (order 2, plain c=1 double scattering): exclusive-cumsum
  field construction (oracle vs brute force 1e-12), EXACT symmetric-
  difference direction fields F(a) = F + aD₁ + a²D₂, quartic intensity
  searches (gaussian reuses the joint octic). Dispatch via
  `born_iterations=2`, alternating scalar only, `use_born_coeffs` off.
  ~30 s/iter = 2.5× M=1; ahead of M=1 at matched iteration from iter 1
  (L_G 4.98 vs 5.12 at it 2). Six oracle tests; M=1 path bit-identical
  after the addition.
- **ML-MS** — exact multislice member; matched the paper's
  (N²+6N−3)|B| count exactly; ~6× ML-ISS wall at N=12. Retired.
- **Conjugate gradients REMOVED** (paper framing: "step size only, no
  conjugate gradients"); plain repeated updates on the same batch remain;
  everything re-verified bit-identical after removal.

## Campaign outcomes (13 full 100-iter runs, A100, seed 42, tBL-WSe2)

- Poisson-NLL ordering: **ML-ISS-poisson best (−1,224,930)** < ML-MS <
  ML-ISS-gaussian < BLISS < Adam-MS < Adam-ISS. BLISS/ML-ISS/ML-MS all
  reach the Adam-MS-100-iter likelihood target at ITERATION 1; ML-ISS
  reaches the like-for-like target 4.5× faster in wall time than ML-MS.
- Undamped exact steps = fastest starter, limit-cycle plateau finisher:
  amplitude ML-ISS damp 1.0 plateaus at 0.3829 amp-metric vs BLISS
  0.3749; damp 0.5 closes it to 0.3763. Damping is load-bearing
  ASYMPTOTICALLY only — 10-iter ablations show the opposite.
- First-sweep BLISS advantage = the amplitude OBJECTIVE (H2), not the
  weights (H1 refuted: less dark-field weight is worse), not
  damping-in-sweep-1 (H3), not negative data (H4: loader clips at 0).
- LISS/BLISS probe response already uses updated slices (not stale);
  the l1 sparse term is log-only in all line-search engines.

## Engine-era support machinery that outlived the study

- **ISS illumination preconditioner hazard**: recon_step's precond
  (added e53bd7d) divides entrance-slice gradients by ~1e-4 at batch 1
  and destroys entrance slices — measured on PSO born6: z-sum phase corr
  0.94→0.82, probe intensity corr 0.86→0.31 on otherwise identical runs.
  **`PTYRAD_DISABLE_ISS_PRECOND=1` is mandatory for every batch-1
  Born/ISS run**; the env gate is recorded in
  `demo/params/ptyrad_iss_precond_env_gate.patch`.
- **Paper-figure reproduction**: originals reproduce at commit aa212b7
  exactly; e53bd7d's precond breaks Adam-ISS with the paper-era lrs
  (loss RISES iters 2–5). `PTYRAD_DISABLE_ISS_PRECOND=1` at HEAD ≡
  aa212b7. Corrected Adam-ISS final: 0.3762 (= paper), L_G 4.578,
  5.8 s/iter. Iter-1 sweep-average is machine-dependent (A100 0.4585 vs
  A4000 0.472; iters 2+ identical).
- **Cost figure** (all-analytic adjoints, validated to 1e-16; data +
  scripts in `A100_results/`): production ISSLowMemFunction pays
  3N|B|+N+|B| transforms (fft2(phig) per view — no batch-sum before the
  shared probe path); a lean chunk-1 hand adjoint hits the table's
  2N|B|+2N+2 and lands between parallel ISS and multislice at batch 32
  (32.4 vs 28.4 vs 35.2 ms, N=64) at 0.50 GB. At 128² FFTs are
  bandwidth-bound: a 2× transform-count advantage compresses to
  ~1.1–1.24× wall. (Unapplied production fix candidate: batch-sum phig.)
- **Regression discipline**: the bit-for-bit engine regression harness
  (`regression_engines.py --capture/--check`) lives with the baselines in
  `~/ML-ISS/regression/*.npz` (LISS, BLISS, BLISS+shifts, ML-ISS
  alternating) — mliss-only since the retirement; the mliss baseline was
  re-verified bit-identical after each refactor.
- **Seeding**: PtyRADSolver seeds position jitter ONLY via the
  constructor seed argument (cli.py passes init_params.random_seed);
  paper-era scripts were unseeded — canvas can differ ±1 px between runs.
  Always pass seed= in new drivers.

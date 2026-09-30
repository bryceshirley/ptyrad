# Lin–Duda wide-angle correction — what was tried, what survived

Condensed 2026-09-30 from `lin_duda_test_report.md` (ptychobench 3D
test/benchmark, 2026-09-27/28), `report.md` (MultiLinDuda in
Waller-Lab/multi-layer-born, 2026-09-28), and `tem_ptycho_application.md`
(the proposal; full reports deleted in the 2026-09-30 cleanup — this
condensation is the surviving record). The implementations were deleted
too (the multi-layer-born clone + branch `linduda-lt` + its patch, the
ptychobench checkout and worktrees) — reviving means reimplementing from
the math here.

## What Lin–Duda is

Lin & Duda, JASA-EL (2012); Lin, Duda & Newhall, J. Comput. Acoustics 21,
1250018 (2013) — underwater-acoustics parabolic equation, but the math is
the generic one-way wave equation. One-way marching
∂ₓu = i k₀{−1 + √(n² + k₀⁻²∇⊥²)}u. With ε = n²−1, μ = k₀⁻²∇⊥²:

- Q₁ (Tappert/paraxial): 1 + ε/2 + μ/2
- Q₂ (Feit–Fleck = every multislice): −1 + √(1+ε) + √(1+μ)
- **Q₃ (Lin–Duda): Q₂ − ½(NL + LN)**, with N = √(1+ε)−1 (real-space
  diagonal) and L = √(1+μ)−1 (Fourier multiplier −1+√(1−(|k⊥|/k₀)²)).

The cross term is evaluated as exp(Ω), Ω = i k₀ dz·C, by a Taylor
recursion u_m = Ω u_{m−1}/m truncated at order M (their Eqs. 8–9; they
observe convergence to 1e-6 within ≤5 terms). Error bound (their Eq. 6):
|E₃| ≤ 2|Δn||cos γ−1|(|Δn| + |cos γ−1|) — it only matters where strong
contrast and wide angles occur SIMULTANEOUSLY.

**The key structural fact** (why this outlived the splitting studies): the
cross term is a **Hamiltonian** correction, not a splitting correction. As
dz→0 multislice converges to the Feit–Fleck generator L+N, not to
√(1+ε+μ)−1. No slice refinement and no splitting order (Strang, Chin —
cf. high_order_splittings.md: fourth order buys nothing on real PSO) can
recover −½(LN+NL). Physically it makes slice transmission angle-dependent
("thick phase grating"); a static multiplicative object cannot absorb it,
so its data signature is residual structure vs detector angle, ∝θ².

## ptychobench implementation test (3D, A100; 2026-09-27/28)

All Phase-0 correctness tests passed on the user's implementation
(`LinDudaOperator`, split-step exp(dzL)·exp(dzN)·exp(Ω)):
generator identities vs dense construction ≤8.6e-15, residual ∝s³ (slopes
2.98–2.99); splitting order — lie-trotter global slope **1.05**, the
stashed `splitting="symmetric"` (Strang) **2.16** with Taylor M≥2 (M=1
costs it second order, slope 1.79); unitarity drift/step at M2 ≤9e-16;
f64 gradcheck clean both splittings. complex64 floors measured: ~6-10e-6
over 64 steps (relevant to MPS runs). ptychobench suite 365 passed.

**Real bug found+fixed:** `torch.asarray` in `_as_field_array` DETACHES
autograd under torch 2.10/array-api-compat 1.15 — the split-step gradient
to ε was silently severed, and H4's surviving cross-term-only gradient was
wrong by a factor (the worst failure mode: plausible partial gradient).
Fixed with the differentiable `.to` cast; forward bit-identical;
minimal reproducer: `torch.asarray(x*2, dtype=complex128).grad_fn is None`.

T2 structural pattern (rel. error vs 64 dense exact steps): paraxial wins
only near-cutoff bound duct modes (5.4e-9 vs H2 4.9e-2); Feit–Fleck exact
in free space; Lin–Duda good everywhere; atom column at 80 mrad tilt:
H1 0.65 / H2 0.42 / **H4 0.040**. No leaky-duct case was run.

### Where the kernel matters: visible-light IDT at NA≈1 (decisive)

Waller-Lab C. elegans FOV_01 (λ=0.532 µm, NA 1.056, 60 slices × 0.5 µm,
120 LED angles to illumination NA 0.994, measured amplitudes):
- Forward error vs exact one-way reference (median over 12 angles,
  intensity relerr): H1 0.605, H2 0.273, H4-LT 0.185, **H4sym 0.049** —
  5.6× better than the H2 kernel their volume was reconstructed with.
- Against their MEASURED images (only the kernel swapped): H2 0.191 →
  **H4sym 0.146**, sitting on top of the exact-model curve (0.145) at
  every NA.
- Slice-spacing sweep: H4sym at 1.0 µm slices beats H2 at 0.5 µm at every
  NA — the better kernel buys a 2× coarser z-grid.
- Reconstruction at identical budget (Adam, 3D-TV, 30 epochs): train
  cost/angle H1 12,946 / H2 6,834 / **H4sym 5,844** (−14% vs H2); held-out
  12,934 vs 12,759 (H4sym ≥ H2, not converged); price ~2× wall, ~2.6× mem.

### Where it does not matter: 20 keV electrons

abTEM Au island on MoS₂, 20 keV, detector to 175 mrad: kernel-vs-kernel
CBED differences are rms z ≤ 0.041 at 1e5 e/Å² (frac|z|>1 = 0 in every
configuration). What dominates instead: z-slicing (0.65/0.29 intensity
relerr at 20/5 Å) and a strong-phase band-limit ambiguity common to ALL
split-step kernels (~0.21 from exact even at 2.5 Å slices; collapses to
1e-3 at 0.1× potential — the near-singular column-core transmissions).
The generator-level advantage is real (T2) but cannot be cashed at 20 keV
in split-step form; a Krylov-evaluated forward would need ~10³× the cost.
The exact reference itself isn't slicing-converged at 20 keV (2.5 vs
1.25 Å slicing differs 8–15% in intensity).

## MultiLinDuda in multi-layer-born (optics, ArrayFire; 2026-09-28)

`MultiLinDuda(MultiPhaseContrast)` (Lie–Trotter: ψ ← K·exp(iσδn)·e^Ω ψ,
screens ON slice planes) + `MultiLinDudaSym` (paper's Strang), with
HAND-DERIVED adjoint/gradient (no autograd in that code base): Eᴴ via the
downward recursion w_{m−1} = ν_s + Ωᴴw_m/m with Lᴴ = conj(ℓ) multiplier
(exact discrete adjoint incl. evanescent band), cross-term object gradient
g_N = Σ_m (1/m)ᾱ[conj(u_{m−1})⊙Lᴴw_m + conj(Lu_{m−1})⊙w_m]. Verified:
δn≡0 agrees with multislice to 0 ulp; f64 numerical-vs-analytic gradient
~8e-7 (same level as the repo's own adjoint).

Results (repo phantom, MultiBorn-generated noise-free amplitudes —
inverse crime favouring Born; 150 FISTA iters, hold-out every 10th angle):
- **Bead, 143 bright-field angles:** train cost — multislice 16.9, Born
  10.7, LinDuda-LT 10.7, sym 10.44 (the cross term closes essentially the
  whole multislice→Born training gap at the multislice parametrisation);
  held-out — Born 4.17 < LinDuda 5.84–5.88 < multislice 6.22 (~6% better
  than multislice; Born's lead = its own data).
- **Full 293 angles:** the per-angle curve is the physical result — at
  NA 0–0.3 multislice ≈ Lin–Duda (nothing for the cross term to fix at
  normal incidence); the advantage grows monotonically with illumination
  NA. **The symmetric variant is WORSE than multislice at low NA** (its
  merged-half-step screens sit dz/2 off the slice planes = defocus
  mismatch) → Lie–Trotter ordering is the right choice for slice-based
  objects.
- **Shepp–Logan (stronger object):** LinDuda beats multislice on every
  metric (train −47%, held-out −10%, RMSE 0.00551 vs 0.00565, flatter
  axial profile) but stays a fraction of the multislice→Born gap on
  Born-generated data.
- **Taylor order:** converged by M≈3 (field change 3.4e-4 M1→2, 1.4e-5
  2→3, then float32 floor); M=6/8 reproduce M=2 exactly at 2.7× cost.
  **M=1 suffices at biological contrast** (k·dz·|N| ≈ 0.1).
- Volume RMSE vs truth was model-independent on the weak bead (~0.0022):
  missing cone, not the forward model, limits it.

Env recipe that worked (no root): ArrayFire 3.9.0 Linux installer
extracted locally (AF_PATH + LD_LIBRARY_PATH=lib64), pip arrayfire 3.8.0,
numpy 1.26.4/scipy 1.14.1 pins, stub tkinter into the venv. The
closed-access PDFs came via Wayback CDX capture of the scitation PDF URL.

## TEM/ptycho application (the open proposal)

Magnitude table (TF-Yukawa projected potentials, Φ_x ≈ φ_column(b)·θ²/2,
b = impact parameter; last column = Fresnel-kernel error k·t·θ⁴/8):

| regime | θ_max | Φ_x @ b=0.15 Å | @ b=0.3 Å | Fresnel err |
|---|---|---|---|---|
| PSO 300 kV, 21 nm, Pr | 86 mrad | 0.064 rad | 0.013 | 0.46 rad |
| tBL-WSe₂ 80 kV bilayer | 107 | 0.006 | 0.001 | 0.03 |
| **Au[001] 20 nm, 80 kV** | 110 | **0.34** | 0.063 | 0.55 |
| **W foil 30 nm, 60 kV** | 120 | **0.46** | 0.087 | 1.0 |
| SrTiO₃-class 300 kV 20 nm | 86 | 0.051 | 0.013 | 0.44 |

Readings: 2D materials hopeless at any kV; **win regime = thick (≥15 nm)
+ heavy (Z≳55) + ≤100 kV + collection ≥100 mrad** (0.05–0.5 rad of Z- and
angle-dependent column phase = a systematic bias masquerading as charge
redistribution or wrong DWF in quantitative-potential MEP); PSO is
marginal but the cheapest real-data test; **the paraxial-kernel error is
~10× the cross term everywhere** → exact kernel is a prerequisite.

Ranked plan: (B) first — real-PSO residual-vs-angle diagnostic with a
Q₃-LT M=1 port in ptyrad (autograd adjoint free, ~6 extra FFTs/slice;
go/no-go: 0.01–0.06 rad at Pr columns); (C) parallel — exact-√(1+ε+μ)
reference on a small supercell (or Bloch waves / ML-ISS full-wave; no
inverse crime); (A) if signal — the publishable claim: "MEP potential
maps carry a Z-dependent bias of X e/Å at 80 kV/20 nm; a 2×-cost
thick-phase-grating operator removes it."

Not worth doing: 2D materials; anything on the Fresnel kernel; more
splitting-order work; chasing depth resolution (dz_min is aperture-set).

`solver_type='linduda'` in ptyrad (`linduda_forward`, `linduda_order`) is
kept for plan (B); it was never validated in these studies. Literature
anchors in the proposal are from model knowledge (web access was blocked)
— verify before citing.

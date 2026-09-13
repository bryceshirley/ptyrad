# PORT SPEC — exact line-search step for PtyRAD

Target: add the exact quartic line search (joint all-slice object step, then probe step
against the already-updated field) to PtyRAD's first-Born / single-scattering engine.

Reference implementation: ptypy `FastFirstBornTorch`
(`~/ptypy/ptypy/custom/fastfirstborntorch.py`), with derivations in
`~/ptypy/firstborn_solver_maths_and_constraints.md`. Authority order: the .md for the
maths, the .py for behaviour, this spec for what the PtyRAD port should do. Where this
spec and the ptypy source differ, this spec wins — the differences are deliberate and
are flagged as such.

---

## 0. Scope, and why this port is small

PtyRAD is autodiff-native. **Do not port the hand-derived adjoint.** The ptypy engine
computes `acc_grad` and `grad_P` by hand because ptypy has no autograd; PtyRAD does, and
the paper's §5.4 cross-implementation check depends on the two codebases deriving
gradients independently. Hand-porting the adjoint destroys that check and doubles the
surface area for no gain.

What actually has to be added:

1. A preconditioner `K_j` applied to the autograd gradient.
2. The direction response `D` — obtained from **one extra forward evaluation**, not from
   any new adjoint.
3. The four quartic coefficients and a host-side cubic solve.
4. A complex-domain step, written back into PtyRAD's storage.
5. The probe step, taken against the updated field.

Everything else PtyRAD already has.

## 1. The structural facts the whole thing rests on

Referred to the entrance plane, N slices at depths `z_j`:

```
g_j  = O_j − 1                                  # chord perturbation
phi_j = IFFT[ H_{z_j} · FFT[P] ]                # unscattered illumination; depends only on P
psi  = P + Σ_j IFFT[ H_{−z_j} · FFT[ g_j · phi_j ] ]
F    = ff_fw(psi)                               # detector field, (B,M,ny,nx)
```

Two facts, and they are the entire justification for the method:

- **`F` is exactly affine in `{g_j}`.** Slices enter as a sum, not an ordered product.
  Hence along any direction `d`, the far field moves on a straight line, and the Gaussian
  intensity cost is an exact quartic in the scalar step — including a **joint** step over
  all N slices at once. This is Born-only; multislice is quartic only along single-slice
  directions.
- **`F` is exactly linear in `P`** at fixed object.

Both are exploited below to avoid writing any new derivative code.

## 2. Getting `D` and `D_P` without an adjoint

**Object direction response.** Because `F` is affine in `g`,

```
D = F(g + d) − F(g)
```

holds **exactly**, for any `d`, with no small-step assumption. `F(g)` is already computed;
`D` therefore costs one extra forward pass. That is the "one extra forward half-pass" the
paper quotes.

Equivalently and identically, `D = ff_fw( Σ_j Prop_{−z_j}[ d_j · phi_j ] )` — the identity
path cancels in the difference. Use whichever fits PtyRAD's forward better; prefer the
difference form, since it reuses the existing code path verbatim and cannot drift from it.

**Probe direction response.** Because `F` is *linear* in `P` (no constant term — every
term carries a factor of `P`),

```
D_P = F(q ; g2)
```

i.e. run the ordinary forward with the probe direction `q` substituted for `P`, at the
**updated** object `g2`. Not a difference, not a JVP — one forward evaluation. Note that
`phi` must be rebuilt from `q` inside that evaluation.

## 3. The batch update

Design point is **batch size 1**: probe updated every view. Larger batches average the
probe update and stall in a spiky-probe local minimum. Support B ≥ 1, optimise for B = 1.

**Step 1 — forward.** Standard PtyRAD forward → `F`, model intensity `u`.

**Step 2 — direction.** Autograd gradient of the direction objective (see §5 on which
objective), preconditioned:

```
K_j    = Σ_m |phi_j|²                      # view-independent; scatter-accumulate over the batch
peak_K = per-slice spatial max of K_j
dn     = peak_K                            if object_denom == 'max'   (winning recipe)
       = K_j + peak_K · denom_reg          if 'local'
d      = grad / clamp_min(dn, eps)
if momentum > 0:  d += momentum · disp_prev        # heavy ball BEFORE the search
```

`'max'` starves off-focus pixels; `'local'` floors at `denom_reg·peak` per slice and
degenerates to `'max'` on a caustic slice. Use `'max'` unless testing.

**Step 3 — response.** `D = F(g + d) − F(g)` (§2).

**Step 4 — coefficients.** Per pixel, `I(a) = u + 2av + a²w` exactly, with

```
v = Σ_m Re( conj(F) · D )
w = Σ_m |D|²
e = u − I_dat                     # SIGN: model minus data. Flipped, it minimises nothing.
ω = mask / (I_dat + 1)
```

`Q(a) = Σ ω (e + 2av + a²w)²` is an exact quartic, and `¼ dQ/da = c0 + c1 a + c2 a² + c3 a³`:

```
c0 = Σ ω e v
c1 = Σ ω (e w + 2v²)
c2 = 3 Σ ω v w
c3 = Σ ω w²
```

**All four sums are accumulated in float64, cast before the products.** See §6 — this is
the single most dangerous line in the port.

**Step 5 — cubic solve** (host-side, `np.roots`, negligible cost):

1. Any non-finite coefficient → fallback `alpha/N`.
2. Strip leading `|c| < 1e-300`; fewer than 2 remaining → fallback.
3. Real roots only (`|imag| ≤ 1e-8·(1+|real|)`); none → fallback.
4. Evaluate true `Q(a)` at each real root and at `a = 0`; take the root with lowest `Q`
   **strictly below `Q(0)`**, else fallback.

Then `a *= ls_damp` (the fallback is damped too). Optional symmetric clip at `max_step`.
**Log every accepted `a`** — see §6.

**Step 6 — object step, in complex.** One scalar `a` applied jointly to all N slices.
There is no per-slice loop of steps anywhere in the object update. See §4 on the
parameterisation hazard.

```
O ← O + a · d          # complex
disp_prev = a · d
```

**Step 7 — probe branch** (if the probe is being updated this batch):

```
F ← F + a·D            # exact, no re-forward
u ← u + 2av + a²w      # exact
e ← u − I_dat
grad_P = autograd gradient w.r.t. P, at the pre-step object
K_P    = Σ_{N,B} |O|²  (pre-step O)          # the correct probe diagonal
q      = grad_P / clamp_min( peak(K_P), eps )      # probe_denom 'max' by default;
                                                   # 'local' spikes tight high-NA probes
D_P    = F(q ; g2)                                 # §2, at the UPDATED object
v_p    = Σ_m Re(conj(F)·D_P);  w_p = Σ_m |D_P|²
b      = line_search(e, v_p, w_p, ω, fallback = beta/N)
P ← P + b·q
```

The exact update of `F` and `u` in the first two lines is what handles the bilinear
object–probe cross term exactly, with no re-evaluation. This is the reason the probe step
follows the object step rather than preceding it.

Invalidate the `phi` cache after any probe change, including after probe constraints.

---

## 4. PtyRAD-specific hazards

These four are where the port will actually fail. Read them before writing code.

### 4.1 The object must be stepped in complex — and the phase branch must survive

PtyRAD stores the object as separate float32 amplitude and phase tensors. The quartic is
exact only if the step is taken in **complex `O`**. Applying it in (amp, phase) coordinates
makes the parameter-to-field map non-affine, `I(a) = u + 2av + a²w` becomes false, and the
line search silently degrades into an uncontrolled approximation that still appears to
work. Oracle test 4 is the tripwire; make it the first test that passes.

So: build complex `O` from (amp, phase), step in complex, decompose back.

**But check the phase range before trusting the round trip.** `angle()` returns the
principal value in (−π, π]. If PtyRAD's stored per-slice phase ever exceeds π — accumulated
over iterations, or on a strongly scattering slice — then complex → `angle()` → stored
phase wraps and destroys it. Inspect the stored phase range on a real run first. If it
stays well inside the branch (the specimens in the paper run at Φ_tot ≈ 0.14–0.4 rad, so it
should), the round trip is safe and you can say so in a comment. If not, track the branch
explicitly rather than hoping.

The probe is **not** affected: `view_as_real` is a linear reparameterisation, so stepping in
those coordinates is equivalent to stepping in complex.

### 4.2 Pin down the intensity normalisation once

PtyRAD's model DP is `fftshift2( Σ_{pmode,omode} |·|² · omode_occu/(Nx·Ny) ) + eps` — a
normalised, fftshifted intensity. ptypy's `u` is the raw, unshifted `Σ_m |F_m|²`.

Write `u_p = c · fftshift(u) + eps` with `c = omode_occu/(Nx·Ny)`. Then the quartic holds
exactly in PtyRAD units provided `v` and `w` carry the *same* transformation:

```
v_p = c · fftshift(v)        w_p = c · fftshift(w)        e = u_p − I_dat,p
```

The `+eps` floor is a constant and drops out of `v` and `w`, but it is in `u_p`, so it must
be in `e`. Define this map in one place and use it everywhere. Oracle test 3 catches a
mismatch.

### 4.3 float32 will bite here harder than it did in ptypy

PtyRAD is float32 throughout by design. Early in a reconstruction the residuals are large
enough that `w²` overflows float32 (`e ~ 1e10` ⇒ `w² ~ 1e40` > float32 max) ⇒ `c3 = inf` ⇒
the isfinite guard fires ⇒ **silent, permanent fallback to the fixed step**, with no error
and no obvious symptom.

Promote `e, v, w, ω` to float64 **before forming the products**. Casting after `.sum()`
does not help. Keep the isfinite guard anyway.

The only cheap detector is the step log: a healthy run shows live spread in the accepted
`a`; a run stuck in fallback shows every value clustered at exactly `ls_damp · alpha/N`.
Build that logging in from the start, not as a debugging afterthought.

### 4.4 Check where `constraint_fn` fires

Step 7 assumes `F ← F + a·D` is still valid when the probe line search runs. If PtyRAD's
`constraint_fn` (ortho_pmode, probe_mask_k, fix_probe_int, object constraints) fires between
the object step and the probe step, `F` and `e` are stale and the probe step is solving the
wrong problem. PtyRAD applies constraints per iteration rather than per batch, so this is
probably fine — verify it rather than assuming, and keep the `constraint_fn` call site
unchanged.

---

## 5. One design decision to make explicitly

In ptypy, the *direction* is the preconditioned gradient of the **amplitude** cost
`E_amp = Σ m (√u − √I_dat)²`, while the *step* exactly minimises the Gaussian **intensity**
quartic `Q`. They are different objectives, and `ls_damp = 0.5` is what reconciles them —
at 1.0 the ptypy run plateaus an order of magnitude high. That damping is load-bearing and
is not a fudge factor for instability.

For PtyRAD you have three options. Pick one deliberately and record which:

- **(a) Match ptypy.** Use `E_amp` explicitly for the direction, keep `ls_damp = 0.5`. Best
  for parity testing against the ptypy engine; the damping is already validated.
- **(b) Use `Q` for both.** Direction and step then share an objective and `ls_damp` should
  be near 1.0. Arguably cleaner, but it is a different algorithm from the one the paper
  measured, and the damping needs re-checking.
- **(c) Use PtyRAD's own loss for the direction.** Then the mismatch is neither of the
  above and `ls_damp` must be retuned from scratch. Only do this if there is a reason.

Default to (a). Expose the choice as a knob.

## 6. Numerical safety

| item | rule |
|---|---|
| **float64 promotion of `e, v, w, ω`** | **Required, cast before the products.** See §4.3. |
| isfinite guard on coefficients | keep even after promoting |
| leading-coefficient strip | drop `\|c\| < 1e-300`; <2 remaining → fallback |
| real-root tolerance | `\|imag\| ≤ 1e-8·(1+\|real\|)` |
| root acceptance | must strictly beat `Q(0)`, else fallback |
| model-intensity floor | `clamp_min(u, 1e-12)` inside any sqrt |
| mask handling | residual weight zeroed where `mask == 0` |
| denominator floors | `clamp_min(≈1.19e-7)` on `dn`, `dn_p` |
| error accumulators | float64 |
| step logging | every accepted `a` and `b`, per batch — the only fallback detector |
| watchdog | probe peak/mean intensity per iteration; runaway = spike |

## 7. Test oracle — write these before any production code

The parity tests are the real specification. Tests 2, 4 and 7 each catch a failure that is
silent at runtime. Use torch autograd as reference only; no autodiff in the production step
beyond the gradient itself.

1. **Direction gradient.** Autograd gradient of the chosen direction objective (§5) w.r.t.
   complex `O`, with a mask case and M > 1 modes. Torch convention: `z.grad = 2·∂L/∂z̄`, so
   the descent direction is `−z.grad`.
2. **`dQ/da` at 0.** `4·c0` equals `d/da Q(F + aD)|₀` from autograd or a central difference.
   The flipped sign `e = I_dat − u` fails this test — that is what it is for.
3. **Full quartic parity.** For several `a` including the returned root: `Q` from
   `(e + 2av + a²w)` equals `Q` from rebuilding `|F + aD|²` equals
   `Q(0) + 4(c0 a + c1 a²/2 + c2 a³/3 + c3 a⁴/4)`. Run this in PtyRAD units to pin §4.2.
4. **Affine exactness.** After the object step, a full re-forward gives
   `F_forward == F + a·D` and `u_forward == u + 2av + a²w` to machine precision. **This is
   the parameterisation tripwire of §4.1** — any leakage means the step was taken in
   (amp, phase), or the gather/scatter windows are wrong.
5. **Probe response.** `D_P == F(q ; g2)` matches a forward difference in `P` (exact, since
   `F` is linear in `P`), and `dQ/db|₀ = 4·c0⁽ᵖ⁾` at the updated field.
6. **Solver vs brute force.** Dense-grid argmin of `Q(a)` matches the selected root.
   Degenerate inputs (`D = 0`, fully masked, `c3 = inf`) return exactly `ls_damp·alpha/N`
   (resp. `beta/N`).
7. **Overflow regression.** Synthetic `e ~ 1e10`, `w ~ 1e10` at float32: without promotion
   `c3` overflows and the run falls back; with float64-before-products it must not.
8. **Joint-step invariant.** A single scalar `a` applies to all N slices — the per-slice
   step trace is constant across `j`. Guards against reintroducing a per-slice sweep, and
   marks the vector-search seam of §9.
9. **Phase branch.** Round-trip complex → (abs, angle) → complex is lossless on the actual
   stored object range (§4.1).

## 8. Knobs

| knob | default | role |
|---|---|---|
| alpha | 1.0 | object fallback step `α/N`, on cubic degeneracy only |
| beta | 1.0 | probe fallback step `β/N`, same condition |
| **ls_damp** | **0.5** | multiplier on every step, exact and fallback. Load-bearing (§5) |
| max_step | 0.0 (off) | symmetric clip on the damped step |
| object_denom | 'max' | `peak_K` vs `K_j + peak·reg` per slice |
| probe_denom | 'max' | decoupled; 'local' spikes high-NA probes |
| denom_reg | 0.01 | floor as a fraction of per-slice peak `K` |
| momentum | 0.0 | heavy-ball mix into the direction, pre-search |
| batch_size | 1 | design point |
| direction_objective | 'amplitude' | §5; 'intensity' or 'ptyrad_loss' |

Heavy ball goes **before** the line search (`d += μ·disp_prev`; `disp_prev = a·d`) — with an
exact step this behaves like conjugate gradient, and it differs from the fixed-step parent's
momentum scaling.

## 9. Future seam — N-D vector line search (do not implement)

Restrict the cost to `g + Σ_j a_j d_j` with per-slice steps `a ∈ R^N`. `Q(a)` remains an
exact multivariate quartic, since Born is quartic along *any* direction. It needs per-slice
responses `D_j`, the masked sums `Re(F*·D_j)`, the N×N Gram `Re(D_j*·D_k)`, and `|D_j|²`;
stationarity is a coupled cubic system.

**Structure for it now, implement it later.** Accumulate `D` as an explicit sum over a
per-slice container so `D_j` stays addressable, and put the Gram behind a flag. Cost: nothing
today.

Motivation, and it is concrete: one scalar step cannot serve a stack whose `K` varies by ~4
orders of magnitude within a focal slice — and on the real two-slice X-ray data the scalar
exact step underperforms its N = 30 showing. At N = 2 the vector search is a bivariate
quartic, which is small enough to be the natural place to demonstrate this.
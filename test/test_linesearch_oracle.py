"""
Oracle tests for the exact quartic line search (LINESEARCH_BORN_SPEC.md §7).

These nine tests are the specification for `ptyrad.linesearch` and were written
BEFORE the production code exists. Tests 1-8 fail with ModuleNotFoundError until
`src/ptyrad/linesearch.py` is implemented; test 9 is a pure-torch hazard check
(§4.1) and passes immediately. Tests 2, 4 and 7 each catch a failure that is
silent at runtime — do not weaken their tolerances.

Expected production API (module `ptyrad.linesearch`), the contract for the
implementation session:

    firstborn_fields(object_patches, probe, H) -> F
        Complex detector-plane field per mode, UNSHIFTED k-space, shape
        (B, pmode, omode, Ny, Nx). Same maths as
        forward_models.born.firstborn_forward up to (and excluding) the
        intensity reduction. Dtype-preserving: float64/complex128 inputs give
        complex128 fields.

    dp_from_fields(F, omode_occu, eps=1e-10) -> dp
        The §4.2 unit map, matching firstborn_forward EXACTLY:
        fftshift2( sum_{pmode,omode} |F|^2 * omode_occu/(Nx*Ny) + eps ),
        shape (B, Ny, Nx).

    response_terms(F, D, omode_occu) -> (v, w)
        v = fftshift2( sum_{pmode,omode} c_o * Re(conj(F) * D) )
        w = fftshift2( sum_{pmode,omode} c_o * |D|^2 ),   c_o = occu_o/(Nx*Ny).
        Same transformation as dp_from_fields but WITHOUT the +eps (it is a
        constant and lives in u only, hence in e = u - I_dat).

    direction_response(object_patches, d, probe, H, per_slice=False) -> D
        Object direction response. d is complex, (B, omode, Nz, Ny, Nx).
        per_slice=False: (B, pmode, omode, Ny, Nx), equal to
        F(g + d) - F(g) exactly (spec §2). per_slice=True: the §9 seam —
        (B, pmode, omode, Nz, Ny, Nx) with .sum(dim=3) equal to the total.

    direction_gradient(obj_complex, probe, H, I_dat, mask, omode_occu,
                       objective="amplitude") -> descent
        Descent direction w.r.t. complex O in torch's Wirtinger convention:
        descent == -O.grad where O.grad = 2*dL/d(conj(O)) of
        L = sum( mask * (sqrt(clamp_min(u_p, 1e-12)) - sqrt(I_dat))^2 ),
        u_p = dp_from_fields(fields(O)). mask=None means weight 1.
        (ptypy's acc_grad is -dL/d(conj(O)) — a factor 2 smaller. The exact
        search absorbs positive scalings; only the fallback step magnitude
        changes. PtyRAD pins the torch-native convention.)

    quartic_coeffs(e, v, w, omega) -> (c0, c1, c2, c3) python floats
        c0 = S[w e v], c1 = S[w (e w + 2 v^2)] ... per spec §3 step 4, where
        S = float64 sum with ALL FOUR inputs cast to float64 BEFORE any
        product (§4.3/§6 — casting after .sum() does not count).

    solve_step_cubic(c0, c1, c2, c3, Qfun, fallback) -> a_raw
        §3 step 5: isfinite guard, strip |c|<1e-300, real roots
        (|imag| <= 1e-8*(1+|real|)), accept only strictly below Q(0),
        else fallback. No damping here.

    line_search(e, v, w, omega, fallback, ls_damp=0.5, max_step=0.0) -> a
        quartic_coeffs + solve_step_cubic, then a *= ls_damp (fallback damped
        too), optional symmetric clip. Returns a python float.

    apply_object_step(obja, objp, a, d) -> None (in-place)
        §3 step 6 / §4.1: O = polar(obja, objp); O += a*d IN COMPLEX;
        obja/objp overwritten with abs/angle. Storage shape (omode, Nz, Ny, Nx),
        d complex of the same shape. One scalar `a` for all slices.

All tests are CPU + eager (torch.compile disabled) and use synthetic tensors
only — no reconstructions.
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import pytest
import torch

torch._dynamo.config.disable = True

from torch.fft import fft2, fftshift, ifft2

torch.manual_seed(0)

EPS_DP = 1e-10  # eps added inside firstborn_forward's intensity reduction


def _ls():
    """Import the production module inside each test so that tests 1-8 fail
    (not error at collection) with ModuleNotFoundError until it exists."""
    import ptyrad.linesearch as m

    return m


# --------------------------------------------------------------------------- #
# Local reference implementations (independent oracle, mirrors born.py maths)  #
# --------------------------------------------------------------------------- #


def fftshift2(x):
    return fftshift(x, dim=(-2, -1))


def ref_fields_from_complex(O, probe, H):
    """Detector-plane field from complex object O (B, omode, Nz, Ny, Nx).
    Mirrors forward_models/born.py::firstborn_forward lines 44-77."""
    Ny, Nx = O.shape[-2:]
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    psi = ifft2(H * probe_k)  # (B, pmode, 1|omode, Nz, Ny, Nx)
    g = (O - 1.0).unsqueeze(1)  # (B, 1, omode, Nz, Ny, Nx)
    scattered = torch.sum(fft2(g * psi) * H.conj(), dim=3)
    return probe_k.squeeze(3) + scattered  # (B, pmode, omode, Ny, Nx)


def ref_fields(obj_ap, probe, H):
    """Same, from PtyRAD (amp, phase) patches (B, omode, Nz, Ny, Nx, 2)."""
    O = torch.polar(obj_ap[..., 0], obj_ap[..., 1])
    return ref_fields_from_complex(O, probe, H)


def ref_dp(F, omode_occu, eps=EPS_DP):
    """§4.2 unit map: PtyRAD model DP from the per-mode detector field."""
    Ny, Nx = F.shape[-2:]
    norm_weight = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    return fftshift2(torch.sum(F.abs().square() * norm_weight, dim=(1, 2)) + eps)


def ref_vw(F, D, omode_occu):
    """v, w carrying the same §4.2 transformation as ref_dp (no eps)."""
    Ny, Nx = F.shape[-2:]
    norm_weight = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    v = fftshift2(torch.sum((F.conj() * D).real * norm_weight, dim=(1, 2)))
    w = fftshift2(torch.sum(D.abs().square() * norm_weight, dim=(1, 2)))
    return v, w


def make_H(Nz, Ny, Nx, dtype=torch.complex64, dz_phase=0.31):
    """Unit-modulus Fresnel-like propagator stack H_j = H^j (entrance plane at
    j=0), shape (1, 1, 1, Nz, Ny, Nx) — matches models.get_propagators_3d."""
    ky = torch.fft.fftfreq(Ny, dtype=torch.float64)
    kx = torch.fft.fftfreq(Nx, dtype=torch.float64)
    chi = dz_phase * (ky[:, None] ** 2 + kx[None, :] ** 2) * (Ny * Nx) ** 0.5
    H1 = torch.exp(-1j * chi)
    j = torch.arange(Nz, dtype=torch.float64).view(Nz, 1, 1)
    return (H1**j).view(1, 1, 1, Nz, Ny, Nx).to(dtype)


def make_probe(B, pmode, Ny, Nx, dtype=torch.complex64, seed=1):
    """Smooth band-limited random probe with unequal mode powers."""
    g = torch.Generator().manual_seed(seed)
    ky = torch.fft.fftfreq(Ny, dtype=torch.float64)
    kx = torch.fft.fftfreq(Nx, dtype=torch.float64)
    aperture = ((ky[:, None] ** 2 + kx[None, :] ** 2).sqrt() < 0.2).to(torch.complex128)
    coef = torch.randn(B, pmode, Ny, Nx, 2, generator=g, dtype=torch.float64)
    pk = torch.view_as_complex(coef) * aperture
    probe = ifft2(pk)
    probe = probe / probe.abs().square().sum(dim=(-2, -1), keepdim=True).sqrt()
    powers = torch.linspace(1.0, 0.3, pmode).view(1, -1, 1, 1)
    return (probe * powers).to(dtype)


def make_object(B, omode, Nz, Ny, Nx, dtype=torch.float32, seed=2, phase_max=0.4):
    """(amp, phase) patches in the regime measured on real runs (see test 9):
    amp near 1, phase in [0, phase_max]."""
    g = torch.Generator().manual_seed(seed)
    amp = 1.0 + 0.01 * torch.randn(B, omode, Nz, Ny, Nx, generator=g, dtype=torch.float64)
    phase = phase_max * torch.rand(B, omode, Nz, Ny, Nx, generator=g, dtype=torch.float64)
    return torch.stack([amp, phase], dim=-1).to(dtype)


def make_direction(B, omode, Nz, Ny, Nx, dtype=torch.complex64, seed=3, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    d = torch.view_as_complex(
        torch.randn(B, omode, Nz, Ny, Nx, 2, generator=g, dtype=torch.float64)
    )
    return (scale * d).to(dtype)


def make_data(obj_seed, B, omode, Nz, Ny, Nx, pmode, occu, dtype=torch.float64):
    """Synthetic measured DP: forward of a *different* object, so residuals are
    nonzero but realistic. Returns (I_dat, mask, omega) in PtyRAD units."""
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    obj_true = make_object(B, omode, Nz, Ny, Nx, dtype=dtype, seed=obj_seed, phase_max=0.5)
    probe = make_probe(B, pmode, Ny, Nx, dtype=cdtype, seed=11)
    H = make_H(Nz, Ny, Nx, dtype=cdtype)
    I_dat = ref_dp(ref_fields(obj_true, probe, H), occu.to(dtype)).clamp_min(0.0)
    mask = torch.ones_like(I_dat)
    mask[..., : Ny // 4, : Nx // 4] = 0.0  # dead corner block
    omega = mask / (I_dat + 1.0)
    return I_dat, mask, omega


def ref_Q(e, v, w, omega, a):
    r = e + (2.0 * a) * v + (a * a) * w
    return float((omega * r * r).sum())


# --------------------------------------------------------------------------- #
# 1. Direction gradient                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("use_mask", [True, False])
def test_1_direction_gradient(use_mask):
    """§7.1: production descent direction of the amplitude objective (§5 option
    (a)) equals -O.grad from an independent autograd reference, with a mask
    case and M > 1 probe modes. Torch convention: z.grad = 2*dL/d(conj(z)),
    descent = -z.grad."""
    ls = _ls()
    B, omode, Nz, Ny, Nx, pmode = 2, 1, 3, 32, 32, 2
    occu = torch.ones(omode, dtype=torch.float64)
    probe = make_probe(B, pmode, Ny, Nx, dtype=torch.complex128, seed=5)
    H = make_H(Nz, Ny, Nx, dtype=torch.complex128)
    obj = make_object(B, omode, Nz, Ny, Nx, dtype=torch.float64, seed=6)
    I_dat, mask, _ = make_data(7, B, omode, Nz, Ny, Nx, pmode, occu)
    mask_arg = mask if use_mask else None

    # Reference: autograd of L = sum m*(sqrt(u_p) - sqrt(I_dat))^2 w.r.t. complex O
    O = torch.polar(obj[..., 0], obj[..., 1]).detach().requires_grad_(True)
    u_p = ref_dp(ref_fields_from_complex(O, probe, H), occu)
    resid = u_p.clamp_min(1e-12).sqrt() - I_dat.sqrt()
    L = (resid.square() if mask_arg is None else mask * resid.square()).sum()
    L.backward()
    ref_descent = -O.grad

    descent = ls.direction_gradient(
        O.detach(), probe, H, I_dat, mask_arg, occu, objective="amplitude"
    )
    assert descent.shape == O.shape
    scale = ref_descent.abs().max()
    assert torch.allclose(descent, ref_descent, rtol=1e-9, atol=1e-12 * scale)
    # and it is genuinely a descent direction of L
    assert float((descent.conj() * O.grad).real.sum()) < 0.0


# --------------------------------------------------------------------------- #
# 2. dQ/da at 0 — the sign tripwire (silent at runtime)                        #
# --------------------------------------------------------------------------- #


def test_2_dQda_at_zero():
    """§7.2: 4*c0 equals d/da Q(F + aD)|_0 from autograd. The flipped residual
    e = I_dat - u produces -4*c0 and MUST fail — that is what this test is
    for (a sign flip 'minimises nothing' but still iterates plausibly)."""
    ls = _ls()
    B, omode, Nz, Ny, Nx, pmode = 2, 1, 3, 32, 32, 2
    occu = torch.ones(omode, dtype=torch.float64)
    probe = make_probe(B, pmode, Ny, Nx, dtype=torch.complex128, seed=8)
    H = make_H(Nz, Ny, Nx, dtype=torch.complex128)
    obj = make_object(B, omode, Nz, Ny, Nx, dtype=torch.float64, seed=9)
    I_dat, _, omega = make_data(10, B, omode, Nz, Ny, Nx, pmode, occu)

    F = ref_fields(obj, probe, H)
    d = make_direction(B, omode, Nz, Ny, Nx, dtype=torch.complex128, seed=12, scale=0.02)
    obj_stepped = torch.polar(obj[..., 0], obj[..., 1]) + d
    D = ref_fields_from_complex(obj_stepped, probe, H) - F  # exact response

    u_p = ref_dp(F, occu)
    v_p, w_p = ref_vw(F, D, occu)
    e = u_p - I_dat

    c0, c1, c2, c3 = ls.quartic_coeffs(e, v_p, w_p, omega)

    # autograd reference for dQ/da at a = 0, rebuilding the intensity from fields
    a = torch.zeros((), dtype=torch.float64, requires_grad=True)
    Qa = (omega * (ref_dp(F + a * D, occu) - I_dat).square()).sum()
    Qa.backward()
    dQda = float(a.grad)

    assert 4.0 * c0 == pytest.approx(dQda, rel=1e-9)

    # sign tripwire: the flipped convention fails this identity
    c0_flip, *_ = ls.quartic_coeffs(I_dat - u_p, v_p, w_p, omega)
    assert abs(4.0 * c0_flip - dQda) > 1e3 * abs(dQda) * 1e-9  # not a rounding miss
    assert 4.0 * c0_flip != pytest.approx(dQda, rel=1e-6)


# --------------------------------------------------------------------------- #
# 3. Full quartic parity, in PtyRAD units — pins §4.2                          #
# --------------------------------------------------------------------------- #


def test_3_quartic_parity_ptyrad_units():
    """§7.3: Q from (e + 2av + a^2 w), Q from rebuilding |F + aD|^2 through the
    §4.2 unit map, and the Taylor form Q(0) + 4(c0 a + c1 a^2/2 + c2 a^3/3 +
    c3 a^4/4) agree for several a including the returned root. omode = 2 with
    unequal occupancies so the per-omode weights are pinned inside the mode
    sum (they are NOT a single scalar c when omode > 1)."""
    ls = _ls()
    B, omode, Nz, Ny, Nx, pmode = 2, 2, 3, 32, 32, 2
    occu = torch.tensor([0.7, 0.3], dtype=torch.float64)
    probe = make_probe(B, pmode, Ny, Nx, dtype=torch.complex128, seed=13)
    H = make_H(Nz, Ny, Nx, dtype=torch.complex128)
    obj = make_object(B, omode, Nz, Ny, Nx, dtype=torch.float64, seed=14)
    I_dat, _, omega = make_data(15, B, omode, Nz, Ny, Nx, pmode, occu)

    F = ls.firstborn_fields(obj, probe, H)
    assert F.shape == (B, pmode, omode, Ny, Nx)
    assert torch.allclose(F, ref_fields(obj, probe, H), rtol=1e-10, atol=1e-12)

    d = make_direction(B, omode, Nz, Ny, Nx, dtype=torch.complex128, seed=16, scale=0.03)
    D = ls.direction_response(obj, d, probe, H)
    u_p = ls.dp_from_fields(F, occu)
    v_p, w_p = ls.response_terms(F, D, occu)
    e = u_p - I_dat
    c0, c1, c2, c3 = ls.quartic_coeffs(e, v_p, w_p, omega)

    def Qfun(a):
        return ref_Q(e, v_p, w_p, omega, a)

    a_root = ls.solve_step_cubic(c0, c1, c2, c3, Qfun, fallback=1.0 / Nz)

    Q0 = Qfun(0.0)
    for a in [0.0, a_root, 0.5 * a_root, -0.3 * a_root, 1.7 * a_root, 0.05, -0.05]:
        q_coeff = Qfun(a)
        q_field = float((omega * (ls.dp_from_fields(F + a * D, occu) - I_dat).square()).sum())
        q_taylor = Q0 + 4.0 * (c0 * a + c1 * a**2 / 2.0 + c2 * a**3 / 3.0 + c3 * a**4 / 4.0)
        assert q_field == pytest.approx(q_coeff, rel=1e-9), f"field vs coeff at a={a}"
        assert q_taylor == pytest.approx(q_coeff, rel=1e-9), f"taylor vs coeff at a={a}"


def test_3b_unit_map_matches_production_forward():
    """§4.2 pin against the actual production forward: dp_from_fields applied
    to firstborn_fields must reproduce forward_models.born.firstborn_forward
    (float32, eager) to float32 precision, including the +eps floor and the
    omode occupancy weights."""
    ls = _ls()
    from ptyrad.forward_models import firstborn_forward

    B, omode, Nz, Ny, Nx, pmode = 2, 2, 3, 32, 32, 2
    occu = torch.tensor([0.7, 0.3], dtype=torch.float32)
    probe = make_probe(B, pmode, Ny, Nx, dtype=torch.complex64, seed=17)
    H = make_H(Nz, Ny, Nx, dtype=torch.complex64)
    obj = make_object(B, omode, Nz, Ny, Nx, dtype=torch.float32, seed=18)

    dp_prod = firstborn_forward(obj, probe, H, occu)
    dp_ours = ls.dp_from_fields(ls.firstborn_fields(obj, probe, H), occu)
    assert dp_ours.shape == dp_prod.shape
    scale = dp_prod.abs().max()
    assert torch.allclose(dp_ours, dp_prod, rtol=1e-5, atol=1e-6 * scale)


# --------------------------------------------------------------------------- #
# 4. Affine exactness — the §4.1 parameterisation tripwire (silent at runtime) #
# --------------------------------------------------------------------------- #


def test_4_affine_exactness():
    """§7.4: after apply_object_step, a full re-forward gives F == F + a*D and
    u == u + 2av + a^2 w to (float32) machine precision, for ANY a — both the
    line-search root and a deliberately large fixed step. Leakage means the
    step was taken in (amp, phase) coordinates. Run at production float32,
    B = 1 (the design point), storage-shaped tensors."""
    ls = _ls()
    B, omode, Nz, Ny, Nx, pmode = 1, 1, 3, 32, 32, 2
    occu = torch.ones(omode, dtype=torch.float32)
    probe = make_probe(B, pmode, Ny, Nx, dtype=torch.complex64, seed=19)
    H = make_H(Nz, Ny, Nx, dtype=torch.complex64)
    I_dat64, _, omega64 = make_data(20, B, omode, Nz, Ny, Nx, pmode, occu.double())
    I_dat = I_dat64.float()
    omega = omega64.float()

    # large enough that a (amp,phase)-coordinate step would miss by >> float32 eps
    for a_mode in ("linesearch", "fixed"):
        obja = make_object(B, omode, Nz, Ny, Nx, dtype=torch.float32, seed=21)[0, ..., 0]
        objp = make_object(B, omode, Nz, Ny, Nx, dtype=torch.float32, seed=21)[0, ..., 1]
        patches = torch.stack([obja, objp], dim=-1).unsqueeze(0)

        F = ls.firstborn_fields(patches, probe, H)
        u_p = ls.dp_from_fields(F, occu)
        d = make_direction(B, omode, Nz, Ny, Nx, dtype=torch.complex64, seed=22, scale=0.4)
        D = ls.direction_response(patches, d, probe, H)
        v_p, w_p = ls.response_terms(F, D, occu)
        e = u_p - I_dat

        if a_mode == "linesearch":
            a = ls.line_search(e, v_p, w_p, omega, fallback=1.0 / Nz, ls_damp=0.5)
        else:
            a = 0.5  # exactness must hold for any a, not just the minimiser

        ls.apply_object_step(obja, objp, float(a), d[0])
        patches2 = torch.stack([obja, objp], dim=-1).unsqueeze(0)
        F2 = ls.firstborn_fields(patches2, probe, H)
        u2 = ls.dp_from_fields(F2, occu)

        F_pred = F + a * D
        u_pred = u_p + (2.0 * a) * v_p + (a * a) * w_p
        f_err = (F2 - F_pred).abs().max() / F.abs().max()
        u_err = (u2 - u_pred).abs().max() / u_p.abs().max()
        assert f_err < 5e-6, f"field leakage {f_err:.2e} (a_mode={a_mode}, a={a})"
        assert u_err < 2e-5, f"intensity leakage {u_err:.2e} (a_mode={a_mode}, a={a})"

    # Teeth: the WRONG update — linearised step in (amp, phase) coordinates —
    # must be caught by the tolerance above by a wide margin.
    obja = make_object(B, omode, Nz, Ny, Nx, dtype=torch.float32, seed=21)[0, ..., 0]
    objp = make_object(B, omode, Nz, Ny, Nx, dtype=torch.float32, seed=21)[0, ..., 1]
    patches = torch.stack([obja, objp], dim=-1).unsqueeze(0)
    F = ls.firstborn_fields(patches, probe, H)
    d = make_direction(B, omode, Nz, Ny, Nx, dtype=torch.complex64, seed=22, scale=0.4)
    D = ls.direction_response(patches, d, probe, H)
    a = 0.5
    dO = a * d[0]
    delta = dO * torch.polar(torch.ones_like(objp), -objp)
    obja_w = obja + delta.real
    objp_w = objp + delta.imag / obja.clamp_min(1e-6)
    F_wrong = ls.firstborn_fields(torch.stack([obja_w, objp_w], dim=-1).unsqueeze(0), probe, H)
    wrong_err = (F_wrong - (F + a * D)).abs().max() / F.abs().max()
    assert wrong_err > 100 * 5e-6, "tripwire has lost its teeth — enlarge a or d"


# --------------------------------------------------------------------------- #
# 5. Probe response — linear in P, evaluated at the UPDATED object             #
# --------------------------------------------------------------------------- #


def test_5_probe_response():
    """§7.5: D_P = F(q; g2) equals the forward difference in P exactly (F is
    linear in P — no small-step assumption), and dQ/db|_0 = 4*c0_p at the
    exactly-updated field F + a*D."""
    ls = _ls()
    B, omode, Nz, Ny, Nx, pmode = 1, 1, 3, 32, 32, 2
    occu = torch.ones(omode, dtype=torch.float64)
    probe = make_probe(B, pmode, Ny, Nx, dtype=torch.complex128, seed=23)
    H = make_H(Nz, Ny, Nx, dtype=torch.complex128)
    obj = make_object(B, omode, Nz, Ny, Nx, dtype=torch.float64, seed=24)
    I_dat, _, omega = make_data(25, B, omode, Nz, Ny, Nx, pmode, occu)

    # object step (a, d) taken first — probe search runs against the updated field
    F = ls.firstborn_fields(obj, probe, H)
    d = make_direction(B, omode, Nz, Ny, Nx, dtype=torch.complex128, seed=26, scale=0.03)
    D = ls.direction_response(obj, d, probe, H)
    u_p = ls.dp_from_fields(F, occu)
    v_p, w_p = ls.response_terms(F, D, occu)
    e = u_p - I_dat
    a = ls.line_search(e, v_p, w_p, omega, fallback=1.0 / Nz, ls_damp=0.5)

    # exact updates, no re-forward (§3 step 7)
    F_upd = F + a * D
    u_upd = u_p + (2.0 * a) * v_p + (a * a) * w_p
    e_upd = u_upd - I_dat

    # updated object g2 as complex patches
    O2 = torch.polar(obj[..., 0], obj[..., 1]) + a * d
    obj2 = torch.stack([O2.abs(), O2.angle()], dim=-1)

    # probe direction and its response: one forward with q in place of P
    q = 0.05 * make_probe(B, pmode, Ny, Nx, dtype=torch.complex128, seed=27)
    D_P = ls.firstborn_fields(obj2, q, H)

    # linearity in P: the forward difference is EXACT
    diff = ls.firstborn_fields(obj2, probe + q, H) - ls.firstborn_fields(obj2, probe, H)
    assert torch.allclose(D_P, diff, rtol=1e-9, atol=1e-11 * F.abs().max())

    # consistency of F_upd with a true re-forward at g2 (sanity, float64 tight)
    assert torch.allclose(F_upd, ls.firstborn_fields(obj2, probe, H), rtol=1e-9,
                          atol=1e-10 * F.abs().max())

    v_q, w_q = ls.response_terms(F_upd, D_P, occu)
    c0p, *_ = ls.quartic_coeffs(e_upd, v_q, w_q, omega)

    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    Qb = (omega * (ls.dp_from_fields(F_upd + b * D_P, occu) - I_dat).square()).sum()
    Qb.backward()
    assert 4.0 * c0p == pytest.approx(float(b.grad), rel=1e-9)


# --------------------------------------------------------------------------- #
# 6. Solver vs brute force, and degenerate fallbacks                           #
# --------------------------------------------------------------------------- #


def test_6_solver_vs_brute_force():
    """§7.6: dense-grid argmin of Q(a) matches the selected root; degenerate
    inputs (D = 0, fully masked, c3 = inf) return exactly ls_damp*alpha/N."""
    ls = _ls()
    B, omode, Nz, Ny, Nx, pmode = 2, 1, 3, 32, 32, 2
    occu = torch.ones(omode, dtype=torch.float64)
    probe = make_probe(B, pmode, Ny, Nx, dtype=torch.complex128, seed=28)
    H = make_H(Nz, Ny, Nx, dtype=torch.complex128)
    obj = make_object(B, omode, Nz, Ny, Nx, dtype=torch.float64, seed=29)
    I_dat, _, omega = make_data(30, B, omode, Nz, Ny, Nx, pmode, occu)

    F = ref_fields(obj, probe, H)
    d = make_direction(B, omode, Nz, Ny, Nx, dtype=torch.complex128, seed=31, scale=0.03)
    D = ref_fields_from_complex(torch.polar(obj[..., 0], obj[..., 1]) + d, probe, H) - F
    u_p = ref_dp(F, occu)
    v_p, w_p = ref_vw(F, D, occu)
    e = u_p - I_dat

    c0, c1, c2, c3 = ls.quartic_coeffs(e, v_p, w_p, omega)

    def Qfun(a):
        return ref_Q(e, v_p, w_p, omega, a)

    a_root = ls.solve_step_cubic(c0, c1, c2, c3, Qfun, fallback=1.0 / Nz)

    span = 2.0 * abs(a_root) + 1.0
    grid = np.linspace(-span, span, 40001)
    qs = np.array([Qfun(a) for a in grid])
    a_grid = grid[qs.argmin()]
    assert abs(a_root - a_grid) <= (grid[1] - grid[0]) * 1.5
    assert Qfun(a_root) <= qs.min() + 1e-9 * abs(Qfun(0.0))
    assert Qfun(a_root) < Qfun(0.0)

    # ---- degenerate inputs: exact damped fallback -------------------------
    alpha, ls_damp = 1.0, 0.5
    fb = alpha / Nz
    zero = torch.zeros_like(e)

    # D = 0  =>  v = w = 0
    a = ls.line_search(e, zero, zero, omega, fallback=fb, ls_damp=ls_damp)
    assert a == ls_damp * fb

    # fully masked
    a = ls.line_search(e, v_p, w_p, torch.zeros_like(omega), fallback=fb, ls_damp=ls_damp)
    assert a == ls_damp * fb

    # c3 = inf (float64 overflow inside the products) => isfinite guard
    w_huge = torch.full_like(w_p, 1e200)
    a = ls.line_search(e, v_p, w_huge, omega, fallback=fb, ls_damp=ls_damp)
    assert a == ls_damp * fb


# --------------------------------------------------------------------------- #
# 7. Overflow regression — the silent permanent-fallback failure (§4.3)        #
# --------------------------------------------------------------------------- #


def test_7_float32_overflow_regression():
    """§7.7: with float32 inputs at early-reconstruction magnitudes, the naive
    float32 accumulation of c3 = sum(omega * w * w) overflows to inf and the
    run silently falls back forever. quartic_coeffs must promote e, v, w,
    omega to float64 BEFORE the products and return finite values matching a
    float64 reference. (Spec quotes e ~ 1e10 => w^2 ~ 1e40; magnitudes below
    are chosen so the naive path genuinely overflows in float32.)"""
    ls = _ls()
    g = torch.Generator().manual_seed(32)
    shape = (2, 32, 32)
    e32 = (1e10 * (1.0 + torch.rand(shape, generator=g))).float()
    v32 = (1e14 * (torch.rand(shape, generator=g) - 0.5)).float()
    w32 = (1e19 * (1.0 + torch.rand(shape, generator=g))).float()
    omega32 = torch.ones(shape, dtype=torch.float32)  # dark pixels: I_dat ~ 0

    # precondition: the naive float32 path really does overflow (else this
    # test has no teeth)
    naive_c3 = (omega32 * w32 * w32).sum()
    assert not torch.isfinite(naive_c3), "synthetic magnitudes no longer overflow float32"

    c = ls.quartic_coeffs(e32, v32, w32, omega32)
    assert all(np.isfinite(ci) for ci in c), f"coefficients not finite: {c}"

    e64, v64, w64, o64 = (t.double() for t in (e32, v32, w32, omega32))
    ref = (
        float((o64 * e64 * v64).sum()),
        float((o64 * (e64 * w64 + 2.0 * v64 * v64)).sum()),
        3.0 * float((o64 * v64 * w64).sum()),
        float((o64 * w64 * w64).sum()),
    )
    for ci, ri in zip(c, ref, strict=True):
        assert ci == pytest.approx(ri, rel=1e-10)

    # and the solver must NOT take the fallback branch on these coefficients
    def Qfun(a):
        return ref_Q(e64, v64, w64, o64, a)

    fb = 12345.678  # sentinel
    a = ls.solve_step_cubic(*c, Qfun, fallback=fb)
    assert a != fb


# --------------------------------------------------------------------------- #
# 8. Joint-step invariant + the §9 per-slice seam                              #
# --------------------------------------------------------------------------- #


def test_8_joint_step_and_per_slice_seam():
    """§7.8: one scalar a applies to ALL N slices (no per-slice sweep), and D
    is accumulated from an addressable per-slice container whose sum equals
    the difference-form response (the §9 vector-search seam)."""
    ls = _ls()
    B, omode, Nz, Ny, Nx, pmode = 1, 1, 4, 32, 32, 2
    occu = torch.ones(omode, dtype=torch.float32)
    probe = make_probe(B, pmode, Ny, Nx, dtype=torch.complex64, seed=33)
    H = make_H(Nz, Ny, Nx, dtype=torch.complex64)

    obja = make_object(B, omode, Nz, Ny, Nx, dtype=torch.float32, seed=34)[0, ..., 0]
    objp = make_object(B, omode, Nz, Ny, Nx, dtype=torch.float32, seed=34)[0, ..., 1]
    patches = torch.stack([obja, objp], dim=-1).unsqueeze(0)
    d = make_direction(B, omode, Nz, Ny, Nx, dtype=torch.complex64, seed=35, scale=0.1)

    # --- per-slice seam: sum of per-slice responses == difference form -----
    D = ls.direction_response(patches, d, probe, H, per_slice=False)
    D_slices = ls.direction_response(patches, d, probe, H, per_slice=True)
    assert D_slices.shape == (B, pmode, omode, Nz, Ny, Nx)
    assert torch.allclose(D_slices.sum(dim=3), D, rtol=1e-5, atol=1e-5 * D.abs().max())
    # slices genuinely distinct (guards against a broadcast D/Nz cheat)
    assert (D_slices[:, :, :, 0] - D_slices[:, :, :, 1]).abs().max() > 1e-3 * D.abs().max()

    # difference form against the production forward
    O = torch.polar(obja, objp).unsqueeze(0)
    obj_stepped = torch.stack([(O + d).abs().squeeze(0), (O + d).angle().squeeze(0)], dim=-1)
    D_diff = ls.firstborn_fields(obj_stepped.unsqueeze(0), probe, H) - ls.firstborn_fields(
        patches, probe, H
    )
    assert torch.allclose(D, D_diff, rtol=1e-4, atol=2e-5 * D.abs().max())

    # --- joint step: per-slice fitted step is one constant scalar ----------
    a = 0.37
    O_before = torch.polar(obja, objp).clone()
    ls.apply_object_step(obja, objp, a, d[0])
    O_after = torch.polar(obja, objp)
    dO = O_after - O_before  # (omode, Nz, Ny, Nx)
    for j in range(Nz):
        num = (dO[:, j] * d[0][:, j].conj()).real.sum()
        den = d[0][:, j].abs().square().sum()
        a_j = float(num / den)
        assert a_j == pytest.approx(a, rel=1e-4), f"slice {j} stepped by {a_j}, not {a}"


# --------------------------------------------------------------------------- #
# 9. Phase branch — round trip on the measured stored-object range (§4.1)      #
# --------------------------------------------------------------------------- #


def test_9_phase_branch_round_trip():
    """§7.9: complex -> (abs, angle) -> complex is lossless on the actual
    stored object range. Measured from existing reconstructions (2026-09):

      PSO_born_paper  model_iter0200: per-slice phase in [0, 0.836] rad,
                                      amp in [0.9818, 1.0035]   (21 slices)
      tBL_WSe2_born   model_iter0200: per-slice phase in [0, 0.185] rad,
                                      amp in [0.9957, 1.0028]   (12 slices)

    Stored phase is one-sided (objp_postiv constraint) and stays far inside
    the (-pi, pi] branch of angle(), so the §4.1 round trip is safe. Also
    documents the wrap hazard the guard exists for."""
    g = torch.Generator().manual_seed(36)
    shape = (1, 21, 64, 64)
    # measured range with margin: amp [0.9, 1.05], phase [0, 1.5] rad
    amp = (0.9 + 0.15 * torch.rand(shape, generator=g)).float()
    phase = (1.5 * torch.rand(shape, generator=g)).float()

    O = torch.polar(amp, phase)
    amp2, phase2 = O.abs(), O.angle()
    assert torch.allclose(amp2, amp, rtol=0, atol=2e-6)
    assert torch.allclose(phase2, phase, rtol=0, atol=2e-6)

    # a second step/decompose cycle stays lossless (iterated round trips)
    O2 = torch.polar(amp2, phase2)
    assert torch.allclose(O2.real, O.real, rtol=0, atol=2e-6)
    assert torch.allclose(O2.imag, O.imag, rtol=0, atol=2e-6)

    # the hazard the spec warns about: beyond pi, angle() wraps and the stored
    # phase is destroyed. This is why the range above had to be measured.
    phase_big = torch.tensor([3.5], dtype=torch.float32)  # > pi
    wrapped = torch.polar(torch.ones(1), phase_big).angle()
    assert not torch.allclose(wrapped, phase_big, atol=1e-3)
    assert wrapped == pytest.approx(3.5 - 2 * np.pi, abs=1e-6)

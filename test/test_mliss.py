"""
Oracle tests for ML-ISS (ptyrad.mliss) — task-spec tests 2, 3 and 4:

- Gradient check: the ML-ISS direction, before preconditioning, against an
  INDEPENDENT analytic construction of the L_G detector residual
  (w/sigma^2)(I_data - I_model) psi_hat backprojected through the branch
  adjoint, on a random N = 3 slice, 2 probe-mode, 3-position problem, to
  round-off (float64).
- Line-search check (object): L_G at ~20 gammas by explicit forward passes
  matches the quartic built from (e, v, w) to round-off, and the chosen
  gamma is its minimiser. Repeated for the probe step at the UPDATED slices.
- Thin-sample check: with N = 1 the ISS field reduces to the thin-sample
  model (O' = O, D_0 = D) and the ML-ISS object step equals a hand-built
  thin-sample implementation.
- Objective consistency: the direction and the line search use the same
  omega = mask/sigma^2 (structural: Q(a) == 2 L_G(a) exactly).
- Monotonicity on the small model: non-fallback steps never raise L_G
  (verified by fresh forward evaluation, not the quartic).

CPU + eager, synthetic tensors only.
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import pytest
import torch

torch._dynamo.config.disable = True

from torch.fft import fft2, fftshift, ifft2, ifftshift

import ptyrad.mliss as ml
from ptyrad.mliss import (
    _fields_from_complex,
    direction_response,
    dp_from_fields,
    iss_fields,
    object_denominator,
    unscattered_illumination,
)

torch.manual_seed(0)

B, PMODE, OMODE, NZ, NY, NX = 3, 2, 1, 3, 16, 16
C_PER_UNIT = 5.0e3  # counts per normalised unit for these tests


def _mk(dtype=torch.complex128, seed=3):
    g = torch.Generator().manual_seed(seed)
    rdt = torch.float64 if dtype == torch.complex128 else torch.float32
    amp = 1.0 + 0.02 * torch.randn(B, OMODE, NZ, NY, NX, generator=g, dtype=rdt)
    phs = 0.3 * torch.rand(B, OMODE, NZ, NY, NX, generator=g, dtype=rdt)
    O = torch.polar(amp, phs)
    pr = torch.randn(1, PMODE, NY, NX, generator=g, dtype=rdt)
    pi = torch.randn(1, PMODE, NY, NX, generator=g, dtype=rdt)
    P = (pr + 1j * pi).to(dtype) / (NY * NX) ** 0.5
    ky = torch.fft.fftfreq(NY, dtype=torch.float64)
    kx = torch.fft.fftfreq(NX, dtype=torch.float64)
    chi = 0.7 * (ky[:, None] ** 2 + kx[None, :] ** 2) * (NY * NX) ** 0.5
    H1 = torch.exp(-1j * chi).to(dtype)
    j = torch.arange(NZ, dtype=rdt).view(NZ, 1, 1)
    H = (H1**j).view(1, 1, 1, NZ, NY, NX)
    occu = torch.ones(OMODE, dtype=rdt)
    # data from a perturbed object, plus a floor so sigma^2 is well-defined
    O2 = O * torch.polar(torch.ones_like(amp), 0.05 * torch.randn(*amp.shape, generator=g, dtype=rdt))
    I_dat = dp_from_fields(_fields_from_complex(O2, P, H), occu).clamp_min(0.0)
    mask = (torch.rand(B, NY, NX, generator=g, dtype=rdt) > 0.1).to(rdt)
    return O, P, H, occu, I_dat, mask


def test_gradient_matches_analytic_residual():
    """-O.grad of L_G == 2 sum_m occu_m conj(phi_jm) IFFT[H_j F_m rho], with
    rho = ifftshift(omega (I_dat - u)) — the (w/sigma^2)(I_data - I_model)
    psi_hat residual of the spec, backprojected through the branch adjoint.
    Analogous closed form for the probe. Round-off in float64."""
    O, P, H, occu, I_dat, mask = _mk()
    sigma2 = ml.mliss_sigma2(I_dat, C_PER_UNIT)

    d_eng, q_eng = ml.mliss_direction(O, P, H, I_dat, mask, sigma2, occu)

    F = _fields_from_complex(O, P, H)
    u = dp_from_fields(F, occu)
    omega = mask / sigma2
    rho = ifftshift(omega * (I_dat - u), dim=(-2, -1))  # (B, Ny, Nx)
    rho = rho.view(B, 1, 1, NY, NX).to(F.dtype)

    phi = unscattered_illumination(P, H)  # (1, pmode, 1, Nz, Ny, Nx)
    cw = (occu / (NY * NX)).view(1, 1, -1, 1, 1, 1)
    # object residual backprojection (adjoint of fft2 is (NxNy) ifft2)
    # H is (1,1,1,Nz,Ny,Nx) with pmode/omode singletons; phi is
    # (1,pmode,1,Nz,Ny,Nx) — both broadcast against (B,pm,om,Nz,Ny,Nx)
    T = ifft2(H * (F * rho.view(B, 1, 1, NY, NX)).unsqueeze(3))
    d_ref = 2.0 * (NY * NX) * (cw * phi.conj() * T).sum(dim=1)

    assert torch.allclose(d_eng, d_ref, rtol=1e-10, atol=1e-12), (
        f"object direction mismatch: {(d_eng - d_ref).abs().max()}"
    )

    # probe residual: F is linear in P -> descent is the adjoint of
    # P -> F(P; g) applied to the residual; verify with autograd of an
    # INDEPENDENT forward (production iss path via iss_fields).
    P2 = P.detach().clone().requires_grad_(True)
    patches = torch.stack([O.abs(), O.angle()], dim=-1)
    F2 = iss_fields(patches, P2, H)
    L2 = ml.mliss_loss(dp_from_fields(F2, occu), I_dat, mask, sigma2)
    L2.backward()
    assert torch.allclose(q_eng, -P2.grad, rtol=1e-10, atol=1e-12)


def test_object_line_search_matches_explicit_forwards():
    """Quartic from (e, v, w) == L_G(gamma) by explicit forward passes at 20
    gammas (round-off), and the returned step is the quartic's minimiser."""
    O, P, H, occu, I_dat, mask = _mk()
    sigma2 = ml.mliss_sigma2(I_dat, C_PER_UNIT)
    omega = mask / sigma2

    g = torch.Generator().manual_seed(9)
    d = (
        torch.randn(B, OMODE, NZ, NY, NX, generator=g, dtype=torch.float64)
        + 1j * torch.randn(B, OMODE, NZ, NY, NX, generator=g, dtype=torch.float64)
    ) * 0.02

    F = _fields_from_complex(O, P, H)
    u = dp_from_fields(F, occu)
    e = u - I_dat
    D = direction_response(None, d, P, H)
    from ptyrad.mliss import response_terms

    v, w = response_terms(F, D, occu)

    gammas = torch.linspace(-2.0, 2.0, 21, dtype=torch.float64)
    for gam in gammas:
        L_fwd = ml.mliss_loss(
            dp_from_fields(_fields_from_complex(O + gam * d, P, H), occu),
            I_dat,
            mask,
            sigma2,
        )
        L_quart = 0.5 * (omega * (e + 2 * gam * v + gam * gam * w) ** 2).sum()
        assert torch.allclose(L_fwd, L_quart, rtol=1e-11, atol=1e-14), (
            f"gamma={gam}: forward {L_fwd} vs quartic {L_quart}"
        )

    a, fb = ml.ml_line_search(e, v, w, omega, fallback=1.0 / NZ, damp=1.0)
    assert not fb
    # chosen gamma is the dense-grid minimiser of the quartic
    dense = torch.linspace(a - 1.0, a + 1.0, 20001, dtype=torch.float64)
    Ld = (
        0.5
        * (
            omega.unsqueeze(0)
            * (
                e.unsqueeze(0)
                + 2 * dense.view(-1, 1, 1, 1) * v.unsqueeze(0)
                + dense.view(-1, 1, 1, 1) ** 2 * w.unsqueeze(0)
            )
            ** 2
        ).sum(dim=(1, 2, 3))
    )
    a_dense = float(dense[Ld.argmin()])
    assert a == pytest.approx(a_dense, abs=2e-4)
    # and it is a stationary point: dL/dgamma(a) ~ 0
    r = e + 2 * a * v + a * a * w
    dL = (omega * r * (2 * v + 2 * a * w)).sum()
    assert abs(float(dL)) < 1e-8 * float((omega * r * r).sum() + 1.0)


def test_probe_line_search_uses_updated_slices():
    """Probe-step quartic against explicit forwards with the UPDATED object
    O + a d and probe P + gamma q (a_P = sum_j D_j[(O'_j + a d_j) Prop d_P])."""
    O, P, H, occu, I_dat, mask = _mk()
    sigma2 = ml.mliss_sigma2(I_dat, C_PER_UNIT)
    omega = mask / sigma2

    g = torch.Generator().manual_seed(21)
    d = (
        torch.randn(B, OMODE, NZ, NY, NX, generator=g, dtype=torch.float64)
        + 1j * torch.randn(B, OMODE, NZ, NY, NX, generator=g, dtype=torch.float64)
    ) * 0.02
    q = (
        torch.randn(1, PMODE, NY, NX, generator=g, dtype=torch.float64)
        + 1j * torch.randn(1, PMODE, NY, NX, generator=g, dtype=torch.float64)
    ) * (0.02 / (NY * NX) ** 0.5)

    a = 0.37  # some object step
    O2 = O + a * d
    F2 = _fields_from_complex(O2, P, H)
    u2 = dp_from_fields(F2, occu)
    e2 = u2 - I_dat
    D_P = _fields_from_complex(O2, q, H)  # F linear in P, at UPDATED slices
    from ptyrad.mliss import response_terms

    v_p, w_p = response_terms(F2, D_P, occu)

    for gam in torch.linspace(-3.0, 3.0, 19, dtype=torch.float64):
        L_fwd = ml.mliss_loss(
            dp_from_fields(_fields_from_complex(O2, P + gam * q, H), occu),
            I_dat,
            mask,
            sigma2,
        )
        L_quart = 0.5 * (omega * (e2 + 2 * gam * v_p + gam * gam * w_p) ** 2).sum()
        assert torch.allclose(L_fwd, L_quart, rtol=1e-11, atol=1e-14)

    b, fb = ml.ml_line_search(e2, v_p, w_p, omega, fallback=1.0 / NZ, damp=1.0)
    assert not fb
    r = e2 + 2 * b * v_p + b * b * w_p
    dL = (omega * r * (2 * v_p + 2 * b * w_p)).sum()
    assert abs(float(dL)) < 1e-8 * float((omega * r * r).sum() + 1.0)
    # strictly lowers L_G vs gamma = 0 (fresh forward, not the quartic)
    L0 = ml.mliss_loss(u2, I_dat, mask, sigma2)
    Lb = ml.mliss_loss(
        dp_from_fields(_fields_from_complex(O2, P + b * q, H), occu), I_dat, mask, sigma2
    )
    assert float(Lb) < float(L0)


def test_thin_sample_reduction_N1():
    """N = 1: the ISS field must equal the thin-sample model
    F = FFT[P] + FFT[(O - 1) P] = FFT[O * P] (H_0 = 1, so D_0 = D), and an
    ML-ISS object step equals a hand-built thin-sample update."""
    g = torch.Generator().manual_seed(5)
    amp = 1.0 + 0.02 * torch.randn(B, OMODE, 1, NY, NX, generator=g, dtype=torch.float64)
    phs = 0.3 * torch.rand(B, OMODE, 1, NY, NX, generator=g, dtype=torch.float64)
    O = torch.polar(amp, phs)
    pr = torch.randn(1, PMODE, NY, NX, generator=g, dtype=torch.float64)
    pi = torch.randn(1, PMODE, NY, NX, generator=g, dtype=torch.float64)
    P = (pr + 1j * pi) / (NY * NX) ** 0.5
    H = torch.ones(1, 1, 1, 1, NY, NX, dtype=torch.complex128)
    occu = torch.ones(OMODE, dtype=torch.float64)

    F = _fields_from_complex(O, P, H)
    F_thin = fft2(O[:, None, :, 0] * P[:, :, None])  # (B, pmode, omode, Ny, Nx)
    assert torch.allclose(F, F_thin, rtol=1e-12, atol=1e-12)

    # a full ML-ISS tensor-level update at N = 1 must run and step sensibly
    I_dat = dp_from_fields(F, occu).clamp_min(0.0) * (
        1.0 + 0.05 * torch.rand(B, NY, NX, generator=g, dtype=torch.float64)
    )
    sigma2 = ml.mliss_sigma2(I_dat, C_PER_UNIT)
    omega = 1.0 / sigma2
    d_eng, _ = ml.mliss_direction(O, P, H, I_dat, None, sigma2, occu)
    phi = unscattered_illumination(P, H)
    dn = object_denominator(phi)
    d = d_eng / dn
    D = direction_response(None, d, P, H)
    from ptyrad.mliss import response_terms

    u = dp_from_fields(F, occu)
    v, w = response_terms(F, D, occu)
    a, fb = ml.ml_line_search(u - I_dat, v, w, omega, fallback=1.0, damp=1.0)
    assert not fb
    L0 = ml.mliss_loss(u, I_dat, None, sigma2)
    La = ml.mliss_loss(
        dp_from_fields(_fields_from_complex(O + a * d, P, H), occu), I_dat, None, sigma2
    )
    assert float(La) < float(L0)
    # thin-sample hand model gives the identical quartic coefficients
    D_thin = fft2(d[:, None, :, 0] * P[:, :, None])
    assert torch.allclose(D, D_thin, rtol=1e-12, atol=1e-12)


def test_batch_update_monotone_and_consistent():
    """Full alternating updates on float32 tensors: L_G (fresh forward) never
    increases over an update sweep with non-fallback steps, and the object
    displacement is exactly a * (direction / K_j) — i.e. what was searched is
    what was stepped."""
    O, P, H, occu, I_dat, mask = _mk(dtype=torch.complex64, seed=13)
    obja = O[0].abs().clone()  # single view (B = 1 tensor-level API)
    objp = O[0].angle().clone()
    I1 = I_dat[[0]].float()
    H32, P32 = H.to(torch.complex64), P.to(torch.complex64)
    occ32 = occu.float()
    cfg = ml.MLISSConfig(counts_per_unit=C_PER_UNIT)
    st = ml.MLISSState()

    sigma2 = ml.mliss_sigma2(I1, cfg.counts_per_unit)

    def LG():
        pat = torch.stack([obja, objp], dim=-1).unsqueeze(0)
        u = dp_from_fields(iss_fields(pat, P32, H32), occ32)
        return float(ml.mliss_loss(u, I1, None, sigma2))

    probe = P32.clone()
    prev = LG()
    for it in range(6):
        # direction/step consistency probe on the first update
        if it == 0:
            d_pre, _ = ml.mliss_direction(
                torch.polar(obja, objp).unsqueeze(0), probe, H32, I1, None, sigma2, occ32
            )
            dn = object_denominator(unscattered_illumination(probe, H32))
            obja0 = obja.clone()
            objp0 = objp.clone()
        probe, diag = ml.mliss_batch_update(
            obja, objp, probe, H32, I1, None, occ32, config=cfg, state=st, update_probe=True
        )
        if it == 0 and not diag["fb_o"]:
            O_expect = torch.polar(obja0, objp0) + diag["a"] * (d_pre[0] / dn)
            assert torch.allclose(torch.polar(obja, objp), O_expect, rtol=1e-5, atol=1e-7)
        cur = LG()
        if not diag["fb_o"] and not diag["fb_p"]:
            assert cur <= prev * (1 + 1e-5), f"L_G rose on non-fallback step: {prev} -> {cur}"
        prev = cur
    assert len(st.steps_o) == 6 and len(st.steps_p) == 6
    assert st.fallback_o < 6  # the search must actually engage


def test_sigma2_requires_counts_scale():
    with pytest.raises(ValueError):
        ml.mliss_sigma2(torch.zeros(2), None)
    s = ml.mliss_sigma2(torch.zeros(2), 4.0)
    assert torch.allclose(s, torch.full((2,), 0.25))


def test_joint_line_search_matches_explicit_forwards():
    """Task-2 joint step: for random d_j and d_{P,m} (N = 3 slices, 2 probe
    modes, 3 positions), L_G at ~20 gammas by explicit forward passes at
    (O + g d, P + g d_P) matches the degree-8 polynomial built from
    A0..A4 to round-off, and the chosen gamma is its minimiser."""
    O, P, H, occu, I_dat, mask = _mk(seed=31)
    sigma2 = ml.mliss_sigma2(I_dat, C_PER_UNIT)
    omega = mask / sigma2

    g = torch.Generator().manual_seed(77)
    d = (
        torch.randn(B, OMODE, NZ, NY, NX, generator=g, dtype=torch.float64)
        + 1j * torch.randn(B, OMODE, NZ, NY, NX, generator=g, dtype=torch.float64)
    ) * 0.02
    q = (
        torch.randn(1, PMODE, NY, NX, generator=g, dtype=torch.float64)
        + 1j * torch.randn(1, PMODE, NY, NX, generator=g, dtype=torch.float64)
    ) * (0.02 / (NY * NX) ** 0.5)

    F = _fields_from_complex(O, P, H)
    u = dp_from_fields(F, occu)
    e = u - I_dat
    from ptyrad.mliss import response_terms

    a_f = _fields_from_complex(O, q, H) + direction_response(None, d, P, H)
    b_f = direction_response(None, d, q, H)
    va, wa = response_terms(F, a_f, occu)
    vb, wb = response_terms(F, b_f, occu)
    ab = response_terms(a_f, b_f, occu)[0]

    # psi(g) = psi + g a + g^2 b holds exactly: check the field expansion once
    gam0 = 0.63
    F_gam = _fields_from_complex(O + gam0 * d, P + gam0 * q, H)
    assert torch.allclose(F_gam, F + gam0 * a_f + gam0**2 * b_f, rtol=1e-11, atol=1e-12)

    for gam in torch.linspace(-1.5, 1.5, 21, dtype=torch.float64):
        L_fwd = ml.mliss_loss(
            dp_from_fields(_fields_from_complex(O + gam * d, P + gam * q, H), occu),
            I_dat,
            mask,
            sigma2,
        )
        r = e + 2 * gam * va + gam**2 * (2 * vb + wa) + 2 * gam**3 * ab + gam**4 * wb
        L_poly = 0.5 * (omega * r * r).sum()
        assert torch.allclose(L_fwd, L_poly, rtol=1e-11, atol=1e-14), (
            f"gamma={gam}: forward {L_fwd} vs octic {L_poly}"
        )

    gsel, fb = ml.ml_joint_line_search(e, va, wa, vb, ab, wb, omega, fallback=1.0 / NZ)
    assert not fb
    # stationary and the global dense-grid minimiser
    dense = torch.linspace(gsel - 1.0, gsel + 1.0, 20001, dtype=torch.float64)
    rr = (
        e.unsqueeze(0)
        + 2 * dense.view(-1, 1, 1, 1) * va.unsqueeze(0)
        + dense.view(-1, 1, 1, 1) ** 2 * (2 * vb + wa).unsqueeze(0)
        + 2 * dense.view(-1, 1, 1, 1) ** 3 * ab.unsqueeze(0)
        + dense.view(-1, 1, 1, 1) ** 4 * wb.unsqueeze(0)
    )
    Ld = 0.5 * (omega.unsqueeze(0) * rr * rr).sum(dim=(1, 2, 3))
    assert gsel == pytest.approx(float(dense[Ld.argmin()]), abs=2e-4)
    L0 = ml.mliss_loss(u, I_dat, mask, sigma2)
    Lsel = ml.mliss_loss(
        dp_from_fields(_fields_from_complex(O + gsel * d, P + gsel * q, H), occu),
        I_dat, mask, sigma2,
    )
    assert float(Lsel) < float(L0)


def test_joint_batch_update_descends():
    """Full joint-mode update on float32 tensors descends L_G (fresh forward)
    and moves object AND probe with one gamma."""
    O, P, H, occu, I_dat, mask = _mk(dtype=torch.complex64, seed=17)
    obja = O[0].abs().clone()
    objp = O[0].angle().clone()
    I1 = I_dat[[0]].float()
    H32, P32, occ32 = H.to(torch.complex64), P.to(torch.complex64), occu.float()
    cfg = ml.MLISSConfig(counts_per_unit=C_PER_UNIT, step_mode="joint")
    st = ml.MLISSState()
    sigma2 = ml.mliss_sigma2(I1, cfg.counts_per_unit)

    def LG(probe):
        pat = torch.stack([obja, objp], dim=-1).unsqueeze(0)
        u = dp_from_fields(iss_fields(pat, probe, H32), occ32)
        return float(ml.mliss_loss(u, I1, None, sigma2))

    probe = P32.clone()
    prev = LG(probe)
    p_before = probe.clone()
    o_before = torch.polar(obja, objp).clone()
    for _ in range(4):
        probe, diag = ml.mliss_batch_update(
            obja, objp, probe, H32, I1, None, occ32, config=cfg, state=st, update_probe=True
        )
        assert diag["b"] == diag["a"]  # one gamma moves both
        cur = LG(probe)
        if not diag["fb_o"]:
            assert cur <= prev * (1 + 1e-5), f"L_G rose on joint non-fallback step: {prev} -> {cur}"
        prev = cur
    assert not torch.equal(probe, p_before)
    assert not torch.equal(torch.polar(obja, objp), o_before)
    assert len(st.steps_o) == 4 and len(st.steps_p) == 0


def test_poisson_gradient_and_line_search():
    """objective='poisson': the direction gradient of L_P matches (a) an
    independent autodiff through the production iss_fields path and (b) the
    analytic residual backprojection with weight c(I/u - 1); and L_P(gamma)
    from the (u, v, w) response terms matches explicit forward passes to
    round-off, with the chosen gamma its minimiser."""
    O, P, H, occu, I_dat, mask = _mk(seed=41)
    c = C_PER_UNIT

    # (a) engine-style gradient (same ops as the update path) ...
    O1 = O.detach().clone().requires_grad_(True)
    F1 = _fields_from_complex(O1, P, H)
    L1 = ml._lp_loss(dp_from_fields(F1, occu), I_dat, mask, c)
    L1.backward()
    # ... vs independent autodiff through the production iss path
    O2p = torch.stack([O.abs(), O.angle()], dim=-1)
    P2 = P.detach().clone().requires_grad_(True)
    F2 = iss_fields(O2p, P2, H)
    u2 = dp_from_fields(F2, occu)
    L2 = ml._lp_loss(u2, I_dat, mask, c)
    L2.backward()

    # (b) analytic residual: dL_P/du = c w (1 - I/u_floored) -> descent
    # weight rho = ifftshift(mask * c * (I/u - 1)); same backprojection as
    # the Gaussian oracle test
    F = _fields_from_complex(O, P, H)
    u = dp_from_fields(F, occu)
    W = mask * c * (I_dat.clamp_min(0) / (c * u).clamp_min(1e-12).div(c) - 1.0)
    rho = ifftshift(W, dim=(-2, -1)).view(B, 1, 1, NY, NX).to(F.dtype)
    phi = unscattered_illumination(P, H)
    cw = (occu / (NY * NX)).view(1, 1, -1, 1, 1, 1)
    T = ifft2(H * (F * rho).unsqueeze(3))
    d_ref = 2.0 * (NY * NX) * (cw * phi.conj() * T).sum(dim=1)

    d_eng = -O1.grad
    assert torch.allclose(d_eng, d_ref, rtol=1e-9, atol=1e-11)

    # probe gradient vs independent path
    P1 = P.detach().clone().requires_grad_(True)
    F1b = _fields_from_complex(O, P1, H)
    L1b = ml._lp_loss(dp_from_fields(F1b, occu), I_dat, mask, c)
    L1b.backward()
    assert torch.allclose(-P1.grad, -P2.grad, rtol=1e-10, atol=1e-12)

    # L_P(gamma) from response terms vs explicit forwards, ~20 gammas
    g = torch.Generator().manual_seed(51)
    d = (
        torch.randn(B, OMODE, NZ, NY, NX, generator=g, dtype=torch.float64)
        + 1j * torch.randn(B, OMODE, NZ, NY, NX, generator=g, dtype=torch.float64)
    ) * 0.02
    D = direction_response(None, d, P, H)
    from ptyrad.mliss import response_terms

    v, w = response_terms(F, D, occu)
    for gam in torch.linspace(-1.0, 1.0, 21, dtype=torch.float64):
        L_fwd = ml._lp_loss(
            dp_from_fields(_fields_from_complex(O + gam * d, P, H), occu), I_dat, mask, c
        )
        cu = (c * (u + 2 * gam * v + gam * gam * w)).clamp_min(1e-12)
        L_uvw = (mask * (cu - c * I_dat.clamp_min(0) * torch.log(cu))).sum()
        assert torch.allclose(L_fwd, L_uvw, rtol=1e-11, atol=1e-11), f"gamma={gam}"

    # minimiser: use the descent direction, then check stationarity on a grid
    d_desc = d_eng / d_eng.abs().max() * 0.05
    D2 = direction_response(None, d_desc, P, H)
    v2, w2 = response_terms(F, D2, occu)
    a, fb, nev = ml.lp_line_search(u, v2, w2, I_dat, mask, c, fallback=1.0 / NZ)
    assert not fb and nev < 80
    dense = np.linspace(a - 0.5, a + 0.5, 4001)
    vals = []
    for x in dense:
        cu = (c * (u + 2 * x * v2 + x * x * w2)).clamp_min(1e-12)
        vals.append(float((mask * (cu - c * I_dat.clamp_min(0) * torch.log(cu))).sum()))
    assert a == pytest.approx(float(dense[int(np.argmin(vals))]), abs=5e-4)


def test_poisson_batch_update_descends():
    """Full alternating updates with objective='poisson' descend L_P."""
    O, P, H, occu, I_dat, mask = _mk(dtype=torch.complex64, seed=23)
    obja, objp = O[0].abs().clone(), O[0].angle().clone()
    I1 = I_dat[[0]].float()
    H32, P32, occ32 = H.to(torch.complex64), P.to(torch.complex64), occu.float()
    cfg = ml.MLISSConfig(counts_per_unit=C_PER_UNIT, objective="poisson")
    st = ml.MLISSState()

    def LP():
        pat = torch.stack([obja, objp], dim=-1).unsqueeze(0)
        u = dp_from_fields(iss_fields(pat, P32, H32), occ32)
        return float(ml._lp_loss(u, I1, None, C_PER_UNIT))

    probe = P32.clone()
    prev = LP()
    for _ in range(5):
        probe, diag = ml.mliss_batch_update(
            obja, objp, probe, H32, I1, None, occ32, config=cfg, state=st, update_probe=True
        )
        cur = LP()
        if not diag["fb_o"] and not diag["fb_p"]:
            assert cur <= prev * (1 + 1e-6), f"L_P rose: {prev} -> {cur}"
        prev = cur
    assert len(st.ls_evals) == 10  # object + probe searches logged


def test_vector_slice_step():
    """N-dimensional per-slice step: the multivariate quartic equals explicit
    forwards at random gamma vectors (round-off, float64); the Newton solution
    strictly improves on the scalar step and has ~zero gradient; N=1 reduces
    to the scalar cubic root."""
    O, P, H, occu, I_dat, mask = _mk(seed=61)
    sigma2 = ml.mliss_sigma2(I_dat, C_PER_UNIT)
    omega = mask / sigma2

    # descent direction (as the engine builds it)
    d_eng, _ = ml.mliss_direction(O, P, H, I_dat, mask, sigma2, occu)
    dn = object_denominator(unscattered_illumination(P, H))
    d = d_eng / dn
    D_slices = direction_response(None, d, P, H, per_slice=True)
    F = _fields_from_complex(O, P, H)
    u = dp_from_fields(F, occu)
    e = u - I_dat

    gvec, fb, vx = ml.ml_vector_line_search(e, F, D_slices, omega, occu, fallback=1.0 / NZ)
    assert not fb
    assert vx["L_vector"] < vx["L_scalar"] < vx["L0"]

    # exactness: quartic residual == explicit forward at arbitrary gamma vecs
    g = torch.Generator().manual_seed(3)
    for _ in range(5):
        gam = 0.5 * torch.randn(NZ, generator=g, dtype=torch.float64)
        O2 = O + torch.einsum("j,bojyx->bojyx", gam.to(O.real.dtype) * 0 + 1, d * 0)  # placeholder
        O2 = O + (d * gam.view(1, 1, -1, 1, 1))
        L_fwd = ml.mliss_loss(
            dp_from_fields(_fields_from_complex(O2, P, H), occu), I_dat, mask, sigma2
        )
        Ffull = F + (D_slices * gam.view(1, 1, 1, -1, 1, 1)).sum(dim=3)
        r = dp_from_fields(Ffull, occu) - I_dat
        L_q = 0.5 * (omega * r * r).sum()
        assert torch.allclose(L_fwd, L_q, rtol=1e-11, atol=1e-13)

    # gradient ~ 0 at the vector solution (finite differences per slice)
    def LG_at(gam):
        O2 = O + (d * gam.view(1, 1, -1, 1, 1).to(O.dtype))
        return float(ml.mliss_loss(
            dp_from_fields(_fields_from_complex(O2, P, H), occu), I_dat, mask, sigma2))
    base = LG_at(gvec)
    for j in range(NZ):
        eps = 1e-5
        gp = gvec.clone(); gp[j] += eps
        gm = gvec.clone(); gm[j] -= eps
        deriv = (LG_at(gp) - LG_at(gm)) / (2 * eps)
        assert abs(deriv) < 1e-5 * (1 + abs(base)), f"slice {j}: dL/dgamma_j = {deriv}"

    # N = 1 reduction: vector step == scalar step
    O1 = O[:, :, :1]
    H1 = H[:, :, :, :1]
    d1 = d[:, :, :1]
    D1 = direction_response(None, d1, P, H1, per_slice=True)
    F1 = _fields_from_complex(O1, P, H1)
    e1 = dp_from_fields(F1, occu) - I_dat
    g1, fb1, vx1 = ml.ml_vector_line_search(e1, F1, D1, omega, occu, fallback=1.0)
    from ptyrad.mliss import response_terms
    v1, w1 = response_terms(F1, D1[:, :, :, 0], occu)
    a1, _ = ml.ml_line_search(e1, v1, w1, omega, fallback=1.0)
    assert float(g1[0]) == pytest.approx(a1, rel=1e-6)

"""Oracle tests for the ML-ISS2 (order-2) extension of ptyrad.mliss.

Run: TORCHDYNAMO_DISABLE=1 .venv/bin/python -m pytest test/test_mliss_m2.py -q
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch
from torch.fft import fft2, ifft2

from ptyrad.mliss import (
    MLISSConfig,
    _fields_from_complex,
    _fields_from_complex_m2,
    _m2_direction_fields,
    _quartic_I,
    dp_from_fields,
    la_line_search,
    la_line_search_quartic,
    mliss_batch_update,
    mliss_loss,
    mliss_sigma2,
    response_terms,
)

torch.manual_seed(0)
B, PM, OM, NZ, NY, NX = 2, 2, 1, 5, 32, 32
DT = torch.complex128


def _setup():
    # unimodular H with the group property H_a H_b = H_{a+b}: H_j = base**j
    phase = torch.rand(NY, NX, dtype=torch.float64) * 2 * np.pi
    base = torch.polar(torch.ones(NY, NX, dtype=torch.float64), phase)
    zj = torch.arange(NZ).view(1, 1, 1, NZ, 1, 1)
    H = (base ** zj).to(DT)
    probe = (torch.randn(1, PM, NY, NX, dtype=torch.float64)
             + 1j * torch.randn(1, PM, NY, NX, dtype=torch.float64)).to(DT)
    O = (1.0 + 0.15 * (torch.randn(B, OM, NZ, NY, NX, dtype=torch.float64)
                       + 1j * torch.randn(B, OM, NZ, NY, NX, dtype=torch.float64))).to(DT)
    occu = torch.ones(OM, dtype=torch.float64)
    return H, base, probe, O, occu


def _brute_force_m2(O, probe, H, base):
    """Independent double loop: first + second order, detector-referred with
    the same global-H_N convention (H_j^* back-reference)."""
    Ny, Nx = O.shape[-2:]
    probe_k = fft2(probe).view(-1, probe.shape[1], 1, 1, Ny, Nx)
    psi = ifft2(H * probe_k)  # (B|1, pmode, omode, Nz, Ny, Nx) via broadcast
    g = (O - 1.0).unsqueeze(1)
    F = probe_k.squeeze(3).clone().expand(O.shape[0], -1, -1, -1, -1).clone()
    for j in range(NZ):
        F = F + fft2(g[:, :, :, j] * psi[..., j, :, :]) * H[..., j, :, :].conj()
        for k in range(j):
            once = g[:, :, :, k] * psi[..., k, :, :]  # scattered at k
            prop = ifft2((base ** (j - k)).to(DT) * fft2(once))  # to slice j
            F = F + fft2(g[:, :, :, j] * prop) * H[..., j, :, :].conj()
    return F


def test_m2_forward_matches_brute_force():
    H, base, probe, O, _ = _setup()
    F_fast = _fields_from_complex_m2(O, probe, H)
    F_ref = _brute_force_m2(O, probe, H, base)
    err = (F_fast - F_ref).abs().max() / F_ref.abs().max()
    assert err < 1e-12, f"cumsum M=2 forward disagrees with brute force: {err:.3e}"


def test_m2_reduces_to_m1_plus_second_order():
    H, _, probe, O, _ = _setup()
    # single slice: no pair (k < j), M=2 == M=1 exactly
    F1 = _fields_from_complex(O[:, :, :1], probe, H[..., :1, :, :])
    F2 = _fields_from_complex_m2(O[:, :, :1], probe, H[..., :1, :, :])
    assert (F1 - F2).abs().max() < 1e-12
    # second order is homogeneous of degree 2: scaling g by t scales it by t^2
    t = 0.5
    Ot = 1.0 + t * (O - 1.0)
    sec = _fields_from_complex_m2(O, probe, H) - _fields_from_complex(O, probe, H)
    sec_t = _fields_from_complex_m2(Ot, probe, H) - _fields_from_complex(Ot, probe, H)
    err = (sec_t - t**2 * sec).abs().max() / sec.abs().max()
    assert err < 1e-12, f"second order not homogeneous degree 2: {err:.3e}"


def test_direction_fields_exact():
    H, _, probe, O, _ = _setup()
    d = 0.05 * (torch.randn_like(O.real) + 1j * torch.randn_like(O.real)).to(DT)
    F = _fields_from_complex_m2(O, probe, H)
    D1, D2 = _m2_direction_fields(O, d, probe, H, F)
    for a in (0.37, -1.4, 2.0):
        F_a = _fields_from_complex_m2(O + a * d, probe, H)
        pred = F + a * D1 + a * a * D2
        err = (F_a - pred).abs().max() / F_a.abs().max()
        assert err < 1e-11, f"F(a) != F + aD1 + a^2 D2 at a={a}: {err:.3e}"


def test_quartic_intensity_coefficients():
    H, _, probe, O, occu = _setup()
    d = 0.05 * (torch.randn_like(O.real) + 1j * torch.randn_like(O.real)).to(DT)
    F = _fields_from_complex_m2(O, probe, H)
    D1, D2 = _m2_direction_fields(O, d, probe, H, F)
    u = dp_from_fields(F, occu)
    va, wa = response_terms(F, D1, occu)
    vb, wb = response_terms(F, D2, occu)
    ab = response_terms(D1, D2, occu)[0]
    A1, A2, A3, A4 = _quartic_I(u, va, wa, vb, ab, wb)
    for a in (0.6, -0.9):
        u_a = dp_from_fields(F + a * D1 + a * a * D2, occu).double()
        pred = u.double() + a * (A1 + a * (A2 + a * (A3 + a * A4)))
        err = (u_a - pred).abs().max() / u_a.abs().max()
        assert err < 1e-11, f"quartic intensity coefficients wrong at a={a}: {err:.3e}"


def test_quartic_amplitude_search_reduces_to_quadratic():
    H, _, probe, O, occu = _setup()
    d = 0.05 * (torch.randn_like(O.real) + 1j * torch.randn_like(O.real)).to(DT)
    F = _fields_from_complex_m2(O, probe, H)
    D1, _ = _m2_direction_fields(O, d, probe, H, F)
    u = dp_from_fields(F, occu)
    va, wa = response_terms(F, D1, occu)
    z = torch.zeros_like(va)
    I_dat = (u * (1.0 + 0.1 * torch.rand_like(u))).detach()
    a_q, fb_q, _ = la_line_search_quartic(u, va, wa, z, z, z, I_dat, None, fallback=0.1)
    a_2, fb_2, _ = la_line_search(u, va, wa, I_dat, None, fallback=0.1)
    assert fb_q == fb_2
    assert abs(a_q - a_2) < 1e-8 * (1 + abs(a_2)), f"{a_q} vs {a_2}"


def _engine_setup(dtype=torch.float32):
    torch.manual_seed(1)
    H, _, probe, O, occu = _setup()
    obja = O.abs()[0].to(dtype)
    objp = O.angle()[0].to(dtype)
    probe32 = probe.to(torch.complex64)
    H32 = H.to(torch.complex64)
    # synthetic data from a perturbed object, through the M=2 model itself
    O_true = O * torch.exp(1j * 0.05 * torch.randn_like(O.real)).to(DT)
    I_dat = dp_from_fields(
        _fields_from_complex_m2(O_true, probe, H), occu
    ).to(dtype)[:1]
    return obja, objp, probe32, H32, I_dat, occu.to(dtype)


def test_batch_update_order2_decreases_objective():
    for objective in ("gaussian", "amplitude"):
        obja, objp, probe, H, I_dat, occu = _engine_setup()
        cfg = MLISSConfig(counts_per_unit=558.0, objective=objective)
        L0_field = _fields_from_complex_m2(torch.polar(obja, objp).unsqueeze(0), probe, H)
        s2 = mliss_sigma2(I_dat, cfg.counts_per_unit)
        L0 = float(mliss_loss(dp_from_fields(L0_field, occu), I_dat, None, s2))
        probe_new, diag = mliss_batch_update(
            obja, objp, probe, H, I_dat, None, occu, cfg, order=2,
        )
        F1 = _fields_from_complex_m2(torch.polar(obja, objp).unsqueeze(0), probe_new, H)
        L1 = float(mliss_loss(dp_from_fields(F1, occu), I_dat, None, s2))
        assert L1 < L0, f"[{objective}] L_G did not decrease: {L0:.6e} -> {L1:.6e}"

"""
Oracle tests for the polynomial line search at Born order M > 1
(docs/polyls_hybrid_order_plan.md §5).

The key gate: the degree-4M polynomial built from born_ray_coeffs +
poly_response_terms IS the Gaussian intensity loss along the ray g + a*d —
verified against a brute-force loss evaluation (full coeffs-aware forward at
the stepped object) at random step values, in float64, for M in {2, 3, 4},
including Born coefficients c != 1 and a linear detector postmap. The model
layer must not be unguarded until these pass (plan §4.3).

Also pinned here:
  - born_fields is the FIELD version of born_forward (production parity);
  - born_ray_coeffs reduces to direction_response at M = 1;
  - poly_line_search is bit-identical to line_search at M = 1;
  - descent + exact damped fallback at M = 3;
  - probe-step exactness at M > 1: F(a*) and u(a*) from the polynomial equal
    a fresh forward at the stepped object (the no-re-forward trick).

CPU + eager (torch.compile disabled), synthetic tensors only.
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import pytest
import torch

torch._dynamo.config.disable = True

import ptyrad.linesearch as ls
from ptyrad.forward_models.born_helpers import born_fields, born_fields_from_complex
from test.test_born_detector_gram import _setup

torch.manual_seed(0)


def _direction(B, omode, Nz, Ny, Nx, seed=3, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    d = torch.view_as_complex(
        torch.randn(B, omode, Nz, Ny, Nx, 2, generator=g, dtype=torch.float64)
    )
    return scale * d


def _coeffs(M, seed=4):
    """Random complex Born coefficients near 1 (the refit regime)."""
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(M, 2, generator=g, dtype=torch.float64) * 0.35
    c[:, 0] += 1.0
    return torch.complex(c[:, 0], c[:, 1])


def _postmap(x):
    """A LINEAR intensity postmap (stand-in for the detector blur — linearity
    is the property the quartic/polynomial exactness rests on)."""
    return x + 0.4 * torch.roll(x, shifts=(1, 2), dims=(-2, -1)) - 0.1 * x.flip(-1)


def _data(patches, probe, H, occu, M, seed=9):
    """Positive measured DP from a DIFFERENT object so residuals are nonzero."""
    true_patches, _, _, _ = _setup(
        patches.shape[0], patches.shape[1], patches.shape[2],
        patches.shape[3], patches.shape[4], probe.shape[1], probe.shape[0],
        phase_scale=0.5, seed=seed,
    )
    F = born_fields(true_patches, probe, H, M)
    return ls.dp_from_fields(F, occu).clamp_min(0.0)


# --------------------------------------------------------------------------- #
# 1. born_fields — production parity and the M = 1 reduction                   #
# --------------------------------------------------------------------------- #


def test_born_fields_matches_born_forward():
    """The §4.2 unit map applied to born_fields must reproduce the production
    born_forward exactly (same recursion, stopped before |.|^2), with and
    without coefficients."""
    from ptyrad.forward_models import born_forward

    patches, probe, H, occu = _setup(2, 2, 5, 16, 16, 2, 1, phase_scale=0.8)
    patches32 = patches.float()
    probe32 = probe.to(torch.complex64)
    H32 = H.to(torch.complex64)
    occu32 = occu.float()
    for n_max, coeffs in [(1, None), (3, None), (4, torch.stack(
        [_coeffs(4).real.float(), _coeffs(4).imag.float()], dim=-1))]:
        dp_prod = born_forward(patches32, probe32, H32, occu32, n_max=n_max, coeffs=coeffs)
        F = born_fields(patches32, probe32, H32, n_max, coeffs)
        dp_ours = ls.dp_from_fields(F, occu32)
        scale = dp_prod.abs().max()
        assert torch.allclose(dp_ours, dp_prod, rtol=1e-5, atol=1e-6 * scale), (
            f"born_fields diverges from born_forward at n_max={n_max}, "
            f"coeffs={'set' if coeffs is not None else 'None'}"
        )


def test_born_fields_equals_iss_fields_at_order1():
    patches, probe, H, occu = _setup(2, 1, 4, 16, 16, 2, 1)
    F1 = born_fields(patches, probe, H, 1)
    F_iss = ls.iss_fields(patches, probe, H)
    assert torch.allclose(F1, F_iss, rtol=1e-11, atol=1e-13 * F_iss.abs().max())


def test_ray_coeffs_reduce_to_direction_response_at_M1():
    """born_ray_coeffs at M = 1: F_0 is the base field and F_1 equals the
    affine direction response F(g + d) - F(g) to float64 precision."""
    patches, probe, H, occu = _setup(2, 1, 4, 16, 16, 2, 1)
    d = _direction(2, 1, 4, 16, 16, scale=0.05)
    F0 = born_fields(patches, probe, H, 1)
    F_stack = ls.born_ray_coeffs(patches, d, probe, H, coeffs=None, M=1, F0=F0)
    assert F_stack.shape[0] == 2
    D = ls.direction_response(patches, d, probe, H)
    s = D.abs().max()
    assert torch.allclose(F_stack[0], F0, rtol=1e-12, atol=1e-13 * s)
    assert torch.allclose(F_stack[1], D, rtol=1e-9, atol=1e-11 * s)


# --------------------------------------------------------------------------- #
# 2. The polynomial oracle — Q(a) IS the loss along the ray (the gate)         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("M", [2, 3, 4])
@pytest.mark.parametrize("use_coeffs,use_postmap", [(False, False), (True, False), (True, True)])
def test_poly_oracle_matches_brute_force(M, use_coeffs, use_postmap):
    """Q(a) from poly_response_terms + the coefficient sums matches the
    brute-force loss (full coeffs-aware forward at g + a*d) at random steps,
    rtol ~1e-9, float64. Includes Born coeffs c != 1 and a linear postmap
    threaded identically through u and the U_s."""
    B, omode, Nz, Ny, Nx, pmode = 2, 1, 6, 16, 16, 2
    patches, probe, H, occu = _setup(B, omode, Nz, Ny, Nx, pmode, 1, phase_scale=0.7)
    coeffs = _coeffs(M) if use_coeffs else None
    postmap = _postmap if use_postmap else (lambda x: x)
    d = _direction(B, omode, Nz, Ny, Nx, scale=0.06)
    I_dat = _data(patches, probe, H, occu, M)
    omega = 1.0 / (I_dat + 1.0)

    O = torch.polar(patches[..., 0], patches[..., 1])
    F0 = born_fields_from_complex(O, probe, H, M, coeffs)
    u0 = postmap(ls.dp_from_fields(F0, occu))
    e = u0 - I_dat

    F_stack = ls.born_ray_coeffs(patches, d, probe, H, coeffs=coeffs, M=M, F0=F0)
    U_list = [postmap(u) for u in ls.poly_response_terms(F_stack, occu)]
    assert len(U_list) == 2 * M

    gk = ls._poly_dq_terms(e, U_list, omega).tolist()
    Q0 = float((omega * e * e).sum())

    rng = np.random.default_rng(11)
    steps = list(0.7 * rng.standard_normal(5)) + [1.5, -1.2]
    for a in steps:
        # brute force: full forward at the stepped object
        u_a = postmap(ls.dp_from_fields(
            born_fields_from_complex(O + a * d, probe, H, M, coeffs), occu
        ))
        q_brute = float((omega * (u_a - I_dat).square()).sum())
        # coefficient form
        q_coeff = float(ls._poly_q_at(
            e, U_list, omega, torch.tensor([a], dtype=torch.float64)
        )[0])
        # Taylor form from the dQ/da coefficients: Q(a) = Q(0) + sum g_k a^{k+1}/(k+1)
        q_taylor = Q0 + sum(g * a ** (k + 1) / (k + 1) for k, g in enumerate(gk))
        assert q_coeff == pytest.approx(q_brute, rel=1e-9), f"coeff vs brute at a={a}"
        assert q_taylor == pytest.approx(q_brute, rel=1e-9), f"taylor vs brute at a={a}"


# --------------------------------------------------------------------------- #
# 3. M = 1 regression — bit identity with the quartic path                     #
# --------------------------------------------------------------------------- #


def test_poly_line_search_bit_identical_at_M1():
    """poly_line_search given [U_1, U_2] = [2v, w] must return the SAME float
    as line_search(e, v, w, ...) — exact equality, not approx (the delegation
    plus the exact power-of-two scaling guarantee it)."""
    B, omode, Nz, Ny, Nx, pmode = 2, 1, 4, 16, 16, 2
    patches, probe, H, occu = _setup(B, omode, Nz, Ny, Nx, pmode, 1)
    d = _direction(B, omode, Nz, Ny, Nx, scale=0.05)
    I_dat = _data(patches, probe, H, occu, 1)
    omega = 1.0 / (I_dat + 1.0)

    F = born_fields(patches, probe, H, 1)
    D = ls.direction_response(patches, d, probe, H)
    v, w = ls.response_terms(F, D, occu)
    e = ls.dp_from_fields(F, occu) - I_dat

    log_q, log_p = [], []
    a_quartic = ls.line_search(e, v, w, omega, fallback=1.0 / Nz, step_log=log_q)
    a_poly = ls.poly_line_search(e, [2.0 * v, w], omega, fallback=1.0 / Nz, step_log=log_p)
    assert a_poly == a_quartic  # bit-identical
    assert log_p == log_q

    # degenerate inputs delegate to the same damped fallback
    zero = torch.zeros_like(v)
    assert ls.poly_line_search(e, [zero, zero], omega, fallback=1.0 / Nz) == 0.5 / Nz


def test_generic_engine_agrees_with_quartic_at_M1():
    """The GENERIC degree-4M machinery (bypassing the M = 1 delegation) must
    agree with the quartic coefficients up to the exact factor 4 and produce
    the same selected root to near machine precision."""
    B, omode, Nz, Ny, Nx, pmode = 2, 1, 4, 16, 16, 2
    patches, probe, H, occu = _setup(B, omode, Nz, Ny, Nx, pmode, 1)
    d = _direction(B, omode, Nz, Ny, Nx, scale=0.05)
    I_dat = _data(patches, probe, H, occu, 1)
    omega = 1.0 / (I_dat + 1.0)

    F = born_fields(patches, probe, H, 1)
    D = ls.direction_response(patches, d, probe, H)
    v, w = ls.response_terms(F, D, occu)
    e = ls.dp_from_fields(F, occu) - I_dat

    gk = ls._poly_dq_terms(e, [2.0 * v, w], omega).tolist()
    c = ls.quartic_coeffs(e, v, w, omega)
    for k in range(4):
        assert gk[k] == pytest.approx(4.0 * c[k], rel=1e-12)

    roots_gen = sorted(ls._real_roots_ascending(gk))
    roots_qua = sorted(ls._real_cubic_roots(*c))
    assert len(roots_gen) == len(roots_qua)
    for rg, rq in zip(roots_gen, roots_qua, strict=True):
        assert rg == pytest.approx(rq, rel=1e-10)


# --------------------------------------------------------------------------- #
# 4. Descent sanity and fallback at M = 3                                      #
# --------------------------------------------------------------------------- #


def test_descent_and_fallback_at_M3():
    M = 3
    B, omode, Nz, Ny, Nx, pmode = 1, 1, 6, 16, 16, 2
    patches, probe, H, occu = _setup(B, omode, Nz, Ny, Nx, pmode, 1, phase_scale=0.7)
    coeffs = _coeffs(M)
    I_dat = _data(patches, probe, H, occu, M)
    omega = 1.0 / (I_dat + 1.0)
    O = torch.polar(patches[..., 0], patches[..., 1])

    # a genuine descent direction: preconditioned negative gradient of the
    # intensity objective through the coeffs-aware order-M forward
    O_leaf = O.detach().clone().requires_grad_(True)
    F = born_fields_from_complex(O_leaf, probe, H, M, coeffs)
    u = ls.dp_from_fields(F, occu)
    L = (omega * (u - I_dat).square()).sum()
    L.backward()
    d = -O_leaf.grad

    F0 = F.detach()
    e = u.detach() - I_dat
    F_stack = ls.born_ray_coeffs(patches, d, probe, H, coeffs=coeffs, M=M, F0=F0)
    U_list = ls.poly_response_terms(F_stack, occu)
    a = ls.poly_line_search(e, U_list, omega, fallback=1.0 / Nz, ls_damp=0.5)
    assert a != 0.5 / Nz  # solver live, not fallback

    u_a = ls.dp_from_fields(
        born_fields_from_complex(O + a * d, probe, H, M, coeffs), occu
    )
    Q0 = float((omega * e * e).sum())
    Qa = float((omega * (u_a - I_dat).square()).sum())
    assert Qa < Q0, f"accepted step does not descend: {Q0:.6e} -> {Qa:.6e}"

    # zero direction -> exact damped fallback (degenerate polynomial)
    F_stack0 = ls.born_ray_coeffs(patches, torch.zeros_like(d), probe, H,
                                  coeffs=coeffs, M=M, F0=F0)
    U0 = ls.poly_response_terms(F_stack0, occu)
    a0 = ls.poly_line_search(e, U0, omega, fallback=1.0 / Nz, ls_damp=0.5)
    assert a0 == 0.5 / Nz


# --------------------------------------------------------------------------- #
# 5. Probe-step exactness at M > 1 — the no-re-forward trick                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("M", [2, 3])
def test_probe_step_exactness(M):
    """F(a*) = sum a*^k F_k and u(a*) from the polynomial equal a fresh
    forward at the stepped object, float64 — the precondition for running the
    (still quartic) probe step against the updated field without re-forward."""
    B, omode, Nz, Ny, Nx, pmode = 1, 1, 5, 16, 16, 2
    patches, probe, H, occu = _setup(B, omode, Nz, Ny, Nx, pmode, 1, phase_scale=0.6)
    coeffs = _coeffs(M)
    d = _direction(B, omode, Nz, Ny, Nx, scale=0.05)
    I_dat = _data(patches, probe, H, occu, M)
    omega = 1.0 / (I_dat + 1.0)
    O = torch.polar(patches[..., 0], patches[..., 1])

    F0 = born_fields_from_complex(O, probe, H, M, coeffs)
    u0 = ls.dp_from_fields(F0, occu)
    e = u0 - I_dat
    F_stack = ls.born_ray_coeffs(patches, d, probe, H, coeffs=coeffs, M=M, F0=F0)
    U_list = ls.poly_response_terms(F_stack, occu)
    a = ls.poly_line_search(e, U_list, omega, fallback=1.0 / Nz, ls_damp=0.5)

    F_pred = ls.ray_field_at(F_stack, a)
    u_pred = ls.ray_intensity_at(u0, U_list, a)
    F_fwd = born_fields_from_complex(O + a * d, probe, H, M, coeffs)
    u_fwd = ls.dp_from_fields(F_fwd, occu)
    assert torch.allclose(F_pred, F_fwd, rtol=1e-9, atol=1e-10 * F_fwd.abs().max())
    assert torch.allclose(u_pred, u_fwd, rtol=1e-9, atol=1e-11 * u_fwd.abs().max())

    # and the probe response at the updated object: linear in P at EVERY order
    q = 0.05 * probe.roll(1, dims=-1)
    O2 = O + a * d
    D_P = born_fields_from_complex(O2, q, H, M, coeffs)
    diff = (
        born_fields_from_complex(O2, probe + q, H, M, coeffs)
        - born_fields_from_complex(O2, probe, H, M, coeffs)
    )
    assert torch.allclose(D_P, diff, rtol=1e-9, atol=1e-11 * F_fwd.abs().max())

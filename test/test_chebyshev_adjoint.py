"""
Gradient correctness of the analytic Chebyshev adjoint (no-tape) against
autograd, and sanity of the ePIE-preconditioned update.

The forward in chebyshev_adjoint.py is pure differentiable torch, so
autograd through IT is the reference for the exact-VJP test; a separate
test pins its exit intensities to the production multislice_forward_chebyshev.
"""

import numpy as np
import pytest
import torch

from ptyrad.forward_models.chebyshev import chebyshev_setup, multislice_forward_chebyshev
from ptyrad.forward_models.chebyshev_adjoint import (
    _cheb_epie,
    chebyshev_forward_ckpt,
    chebyshev_vjp,
)
from ptyrad.utils import fftshift2

NPIX = 32
DX = 0.2
LAMBD = 0.0197  # 300 kV electrons, Angstrom
DZ_SLICE = 8.0
NZ = 3  # chebyshev needs 3 slices minimum in production configs


def _problem(dtype, n=2, pmode=2, omode=2, seed=0):
    torch.manual_seed(seed)
    rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
    # smooth random object: small phase, mild absorption
    def smooth(shape):
        x = torch.randn(*shape, NPIX // 4, NPIX // 4, dtype=torch.float64)
        x = torch.nn.functional.interpolate(
            x.reshape(-1, 1, NPIX // 4, NPIX // 4), size=(NPIX, NPIX), mode="bicubic"
        ).reshape(*shape, NPIX, NPIX)
        return x

    phase = 0.3 * smooth((n, omode, NZ))
    amp = (1.0 - 0.05 * smooth((n, omode, NZ)).abs()).clamp(0.7, 1.0)
    obj = torch.stack([amp, phase], dim=-1).to(rdtype)

    yy, xx = torch.meshgrid(
        torch.arange(NPIX) - NPIX / 2, torch.arange(NPIX) - NPIX / 2, indexing="ij"
    )
    r2 = (yy**2 + xx**2).double()
    probe = torch.stack(
        [torch.exp(-r2 / (2 * (3.0 + 2 * m) ** 2)) * torch.exp(1j * 0.1 * m * r2 / NPIX)
         for m in range(pmode)]
    )[None].to(dtype)
    return obj, probe


def _kernel(dtype, n_slab):
    gs = NZ // n_slab
    groups = tuple((i * gs, (i + 1) * gs) for i in range(n_slab))
    kernel = chebyshev_setup(
        (NPIX, NPIX), dx=DX, lambd=LAMBD, dz_slab=gs * DZ_SLICE,
        phase_cap=(-1.5, 1.5), amp_cap=(0.6, 1.2), tol=1e-8,
        dtype=dtype, verbose=False,
    )
    return kernel, groups


@pytest.mark.parametrize("n_slab", [3, 1])
def test_exact_vjp_vs_autograd_f64(n_slab):
    dtype = torch.complex128
    obj, probe = _problem(dtype)
    kernel, groups = _kernel(dtype, n_slab)
    occu = torch.full((obj.shape[1],), 1.0 / obj.shape[1], dtype=torch.float64)

    obj_ad = obj.clone().requires_grad_(True)
    probe_r = torch.view_as_real(probe).clone().requires_grad_(True)

    T, *_ = chebyshev_forward_ckpt(obj_ad, torch.view_as_complex(probe_r), *kernel, groups)
    # generic real loss through an arbitrary fixed cotangent AND the
    # intensity map: L = |T - T_t|^2 exercises a dense, structured lam
    T_t = T.detach() * torch.exp(
        1j * 0.1 * torch.randn_like(T.real)
    )
    L = (T - T_t).abs().square().sum()
    L.backward()

    lam = (T - T_t).detach()  # dL = 2 Re <lam, dT>
    ga, gp, gprobe = chebyshev_vjp(obj, probe, *kernel, groups, lam)

    for got, want, name in [
        (ga, obj_ad.grad[..., 0], "obja"),
        (gp, obj_ad.grad[..., 1], "objp"),
        (torch.view_as_real(2.0 * gprobe), probe_r.grad, "probe"),
    ]:
        rel = (got - want).norm() / want.norm()
        assert rel < 1e-11, f"{name}: rel err {rel:.3e}"


def test_exact_vjp_through_intensities_f64():
    """Full-chain check against autograd through the PRODUCTION forward:
    L = sum(w * dp) with dp from multislice_forward_chebyshev."""
    dtype = torch.complex128
    obj, probe = _problem(dtype)
    kernel, groups = _kernel(dtype, 3)
    omode = obj.shape[1]
    occu = torch.full((omode,), 1.0 / omode, dtype=torch.float64)
    w = torch.rand(obj.shape[0], NPIX, NPIX, dtype=torch.float64)

    obj_ad = obj.clone().requires_grad_(True)
    probe_r = torch.view_as_real(probe).clone().requires_grad_(True)
    dp = multislice_forward_chebyshev(
        obj_ad, torch.view_as_complex(probe_r), *kernel, groups, omode_occu=occu
    )
    (dp * w).sum().backward()

    # analytic: lam = d/d(Tbar) of sum(w*dp) = unshift(w)*occu*scale*T
    T, eps_ts, nm1s, ckpts = chebyshev_forward_ckpt(obj, probe, *kernel, groups)
    scale = 1.0 / (NPIX * NPIX)
    lam = fftshift2(w)[:, None, None] * occu.view(1, 1, -1, 1, 1) * scale * T
    ga, gp, gprobe = chebyshev_vjp(
        obj, probe, *kernel, groups, lam, forward_state=(eps_ts, nm1s, ckpts)
    )

    for got, want, name in [
        (ga, obj_ad.grad[..., 0], "obja"),
        (gp, obj_ad.grad[..., 1], "objp"),
        (torch.view_as_real(2.0 * gprobe), probe_r.grad, "probe"),
    ]:
        rel = (got - want).norm() / want.norm()
        assert rel < 1e-11, f"{name}: rel err {rel:.3e}"


def test_vjp_f32_matches_f64_reference():
    """float32 run (production dtype): the reverse reconstruction shares the
    forward's growth modes, so the gradient should track the f64 reference
    to float32 roundoff levels."""
    obj64, probe64 = _problem(torch.complex128)
    k64, groups = _kernel(torch.complex128, 3)
    T64, *_ = chebyshev_forward_ckpt(obj64, probe64, *k64, groups)
    lam64 = T64 * 0.1

    ga64, gp64, gpr64 = chebyshev_vjp(obj64, probe64, *k64, groups, lam64)

    k32, _ = _kernel(torch.complex64, 3)
    obj32, probe32 = obj64.float(), probe64.to(torch.complex64)
    T32, *_ = chebyshev_forward_ckpt(obj32, probe32, *k32, groups)
    ga32, gp32, gpr32 = chebyshev_vjp(obj32, probe32, *k32, groups, T32 * 0.1)

    for got, want, name, tol in [
        (ga32.double(), ga64, "obja", 2e-4),
        (gp32.double(), gp64, "objp", 2e-4),
        (gpr32.to(torch.complex128), gpr64, "probe", 2e-4),
    ]:
        rel = (got - want).norm() / want.norm()
        assert rel < tol, f"{name}: rel err {rel:.3e}"


def test_forward_matches_production_dp():
    dtype = torch.complex128
    obj, probe = _problem(dtype)
    kernel, groups = _kernel(dtype, 3)
    omode = obj.shape[1]
    occu = torch.full((omode,), 1.0 / omode, dtype=torch.float64)
    dp_prod = multislice_forward_chebyshev(obj, probe, *kernel, groups, omode_occu=occu)
    T, *_ = chebyshev_forward_ckpt(obj, probe, *kernel, groups)
    scale = 1.0 / (NPIX * NPIX)
    dp = torch.sum(
        fftshift2(T).abs().square() * occu[:, None, None], dim=(1, 2)
    ) * scale + 1e-10
    rel = (dp - dp_prod).norm() / dp_prod.norm()
    assert rel < 1e-13, f"dp mismatch {rel:.3e}"


def test_epie_update_reduces_data_error():
    """Noiseless self-consistent problem: repeated updates at one position
    must drive the modulus error down."""
    dtype = torch.complex64
    obj, probe = _problem(dtype, n=1, pmode=1, omode=1, seed=3)
    kernel, groups = _kernel(dtype, 3)
    occu = torch.ones(1)

    # measurements from the true object
    T, *_ = chebyshev_forward_ckpt(obj, probe, *kernel, groups)
    scale = 1.0 / (NPIX * NPIX)
    meas = fftshift2((T.abs().square() * occu.view(1, 1, -1, 1, 1)).sum((1, 2))) * scale

    # perturbed start: phases damped
    cur = obj.clone()
    cur[..., 1] *= 0.5

    def data_err(o):
        Tc, *_ = chebyshev_forward_ckpt(o, probe, *kernel, groups)
        dpc = fftshift2((Tc.abs().square() * occu.view(1, 1, -1, 1, 1)).sum((1, 2))) * scale
        return ((dpc.sqrt() - meas.sqrt()) ** 2).sum().item()

    e0 = data_err(cur)
    for _ in range(100):
        dp_model, cur, _ = _cheb_epie(
            cur, probe, *kernel, groups, occu, meas,
            beta_obj=0.3, beta_probe=0.0, update_probe=False, eps=1e-10,
        )
    e1 = data_err(cur)
    assert e1 < 0.05 * e0, f"data error {e0:.3e} -> {e1:.3e}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

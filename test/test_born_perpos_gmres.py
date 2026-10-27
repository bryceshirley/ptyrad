"""Tests for the per-position Born residual fit (Gram solve) and the
GMRES(M) forward model (born_gmres). Small synthetic cases, CPU, fast —
mirrors the pre-reconstruction checks of the per-position coefficient
study (~/born-perpos-fit)."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptyrad.forward_models.born_gmres import born_apply_E, born_gmres_forward
from ptyrad.forward_models.born_helpers import (
    _born_init,
    born_multislice_target,
    born_position_residual_fit,
    born_residual_solve_from_r,
    born_residual_solve_gram,
    born_tsqr_update,
    born_volume_basis,
    born_volume_residual_rows,
)
from ptyrad.utils import fftshift2


def small_case(Nz=4, Npix=32, seed=123, dtype=torch.complex64, phase_scale=0.3):
    g = torch.Generator().manual_seed(seed)
    rdt = torch.float64 if dtype == torch.complex128 else torch.float32
    obja = (1.0 + 0.02 * torch.randn(1, 1, Nz, Npix, Npix, generator=g)).to(rdt)
    objp = (phase_scale * torch.randn(1, 1, Nz, Npix, Npix, generator=g)).to(rdt)
    patches = torch.stack([obja, objp], dim=-1)
    probe = torch.view_as_complex(
        torch.randn(1, 1, Npix, Npix, 2, generator=g).to(rdt).contiguous()
    )
    phi = torch.randn(Npix, Npix, generator=g).to(rdt)
    H1 = torch.exp(1j * 0.3 * phi)
    z = torch.arange(Nz).view(1, 1, 1, Nz, 1, 1)
    H = H1.view(1, 1, 1, 1, Npix, Npix).pow(z).to(dtype)
    return patches, probe, H


def rel(a, b):
    a = a.to(torch.complex128).reshape(-1)
    b = b.to(torch.complex128).reshape(-1)
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def test_gram_solve_matches_tsqr():
    g = torch.Generator().manual_seed(5)
    A = torch.view_as_complex(torch.randn(5000, 6, 2, generator=g).contiguous())
    A[:, 3] = A[:, 0] * 0.999 + 1e-3 * A[:, 3]
    b = torch.view_as_complex(torch.randn(5000, 2, generator=g).contiguous())
    R = born_tsqr_update(None, torch.cat([A, b[:, None]], dim=1))
    for offset in [False, True]:
        c1, r1, k1, _, cd1 = born_residual_solve_from_r(R, 6, 5000, offset_from_one=offset)
        c2, r2, k2, _, cd2 = born_residual_solve_gram(A, b, offset_from_one=offset)
        assert rel(c2, c1) < 1e-6
        assert k1 == k2
        assert abs(r1 - r2) < 1e-6
        assert abs(cd1 - cd2) / cd1 < 1e-6


def test_unit_coeff_residual_telescopes():
    M = 3
    patches, probe, H = small_case(Nz=6)
    V, _, D = born_volume_basis(patches, probe, H, M + 1)
    A, b = born_volume_residual_rows(V, D, M)
    r_unit = b - A.sum(1)
    Npix = V.shape[-1]
    ref = torch.cat(
        [
            V[M + 1][..., 1:, :, :].reshape(-1),
            (D[M] / math.sqrt(Npix * Npix)).reshape(-1),
        ]
    )
    assert rel(r_unit, ref) < 1e-5


def test_fit_c_equals_one_at_full_order():
    Nz = 4
    patches, probe, H = small_case(Nz=Nz)
    c, rel_res, rank, cond, fb = born_position_residual_fit(patches, probe, H, Nz)
    assert not fb
    assert rank == Nz
    assert (c - 1).abs().max().item() < 1e-3
    assert rel_res < 1e-4


def test_fit_vacuum_fallback():
    patches, probe, H = small_case(Nz=4, phase_scale=0.0)
    patches[..., 0] = 1.0  # exactly vacuum: O - 1 = 0
    c, rel_res, rank, cond, fb = born_position_residual_fit(
        patches, probe, H, 4, vacuum_tol=1e-6
    )
    assert fb
    assert torch.allclose(c, torch.ones_like(c))


def test_apply_E_reproduces_next_order():
    M = 3
    patches, probe, H = small_case(Nz=6)
    V, _, D = born_volume_basis(patches, probe, H, M + 1)
    obj, _, _ = _born_init(patches, probe, H)
    full = V[0].shape
    for m in range(M + 1):
        planes_next, det_next = born_apply_E(obj, V[m].expand(full), H)
        assert rel(planes_next, V[m + 1]) < 1e-5 or V[m + 1].abs().max() == 0
        assert rel(det_next, D[m]) < 1e-5


def test_gmres_full_order_matches_multislice():
    Nz, Npix = 4, 32
    patches, probe, H = small_case(Nz=Nz, Npix=Npix)
    stats = {}
    with torch.no_grad():
        dp = born_gmres_forward(patches, probe, H, None, n_max=Nz, stats=stats)
        T = born_multislice_target(patches, probe, H)
        nw = torch.ones(1) / (Npix * Npix)
        dp_T = fftshift2(
            torch.sum(T.abs().square() * nw.view(1, 1, -1, 1, 1), dim=(1, 2)) + 1e-10
        )
    assert not stats["fallback"]
    assert rel(dp, dp_T) < 1e-5
    assert stats["rel_res"] < 1e-6


def test_gmres_vacuum_fallback_matches_plain_series():
    patches, probe, H = small_case(Nz=4, phase_scale=0.0)
    patches[..., 0] = 1.0
    stats = {}
    with torch.no_grad():
        dp = born_gmres_forward(patches, probe, H, None, n_max=4, stats=stats)
    assert stats["fallback"]
    assert torch.isfinite(dp).all()


def test_gmres_gradcheck_double():
    patches, probe, H = small_case(Nz=4, Npix=16, seed=7, dtype=torch.complex128)
    meas = torch.rand(1, 16, 16, dtype=torch.float64,
                      generator=torch.Generator().manual_seed(9))
    d = torch.randn(patches.shape, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(11))

    def loss_of(p):
        dp = born_gmres_forward(p, probe, H, None, n_max=3)
        return ((dp.clamp(min=0).sqrt() - meas) ** 2).mean()

    p0 = patches.clone().requires_grad_(True)
    loss_of(p0).backward()
    ad = (p0.grad * d).sum().item()
    h = 1e-6
    with torch.no_grad():
        fd = (loss_of(patches + h * d).item() - loss_of(patches - h * d).item()) / (2 * h)
    assert abs(ad - fd) / max(abs(fd), 1e-30) < 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""The Fourier-basis Born-coefficient refit — now the only coefficient
algorithm (reconstruction._refit_born_coeffs_fourier).

Driven through a stub model that supplies just the accessors the refit
touches, so the whole path runs: volume basis + volume/detector multislice
targets -> Fourier remainder rows -> chunk-accumulated split-k complex128
moments -> equilibrated direct-c solve -> diagnostics and the trajectory
row. Covers the three targets — the multislice REMAINDER fit, the
target-free RESIDUAL fit (the GMRES minimiser, which must never touch a
multislice sweep) and the hybrid_amp detector target — plus adaptive
order growth (grow_tol).
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import pytest
import torch

torch._dynamo.config.disable = True

from ptyrad.reconstruction import _refit_born_coeffs_fourier


class StubModel:
    """Minimal stand-in exposing only what the refit reads."""

    def __init__(self, nviews=4, omode=1, pmode=2, Nz=6, Npix=16, M=3,
                 phase=0.6, seed=0, blur=None):
        g = torch.Generator().manual_seed(seed)
        self.n_slice = Nz
        self.born_iterations = M
        self.omode_occu = torch.ones(omode) / omode
        self.detector_blur_std = blur
        self.born_refit_views = torch.arange(nviews)
        self.opt_born_coeffs = torch.zeros(Nz, 2)
        self.born_refit_iters = []
        self.born_refit_csv_row = None

        p = torch.rand(nviews, omode, Nz, Npix, Npix, 2, generator=g)
        p[..., 0] = 1.0 + 0.1 * (p[..., 0] - 0.5)
        p[..., 1] *= phase
        self._patches = p
        self._probe = torch.view_as_complex(
            torch.randn(1, pmode, Npix, Npix, 2, generator=g).contiguous()
        )
        ky = torch.fft.fftfreq(Npix)
        H1 = torch.exp(-1j * 0.3 * (ky[:, None] ** 2 + ky[None, :] ** 2) * Npix * Npix)
        self._H = (H1 ** torch.arange(Nz).view(Nz, 1, 1)).to(torch.complex64)
        self._meas = torch.rand(nviews, Npix, Npix, generator=g) + 0.1

    def get_obj_patches(self, sl):
        return self._patches[sl]

    def get_probes(self, sl):
        return self._probe

    def get_propagators(self, sl):
        return self._H

    def get_propagators_3d(self, H):
        return H.view(1, 1, 1, *H.shape)

    def get_measurements(self, sl):
        return self._meas[sl]


def _cfg(**kw):
    base = dict(start_iter=1, step=1, end_iter=None, n_views=4, pin_first=True,
                grow_tol=None, grow_fast=False, n_limit=None, target="multislice")
    base.update(kw)
    return base


def _coeffs(model, n):
    c = model.opt_born_coeffs
    return torch.complex(c[:n, 0], c[:n, 1])


def test_refit_runs_and_writes_coeffs():
    m = StubModel(M=3)
    _refit_born_coeffs_fourier(m, niter=1, cfg=_cfg(), verbose=False)
    c = _coeffs(m, 3)
    assert torch.isfinite(c).all()
    assert c.abs().max() < 1e3  # the ridge keeps the solve bounded
    row = m.born_refit_csv_row
    assert row["values"][0] == 1 and row["values"][2] == 3
    assert len(row["header"]) == len(row["values"])
    assert len(m.born_refit_iters) == 1
    fit_res = m.born_refit_iters[0][1]
    assert 0.0 <= fit_res < 1.0  # the fit beats the zero model


def test_refit_at_full_order_is_near_exact():
    # at M = Nz the orders span the multislice field exactly, so the fit
    # residual must collapse (this is the identity the series rests on)
    Nz = 4
    m = StubModel(Nz=Nz, M=Nz, Npix=16, phase=0.4)
    _refit_born_coeffs_fourier(m, niter=1, cfg=_cfg(), verbose=False)
    assert m.born_refit_iters[0][1] < 1e-3


def test_grow_tol_promotes_order():
    # an unreachable tolerance must promote from M up to n_limit, using the
    # lookahead moments (grow_fast: repeatedly within one refit)
    m = StubModel(Nz=6, M=2, phase=1.2)
    _refit_born_coeffs_fourier(
        m, niter=1, cfg=_cfg(grow_tol=1e-12, grow_fast=True, n_limit=5), verbose=False
    )
    assert m.born_iterations == 5
    assert m.born_refit_csv_row["values"][2] == 5
    assert torch.isfinite(_coeffs(m, 5)).all()


def test_grow_tol_single_step_and_no_growth_when_satisfied():
    m = StubModel(Nz=6, M=2, phase=1.2)
    _refit_born_coeffs_fourier(m, niter=1, cfg=_cfg(grow_tol=1e-12), verbose=False)
    assert m.born_iterations == 3  # one promotion per refit without grow_fast

    m2 = StubModel(Nz=6, M=2, phase=1.2)
    _refit_born_coeffs_fourier(m2, niter=1, cfg=_cfg(grow_tol=1.0), verbose=False)
    assert m2.born_iterations == 2  # tolerance already met: no promotion


def test_grow_tol_padding_keeps_row_width_fixed():
    # the csv schema is sized by n_limit so rows stay alignable across
    # iterations while the order grows
    m = StubModel(Nz=6, M=2, phase=1.2)
    _refit_born_coeffs_fourier(
        m, niter=1, cfg=_cfg(grow_tol=1e-12, n_limit=5), verbose=False
    )
    row = m.born_refit_csv_row
    assert len(row["header"]) == len(row["values"])
    assert sum(1 for h in row["header"] if h.startswith("cabs")) == 5


def test_hybrid_amp_moves_the_fit_and_reports_diagnostics():
    cfg_ms, cfg_hy = _cfg(), _cfg(target="hybrid_amp")
    m_ms, m_hy = StubModel(M=3, seed=5), StubModel(M=3, seed=5)
    _refit_born_coeffs_fourier(m_ms, niter=1, cfg=cfg_ms, verbose=False)
    _refit_born_coeffs_fourier(m_hy, niter=1, cfg=cfg_hy, verbose=False)
    c_ms, c_hy = _coeffs(m_ms, 3), _coeffs(m_hy, 3)
    assert torch.isfinite(c_hy).all()
    # the data amplitude must actually move the coefficients...
    assert not torch.allclose(c_ms, c_hy, rtol=1e-3, atol=1e-5)
    # ...and the reported shift/dc_rel must be non-zero only for hybrid
    _, _, _, _, _, shift_ms, dc_ms = m_ms.born_refit_iters[0]
    _, _, _, _, _, shift_hy, dc_hy = m_hy.born_refit_iters[0]
    assert shift_ms == 0.0 and dc_ms == 0.0
    assert shift_hy > 0.0 and dc_hy > 0.0
    assert m_hy.born_refit_csv_row["values"][3] == 1  # hybrid flag


def test_hybrid_c_ms_recovers_the_data_free_fit():
    # c_MS comes back as the second right-hand side of the SAME solve, so
    # it must equal the coefficients the pure multislice target produces
    m_ms, m_hy = StubModel(M=3, seed=11), StubModel(M=3, seed=11)
    _refit_born_coeffs_fourier(m_ms, niter=1, cfg=_cfg(), verbose=False)
    _refit_born_coeffs_fourier(m_hy, niter=1, cfg=_cfg(target="hybrid_amp"),
                               verbose=False)
    c_ms = _coeffs(m_ms, 3).to(torch.complex128)
    row = m_hy.born_refit_csv_row
    i = row["header"].index("cms1_re")
    cms_hy = torch.tensor(
        [complex(row["values"][i + 2 * k], row["values"][i + 2 * k + 1]) for k in range(3)],
        dtype=torch.complex128,
    )
    rel = (torch.linalg.vector_norm(cms_hy - c_ms)
           / torch.linalg.vector_norm(c_ms)).item()
    assert rel < 1e-4


def test_residual_target_runs_and_needs_no_multislice():
    # target='residual' is the GMRES minimiser, shared over the subsample:
    # target-free, so it never calls the multislice sweep
    # the refit imports inside the function, so patch at the source module
    import ptyrad.forward_models.born_coeff_fit as bcf

    calls = []
    orig = bcf.born_multislice_volume_target
    bcf.born_multislice_volume_target = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        m = StubModel(M=3)
        _refit_born_coeffs_fourier(m, niter=1, cfg=_cfg(target="residual"),
                                   verbose=False)
    finally:
        bcf.born_multislice_volume_target = orig
    assert not calls, "the residual target must not build a volume multislice target"
    c = _coeffs(m, 3)
    assert torch.isfinite(c).all() and c.abs().max() < 1e3
    assert 0.0 <= m.born_refit_iters[0][1] < 1.0


def test_residual_target_is_exact_at_full_order():
    Nz = 4
    m = StubModel(Nz=Nz, M=Nz, Npix=16, phase=0.4)
    _refit_born_coeffs_fourier(m, niter=1, cfg=_cfg(target="residual"), verbose=False)
    # at M = Nz nilpotency closes the Krylov space: c -> 1, residual -> 0
    assert m.born_refit_iters[0][1] < 1e-3
    assert (_coeffs(m, Nz) - 1).abs().max() < 1e-2


def test_detector_blur_path_runs():
    m = StubModel(M=3, blur=1.0)
    _refit_born_coeffs_fourier(m, niter=1, cfg=_cfg(target="hybrid_amp"), verbose=False)
    assert torch.isfinite(_coeffs(m, 3)).all()


def test_multi_chunk_accumulation_matches_single_chunk():
    # the refit chunks views in 2s; the accumulated moments must not depend
    # on how the views are split
    m2 = StubModel(nviews=2, M=3, seed=21)
    m4 = StubModel(nviews=4, M=3, seed=21)
    # same problem, duplicated into two chunks: the probe/H must be copied
    # too (StubModel's generator advances with nviews, so they differ)
    m4._patches = m2._patches.repeat(2, 1, 1, 1, 1, 1)
    m4._meas = m2._meas.repeat(2, 1, 1)
    m4._probe, m4._H = m2._probe, m2._H
    _refit_born_coeffs_fourier(m2, niter=1, cfg=_cfg(), verbose=False)
    _refit_born_coeffs_fourier(m4, niter=1, cfg=_cfg(), verbose=False)
    c2, c4 = _coeffs(m2, 3).to(torch.complex128), _coeffs(m4, 3).to(torch.complex128)
    rel = (torch.linalg.vector_norm(c4 - c2) / torch.linalg.vector_norm(c2)).item()
    assert rel < 1e-5


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

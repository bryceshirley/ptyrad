"""
Oracle tests for ML-MS (ptyrad.mlms):

- Polynomial/Horner check (N = 3 and N = 6, 2 probe modes, float64): the
  multislice L_G at ~20 gammas by explicit forward passes matches the Horner
  evaluation on the detector coefficient fields to round-off, and the
  bracket+Brent gamma is the minimiser to within tolerance.
- N = 1, zero defocus: ML-MS and ML-ISS give the same step and the same
  update to round-off-ish float32 tolerance (the models coincide there).
- Transform count: the logged transforms per position per mode equal the
  Table 2 multislice line: gradient 4N-2, sweep N^2, probe 2N-1,
  total N^2 + 6N - 3.

CPU + eager, synthetic tensors only.
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import pytest
import torch

torch._dynamo.config.disable = True

import ptyrad.mliss as ml
import ptyrad.mlms as mm
from ptyrad.linesearch import dp_from_fields
from ptyrad.models import PtychoAD

torch.manual_seed(0)

B, PMODE, OMODE, NY, NX = 3, 2, 1, 16, 16
C_PER_UNIT = 5.0e3


def _mk(NZ, seed=3):
    g = torch.Generator().manual_seed(seed)
    amp = 1.0 + 0.02 * torch.randn(B, OMODE, NZ, NY, NX, generator=g, dtype=torch.float64)
    phs = 0.3 * torch.rand(B, OMODE, NZ, NY, NX, generator=g, dtype=torch.float64)
    O = torch.polar(amp, phs)
    pr = torch.randn(1, PMODE, NY, NX, generator=g, dtype=torch.float64)
    pi = torch.randn(1, PMODE, NY, NX, generator=g, dtype=torch.float64)
    P = (pr + 1j * pi) / (NY * NX) ** 0.5
    ky = torch.fft.fftfreq(NY, dtype=torch.float64)
    kx = torch.fft.fftfreq(NX, dtype=torch.float64)
    chi = 0.7 * (ky[:, None] ** 2 + kx[None, :] ** 2) * (NY * NX) ** 0.5
    H2 = torch.exp(-1j * chi).view(1, NY, NX)
    occu = torch.ones(OMODE, dtype=torch.float64)
    O2 = O * torch.polar(
        torch.ones_like(amp), 0.05 * torch.randn(*amp.shape, generator=g, dtype=torch.float64)
    )
    I_dat = dp_from_fields(mm.ms_fields(O2, P, H2), occu).clamp_min(0.0)
    return O, P, H2, occu, I_dat


@pytest.mark.parametrize("NZ", [3, 6])
def test_horner_matches_explicit_forwards(NZ):
    O, P, H2, occu, I_dat = _mk(NZ)
    sigma2 = ml.mliss_sigma2(I_dat, C_PER_UNIT)
    omega = 1.0 / sigma2

    g = torch.Generator().manual_seed(11)
    d = (
        torch.randn(B, OMODE, NZ, NY, NX, generator=g, dtype=torch.float64)
        + 1j * torch.randn(B, OMODE, NZ, NY, NX, generator=g, dtype=torch.float64)
    ) * 0.02

    F, psi_inc = mm.ms_fields(O, P, H2, keep_incident=True)
    Fk = mm.ms_poly_coeffs(O, d, P, H2, psi_inc, F)
    assert len(Fk) == NZ + 1

    from torch.fft import ifftshift

    I_us = ifftshift(I_dat, dim=(-2, -1))
    om_us = ifftshift(omega, dim=(-2, -1))
    lg = mm._lg_of_gamma_factory(Fk, I_us, om_us, occu)

    for gam in torch.linspace(-1.2, 1.2, 21, dtype=torch.float64):
        L_fwd = ml.mliss_loss(
            dp_from_fields(mm.ms_fields(O + gam * d, P, H2), occu), I_dat, None, sigma2
        )
        L_h = lg(float(gam))
        assert abs(L_h - float(L_fwd)) <= 1e-10 * (abs(float(L_fwd)) + 1.0), (
            f"N={NZ} gamma={gam}: horner {L_h} vs forward {float(L_fwd)}"
        )

    # the brent/minimiser check needs a DESCENT direction (a random d can
    # have its minimum at gamma = 0): use the negative L_G gradient
    O_leaf = O.detach().clone().requires_grad_(True)
    Lg = ml.mliss_loss(dp_from_fields(mm.ms_fields(O_leaf, P, H2), occu), I_dat, None, sigma2)
    Lg.backward()
    d_desc = -O_leaf.grad
    d_desc = d_desc * (0.05 / d_desc.abs().max())
    F2, psi_inc2 = mm.ms_fields(O, P, H2, keep_incident=True)
    Fk2 = mm.ms_poly_coeffs(O, d_desc, P, H2, psi_inc2, F2)
    lg2 = mm._lg_of_gamma_factory(Fk2, I_us, om_us, occu)
    gmin, fmin, nev = mm.brent_min(lg2, step=0.25)
    assert gmin is not None and fmin < lg2(0.0)
    assert nev < 80
    # minimiser within tolerance of a dense grid
    dense = np.linspace(gmin - 0.5, gmin + 0.5, 4001)
    vals = [lg2(x) for x in dense]
    assert gmin == pytest.approx(float(dense[int(np.argmin(vals))]), abs=5e-4)


def _make_pair_models_N1():
    """Two PtychoAD models (born vs multislice) with N=1, H=1 (zero defocus),
    identical data/init."""
    NZ, NOY, NOX = 1, 32, 32
    CROP = np.array([[0, 0], [0, 8], [8, 8]], dtype=np.int32)
    g = torch.Generator().manual_seed(7)
    amp = 1.0 + 0.01 * torch.randn(OMODE, NZ, NOY, NOX, generator=g)
    phase = 0.3 * torch.rand(OMODE, NZ, NOY, NOX, generator=g)
    obj_true = torch.polar(amp, phase)
    ny = nx = 16
    pr = torch.randn(PMODE, ny, nx, generator=g)
    pi = torch.randn(PMODE, ny, nx, generator=g)
    probe = ((pr + 1j * pi) / (ny * nx) ** 0.5).to(torch.complex64)
    H1 = torch.ones(ny, nx, dtype=torch.complex64)
    occu = torch.ones(OMODE)
    meas = []
    for y0, x0 in CROP:
        win = obj_true[:, :, y0 : y0 + ny, x0 : x0 + nx]
        F = mm.ms_fields(win.unsqueeze(0).to(torch.complex64), probe.unsqueeze(0), H1.view(1, ny, nx))
        meas.append(dp_from_fields(F, occu)[0])
    meas = torch.stack(meas).clamp_min(0.0)

    def mk(solver):
        iv = {
            "obj": np.ones((OMODE, NZ, NOY, NOX), dtype=np.complex64),
            "obj_tilts": np.zeros((1, 2), dtype=np.float32),
            "slice_thickness": np.float32(1.0),
            "probe": probe.numpy(),
            "probe_pos_shifts": np.zeros((len(CROP), 2), dtype=np.float32),
            "omode_occu": occu.numpy(),
            "H": H1.numpy(),
            "measurements": meas.numpy(),
            "N_scan_slow": 1,
            "N_scan_fast": 3,
            "crop_pos": CROP,
            "dx": np.float32(0.1),
            "dk": np.float32(0.01),
            "lambd": np.float32(0.0025),
            "random_seed": None,
            "length_unit": "Ang",
            "scan_affine": None,
        }
        mp = {
            "detector_blur_std": None,
            "obj_preblur_std": None,
            "solver_type": solver,
            "born_iterations": 1,
            "update_params": {
                "obja": {"lr": 5e-4, "start_iter": 1},
                "objp": {"lr": 5e-4, "start_iter": 1},
                "obj_tilts": {"lr": 0},
                "slice_thickness": {"lr": 0},
                "probe": {"lr": 1e-4, "start_iter": 1},
                "probe_pos_shifts": {"lr": 0},
            },
            "optimizer_params": {"name": "Adam", "configs": {}},
        }
        return PtychoAD(iv, mp, device="cpu", verbose=False)

    return mk("born"), mk("multislice")


def test_N1_zero_defocus_matches_mliss():
    m_iss, m_ms = _make_pair_models_N1()
    cfg_i = ml.MLISSConfig(counts_per_unit=C_PER_UNIT)
    cfg_m = mm.MLMSConfig(counts_per_unit=C_PER_UNIT)
    st_i, st_m = ml.MLISSState(), mm.MLMSState()
    d1 = ml.mliss_model_update_batched(m_iss, [0, 1, 2], config=cfg_i, state=st_i)
    d2 = mm.mlms_model_update_batched(m_ms, [0, 1, 2], config=cfg_m, state=st_m)
    assert d1["a"] == pytest.approx(d2["a"], rel=2e-4)
    assert d1["b"] == pytest.approx(d2["b"], rel=2e-3)
    assert torch.allclose(m_iss.opt_obja.data, m_ms.opt_obja.data, atol=2e-6)
    assert torch.allclose(m_iss.opt_objp.data, m_ms.opt_objp.data, atol=2e-6)
    assert torch.allclose(m_iss.opt_probe.data, m_ms.opt_probe.data, atol=2e-5)


def test_transform_count_matches_table2():
    """Counted transforms per position per mode == Table 2 multislice line."""
    NZ = 4
    NOY = NOX = 40
    CROP = np.array([[0, 0], [8, 8]], dtype=np.int32)
    g = torch.Generator().manual_seed(9)
    ny = nx = 16
    pr = torch.randn(PMODE, ny, nx, generator=g)
    pi = torch.randn(PMODE, ny, nx, generator=g)
    probe = ((pr + 1j * pi) / (ny * nx) ** 0.5).to(torch.complex64)
    ky = torch.fft.fftfreq(ny)
    kx = torch.fft.fftfreq(nx)
    H1 = torch.exp(-1j * 0.3 * (ky[:, None] ** 2 + kx[None, :] ** 2) * ny).to(torch.complex64)
    occu = torch.ones(OMODE)
    meas = torch.rand(len(CROP), ny, nx).clamp_min(0.01)
    iv = {
        "obj": (np.ones((OMODE, NZ, NOY, NOX)) + 0j).astype(np.complex64),
        "obj_tilts": np.zeros((1, 2), dtype=np.float32),
        "slice_thickness": np.float32(1.0),
        "probe": probe.numpy(),
        "probe_pos_shifts": np.zeros((len(CROP), 2), dtype=np.float32),
        "omode_occu": occu.numpy(),
        "H": H1.numpy(),
        "measurements": meas.numpy(),
        "N_scan_slow": 1,
        "N_scan_fast": 2,
        "crop_pos": CROP,
        "dx": np.float32(0.1),
        "dk": np.float32(0.01),
        "lambd": np.float32(0.0025),
        "random_seed": None,
        "length_unit": "Ang",
        "scan_affine": None,
    }
    mp = {
        "detector_blur_std": None,
        "obj_preblur_std": None,
        "solver_type": "multislice",
        "born_iterations": 1,
        "update_params": {
            "obja": {"lr": 5e-4, "start_iter": 1},
            "objp": {"lr": 5e-4, "start_iter": 1},
            "obj_tilts": {"lr": 0},
            "slice_thickness": {"lr": 0},
            "probe": {"lr": 1e-4, "start_iter": 1},
            "probe_pos_shifts": {"lr": 0},
        },
        "optimizer_params": {"name": "Adam", "configs": {}},
    }
    model = PtychoAD(iv, mp, device="cpu", verbose=False)
    st = mm.MLMSState()
    mm.mlms_model_update_batched(
        model, [0, 1], config=mm.MLMSConfig(counts_per_unit=C_PER_UNIT), state=st
    )
    frames = len(CROP) * PMODE * OMODE  # per-position-per-mode divisor
    N = NZ
    assert st.transforms["gradient"] / frames == 4 * N - 2
    assert st.transforms["sweep"] / frames == N * N
    assert st.transforms["probe"] / frames == 2 * N - 1
    total = sum(st.transforms.values()) / frames
    assert total == N * N + 6 * N - 3  # Table 2, multislice gradient+line step

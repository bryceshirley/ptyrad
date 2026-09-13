"""
Model-layer wiring tests for linesearch_model_update on a real (minimal)
PtychoAD instance: window gather/scatter against the model's own crop grids,
forward consistency with model.forward, descent over view sweeps, and probe
write-back. CPU, eager, synthetic data — no reconstructions.
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import pytest
import torch

torch._dynamo.config.disable = True

from test.test_linesearch_oracle import make_probe

import ptyrad.linesearch as ls
from ptyrad.models import PtychoAD

OMODE, NZ, NY, NX, PMODE = 1, 3, 32, 32, 2
NOY, NOX = 64, 64  # canvas larger than the window so scatter is non-trivial
CROP_POS = np.array([[0, 0], [0, 16], [16, 0], [16, 16]], dtype=np.int32)


def _H1():
    """Unit-modulus single-step propagator (Ny, Nx), entrance plane at j=0."""
    ky = torch.fft.fftfreq(NY, dtype=torch.float64)
    kx = torch.fft.fftfreq(NX, dtype=torch.float64)
    chi = 0.31 * (ky[:, None] ** 2 + kx[None, :] ** 2) * (NY * NX) ** 0.5
    return torch.exp(-1j * chi).to(torch.complex64)


def _make_model():
    """Minimal born PtychoAD: data from a true object, model starts at vacuum."""
    g = torch.Generator().manual_seed(7)
    amp = 1.0 + 0.01 * torch.randn(OMODE, NZ, NOY, NOX, generator=g)
    phase = 0.3 * torch.rand(OMODE, NZ, NOY, NOX, generator=g)
    obj_true = torch.polar(amp, phase)

    probe = make_probe(1, PMODE, NY, NX, seed=41)[0]  # (pmode, Ny, Nx) complex64
    H1 = _H1()
    j = torch.arange(NZ, dtype=torch.float32).view(NZ, 1, 1)
    H3d = (H1 ** j).view(1, 1, 1, NZ, NY, NX)
    occu = torch.ones(OMODE)

    measurements = []
    for y0, x0 in CROP_POS:
        win = obj_true[:, :, y0 : y0 + NY, x0 : x0 + NX]
        patches = torch.stack([win.abs(), win.angle()], dim=-1).unsqueeze(0)
        dp = ls.dp_from_fields(ls.firstborn_fields(patches, probe.unsqueeze(0), H3d), occu)
        measurements.append(dp[0])
    measurements = torch.stack(measurements).clamp_min(0.0)

    n_scans = len(CROP_POS)
    init_variables = {
        "obj": np.ones((OMODE, NZ, NOY, NOX), dtype=np.complex64),  # vacuum start
        "obj_tilts": np.zeros((1, 2), dtype=np.float32),
        "slice_thickness": np.float32(1.0),
        "probe": probe.numpy(),
        "probe_pos_shifts": np.zeros((n_scans, 2), dtype=np.float32),
        "omode_occu": occu.numpy(),
        "H": H1.numpy(),
        "measurements": measurements.numpy(),
        "N_scan_slow": 2,
        "N_scan_fast": 2,
        "crop_pos": CROP_POS,
        "dx": np.float32(0.1),
        "dk": np.float32(0.01),
        "lambd": np.float32(0.0025),
        "random_seed": None,
        "length_unit": "Ang",
        "scan_affine": None,
    }
    model_params = {
        "detector_blur_std": None,
        "obj_preblur_std": None,
        "solver_type": "born",
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
    return PtychoAD(init_variables, model_params, device="cpu", verbose=False)


def test_wiring_forward_consistency():
    """The line search's own forward (gathered window -> firstborn_fields ->
    dp_from_fields) must reproduce model.forward for the same view — any
    mismatch means the crop grids, propagator stack, or unit map are wired
    wrong and the search would descend on the wrong problem."""
    model = _make_model()
    for index in range(len(CROP_POS)):
        idx = torch.as_tensor([index])
        with torch.no_grad():
            dp_model = model(idx)
        gy = model.rpy_grid + model.crop_pos[index, 0]
        gx = model.rpx_grid + model.crop_pos[index, 1]
        obja = model.opt_obja.data[:, :, gy, gx]
        objp = model.opt_objp.data[:, :, gy, gx]
        patches = torch.stack([obja, objp], dim=-1).unsqueeze(0)
        H = model.get_propagators_3d(model.get_propagators(idx)).detach()
        probe = model.get_probes(idx).detach()
        dp_ours = ls.dp_from_fields(ls.firstborn_fields(patches, probe, H), model.omode_occu)
        scale = dp_model.abs().max()
        assert torch.allclose(dp_ours, dp_model, rtol=1e-5, atol=1e-6 * scale), (
            f"view {index}: line-search forward disagrees with model.forward"
        )


def test_view_sweeps_descend_and_scatter_is_local():
    """Five sweeps over the four views must descend monotonically and cut the
    direction objective by well over an order of magnitude (measured: ~40x);
    the accepted steps must be live (§4.3 fallback detector); and pixels
    never covered by any window must stay untouched."""
    model = _make_model()
    cfg = ls.LineSearchConfig()
    st = ls.LineSearchState()

    # rows/cols >= 48 are outside every window (windows span [0,48) x [0,48))
    before_amp = model.opt_obja.data[..., 48:, :].clone()
    before_phs = model.opt_objp.data[..., :, 48:].clone()
    probe_before = torch.view_as_complex(model.opt_probe.data).clone()

    n_sweeps = 5
    sweep_losses = []
    for _ in range(n_sweeps):
        losses = [
            ls.linesearch_model_update(model, index, cfg, st)["loss"]
            for index in range(len(CROP_POS))
        ]
        sweep_losses.append(float(np.mean(losses)))

    assert all(b < a for a, b in zip(sweep_losses, sweep_losses[1:])), (
        f"not monotone: {sweep_losses}"
    )
    assert sweep_losses[-1] < 0.1 * sweep_losses[0], f"weak descent: {sweep_losses}"
    assert torch.equal(model.opt_obja.data[..., 48:, :], before_amp)
    assert torch.equal(model.opt_objp.data[..., :, 48:], before_phs)

    n_updates = n_sweeps * len(CROP_POS)
    assert len(st.steps_o) == n_updates and len(st.steps_p) == n_updates
    fallback = cfg.ls_damp * cfg.alpha / NZ
    assert any(a != pytest.approx(fallback, rel=1e-12) for a in st.steps_o), (
        f"all object steps pinned at fallback {fallback}: silent-fallback signature"
    )

    probe_after = torch.view_as_complex(model.opt_probe.data)
    assert not torch.equal(probe_after, probe_before)  # probe step wrote back
    assert torch.isfinite(probe_after.abs()).all()


def test_object_only_leaves_probe_alone():
    model = _make_model()
    probe_before = torch.view_as_complex(model.opt_probe.data).clone()
    st = ls.LineSearchState()
    diag = ls.linesearch_model_update(model, 0, update_probe=False, state=st)
    assert diag["b"] is None and len(st.steps_p) == 0
    assert torch.equal(torch.view_as_complex(model.opt_probe.data), probe_before)


def test_guards():
    """Multislice models and unported features must be refused loudly rather
    than silently searching on the wrong problem."""
    model = _make_model()

    model.solver_type = "multislice"
    with pytest.raises(ValueError, match="single scattering"):
        ls.linesearch_model_update(model, 0)
    model.solver_type = "born"

    model.born_iterations = 2
    with pytest.raises(ValueError, match="born_iterations=1"):
        ls.linesearch_model_update(model, 0)
    model.born_iterations = 1

    model.obj_preblur_std = 1.0
    with pytest.raises(NotImplementedError, match="obj_preblur"):
        ls.linesearch_model_update(model, 0)
    model.obj_preblur_std = None

    with pytest.raises(NotImplementedError, match="displacement canvas"):
        ls.linesearch_model_update(model, 0, config=ls.LineSearchConfig(momentum=0.5))

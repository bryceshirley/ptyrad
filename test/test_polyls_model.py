"""
Model-layer wiring tests for the ray-polynomial line search at Born order
M > 1 (polyls_hybrid_order_plan.md §4.4): forward consistency with
model.forward through the coeffs-aware order-M engine, descent over view
sweeps, per-view vs batched parity at B = 1, and the order-growth contract
(M and the coefficients are re-read from the model on every call — the
detector refit's grow_tol hook mutates both after a sweep). CPU, eager,
synthetic data — no reconstructions.
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import pytest
import torch

torch._dynamo.config.disable = True

import ptyrad.linesearch as ls
from ptyrad.forward_models.born_helpers import born_fields
from ptyrad.models import PtychoAD
from test.test_linesearch_model import _H1, CROP_POS, NX, NY, NZ, OMODE, PMODE
from test.test_linesearch_oracle import make_probe

COEFFS3 = [[1.0, 0.0], [0.9, 0.15], [0.75, -0.2]]


def _make_model_M(born_iterations, coeffs_init=None, noy=64, nox=64):
    """Minimal born PtychoAD at order M: data from a true object through the
    SAME coeffs-aware order-M forward, model starts at vacuum."""
    g = torch.Generator().manual_seed(7)
    amp = 1.0 + 0.01 * torch.randn(OMODE, NZ, noy, nox, generator=g)
    phase = 0.3 * torch.rand(OMODE, NZ, noy, nox, generator=g)
    obj_true = torch.polar(amp, phase)

    probe = make_probe(1, PMODE, NY, NX, seed=41)[0]
    H1 = _H1()
    j = torch.arange(NZ, dtype=torch.float32).view(NZ, 1, 1)
    H3d = (H1**j).view(1, 1, 1, NZ, NY, NX)
    occu = torch.ones(OMODE)
    coeffs = (
        torch.tensor(coeffs_init, dtype=torch.float32) if coeffs_init is not None else None
    )

    measurements = []
    for y0, x0 in CROP_POS:
        win = obj_true[:, :, y0 : y0 + NY, x0 : x0 + NX]
        patches = torch.stack([win.abs(), win.angle()], dim=-1).unsqueeze(0)
        F = born_fields(patches, probe.unsqueeze(0), H3d, born_iterations, coeffs)
        measurements.append(ls.dp_from_fields(F, occu)[0])
    measurements = torch.stack(measurements).clamp_min(0.0)

    n_scans = len(CROP_POS)
    init_variables = {
        "obj": np.ones((OMODE, NZ, noy, nox), dtype=np.complex64),
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
        "born_iterations": born_iterations,
        "born_coeffs_init": coeffs_init,
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


def test_forward_consistency_M3_with_coeffs():
    """The poly path's own forward (gathered window -> born_fields ->
    dp_from_fields, with the model's coefficients) must reproduce
    model.forward — otherwise the search descends on the wrong problem."""
    model = _make_model_M(3, COEFFS3)
    assert model.use_born_coeffs
    M, coeffs = ls._born_order_and_coeffs(model)
    assert M == 3 and coeffs is not None and coeffs.shape[0] == 3
    for index in range(len(CROP_POS)):
        idx = torch.as_tensor([index])
        with torch.no_grad():
            dp_model = model(idx)
            gy = model.rpy_grid + model.crop_pos[index, 0]
            gx = model.rpx_grid + model.crop_pos[index, 1]
            patches = torch.stack(
                [model.opt_obja.data[:, :, gy, gx], model.opt_objp.data[:, :, gy, gx]],
                dim=-1,
            ).unsqueeze(0)
            H = model.get_propagators_3d(model.get_propagators(idx)).detach()
            probe = model.get_probes(idx).detach()
            dp_ours = ls.dp_from_fields(
                born_fields(patches, probe, H, M, coeffs), model.omode_occu
            )
        scale = dp_model.abs().max()
        assert torch.allclose(dp_ours, dp_model, rtol=1e-5, atol=1e-6 * scale), (
            f"view {index}: poly-path forward disagrees with model.forward"
        )


def test_view_sweeps_descend_M3():
    """Sweeps at M = 3 with coefficients must descend strongly with live
    (non-fallback) steps, and pixels outside every window stay untouched."""
    model = _make_model_M(3, COEFFS3)
    cfg = ls.LineSearchConfig()
    st = ls.LineSearchState()
    before_amp = model.opt_obja.data[..., 48:, :].clone()

    n_sweeps = 5
    sweep_losses = []
    for _ in range(n_sweeps):
        losses = [
            ls.linesearch_model_update(model, index, cfg, st)["loss"]
            for index in range(len(CROP_POS))
        ]
        sweep_losses.append(float(np.mean([float(v) for v in losses])))

    assert sweep_losses[-1] < 0.15 * sweep_losses[0], f"weak descent: {sweep_losses}"
    assert torch.equal(model.opt_obja.data[..., 48:, :], before_amp)

    fallback = cfg.ls_damp * cfg.alpha / NZ
    live = [a for a in st.steps_o if a != pytest.approx(fallback, rel=1e-12)]
    assert len(live) > 0.8 * len(st.steps_o), (
        f"object steps mostly pinned at fallback {fallback}: {st.steps_o}"
    )
    assert torch.isfinite(torch.view_as_complex(model.opt_probe.data).abs()).all()


def test_batched_matches_per_view_at_B1_M3():
    """The joint-batch updater at B = 1 must reproduce the per-view updater
    through the poly path (same window, same F_stack, same scalar step)."""
    m1, m2 = _make_model_M(3, COEFFS3), _make_model_M(3, COEFFS3)
    d1 = ls.linesearch_model_update(m1, 0, state=ls.LineSearchState())
    d2 = ls.linesearch_model_update_batched(m2, [0], state=ls.LineSearchState())
    assert d1["a"] == pytest.approx(d2["a"], rel=1e-5)
    assert d1["b"] == pytest.approx(d2["b"], rel=1e-4)
    assert torch.allclose(m1.opt_obja.data, m2.opt_obja.data, atol=1e-6)
    assert torch.allclose(m1.opt_objp.data, m2.opt_objp.data, atol=1e-6)
    assert torch.allclose(m1.opt_probe.data, m2.opt_probe.data, atol=1e-6)


def test_hybrid_loop_contract():
    """The hybrid division of labor (plan §1, hard rule): within a view sweep
    the Born coefficients are a FROZEN input — nothing in the line search
    writes opt_born_coeffs — and the detector refit, called after the sweep
    (the driver's constraint -> refit ordering), (a) updates the coefficients
    at a non-weak object, and (b) grows born_iterations under grow_tol so the
    NEXT iteration's line search runs at the new order."""
    from ptyrad.reconstruction import refit_born_coeffs

    refit_cfg = {
        "start_iter": 1,
        "step": 1,
        "end_iter": None,
        "n_views": 4,
        "pin_first": False,
        "ridge": 1e-5,
        "method": "detector",
        "target": "multislice",
        "grow_tol": 10.0,  # first refit: no growth (coefficients only)
        "n_limit": 3,
    }
    model = _make_model_M(2, None)
    # non-weak object: start AT the scattering object the data came from, and
    # wire the refit config the way PtychoAD does from model_params
    g = torch.Generator().manual_seed(7)
    amp = 1.0 + 0.01 * torch.randn(OMODE, NZ, 64, 64, generator=g)
    phase = 0.3 * torch.rand(OMODE, NZ, 64, 64, generator=g)
    model.opt_obja.data = amp
    model.opt_objp.data = phase
    model.born_refit = refit_cfg
    model.use_born_coeffs = True
    model.register_buffer(
        "born_refit_views", torch.arange(len(CROP_POS)).long(), persistent=False
    )

    c_pre = model.opt_born_coeffs.data.clone()
    st = ls.LineSearchState()
    for index in range(len(CROP_POS)):  # the view sweep
        ls.linesearch_model_update(model, index, state=st)
    # frozen within the sweep: the line search never touches the coefficients
    assert torch.equal(model.opt_born_coeffs.data, c_pre)
    assert model.born_iterations == 2

    refit_born_coeffs(model, 1, verbose=False)  # driver ordering: after sweep
    assert model.born_iterations == 2  # grow_tol 10: no growth
    assert not torch.allclose(model.opt_born_coeffs.data, c_pre, atol=1e-6), (
        "refit did not move the coefficients at a non-weak object"
    )

    # (b) growth: tighten grow_tol so the residual exceeds it; next sweep at M+1
    model.born_refit["grow_tol"] = 1e-12
    refit_born_coeffs(model, 2, verbose=False)
    assert model.born_iterations == 3
    assert model.opt_born_coeffs.shape[0] == 3
    d = ls.linesearch_model_update(model, 0, state=st)  # re-reads M and c
    assert np.isfinite(d["a"]) and np.isfinite(d["b"])


def test_order_growth_is_honored_between_sweeps():
    """The grow_tol hook promotes born_iterations and resizes opt_born_coeffs
    after a sweep; the next update must run at the new order with the new
    coefficients — no caching of M anywhere in the search path."""
    model = _make_model_M(2, [[1.0, 0.0], [0.9, 0.1]])
    st = ls.LineSearchState()
    d = ls.linesearch_model_update(model, 0, state=st)
    assert np.isfinite(d["a"])

    # simulate _refit_born_coeffs_detector growth: n 2 -> 3, coeffs resized
    model.born_iterations = 3
    c = torch.tensor([[1.0, 0.0], [0.95, 0.05], [0.9, -0.1]], dtype=torch.float32)
    model.opt_born_coeffs.data = c
    M, coeffs = ls._born_order_and_coeffs(model)
    assert M == 3 and coeffs.shape[0] == 3

    d2 = ls.linesearch_model_update(model, 1, state=st)
    assert np.isfinite(d2["a"]) and np.isfinite(d2["b"])

    # mismatch (coeffs shorter than M) is refused loudly
    model.born_iterations = 4
    with pytest.raises(ValueError, match="orders but born_iterations"):
        ls.linesearch_model_update(model, 0, state=st)

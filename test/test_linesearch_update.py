"""
Integration tests for linesearch_batch_update (spec §3 steps 1-7).

The nine §7 oracles in test_linesearch_oracle.py pin the primitives; these
tests exercise the composed batch update: forward -> preconditioned direction
-> response -> quartic -> cubic solve -> joint complex object step -> probe
step against the exactly-updated field. Synthetic tensors, CPU, B = 1 (the
design point). Not asserted: probe recovery toward truth — a joint
probe+object update on a single view is gauge-ambiguous and gauge fixing is
PtyRAD's constraint machinery's job, outside this module.
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import pytest
import torch

torch._dynamo.config.disable = True

from test.test_linesearch_oracle import make_H, make_object, make_probe, ref_dp, ref_fields

import ptyrad.linesearch as ls


def _problem(seed_obj=42, seed_probe=41, phase_max=0.3):
    """Small synthetic single-view problem: data from a true object/probe,
    reconstruction started from vacuum object and a perturbed probe."""
    omode, Nz, Ny, Nx, pmode = 1, 3, 32, 32, 2
    occu = torch.ones(omode)
    H = make_H(Nz, Ny, Nx)
    probe_true = make_probe(1, pmode, Ny, Nx, seed=seed_probe)
    obj_true = make_object(1, omode, Nz, Ny, Nx, seed=seed_obj, phase_max=phase_max)
    I_dat = ref_dp(ref_fields(obj_true, probe_true, H), occu).clamp_min(0.0)
    mask = torch.ones_like(I_dat)
    mask[..., :4, :4] = 0.0
    omega = mask / (I_dat + 1.0)

    obja = torch.ones(omode, Nz, Ny, Nx)
    objp = torch.zeros(omode, Nz, Ny, Nx)
    probe = probe_true + 0.1 * make_probe(1, pmode, Ny, Nx, seed=seed_probe + 2)
    return obja, objp, probe, H, I_dat, mask, omega, occu, Nz


def _Q(obja, objp, probe, H, I_dat, omega, occu):
    patches = torch.stack([obja, objp], dim=-1).unsqueeze(0)
    u = ls.dp_from_fields(ls.iss_fields(patches, probe, H), occu)
    return float((omega * (u - I_dat).square()).sum())


def test_update_descends_and_solver_is_live():
    """Ten updates must reduce the Gaussian intensity cost by orders of
    magnitude, and the accepted step log must show live spread — every value
    clustered at exactly ls_damp*alpha/N is the silent-fallback signature the
    log exists to expose (spec §4.3/§6)."""
    obja, objp, probe, H, I_dat, mask, omega, occu, Nz = _problem()
    cfg = ls.LineSearchConfig()
    st = ls.LineSearchState()

    q0 = _Q(obja, objp, probe, H, I_dat, omega, occu)
    for _ in range(10):
        probe, diag = ls.linesearch_batch_update(
            obja, objp, probe, H, I_dat, mask, occu, cfg, st
        )
    q1 = _Q(obja, objp, probe, H, I_dat, omega, occu)

    assert q1 < 1e-3 * q0, f"cost only moved {q0:.3e} -> {q1:.3e}"
    assert len(st.steps_o) == 10 and len(st.steps_p) == 10
    fallback = cfg.ls_damp * cfg.alpha / Nz
    live = [a for a in st.steps_o if a != pytest.approx(fallback, rel=1e-12)]
    assert len(live) == 10, f"object steps stuck at fallback {fallback}: {st.steps_o}"
    assert max(st.steps_o) - min(st.steps_o) > 0.01  # genuine spread
    assert diag["b"] is not None and diag["probe_peak"] > 0


def test_update_object_only_and_probe_branch_exactness():
    """update_probe=False must leave the probe untouched and log no probe
    steps; and after an update the stored object must reproduce the exactly
    predicted field F + a*D on a re-forward (the §3 step-7 precondition,
    end-to-end through the updater rather than the primitives)."""
    obja, objp, probe, H, I_dat, mask, omega, occu, Nz = _problem()
    cfg = ls.LineSearchConfig()
    st = ls.LineSearchState()

    patches0 = torch.stack([obja, objp], dim=-1).unsqueeze(0)
    F0 = ls.iss_fields(patches0, probe, H)

    probe_out, diag = ls.linesearch_batch_update(
        obja, objp, probe, H, I_dat, mask, occu, cfg, st, update_probe=False
    )
    assert torch.equal(probe_out, probe)
    assert diag["b"] is None and len(st.steps_p) == 0

    # reconstruct what the update did and verify affine exactness end-to-end:
    # the re-forward field must equal F0 + a*D for the direction the updater
    # took (recovered from disp_prev = a*d).
    a = diag["a"]
    d = st.disp_prev / a
    D = ls.direction_response(patches0, d, probe, H)
    patches1 = torch.stack([obja, objp], dim=-1).unsqueeze(0)
    F1 = ls.iss_fields(patches1, probe, H)
    err = (F1 - (F0 + a * D)).abs().max() / F0.abs().max()
    assert err < 5e-6, f"updater leaked out of the affine parameterisation: {err:.2e}"


def test_update_momentum_and_mask_none():
    """Heavy-ball path (momentum mixed in BEFORE the search, disp_prev = a*d)
    and mask=None both run and still descend."""
    obja, objp, probe, H, I_dat, mask, omega, occu, Nz = _problem()
    cfg = ls.LineSearchConfig(momentum=0.5)
    st = ls.LineSearchState()

    omega_none = 1.0 / (I_dat + 1.0)
    q0 = _Q(obja, objp, probe, H, I_dat, omega_none, occu)
    for _ in range(5):
        probe, _ = ls.linesearch_batch_update(
            obja, objp, probe, H, I_dat, None, occu, cfg, st
        )
    q1 = _Q(obja, objp, probe, H, I_dat, omega_none, occu)
    assert q1 < 0.05 * q0
    assert st.disp_prev is not None


def test_update_intensity_objective_knob():
    """§5 option (b) exposed as a knob: direction from Q itself also descends
    (its ls_damp would need retuning for production; here we only assert the
    plumbing works and the search still improves the cost)."""
    obja, objp, probe, H, I_dat, mask, omega, occu, Nz = _problem()
    cfg = ls.LineSearchConfig(direction_objective="intensity")
    st = ls.LineSearchState()
    q0 = _Q(obja, objp, probe, H, I_dat, omega, occu)
    for _ in range(5):
        probe, _ = ls.linesearch_batch_update(
            obja, objp, probe, H, I_dat, mask, occu, cfg, st
        )
    q1 = _Q(obja, objp, probe, H, I_dat, omega, occu)
    assert q1 < 0.2 * q0

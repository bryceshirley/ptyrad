"""Distributed Born forwards (born_dist): P-way probe-mode split and
depth-partitioned slice split (chain AND direct carry) — exactness vs the
single-device born_forward. CPU-only — the split math is device-agnostic."""

import pytest
import torch

from ptyrad.forward_models import (
    born_forward,
    born_forward_dist,
    depth_bounds,
    make_blocks,
    make_modesplit_forward,
    mode_groups,
)


def _inputs(n=2, pmode=4, omode=1, nz=5, npix=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    obj = torch.randn(n, omode, nz, npix, npix, 2, generator=g) * 0.02
    obj[..., 0] += 1.0
    probe = torch.randn(n, pmode, npix, npix, 2, generator=g)
    probe = torch.complex(probe[..., 0], probe[..., 1])
    H = torch.exp(1j * 0.01 * torch.randn(1, 1, 1, nz, npix, npix, generator=g))
    H = H.to(torch.complex64)
    occu = torch.ones(omode)
    return obj, probe, H, occu


@pytest.mark.parametrize("pmode,n_groups", [(4, 2), (12, 4), (12, 3), (3, 4)])
def test_modesplit_matches_single_device(pmode, n_groups):
    obj, probe, H, occu = _inputs(pmode=pmode)
    with torch.compiler.set_stance("force_eager"):
        ref = born_forward(obj, probe, H, occu, n_max=3)
        split = make_modesplit_forward(["cpu"] * n_groups)
        out = split(obj, probe, H, occu, n_max=3)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-7)


def test_modesplit_with_coeffs_and_grad():
    obj, probe, H, occu = _inputs(pmode=4)
    coeffs = torch.tensor([[1.0, 0.0], [0.6, 0.1], [0.2, -0.05]])
    with torch.compiler.set_stance("force_eager"):
        ref = born_forward(obj, probe, H, occu, n_max=3, coeffs=coeffs)
        split = make_modesplit_forward(["cpu", "cpu"])
        o = obj.detach().requires_grad_(True)
        out = split(o, probe, H, occu, n_max=3, coeffs=coeffs)
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-7)
        out.sum().backward()
    assert o.grad is not None and torch.isfinite(o.grad).all()


def test_modesplit_single_group_falls_through():
    obj, probe, H, occu = _inputs(pmode=1)
    with torch.compiler.set_stance("force_eager"):
        ref = born_forward(obj, probe, H, occu, n_max=2)
        split = make_modesplit_forward(["cpu", "cpu"])
        out = split(obj, probe, H, occu, n_max=2)
    assert torch.equal(out, ref)


def _slice_blocks(obj, H, probe, bounds):
    objc_full = (torch.polar(obj[..., 0], obj[..., 1]) - 1.0).unsqueeze(1)
    devs = ["cpu"] * (len(bounds) - 1)
    objc_blocks = [
        objc_full[..., lo:hi, :, :].contiguous() for lo, hi in zip(bounds[:-1], bounds[1:])
    ]
    H_blocks = [H[..., lo:hi, :, :].contiguous() for lo, hi in zip(bounds[:-1], bounds[1:])]
    return make_blocks(objc_blocks, H_blocks, probe, devs)


@pytest.mark.parametrize("carry", ["chain", "direct"])
@pytest.mark.parametrize("bounds", [[0, 5, 8, 11, 13], [0, 7, 13]])
def test_slicesplit_matches_single_device(carry, bounds):
    obj, probe, H, occu = _inputs(pmode=3, nz=13)
    coeffs = torch.tensor([[1.0, 0.0], [0.7, 0.05], [0.3, -0.1], [0.1, 0.02]])
    with torch.compiler.set_stance("force_eager"):
        ref = born_forward(obj, probe, H, occu, n_max=4, coeffs=coeffs)
        blocks = _slice_blocks(obj, H, probe, bounds)
        out = born_forward_dist(blocks, occu, 4, coeffs=coeffs, carry=carry)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-6)


def test_slicesplit_gradients_chain_vs_direct():
    obj, probe, H, occu = _inputs(pmode=2, nz=8)
    bounds = [0, 3, 5, 7, 8]
    grads = {}
    for carry in ("chain", "direct"):
        objc_full = (torch.polar(obj[..., 0], obj[..., 1]) - 1.0).unsqueeze(1)
        oc = objc_full.detach().requires_grad_(True)
        objc_blocks = [oc[..., lo:hi, :, :] for lo, hi in zip(bounds[:-1], bounds[1:])]
        H_blocks = [H[..., lo:hi, :, :].contiguous() for lo, hi in zip(bounds[:-1], bounds[1:])]
        blocks = make_blocks(objc_blocks, H_blocks, probe, ["cpu"] * 4)
        with torch.compiler.set_stance("force_eager"):
            born_forward_dist(blocks, occu, 3, carry=carry).sum().backward()
        assert oc.grad is not None and torch.isfinite(oc.grad).all()
        grads[carry] = oc.grad.clone()
    assert torch.allclose(grads["chain"], grads["direct"], rtol=1e-4, atol=1e-7)


def test_depth_bounds():
    assert depth_bounds(42, 4) == [0, 11, 22, 32, 42]
    b = depth_bounds(42, 4, m=12)
    assert b[0] == 0 and b[-1] == 42 and b[1] >= 12
    assert all(b[i + 1] > b[i] for i in range(4))
    # extreme: M close to Nz leaves 1 slice per later block
    b = depth_bounds(10, 4, m=9)
    assert b == [0, 7, 8, 9, 10]
    with pytest.raises(ValueError):
        depth_bounds(3, 4)


def test_depth_bounds_feed_forward_dist():
    nz, m = 13, 6
    obj, probe, H, occu = _inputs(pmode=2, nz=nz)
    bounds = depth_bounds(nz, 4, m=m)
    with torch.compiler.set_stance("force_eager"):
        ref = born_forward(obj, probe, H, occu, n_max=m)
        out = born_forward_dist(_slice_blocks(obj, H, probe, bounds), occu, m)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-6)


def test_mode_groups_partition():
    assert mode_groups(12, 4) == [(0, 3), (3, 6), (6, 9), (9, 12)]
    assert mode_groups(12, 8) == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
    ]
    for pmode, g in [(12, 5), (4, 2), (3, 4), (1, 3)]:
        grp = mode_groups(pmode, g)
        assert grp[0][0] == 0 and grp[-1][1] == pmode
        assert all(hi > lo for lo, hi in grp)
        assert all(grp[i][1] == grp[i + 1][0] for i in range(len(grp) - 1))

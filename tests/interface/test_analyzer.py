"""Tests for :mod:`ptyrad.analysis` against a synthetic ``model.hdf5``."""

import json

import numpy as np
import pytest

from ptyrad.analysis import (
    Analyzer,
    apply_fov,
    extract_object,
    extract_object_amplitude,
    extract_object_phase,
    extract_probe,
    extract_probe_positions,
    extract_provenance,
    get_probe_center_positions,
    get_scanned_fov_bbox,
)
from ptyrad.io.provenance import save_provenance_to_hdf5
from ptyrad.io.save import save_dict_to_hdf5


# ---------------------------------------------------------------------------
# Fixture: write a tiny synthetic model.hdf5 next to a placeholder meas file
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_payload():
    """Synthetic ``optimizable_tensors`` + ``model_attributes`` consistent enough
    for every getter except ``build_model``."""
    pmode, Hp, Wp = 2, 4, 4
    omode, Nz, Ny, Nx = 1, 1, 12, 14
    # 3x3 scan grid with crop_pos (top-left of probe window) in [0, Ny-Hp]x[0, Nx-Wp]
    cy = np.array([0, 3, 7], dtype=np.int32)
    cx = np.array([0, 4, 9], dtype=np.int32)
    crop_pos = np.array([[y, x] for y in cy for x in cx], dtype=np.int32)
    N = crop_pos.shape[0]

    rng = np.random.default_rng(0)
    obja = rng.uniform(0.8, 1.0, size=(omode, Nz, Ny, Nx)).astype(np.float32)
    objp = rng.uniform(-0.3, 0.3, size=(omode, Nz, Ny, Nx)).astype(np.float32)
    probe_real = rng.standard_normal((pmode, Hp, Wp)).astype(np.float32)
    probe_imag = rng.standard_normal((pmode, Hp, Wp)).astype(np.float32)
    probe = (probe_real + 1j * probe_imag).astype(np.complex64)
    probe_pos_shifts = rng.standard_normal((N, 2)).astype(np.float32) * 0.1

    return {
        "ptyrad_version": "test",
        "optimizable_tensors": {
            "obja": obja,
            "objp": objp,
            "probe": probe,
            "probe_pos_shifts": probe_pos_shifts,
            "obj_tilts": np.zeros((1, 2), dtype=np.float32),
            "slice_thickness": np.float32(2.0),
        },
        "model_attributes": {
            "dx": np.float32(0.25),
            "dk": np.float32(0.1),
            "lambd": np.float32(0.0251),
            "crop_pos": crop_pos,
            "N_scan_slow": np.int32(3),
            "N_scan_fast": np.int32(3),
            "simu_Npix": Hp,
            "meas_Npix": Hp,
        },
        "params": {
            "init_params": {"meas_params": {"path": "synthetic.raw", "key": None}},
            "recon_params": {"NITER": 1},
        },
        "loss_iters": np.array([[1, 0.5], [2, 0.4]], dtype=np.float32),
        "batch_losses": {"loss_single": [0.4]},
        "avg_losses": {"loss_single": np.float32(0.4)},
        "niter": 2,
        "indices": np.arange(N, dtype=np.int64),
        "avg_iter_t": np.float64(0.01),
        # dashboard fields
        "dz_iters": [],
        "lr_iters": {},
        "avg_tilt_iters": {},
        "convergence_iters": {},
    }


@pytest.fixture
def tiny_hdf5(tmp_path, tiny_payload):
    path = tmp_path / "model_iter0001.hdf5"
    save_dict_to_hdf5(tiny_payload, str(path))
    provenance = {
        "probe": [{"action": "Synthetic", "uid": "abcd1234"}],
        "obj": [{"action": "Synthetic"}],
    }
    save_provenance_to_hdf5(path, json.dumps(provenance))
    return path, tiny_payload, provenance


# ---------------------------------------------------------------------------
# Init paths
# ---------------------------------------------------------------------------


def test_init_from_path(tiny_hdf5):
    path, payload, _ = tiny_hdf5
    a = Analyzer(path)
    assert a.path == str(path)
    assert a.niter == payload["niter"]
    assert a.probe_shape == payload["optimizable_tensors"]["probe"].shape[-2:]
    assert a.pmode == 2
    assert a.omode == 1
    assert a.nslice == 1
    assert not a.is_multislice
    np.testing.assert_allclose(a.dx, float(payload["model_attributes"]["dx"]))


def test_init_from_dict(tiny_payload):
    a = Analyzer(tiny_payload)
    assert a.path is None
    assert a.niter == tiny_payload["niter"]


def test_repr_contains_useful_info(tiny_payload):
    a = Analyzer(tiny_payload)
    r = repr(a)
    assert "Analyzer" in r
    assert "pmode=2" in r


# ---------------------------------------------------------------------------
# Object getters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fov", ["full", "crop"])
def test_get_object_shape(tiny_payload, fov):
    a = Analyzer(tiny_payload)
    obj = a.get_object(fov=fov)
    assert obj.dtype == np.complex64
    # Always 4D
    assert obj.ndim == 4
    omode, Nz = obj.shape[0], obj.shape[1]
    assert omode == 1 and Nz == 1


def test_get_object_full_recomposition(tiny_payload):
    obja = tiny_payload["optimizable_tensors"]["obja"]
    objp = tiny_payload["optimizable_tensors"]["objp"]
    expected = (obja * np.exp(1j * objp)).astype(np.complex64)
    np.testing.assert_allclose(
        extract_object(tiny_payload, fov="full"), expected, rtol=1e-5
    )


def test_get_object_crop_matches_bbox(tiny_payload):
    a = Analyzer(tiny_payload)
    bbox = a.get_fov_bbox()
    y_min, y_max, x_min, x_max = bbox
    obj_full = a.get_object(fov="full")
    obj_crop = a.get_object(fov="crop")
    assert obj_crop.shape[-2:] == (y_max - y_min + 1, x_max - x_min + 1)
    np.testing.assert_allclose(
        obj_crop, obj_full[..., y_min : y_max + 1, x_min : x_max + 1]
    )


def test_get_object_phase_no_sign_flip(tiny_payload):
    objp_saved = tiny_payload["optimizable_tensors"]["objp"]
    np.testing.assert_allclose(extract_object_phase(tiny_payload), objp_saved)


def test_get_object_amplitude(tiny_payload):
    obja_saved = tiny_payload["optimizable_tensors"]["obja"]
    np.testing.assert_allclose(extract_object_amplitude(tiny_payload), obja_saved)


def test_get_object_fov_invalid(tiny_payload):
    with pytest.raises(ValueError, match="fov"):
        extract_object(tiny_payload, fov="weird")


def test_get_object_crop_requires_attrs(tiny_payload):
    # Pass only opt_tensors (no model_attrs available anywhere) -> should error
    opt = tiny_payload["optimizable_tensors"]
    with pytest.raises(ValueError, match="crop_pos"):
        extract_object(opt, fov="crop")


# ---------------------------------------------------------------------------
# Probe getter
# ---------------------------------------------------------------------------


def test_get_probe_real_returns_stored(tiny_payload):
    stored = tiny_payload["optimizable_tensors"]["probe"]
    np.testing.assert_allclose(extract_probe(tiny_payload, space="real"), stored)


def test_get_probe_fourier_roundtrip(tiny_payload):
    """ifft(extract_probe('fourier')) should recover extract_probe('real') under the
    fftshift sandwich convention used by plot_probe_modes."""
    from numpy.fft import fftshift, ifft2, ifftshift

    probe_r = extract_probe(tiny_payload, space="real")
    probe_k = extract_probe(tiny_payload, space="fourier")
    recovered = fftshift(
        ifft2(ifftshift(probe_k, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )
    np.testing.assert_allclose(recovered, probe_r, atol=1e-5)


def test_get_probe_invalid_space(tiny_payload):
    with pytest.raises(ValueError, match="space"):
        extract_probe(tiny_payload, space="kspace")


def test_get_probe_as_torch(tiny_payload):
    import torch

    out = extract_probe(tiny_payload, space="real", as_torch=True, device="cpu")
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.complex64


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


def test_get_probe_center_positions_top_left_offset(tiny_payload):
    crop_pos = tiny_payload["model_attributes"]["crop_pos"]
    probe_shape = tiny_payload["optimizable_tensors"]["probe"].shape[-2:]
    centers_no_shift = get_probe_center_positions(crop_pos, probe_shape)
    expected = crop_pos.astype(np.float32) + np.array(
        [probe_shape[0] // 2, probe_shape[1] // 2], dtype=np.float32
    )
    np.testing.assert_allclose(centers_no_shift, expected)


def test_get_probe_positions_includes_shifts(tiny_payload):
    a = Analyzer(tiny_payload)
    pos_with = a.get_probe_positions(include_sub_px_shifts=True)
    pos_without = a.get_probe_positions(include_sub_px_shifts=False)
    shifts = tiny_payload["optimizable_tensors"]["probe_pos_shifts"]
    np.testing.assert_allclose(pos_with - pos_without, shifts, atol=1e-5)


def test_get_probe_positions_units_angstrom(tiny_payload):
    a = Analyzer(tiny_payload)
    px = a.get_probe_positions(units="px")
    pixel = a.get_probe_positions(units="pixel")
    ang = a.get_probe_positions(units="Ang")
    dx = float(np.asarray(tiny_payload["model_attributes"]["dx"]))
    np.testing.assert_allclose(pixel, px)
    np.testing.assert_allclose(ang, px * dx, rtol=1e-6)


def test_get_probe_positions_target_crop_vs_probe(tiny_payload):
    a = Analyzer(tiny_payload)
    pos_probe = a.get_probe_positions(target="probe")
    pos_crop = a.get_probe_positions(target="crop")
    probe_shape = tiny_payload["optimizable_tensors"]["probe"].shape[-2:]
    expected_offset = np.array(
        [probe_shape[0] // 2, probe_shape[1] // 2], dtype=np.float32
    )
    np.testing.assert_allclose(
        pos_probe - pos_crop,
        np.broadcast_to(expected_offset, pos_probe.shape),
        atol=1e-5,
    )


def test_get_probe_positions_fov_crop_local(tiny_payload):
    a = Analyzer(tiny_payload)
    pos_crop = a.get_probe_positions(fov="crop")
    bbox = a.get_fov_bbox()
    y_extent = bbox[1] - bbox[0]
    x_extent = bbox[3] - bbox[2]
    # All coords inside [0, extent] inclusive (we used integer shifts of 0.1 magnitude)
    assert pos_crop[:, 0].min() >= -1.0
    assert pos_crop[:, 0].max() <= y_extent + 1.0
    assert pos_crop[:, 1].min() >= -1.0
    assert pos_crop[:, 1].max() <= x_extent + 1.0


def test_get_probe_positions_missing_shifts_falls_back_to_zero(tiny_payload):
    opt = dict(tiny_payload["optimizable_tensors"])
    opt.pop("probe_pos_shifts")
    pos = extract_probe_positions(
        opt, model_attrs=tiny_payload["model_attributes"], include_sub_px_shifts=True
    )
    crop_pos = tiny_payload["model_attributes"]["crop_pos"]
    probe_shape = tiny_payload["optimizable_tensors"]["probe"].shape[-2:]
    expected = crop_pos.astype(np.float32) + np.array(
        [probe_shape[0] // 2, probe_shape[1] // 2], dtype=np.float32
    )
    np.testing.assert_allclose(pos, expected)


def test_get_probe_positions_invalid_units(tiny_payload):
    a = Analyzer(tiny_payload)
    with pytest.raises(ValueError, match="units"):
        a.get_probe_positions(units="nm")


def test_get_probe_positions_invalid_target(tiny_payload):
    a = Analyzer(tiny_payload)
    with pytest.raises(ValueError, match="target"):
        a.get_probe_positions(target="sample")


# ---------------------------------------------------------------------------
# FOV bbox
# ---------------------------------------------------------------------------


def test_fov_bbox_matches_manual_min_max(tiny_payload):
    crop_pos = tiny_payload["model_attributes"]["crop_pos"]
    probe_shape = tiny_payload["optimizable_tensors"]["probe"].shape[-2:]
    centers = crop_pos + np.array([probe_shape[0] // 2, probe_shape[1] // 2])
    bbox = get_scanned_fov_bbox(crop_pos, probe_shape)
    assert bbox == (
        int(centers[:, 0].min()),
        int(centers[:, 0].max()),
        int(centers[:, 1].min()),
        int(centers[:, 1].max()),
    )


def test_apply_fov_none_is_noop(tiny_payload):
    arr = tiny_payload["optimizable_tensors"]["obja"]
    np.testing.assert_array_equal(apply_fov(arr, None), arr)


# ---------------------------------------------------------------------------
# Loss / keys / provenance
# ---------------------------------------------------------------------------


def test_loss_curves(tiny_payload):
    a = Analyzer(tiny_payload)
    lc = a.get_loss_curves()
    assert lc["niter"] == tiny_payload["niter"]
    np.testing.assert_allclose(lc["loss_iters"], tiny_payload["loss_iters"])


def test_list_keys_includes_optimizable_tensors(tiny_payload):
    a = Analyzer(tiny_payload)
    keys = a.get_keys()
    assert "optimizable_tensors.probe" in keys
    assert "model_attributes.crop_pos" in keys


def test_provenance_roundtrip(tiny_hdf5):
    path, _, prov = tiny_hdf5
    a = Analyzer(path)
    assert a.provenance == prov
    # Reads directly when given a path
    assert extract_provenance(path) == prov


def test_provenance_warns_when_from_dict(tiny_payload):
    a = Analyzer(tiny_payload)
    with pytest.warns(UserWarning, match="dict"):
        assert a.provenance is None
    # Subsequent access should be cached (no second warning)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # would raise if a warning fired
        assert a.provenance is None


def test_provenance_missing_attr_returns_none(tmp_path, tiny_payload):
    path = tmp_path / "no_prov.hdf5"
    save_dict_to_hdf5(tiny_payload, str(path))
    # No save_provenance_to_hdf5 call
    a = Analyzer(path)
    assert a.provenance is None


# ---------------------------------------------------------------------------
# build_model error path (can't actually build without measurements)
# ---------------------------------------------------------------------------


def test_build_model_missing_meas_raises_clear_error(tiny_payload):
    a = Analyzer(tiny_payload)
    with pytest.raises(FileNotFoundError, match="synthetic.raw"):
        a.build_model(device="cpu")


def test_forward_without_build_raises(tiny_payload):
    a = Analyzer(tiny_payload)
    with pytest.raises(RuntimeError, match="build_model"):
        a.forward([0])


def test_forward_defaults_to_no_grad(tiny_payload):
    import torch

    class DummyModel:
        def forward(self, indices, return_raw=False):
            x = torch.ones((), requires_grad=True)
            return x * 2

    a = Analyzer(tiny_payload)
    a._model = DummyModel()
    out = a.forward([0])
    assert not out.requires_grad
    out_grad = a.forward([0], enable_grad=True)
    assert out_grad.requires_grad


def test_plot_summary_without_build_raises(tiny_payload):
    a = Analyzer(tiny_payload)
    with pytest.raises(RuntimeError, match="build_model"):
        a.plot_summary([0])


# ---------------------------------------------------------------------------
# Plot adapters: smoke test (no display)
# ---------------------------------------------------------------------------


def test_plot_loss_smoke(tiny_payload):
    import matplotlib

    matplotlib.use("Agg")
    a = Analyzer(tiny_payload)
    fig = a.plot_loss(show_fig=False, pass_fig=True)
    assert fig is not None


def test_plot_probe_smoke(tiny_payload):
    import matplotlib

    matplotlib.use("Agg")
    a = Analyzer(tiny_payload)
    fig = a.plot_probe(space="fourier", show_fig=False, pass_fig=True)
    assert fig is not None


def test_plot_positions_smoke(tiny_payload):
    import matplotlib

    matplotlib.use("Agg")
    a = Analyzer(tiny_payload)
    fig = a.plot_positions(show_fig=False, pass_fig=True, show_arrow=False)
    assert fig is not None

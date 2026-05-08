import json

import h5py
import numpy as np
import torch

from ptyrad.io.load import load_ptyrad
from ptyrad.io.provenance import save_provenance_to_hdf5
from ptyrad.io.save import save_dict_to_hdf5


def test_save_dict_to_hdf5_and_load_ptyrad_roundtrip(tmp_path):
    path = tmp_path / "model_iter0001.hdf5"
    payload = {
        "ptyrad_version": "test",
        "optimizable_tensors": {
            "objp": torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3),
            "obja": np.ones((1, 1, 2, 3), dtype=np.float32),
            "probe": torch.ones((1, 2, 2), dtype=torch.complex64),
            "probe_pos_shifts": torch.zeros((4, 2), dtype=torch.float32),
        },
        "model_attributes": {
            "dx": torch.tensor(0.25),
            "crop_pos": np.zeros((4, 2), dtype=np.int32),
            "simu_Npix": 2,
        },
        "params": {
            "init_params": {"meas_params": {"path": "synthetic.raw", "key": None}},
            "recon_params": {"NITER": 1},
        },
        "optim_state_dict": {
            "state": {
                0: {
                    "step": torch.tensor(1),
                    "exp_avg": torch.zeros((2,), dtype=torch.float32),
                }
            },
            "param_groups": [{"params": [0], "lr": 0.001}],
        },
        "scheduler_state_dict": None,
        "loss_iters": [(1, 0.5)],
        "avg_losses": {"loss_single": np.float32(0.5)},
        "avg_iter_t": np.float64(0.12),
        "niter": 1,
        "indices": np.arange(4, dtype=np.int64),
        "batch_losses": {"loss_single": [0.5]},
        "notes": ["roundtrip", "strings"],
        "none_value": None,
    }

    save_dict_to_hdf5(payload, str(path))
    provenance = {"probe": [{"action": "Synthetic"}]}
    save_provenance_to_hdf5(path, json.dumps(provenance))

    loaded = load_ptyrad(str(path))

    for key in (
        "optimizable_tensors",
        "model_attributes",
        "params",
        "loss_iters",
        "avg_losses",
        "avg_iter_t",
        "niter",
    ):
        assert key in loaded

    np.testing.assert_array_equal(
        loaded["optimizable_tensors"]["objp"],
        payload["optimizable_tensors"]["objp"].numpy(),
    )
    np.testing.assert_array_equal(
        loaded["model_attributes"]["crop_pos"],
        payload["model_attributes"]["crop_pos"],
    )
    assert loaded["scheduler_state_dict"] is None
    assert loaded["none_value"] is None
    assert "0" in loaded["optim_state_dict"]["state"]

    with h5py.File(path, "r") as h5:
        assert "provenance_json" in h5.attrs
        assert json.loads(h5.attrs["provenance_json"]) == provenance


def test_roundtrip_critical_arrays_are_finite(tmp_path):
    path = tmp_path / "finite_model.hdf5"
    payload = {
        "optimizable_tensors": {
            "objp": np.ones((1, 1, 2, 2), dtype=np.float32),
            "obja": np.ones((1, 1, 2, 2), dtype=np.float32),
            "probe": np.ones((1, 2, 2), dtype=np.complex64),
        },
        "model_attributes": {"dx": 0.1},
        "params": {},
        "loss_iters": [(1, 1.0)],
        "avg_losses": {"loss_single": 1.0},
        "avg_iter_t": 0.01,
        "niter": 1,
    }
    save_dict_to_hdf5(payload, str(path))

    loaded = load_ptyrad(str(path))

    for tensor_name in ("objp", "obja", "probe"):
        arr = np.asarray(loaded["optimizable_tensors"][tensor_name])
        assert np.isfinite(arr).all()

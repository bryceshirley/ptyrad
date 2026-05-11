"""
User-facing wrapper around a PtyRAD ``model.hdf5`` output.

:class:`Analyzer` is a thin delegator: every ``get_*`` method is a 1-2
line call into the pure functions in :mod:`ptyrad.analysis.extract`, and
every plotting method forwards directly to the existing helpers in
:mod:`ptyrad.plotting`. Model rebuild is lazy via
:meth:`Analyzer.build_model`.

Typical use::

    from ptyrad.analysis import Analyzer

    a = Analyzer("path/to/model_iter0100.hdf5")
    obj   = a.get_object(fov="crop")          # complex (omode, Nz, Ny, Nx)
    probe = a.get_probe(space="fourier")      # complex (pmode, Ny, Nx)
    pos   = a.get_probe_positions(units="Ang")  # (N, 2) in Ångströms
    a.plot_dashboard()

    model = a.build_model(device="cpu")   # rebuild for forward passes
    dp    = a.forward([0, 1, 2, 3])
"""

from __future__ import annotations

import logging
import os
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Literal, Mapping

import numpy as np

from ptyrad.analysis.extract import (
    extract_keys,
    extract_loss_curves,
    extract_object,
    extract_object_amplitude,
    extract_object_phase,
    extract_probe,
    extract_probe_positions,
    extract_provenance,
    get_scanned_fov_bbox,
)
from ptyrad.io.hierarchy import get_nested
from ptyrad.io.load import load_ptyrad

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, overrides: Mapping[str, Any] | None) -> dict:
    """Recursively merge ``overrides`` into a deep copy of ``base`` (dict in, dict out)."""
    out = deepcopy(base)
    if not overrides:
        return out
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _history_list(value: Any) -> list:
    """Convert HDF5-restored history arrays back to mutable Python lists.

    The reconstruction loop appends ``(niter, value)`` pairs to
    ``model.loss_iters`` / ``model.dz_iters``. These survive the HDF5
    round-trip as ``(N, 2)`` ndarrays, which this helper re-zips to a
    list of tuples so the rebuilt model is plottable and can continue to
    append histories
    (``model.loss_iters.append(...)`` still works).
    """
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        if value.ndim == 0:
            return [value.item()]
        if value.ndim == 1:
            return value.tolist()
        return [tuple(row.tolist()) for row in value]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


def _history_defaultdict(value: Any) -> defaultdict:
    """Restore a dict-of-history-arrays as the ``defaultdict(list)`` form the model uses.

    Applies to ``lr_iters``, ``avg_tilt_iters``, ``convergence_iters``,
    which the reconstruction loop appends to by key (e.g.
    ``model.lr_iters['name'].append(value)``).
    """
    out = defaultdict(list)
    if isinstance(value, Mapping):
        for key, history in value.items():
            out[key] = _history_list(history)
    return out


def _restore_model_history(model, data: Mapping[str, Any]) -> None:
    """Copy saved iteration histories onto a freshly rebuilt model."""
    model.loss_iters = _history_list(data.get("loss_iters"))
    model.dz_iters = _history_list(data.get("dz_iters"))
    model.lr_iters = _history_defaultdict(data.get("lr_iters"))
    model.avg_tilt_iters = _history_defaultdict(data.get("avg_tilt_iters"))
    model.convergence_iters = _history_defaultdict(data.get("convergence_iters"))


class Analyzer:
    """Wrapper around a loaded PtyRAD ``model.hdf5`` reconstruction.

    Parameters
    ----------
    source : str | os.PathLike | dict
        Either a path to a ``.hdf5`` / ``.h5`` reconstruction output (the
        file is loaded via :func:`ptyrad.io.load.load_ptyrad`) or an
        already-loaded dict. When constructed from a dict the analyzer
        cannot retrieve HDF5 root attributes (provenance); that path
        returns ``None`` from :attr:`provenance` with a one-time
        :class:`UserWarning`.

    Notes
    -----
    Construction is cheap: only the HDF5 datasets/groups are loaded into a
    nested dict (provenance is deferred, the :class:`PtychoModel` is not
    built until :meth:`build_model` is called).

    The class is a *thin delegator*: every ``get_*`` method is a one-line
    forward to the corresponding pure function in
    :mod:`ptyrad.analysis.extract`,
    so users who prefer raw dicts can skip the class entirely and call
    the functions directly.
    """

    def __init__(self, source: str | os.PathLike | dict):
        if isinstance(source, Mapping):
            self._data: dict = dict(source)
            self._path: str | None = None
        else:
            path = os.fspath(source)
            self._data = load_ptyrad(path)
            self._path = path

        self._provenance: dict | None = None
        self._provenance_loaded = False
        self._model = None  # lazy

    # ------------------------------------------------------------------
    # Cheap accessors
    # ------------------------------------------------------------------

    @property
    def data(self) -> dict:
        return self._data

    @property
    def path(self) -> str | None:
        return self._path

    @property
    def opt_tensors(self) -> dict:
        return self._data["optimizable_tensors"]

    @property
    def model_attrs(self) -> dict:
        return self._data["model_attributes"]

    @property
    def params(self) -> dict:
        return self._data["params"]

    @property
    def provenance(self) -> dict | None:
        """Lazy-loaded provenance dict; ``None`` if unavailable.

        Reads the ``provenance_json`` root HDF5 attr via
        :func:`extract.extract_provenance` on first access and caches the
        result. Returns ``None`` (with a one-time :class:`UserWarning`)
        when the :class:`Analyzer` was constructed from an in-memory
        dict, since root attrs are not preserved by
        :func:`ptyrad.io.load.load_ptyrad`.
        """
        if self._provenance_loaded:
            return self._provenance
        self._provenance_loaded = True
        if self._path is None:
            warnings.warn(
                "Analyzer was constructed from an in-memory dict; provenance "
                "is only available when loading from a file path.",
                stacklevel=2,
            )
            return None
        self._provenance = extract_provenance(self._path)
        return self._provenance

    @property
    def dx(self) -> float:
        return float(np.asarray(self.model_attrs["dx"]))

    @property
    def dk(self) -> float:
        return float(np.asarray(self.model_attrs["dk"]))

    @property
    def lambd(self) -> float:
        return float(np.asarray(self.model_attrs["lambd"]))

    @property
    def probe_shape(self) -> tuple[int, int]:
        return tuple(np.asarray(self.opt_tensors["probe"]).shape[-2:])  # type: ignore[return-value]

    @property
    def pmode(self) -> int:
        return int(np.asarray(self.opt_tensors["probe"]).shape[0])

    @property
    def omode(self) -> int:
        return int(np.asarray(self.opt_tensors["objp"]).shape[0])

    @property
    def nslice(self) -> int:
        return int(np.asarray(self.opt_tensors["objp"]).shape[1])

    @property
    def is_multislice(self) -> bool:
        return self.nslice > 1

    @property
    def niter(self) -> int | None:
        return self._data.get("niter")

    @property
    def model(self):
        """The cached :class:`PtychoModel` instance built via :meth:`build_model`, or ``None``."""
        return self._model

    # ------------------------------------------------------------------
    # Getters (delegate to extract.*)
    # ------------------------------------------------------------------

    def get_object(
        self,
        fov: Literal["full", "crop"] = "full",
        *,
        as_torch: bool = False,
        device: str = "cpu",
    ):
        """Complex object ``obja * exp(1j * objp)``; see :func:`extract.extract_object`."""
        return extract_object(
            self._data, fov=fov, as_torch=as_torch, device=device
        )

    def get_object_amplitude(
        self,
        fov: Literal["full", "crop"] = "full",
        *,
        as_torch: bool = False,
        device: str = "cpu",
    ):
        """Object amplitude ``obja``; see :func:`extract.extract_object_amplitude`."""
        return extract_object_amplitude(
            self._data, fov=fov, as_torch=as_torch, device=device
        )

    def get_object_phase(
        self,
        fov: Literal["full", "crop"] = "full",
        *,
        as_torch: bool = False,
        device: str = "cpu",
    ):
        """Object phase ``objp``; see :func:`extract.extract_object_phase`."""
        return extract_object_phase(
            self._data, fov=fov, as_torch=as_torch, device=device
        )

    def get_probe(
        self,
        space: Literal["real", "fourier"] = "real",
        *,
        as_torch: bool = False,
        device: str = "cpu",
    ):
        """Complex probe modes; see :func:`extract.extract_probe`."""
        return extract_probe(self._data, space=space, as_torch=as_torch, device=device)

    def get_probe_positions(
        self,
        fov: Literal["full", "crop"] = "full",
        *,
        target: Literal["probe", "crop"] = "probe",
        units: Literal["px", "pixel", "Ang"] = "px",
        include_sub_px_shifts: bool = True,
        as_torch: bool = False,
        device: str = "cpu",
    ):
        """Probe-center or crop-window positions; see :func:`extract.extract_probe_positions`.

        ``include_sub_px_shifts`` controls whether the optimized
        ``probe_pos_shifts`` offsets are included. ``target='probe'`` returns
        probe centers; ``target='crop'`` returns crop-window top-left
        positions, which can be negative when combined with ``fov='crop'``.

        Power users who hold only a sub-dict (e.g. just
        ``optimizable_tensors`` from a streaming loader) should call the
        standalone :func:`extract.extract_probe_positions` with explicit
        ``model_attrs``.
        """
        return extract_probe_positions(
            self._data,
            fov=fov,
            target=target,
            units=units,
            include_sub_px_shifts=include_sub_px_shifts,
            as_torch=as_torch,
            device=device,
        )

    def get_loss_curves(self) -> dict:
        """Loss-history fields; see :func:`extract.extract_loss_curves`."""
        return extract_loss_curves(self._data)

    def get_keys(self) -> list[str]:
        """Flat dotted-key listing of the loaded data dict."""
        return extract_keys(self._data)

    def get_fov_bbox(self) -> tuple[int, int, int, int]:
        """Inclusive scanned-FOV bbox over probe-center positions.

        Returns ``(y_min, y_max, x_min, x_max)`` matching the slicing of
        the saved ``*_crop`` TIFFs.
        """
        return get_scanned_fov_bbox(
            np.asarray(self.model_attrs["crop_pos"]), self.probe_shape
        )

    # ------------------------------------------------------------------
    # Plot adapters (delegate to ptyrad.plotting.*)
    # ------------------------------------------------------------------

    def plot_loss(self, **kw):
        """Plot the loss curve via :func:`ptyrad.plotting.plot_loss_curves`."""
        from ptyrad.plotting.basic import plot_loss_curves

        return plot_loss_curves(self._data["loss_iters"], **kw)

    def plot_probe(
        self,
        space: Literal["real", "fourier"] = "real",
        amp_or_phase: Literal["amplitude", "phase"] = "amplitude",
        **kw,
    ):
        """Plot probe modes via :func:`ptyrad.plotting.plot_probe_modes`.

        ``space`` and ``amp_or_phase`` map directly to that function's
        ``real_or_fourier`` and ``amp_or_phase`` arguments.
        """
        from ptyrad.plotting.basic import plot_probe_modes

        return plot_probe_modes(
            opt_probe=np.asarray(self.opt_tensors["probe"]),
            amp_or_phase=amp_or_phase,
            real_or_fourier=space,
            **kw,
        )

    def plot_positions(
        self,
        fov: Literal["full", "crop"] = "full",
        *,
        target: Literal["probe", "crop"] = "probe",
        **kw,
    ):
        """Scatter scan positions via :func:`ptyrad.plotting.plot_scan_positions`.

        See :meth:`get_probe_positions` for the ``fov`` / ``target`` semantics.
        """
        from ptyrad.plotting.basic import plot_scan_positions

        return plot_scan_positions(
            pos=self.get_probe_positions(fov=fov, target=target), **kw
        )

    def plot_tilts(self, **kw):
        """Plot per-position object tilts via :func:`ptyrad.plotting.plot_obj_tilts`."""
        from ptyrad.plotting.basic import plot_obj_tilts

        return plot_obj_tilts(
            pos=self.get_probe_positions(),
            tilts=self.opt_tensors.get("obj_tilts"),
            **kw,
        )

    def plot_slice_thickness(self, **kw):
        """Plot slice-thickness evolution via :func:`ptyrad.plotting.plot_slice_thickness`."""
        from ptyrad.plotting.basic import plot_slice_thickness

        return plot_slice_thickness(self._data.get("dz_iters"), **kw)

    def plot_dashboard(self, **kw):
        """Convergence dashboard via :func:`ptyrad.plotting.plot_convergence_dashboard`.

        Pulls ``loss_iters``, ``lr_iters``, ``dz_iters``,
        ``avg_tilt_iters``, ``convergence_iters`` straight from the
        loaded dict — no rebuilt model required.
        """
        from ptyrad.plotting.basic import plot_convergence_dashboard

        return plot_convergence_dashboard(
            loss_iters=self._data.get("loss_iters"),
            lr_iters=self._data.get("lr_iters"),
            dz_iters=self._data.get("dz_iters"),
            avg_tilt_iters=self._data.get("avg_tilt_iters"),
            convergence_iters=self._data.get("convergence_iters"),
            **kw,
        )

    def plot_summary(self, indices, **kw):
        """Summary figure via :func:`ptyrad.plotting.plot_summary`.

        Requires a rebuilt model — call :meth:`build_model` first.
        ``output_path`` is read from ``kw`` (default ``'.'``).
        """
        from ptyrad.plotting.model import plot_summary

        self._require_model("plot_summary")
        return plot_summary(
            output_path=kw.pop("output_path", "."),
            model=self._model,
            niter=self.niter,
            indices=np.asarray(indices),
            init_variables=self._init_variables,
            **kw,
        )

    def plot_forward_pass(self, indices, *, dp_power: float = 0.5, **kw):
        """Forward-pass diagnostic via :func:`ptyrad.plotting.plot_forward_pass`.

        Requires a rebuilt model — call :meth:`build_model` first.
        """
        from ptyrad.plotting.model import plot_forward_pass

        self._require_model("plot_forward_pass")
        return plot_forward_pass(
            model=self._model,
            indices=np.asarray(indices),
            dp_power=dp_power,
            **kw,
        )

    # ------------------------------------------------------------------
    # Lazy model rebuild
    # ------------------------------------------------------------------

    def build_model(
        self,
        device: str = "cuda",
        overrides: Mapping[str, Any] | None = None,
    ):
        """Rebuild a :class:`PtychoModel` from the saved ``params`` and weights.

        Runs :class:`ptyrad.init.initializer.Initializer` against
        ``data['params']['init_params']`` to construct the
        ``init_variables`` dict, instantiates the model, copies the saved
        ``optimizable_tensors`` (``obja``, ``objp``, ``probe``,
        ``probe_pos_shifts``, ``obj_tilts``, ``slice_thickness``) into the
        fresh model, and finally restores the iteration histories
        (``loss_iters``, ``dz_iters``, ``lr_iters``, ``avg_tilt_iters``,
        ``convergence_iters``) so the rebuilt model is plottable via
        :meth:`plot_summary` / :meth:`plot_forward_pass` and suitable for
        forward-pass diagnostics.

        The model and its ``init_variables`` are cached on ``self`` after a
        successful build.

        Parameters
        ----------
        device
            Torch device string (e.g. ``'cpu'``, ``'cuda'``, ``'cuda:0'``).
        overrides
            Dict deep-merged into ``params`` before initialization. The
            common use is to redirect the measurement file path when the
            original measurement data has moved::

                a.build_model(
                    overrides={'init_params': {'meas_params': {'path': '/new/path.raw'}}}
                )

        Raises
        ------
        FileNotFoundError
            If ``init_params['meas_params']['path']`` does not resolve.
            The error names the missing path explicitly rather than
            failing deep in the dataloader.
        """
        import torch

        from ptyrad.core.models.ptycho import PtychoModel
        from ptyrad.init.initializer import Initializer

        params = _deep_merge(self.params, overrides)
        init_params = params["init_params"]
        meas_path = get_nested(
            init_params, key="meas_params.path", safe=True, default=None
        )
        if meas_path is not None and not os.path.exists(meas_path):
            raise FileNotFoundError(
                f"build_model: measurement file referenced by params['init_params']"
                f"['meas_params']['path'] does not exist: {meas_path}. "
                "Pass overrides={'init_params': {'meas_params': {'path': '<new>'}}} "
                "to redirect."
            )

        seed = self._data.get("random_seed")
        init = Initializer(init_params, seed=seed).init_all()
        model = PtychoModel(init.init_variables, params["model_params"], device=device)

        # Copy saved weights into the freshly initialized model
        opt = self.opt_tensors
        with torch.no_grad():
            model.opt_obja.copy_(
                torch.as_tensor(np.asarray(opt["obja"]), dtype=torch.float32, device=device)
            )
            model.opt_objp.copy_(
                torch.as_tensor(np.asarray(opt["objp"]), dtype=torch.float32, device=device)
            )
            probe_c = torch.as_tensor(
                np.asarray(opt["probe"]), dtype=torch.complex64, device=device
            )
            model.opt_probe.copy_(torch.view_as_real(probe_c))
            if "probe_pos_shifts" in opt:
                model.opt_probe_pos_shifts.copy_(
                    torch.as_tensor(
                        np.asarray(opt["probe_pos_shifts"]),
                        dtype=torch.float32,
                        device=device,
                    )
                )
            if "obj_tilts" in opt:
                model.opt_obj_tilts.copy_(
                    torch.as_tensor(
                        np.asarray(opt["obj_tilts"]),
                        dtype=torch.float32,
                        device=device,
                    )
                )
            if "slice_thickness" in opt:
                model.opt_slice_thickness.copy_(
                    torch.as_tensor(
                        np.asarray(opt["slice_thickness"]),
                        dtype=torch.float32,
                        device=device,
                    )
                )

        _restore_model_history(model, self._data)
        self._model = model
        self._init_variables = init.init_variables
        return model

    def forward(
        self,
        indices,
        *,
        return_raw: bool = False,
        enable_grad: bool = False,
    ):
        """Run a forward pass on the rebuilt model.

        Thin wrapper around ``self.model.forward(indices, return_raw=...)``.
        Requires :meth:`build_model` to have been called first; ``indices``
        is coerced to ``np.ndarray`` before being passed through.

        By default this runs under ``torch.no_grad()`` because analysis and
        plotting calls usually do not need autograd graphs. Set
        ``enable_grad=True`` to record gradients for offline custom losses
        or refinement experiments.
        """
        import torch

        self._require_model("forward")
        grad_context = torch.enable_grad() if enable_grad else torch.no_grad()
        with grad_context:
            return self._model.forward(np.asarray(indices), return_raw=return_raw)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_model(self, method: str) -> None:
        if self._model is None:
            raise RuntimeError(
                f"Analyzer.{method}() requires a built model. "
                "Call Analyzer.build_model(device=...) first."
            )

    def __repr__(self) -> str:
        src = self._path if self._path else "<in-memory dict>"
        return (
            f"Analyzer(source={src!r}, niter={self.niter}, "
            f"obj=(omode={self.omode}, nslice={self.nslice}), "
            f"probe=(pmode={self.pmode}, shape={self.probe_shape}))"
        )

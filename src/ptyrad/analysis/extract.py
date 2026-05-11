"""
Pure extraction functions for PtyRAD ``model.hdf5`` outputs.

Every public extraction worker accepts either the full loaded dict (from
:func:`ptyrad.io.load.load_ptyrad`) or pre-sliced sub-dicts
(``opt_tensors``, ``model_attrs``) so callers can use these helpers
without instantiating :class:`ptyrad.analysis.Analyzer`.

Geometry / FOV helpers (``get_probe_center_positions``,
``get_scanned_fov_bbox``, ``apply_fov``) are re-exported on purpose:
they encode the single most error-prone convention in the saved file
(``crop_pos`` is the top-left of the probe window, not the center).
The public position extractor exposes both conventions explicitly via
``target='probe'`` and ``target='crop'``.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Mapping

import numpy as np

from ptyrad.io.hierarchy import get_nested, list_nested_keys


# ---------------------------------------------------------------------------
# Geometry / FOV helpers (single source of truth)
# ---------------------------------------------------------------------------


def _add_shifts(pos: np.ndarray, shifts: np.ndarray | None) -> np.ndarray:
    """Add ``shifts`` to ``pos`` after a shape check. Returns ``pos`` unchanged if shifts is None."""
    if shifts is None:
        return pos
    if shifts.shape != pos.shape:
        raise ValueError(
            f"probe_pos_shifts shape {shifts.shape} does not match positions shape {pos.shape}"
        )
    return pos + shifts


def get_probe_center_positions(
    crop_pos: np.ndarray,
    probe_shape: tuple[int, int],
    probe_pos_shifts: np.ndarray | None = None,
) -> np.ndarray:
    """Convert top-left ``crop_pos`` to probe-center positions.

    The ``crop_pos`` stored in ``model.hdf5`` is the top-left corner of the
    probe window in object pixels — that is the form consumed by
    :meth:`PtychoModel.get_obj_patches`, not the visual probe center. The
    probe center sits ``(Hp // 2, Wp // 2)`` further in.

    Parameters
    ----------
    crop_pos : array-like, shape (N, 2), int
        Top-left ``[y, x]`` of each probe window in object pixels.
    probe_shape : tuple
        Probe shape; only the last two dims ``(Hp, Wp)`` are used.
    probe_pos_shifts : array-like, shape (N, 2), optional
        Optimized sub-pixel offsets; added to the center positions when given.

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)``, ``float32``, ``[y, x]`` order, in object pixels.
    """
    crop_pos = np.asarray(crop_pos)
    Hp, Wp = int(probe_shape[-2]), int(probe_shape[-1])
    offset = np.array([Hp // 2, Wp // 2], dtype=np.float32)
    pos = crop_pos.astype(np.float32) + offset
    shifts_arr = (
        np.asarray(probe_pos_shifts, dtype=np.float32)
        if probe_pos_shifts is not None
        else None
    )
    return _add_shifts(pos, shifts_arr)


def get_scanned_fov_bbox(
    crop_pos: np.ndarray, probe_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Inclusive ``(y_min, y_max, x_min, x_max)`` of probe-center positions.

    The bbox is computed from the integer probe-center positions
    (``crop_pos + probe_shape // 2``) without ``probe_pos_shifts`` — matching
    the bbox math in :func:`ptyrad.io.save.save_results` so the result lines
    up exactly with the saved ``*_crop`` TIFFs (sliced as
    ``[y_min:y_max + 1, x_min:x_max + 1]``).
    """
    centers = get_probe_center_positions(crop_pos, probe_shape).astype(np.int64)
    y_min, y_max = int(centers[:, 0].min()), int(centers[:, 0].max())
    x_min, x_max = int(centers[:, 1].min()), int(centers[:, 1].max())
    return y_min, y_max, x_min, x_max


def apply_fov(arr: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    """Crop the last two dims of ``arr`` to an inclusive ``bbox``.

    ``bbox`` is ``(y_min, y_max, x_min, x_max)`` and both bounds are
    inclusive. Returns ``arr`` unchanged when ``bbox is None``.
    """
    if bbox is None:
        return arr
    y_min, y_max, x_min, x_max = bbox
    return arr[..., y_min : y_max + 1, x_min : x_max + 1]


# ---------------------------------------------------------------------------
# Internal helpers (private to this module)
# ---------------------------------------------------------------------------


def _resolve_opt_tensors(data_or_opt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either the full data dict or the ``optimizable_tensors`` sub-dict."""
    if "optimizable_tensors" in data_or_opt:
        return data_or_opt["optimizable_tensors"]
    return data_or_opt


def _resolve_model_attrs(
    data_or_opt: Mapping[str, Any], model_attrs: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    if model_attrs is not None:
        return model_attrs
    if isinstance(data_or_opt, Mapping) and "model_attributes" in data_or_opt:
        return data_or_opt["model_attributes"]
    return None


def _probe_shape_from(
    opt_tensors: Mapping[str, Any], probe_shape: tuple[int, int] | None
) -> tuple[int, int]:
    if probe_shape is not None:
        return tuple(probe_shape)  # type: ignore[return-value]
    probe = opt_tensors.get("probe")
    if probe is None:
        raise KeyError("opt_tensors has no 'probe' to infer probe_shape from.")
    return tuple(probe.shape[-2:])  # type: ignore[return-value]


def _crop_bbox(
    opt_tensors: Mapping[str, Any],
    model_attrs: Mapping[str, Any] | None,
    probe_shape: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    if model_attrs is None or "crop_pos" not in model_attrs:
        raise ValueError(
            "fov='crop' requires model_attrs with 'crop_pos'. "
            "Pass the full data dict or supply model_attrs explicitly."
        )
    return get_scanned_fov_bbox(
        np.asarray(model_attrs["crop_pos"]),
        _probe_shape_from(opt_tensors, probe_shape),
    )


def _to_torch(arr: np.ndarray, device: str):
    import torch

    return torch.as_tensor(arr, device=device)


# ---------------------------------------------------------------------------
# Object extractors
# ---------------------------------------------------------------------------


def extract_object(
    data_or_opt: Mapping[str, Any],
    *,
    fov: Literal["full", "crop"] = "full",
    model_attrs: Mapping[str, Any] | None = None,
    probe_shape: tuple[int, int] | None = None,
    as_torch: bool = False,
    device: str = "cpu",
):
    """Return the complex object ``obja * exp(1j * objp)``.

    Shape ``(omode, Nz, Ny, Nx)``, ``complex64``. The 4D shape is preserved
    even for single-slice runs (``Nz == 1``) so downstream code can branch on
    ``Nz`` directly; squeeze yourself if you need 2D.

    ``fov='crop'`` clips to the scanned-FOV bbox (see
    :func:`get_scanned_fov_bbox`); it requires ``crop_pos`` from
    ``model_attrs``. ``probe_shape`` is auto-inferred from
    ``opt_tensors['probe']`` and only needs to be passed when callers
    have stripped the probe out of ``opt_tensors``.
    """
    opt = _resolve_opt_tensors(data_or_opt)
    obja = np.asarray(opt["obja"])
    objp = np.asarray(opt["objp"])
    obj = (obja * np.exp(1j * objp)).astype(np.complex64)
    if fov == "crop":
        attrs = _resolve_model_attrs(data_or_opt, model_attrs)
        obj = apply_fov(obj, _crop_bbox(opt, attrs, probe_shape))
    elif fov != "full":
        raise ValueError(f"fov must be 'full' or 'crop', got {fov!r}")
    return _to_torch(obj, device) if as_torch else obj


def extract_object_amplitude(
    data_or_opt: Mapping[str, Any],
    *,
    fov: Literal["full", "crop"] = "full",
    model_attrs: Mapping[str, Any] | None = None,
    probe_shape: tuple[int, int] | None = None,
    as_torch: bool = False,
    device: str = "cpu",
):
    """Return the saved object amplitude ``obja`` (no recomposition).

    Shape ``(omode, Nz, Ny, Nx)``, ``float32``. ``fov`` behaves the same as
    :func:`extract_object`.
    """
    opt = _resolve_opt_tensors(data_or_opt)
    arr = np.asarray(opt["obja"], dtype=np.float32)
    if fov == "crop":
        attrs = _resolve_model_attrs(data_or_opt, model_attrs)
        arr = apply_fov(arr, _crop_bbox(opt, attrs, probe_shape))
    elif fov != "full":
        raise ValueError(f"fov must be 'full' or 'crop', got {fov!r}")
    return _to_torch(arr, device) if as_torch else arr


def extract_object_phase(
    data_or_opt: Mapping[str, Any],
    *,
    fov: Literal["full", "crop"] = "full",
    model_attrs: Mapping[str, Any] | None = None,
    probe_shape: tuple[int, int] | None = None,
    as_torch: bool = False,
    device: str = "cpu",
):
    """Return the saved object phase ``objp`` (no recomposition).

    Shape ``(omode, Nz, Ny, Nx)``, ``float32``. ``fov`` behaves the same as
    :func:`extract_object`. Sign convention follows :class:`PtychoModel`: the
    complex object is ``obja * exp(1j * objp)``; no sign flip is applied
    here.
    """
    opt = _resolve_opt_tensors(data_or_opt)
    arr = np.asarray(opt["objp"], dtype=np.float32)
    if fov == "crop":
        attrs = _resolve_model_attrs(data_or_opt, model_attrs)
        arr = apply_fov(arr, _crop_bbox(opt, attrs, probe_shape))
    elif fov != "full":
        raise ValueError(f"fov must be 'full' or 'crop', got {fov!r}")
    return _to_torch(arr, device) if as_torch else arr


# ---------------------------------------------------------------------------
# Probe extractor
# ---------------------------------------------------------------------------


def extract_probe(
    data_or_opt: Mapping[str, Any],
    *,
    space: Literal["real", "fourier"] = "real",
    as_torch: bool = False,
    device: str = "cpu",
):
    """Return the probe modes. Shape ``(pmode, Ny, Nx)``, ``complex64``.

    ``space='real'`` returns the stored real-space complex wavefunction
    (the form held in ``optimizable_tensors['probe']`` after the
    ``view_as_complex`` post-process at save time).

    ``space='fourier'`` returns
    ``fftshift(fft2(ifftshift(probe), norm='ortho'))`` along the last two
    axes. The fftshift sandwich matches
    :func:`ptyrad.plotting.plot_probe_modes` exactly so amplitudes and
    phases line up between getters and plotters; in particular the
    pre-fftshift to the corner avoids the checkerboard-phase artifact
    that a plain ``fft2`` would produce.
    """
    opt = _resolve_opt_tensors(data_or_opt)
    probe = np.asarray(opt["probe"]).astype(np.complex64)
    if space == "real":
        out = probe
    elif space == "fourier":
        from numpy.fft import fft2, fftshift, ifftshift

        out = fftshift(
            fft2(ifftshift(probe, axes=(-2, -1)), norm="ortho"),
            axes=(-2, -1),
        ).astype(np.complex64)
    else:
        raise ValueError(f"space must be 'real' or 'fourier', got {space!r}")
    return _to_torch(out, device) if as_torch else out


# ---------------------------------------------------------------------------
# Position extractor
# ---------------------------------------------------------------------------


def extract_probe_positions(
    data_or_opt: Mapping[str, Any],
    model_attrs: Mapping[str, Any] | None = None,
    *,
    fov: Literal["full", "crop"] = "full",
    target: Literal["probe", "crop"] = "probe",
    units: Literal["px", "pixel", "Ang"] = "px",
    include_sub_px_shifts: bool = True,
    as_torch: bool = False,
    device: str = "cpu",
):
    """Return positions, shape ``(N, 2)`` as ``[y, x]``, ``float32``.

    Parameters
    ----------
    data_or_opt
        Full data dict from :func:`ptyrad.io.load.load_ptyrad`, or the
        ``optimizable_tensors`` sub-dict. ``model_attrs`` must be supplied
        when the latter is passed.
    model_attrs
        Optional explicit ``model_attributes`` sub-dict. Must contain
        ``crop_pos`` (always) and ``dx`` (only when ``units='Ang'``).
    fov
        ``'full'`` returns positions in the full saved-object coordinate
        frame. ``'crop'`` subtracts the **probe-center** scanned-FOV bbox
        top-left (see :func:`get_scanned_fov_bbox`) so that, regardless of
        ``target``, the returned positions live in the same coordinate
        frame as the array returned by ``extract_object(fov='crop')``. Because
        that cropped FOV is anchored on probe centers, ``target='crop'`` can
        produce negative local coordinates near the top/left edge.
    target
        ``'probe'`` (default) returns probe-center positions
        (``crop_pos + (Hp // 2, Wp // 2)``). ``'crop'`` returns the
        top-left crop-window positions used by
        :meth:`PtychoModel.get_obj_patches`. The two differ by exactly
        ``(Hp // 2, Wp // 2)`` and that offset survives ``fov='crop'``
        because the same bbox is subtracted in both cases.
    units
        ``'px'`` / ``'pixel'`` returns object-space pixels. ``'Ang'``
        multiplies by ``model_attrs['dx']`` to return Ångströms.
    include_sub_px_shifts
        When ``True`` (default), add ``opt_tensors['probe_pos_shifts']``
        if present. These are the optimized sub-pixel offsets relative to
        the integer ``crop_pos`` grid. Older saves without the key silently
        fall back to zero shifts.
    """
    opt = _resolve_opt_tensors(data_or_opt)
    attrs = _resolve_model_attrs(data_or_opt, model_attrs)
    if attrs is None or "crop_pos" not in attrs:
        raise ValueError(
            "extract_probe_positions requires model_attrs with 'crop_pos'. "
            "Pass the full data dict or supply model_attrs explicitly."
        )

    crop_pos = np.asarray(attrs["crop_pos"])
    probe_shape = _probe_shape_from(opt, None)
    shifts = (
        np.asarray(opt["probe_pos_shifts"], dtype=np.float32)
        if include_sub_px_shifts and "probe_pos_shifts" in opt
        else None
    )

    if target == "probe":
        pos = get_probe_center_positions(crop_pos, probe_shape, shifts)
    elif target == "crop":
        pos = _add_shifts(crop_pos.astype(np.float32), shifts)
    else:
        raise ValueError(f"target must be 'probe' or 'crop', got {target!r}")

    if fov == "crop":
        y_min, _, x_min, _ = get_scanned_fov_bbox(crop_pos, probe_shape)
        pos = pos - np.array([y_min, x_min], dtype=np.float32)
    elif fov != "full":
        raise ValueError(f"fov must be 'full' or 'crop', got {fov!r}")

    if units == "Ang":
        dx = attrs.get("dx")
        if dx is None:
            raise ValueError("units='Ang' requires 'dx' in model_attrs.")
        pos = pos * float(np.asarray(dx))
    elif units not in ("px", "pixel"):
        raise ValueError(f"units must be 'px', 'pixel', or 'Ang', got {units!r}")

    return _to_torch(pos, device) if as_torch else pos


# ---------------------------------------------------------------------------
# Misc extractors
# ---------------------------------------------------------------------------


def extract_loss_curves(data: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the loss-history fields from a loaded data dict.

    Returns a ``dict`` with these keys (missing fields become ``None``):

    - ``loss_iters`` — ``(niter, 2)`` ``float64`` ndarray of
      ``(iter_number, total_loss)`` pairs.
    - ``batch_losses`` — ``dict[str, list[float]]`` of per-loss batch
      values from the most recent iteration.
    - ``avg_losses`` — ``dict[str, float]`` of averaged batch losses.
    - ``niter`` — final iteration number (``int``).
    """
    return {
        "loss_iters": data.get("loss_iters"),
        "batch_losses": data.get("batch_losses"),
        "avg_losses": data.get("avg_losses"),
        "niter": data.get("niter"),
    }


def extract_provenance(path: str | os.PathLike) -> dict | None:
    """Read the ``provenance_json`` root attribute from an HDF5 file.

    The provenance JSON is stored as a root-level HDF5 attribute, not as a
    dataset, so :func:`ptyrad.io.load.load_ptyrad` silently drops it. This
    helper opens the file directly via
    :func:`ptyrad.io.provenance.load_provenance_from_h5` and returns the
    parsed dict (with keys like ``'probe'``, ``'obj'``, ``'pos'``,
    ``'tilt'``). Returns ``None`` when the attribute is missing or the
    JSON parse fails.
    """
    from ptyrad.io.provenance import load_provenance_from_h5

    prov = load_provenance_from_h5(str(path))
    return prov if prov else None


def extract_keys(data: Mapping[str, Any], delimiter: str = ".") -> list[str]:
    """Return a flat dotted-key listing of a loaded data dict.

    Thin wrapper over :func:`ptyrad.io.hierarchy.list_nested_keys` that
    accepts a nested ``dict`` (the form returned by ``load_ptyrad``) and
    yields keys like ``'optimizable_tensors.probe'``,
    ``'model_attributes.crop_pos'``.
    """
    return list_nested_keys(data, delimiter=delimiter)


# Re-export ``get_nested`` so callers can resolve arbitrary paths without
# pulling in the ``io.hierarchy`` module separately.
__all__ = [
    "apply_fov",
    "extract_keys",
    "extract_loss_curves",
    "extract_object",
    "extract_object_amplitude",
    "extract_object_phase",
    "extract_probe",
    "extract_probe_positions",
    "extract_provenance",
    "get_nested",
    "get_probe_center_positions",
    "get_scanned_fov_bbox",
]

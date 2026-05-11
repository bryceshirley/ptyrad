"""
User-facing analysis of PtyRAD ``model.hdf5`` outputs.

The main entry point is :class:`Analyzer`. The module-level ``extract_*``
functions in :mod:`ptyrad.analysis.extract` are re-exported here so callers
can work with a raw dict from :func:`ptyrad.io.load.load_ptyrad` without
instantiating the class — useful when only a sub-dict is on hand or when
processing many files in a streaming loop.

The geometry helpers (:func:`get_probe_center_positions`,
:func:`get_scanned_fov_bbox`, :func:`apply_fov`) are public on purpose:
they encode the top-left-vs-center ``crop_pos`` convention that is the
most error-prone part of the saved file format.
"""

from .analyzer import Analyzer
from .extract import (
    apply_fov,
    extract_keys,
    extract_loss_curves,
    extract_object,
    extract_object_amplitude,
    extract_object_phase,
    extract_probe,
    extract_probe_positions,
    extract_provenance,
    get_probe_center_positions,
    get_scanned_fov_bbox,
)

__all__ = [
    "Analyzer",
    "apply_fov",
    "extract_keys",
    "extract_loss_curves",
    "extract_object",
    "extract_object_amplitude",
    "extract_object_phase",
    "extract_probe",
    "extract_probe_positions",
    "extract_provenance",
    "get_probe_center_positions",
    "get_scanned_fov_bbox",
]

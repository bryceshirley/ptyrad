"""
ConvergenceMonitor: periodic convergence metric tracking for optimizable tensors.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class ConvergenceMonitor:
    """
    Tracks convergence of optimizable tensors during ptychographic reconstruction.

    Takes periodic snapshots of tracked tensors and computes the iter-to-iter change
    (relative to the previous snapshot) at each snapshot.

    Tracked tensors: ``obja``, ``objp``, ``probe``, ``probe_pos_shifts``.
    ``slice_thickness`` and ``obj_tilts`` are excluded — they are already tracked every iteration
    via ``model.dz_iters`` and ``model.avg_tilt_iters`` and fed directly to the dashboard.

    For ``obja`` and ``objp``, metrics are computed on the ROI crop (scanned area bounding box)
    only. ``obja`` is transformed as ``1 - obja`` so vacuum → 0 and material → >0. Two scalars
    are stored per tensor per step using percentile masking on the current snapshot: a background
    metric (pixels below ``p_low``) and a signal metric (pixels above ``p_high``). The results
    are stored under keys ``obja_bg``, ``obja_fg``, ``objp_bg``, ``objp_fg``.

    For ``probe``, the fractional intensity change (``sum|ΔI| / sum(I_prev)``) of mode-summed probe intensity is tracked.
    For ``probe_pos_shifts``, the RMS displacement change in Å is tracked.

    Results are stored in ``model.convergence_iters`` as a dict of lists of 2-tuples
    ``(niter, value)``.

    Args:
        params: Parsed ``ConvergenceMonitorParams`` dict (with keys ``tensors``,
            ``every_n_iters``, ``percentile_range``).
        model: ``PtychoModel`` instance. An initial snapshot is taken during ``__init__``
            so the baseline is the state before the first optimizer update.
    """

    # Maps tensor name → metric type used by _compute_metric (obja/objp use _compute_bg_fg_metric)
    _METRIC_TYPE = {
        "probe":            "norm_l1",
        "probe_pos_shifts": "rms",
    }

    def __init__(self, params: dict, model) -> None:
        self._tensors: list          = list(params["tensors"])
        self._every_n: Optional[int] = params.get("every_n_iters")
        self._percentile_range: list = list(params.get("percentile_range", [15.0, 85.0]))

        self._dx: float  = float(model.dx)   # pixel size [Å]; used to convert probe_pos_shifts to Å

        # Precompute scanned-area ROI from all scan positions — matches save.py crop convention.
        # crop_pos stores top-left corners of probe patches; adding probe_half gives center positions.
        with torch.no_grad():
            probe_half = torch.tensor(model.get_complex_probe_view().shape[-2:]).cpu() // 2
            centers    = model.crop_pos.cpu() + probe_half  # (N_scans, 2)
        self._y_min = int(centers[:, 0].min().item())
        self._y_max = int(centers[:, 0].max().item())
        self._x_min = int(centers[:, 1].min().item())
        self._x_max = int(centers[:, 1].max().item())

        self._prev: dict = {}

        for name in self._tensors:
            snaps = self._snapshot(model, name)
            for key, tensor in snaps.items():
                self._prev[key] = tensor.clone()

        logger.info(
            f"### Creating ConvergenceMonitor with {params} ### ")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_step(self, niter: int, save_iters: Optional[int]) -> bool:
        """Return True if a convergence snapshot should be taken at ``niter``."""
        if self._every_n is not None:
            return niter % self._every_n == 0
        if save_iters is not None:
            return niter % save_iters == 0
        return False

    def step(self, model, niter: int) -> None:
        """Compute and record convergence metrics for all tracked tensors."""
        for name in self._tensors:
            snaps = self._snapshot(model, name)
            for key, current in snaps.items():
                if key in ("obja", "objp"):
                    bg_change, fg_change = self._compute_bg_fg_metric(
                        current, self._prev[key], self._percentile_range
                    )
                    model.convergence_iters[f"{key}_bg"].append((niter, bg_change))
                    model.convergence_iters[f"{key}_fg"].append((niter, fg_change))
                else:
                    metric_type = self._METRIC_TYPE[key]
                    iter_change = self._compute_metric(current, self._prev[key], metric_type)
                    if key == "probe_pos_shifts":
                        iter_change *= self._dx
                    model.convergence_iters[key].append((niter, iter_change))
                self._prev[key] = current.clone()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _snapshot(self, model, name: str) -> dict:
        """
        Return a dict of detached CPU float32 tensors for the given parameter name.

        For 'obja': returns ``{"obja": 1 - roi}`` where roi is the scanned-area crop of obja.
        For 'objp': returns ``{"objp": roi}`` (scanned-area crop, no transform).
        For 'probe': returns ``{"probe": probe_intensity}``.
        For others: returns ``{name: tensor}``.
        """
        if name == "obja":
            obj = model.optimizable_tensors["obja"].detach().float()
            roi = obj[:, :, self._y_min - 1:self._y_max, self._x_min - 1:self._x_max]
            return {"obja": (1.0 - roi).cpu()}
        if name == "objp":
            obj = model.optimizable_tensors["objp"].detach().float()
            roi = obj[:, :, self._y_min - 1:self._y_max, self._x_min - 1:self._x_max]
            return {"objp": roi.cpu()}
        if name == "probe":
            probe_c = model.get_complex_probe_view().detach()
            probe_int = probe_c.abs().square().sum(0).cpu()
            return {"probe": probe_int}
        return {name: model.optimizable_tensors[name].detach().float().cpu()}

    @staticmethod
    def _compute_metric(current: torch.Tensor, reference: torch.Tensor, metric_type: str) -> float:
        """Compute a scalar convergence metric between current and reference tensors."""
        diff = current - reference
        if metric_type == "norm_l1":
            # Fractional intensity change: total absolute change as fraction of total reference intensity
            return (diff.abs().sum() / (reference.sum() + 1e-8)).item()
        if metric_type == "rms":
            # Per-position RMS displacement magnitude (for (N, 2) probe_pos_shifts tensors)
            return diff.pow(2).sum(dim=-1).mean().sqrt().item()
        raise ValueError(f"Unknown metric_type: {metric_type!r}")

    @staticmethod
    def _compute_bg_fg_metric(
        current: torch.Tensor,
        reference: torch.Tensor,
        percentile_range: list,
    ) -> tuple:
        """Compute background and signal mean absolute change using percentile masking.

        Percentiles are computed on the flattened current snapshot. Background mask selects
        pixels below p_low (vacuum region); signal mask selects pixels above p_high (material).
        Returns (bg_change, fg_change) as floats.
        """
        flat_curr = current.flatten()
        p_lo = torch.quantile(flat_curr, percentile_range[0] / 100.0).item()
        p_hi = torch.quantile(flat_curr, percentile_range[1] / 100.0).item()
        flat_diff = (current - reference).abs().flatten()
        bg_mask = flat_curr < p_lo
        fg_mask = flat_curr > p_hi
        bg_change = flat_diff[bg_mask].mean().item() if bg_mask.any() else 0.0
        fg_change = flat_diff[fg_mask].mean().item() if fg_mask.any() else 0.0
        
        return bg_change, fg_change


def create_convergence_monitor(convergence_monitor_params, model) -> Optional[ConvergenceMonitor]:
    """
    Factory that returns a ``ConvergenceMonitor`` when ``convergence_monitor_params`` is not None.

    Args:
        convergence_monitor_params: Parsed dict from ``ReconParams.convergence_monitor``,
            or ``None`` to disable monitoring.
        model: ``PtychoModel`` instance used for the initial snapshot.

    Returns:
        A configured ``ConvergenceMonitor``, or None if params is None.
    """
    if convergence_monitor_params is None:
        return None
    return ConvergenceMonitor(convergence_monitor_params, model)

"""
ConvergenceMonitor: periodic convergence metric tracking for optimizable tensors.
"""

import logging
from typing import Optional

import torch

from ptyrad.runtime.logging import report

logger = logging.getLogger(__name__)


class ConvergenceMonitor:
    """
    Tracks convergence of optimizable tensors during ptychographic reconstruction.

    Takes periodic snapshots of tracked tensors and computes the iter-to-iter change
    (relative to the previous snapshot) at each snapshot.

    Tracked tensors: ``obja``, ``objp``, ``probe`` (→ ``probe_amp`` only), ``probe_pos_shifts``.
    ``slice_thickness`` and ``obj_tilts`` are excluded — they are already tracked every iteration
    via ``model.dz_iters`` and ``model.avg_tilt_iters`` and fed directly to the dashboard.

    Results are stored in ``model.convergence_iters`` as a dict of lists of 2-tuples
    ``(niter, iter_change)``.

    Args:
        params: Parsed ``ConvergenceMonitorParams`` dict (with keys ``tensors``,
            ``every_n_iters``, ``threshold``).
        model: ``PtychoModel`` instance. An initial snapshot is taken during ``__init__``
            so the baseline is the state before the first optimizer update.
    """

    # Maps tensor name → metric type used by _compute_metric
    _METRIC_TYPE = {
        "obja":             "rel_frob",
        "objp":             "rel_frob",
        "probe":            "rel_frob",
        "probe_pos_shifts": "rms",
    }

    def __init__(self, params: dict, model) -> None:
        self._tensors: list       = list(params["tensors"])
        self._every_n: Optional[int] = params.get("every_n_iters")
        self._threshold: float    = params["threshold"]
        self._converged: set      = set()

        model.convergence_threshold = self._threshold
        # TODO: extend ConvergenceMonitorParams with per-tensor threshold dict so each tensor
        # can have its own convergence criterion, and expose them as reference lines in the dashboard.

        self._dx: float  = float(model.dx)   # pixel size [Å]; used to convert probe_pos_shifts to Å
        self._prev: dict = {}

        for name in self._tensors:
            snaps = self._snapshot(model, name)
            for key, tensor in snaps.items():
                self._prev[key] = tensor.clone()

        report(
            f"ConvergenceMonitor initialized — tracking: {self._tensors}, "
            f"threshold: {self._threshold:.1e}",
            verbosity="INFO",
        )

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
                metric_type = self._METRIC_TYPE[key]
                iter_change = self._compute_metric(current, self._prev[key], metric_type)
                if key == "probe_pos_shifts":
                    iter_change *= self._dx

                model.convergence_iters[key].append((niter, iter_change))
                self._prev[key] = current.clone()

                if iter_change < self._threshold and key not in self._converged:
                    self._converged.add(key)
                    report(
                        f"[iter {niter}] {key} change {iter_change:.3e} < "
                        f"threshold {self._threshold:.1e} — converged",
                        verbosity="INFO",
                    )

        tracked_keys = self._all_tracked_keys()
        if tracked_keys and tracked_keys.issubset(self._converged):
            report(
                f"[iter {niter}] All monitored tensors have converged "
                f"(threshold {self._threshold:.1e})",
                verbosity="INFO",
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _all_tracked_keys(self) -> set:
        """Return the set of history keys, which matches self._tensors directly."""
        return set(self._tensors)

    @staticmethod
    def _snapshot(model, name: str) -> dict:
        """
        Return a dict of detached CPU float32 tensors for the given parameter name.

        For 'probe', returns {'probe_amp': ..., 'probe_phase': ...}.
        For all others, returns {name: tensor}.
        """
        if name == "probe":
            probe_c = model.get_complex_probe_view().detach()  # keep complex64; abs() returns float32
            return {"probe": probe_c.abs().cpu()}
        return {name: model.optimizable_tensors[name].detach().float().cpu()}

    @staticmethod
    def _compute_metric(current: torch.Tensor, reference: torch.Tensor, metric_type: str) -> float:
        """Compute a scalar convergence metric between current and reference tensors."""
        diff = current - reference
        if metric_type == "rel_frob":
            return (diff.norm() / (reference.norm() + 1e-8)).item()
        if metric_type == "rms":
            # Per-position displacement magnitude (works for (N, 2) tensors)
            return diff.pow(2).sum(dim=-1).mean().sqrt().item()
        raise ValueError(f"Unknown metric_type: {metric_type!r}")


def create_convergence_monitor(convergence_monitor_params, model) -> Optional[ConvergenceMonitor]:
    """
    Factory that returns a ``ConvergenceMonitor`` when ``convergence_monitor_params`` is not None.

    Args:
        convergence_monitor_params: Parsed dict from ``ReconParams.convergence_monitor``,
            or None to disable monitoring.
        model: ``PtychoModel`` instance used for the initial snapshot.

    Returns:
        A configured ``ConvergenceMonitor``, or None if params is None.
    """
    if convergence_monitor_params is None:
        return None
    return ConvergenceMonitor(convergence_monitor_params, model)

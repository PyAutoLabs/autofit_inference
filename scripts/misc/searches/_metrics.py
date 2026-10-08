"""MLTracker — copied from ``autofit_workspace_developer/searches_minimal/_metrics.py``.

The copy is deliberate (no shared package at birth); ruff --fix modernised its type hints
(``X | None``, ``collections.abc``) to this repo's lint. One addition below the copied
class: :data:`TIME_BASIS_OBSERVED` / :data:`TIME_BASIS_ESTIMATED`, the labels a row
carries because ``from_log_l_history`` *interpolates* times (JAX paths) while
``wrap``/``record`` *observe* them (the numpy callback path).

Original docstring follows.

Shared per-evaluation tracker used by every script in this folder.

Wrap a log-likelihood callable with ``MLTracker.wrap`` (or call
``tracker.record(log_l)`` manually) and the tracker stores the log L and
wall-clock time of every evaluation. After the run, ``finalise`` returns
the eval index and wall time at which the running max log L first came
within ``tolerance`` nats of the final maximum -- the "evals to ML" /
"time to ML" headline numbers used in the comparison.

For JAX paths where the likelihood runs inside ``jax.jit`` (and a Python
callback is impossible without forcing a host round-trip), use
``MLTracker.from_log_l_history`` instead with the full per-eval log L
sequence reconstructed from the sampler's dead-point + live-point state.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Optional


class MLTracker:
    """Record per-evaluation log L and wall time, compute evals/time to ML."""

    def __init__(self):
        self.t0 = time.time()
        self.history_log_l: list[float] = []
        self.history_wall: list[float] = []

    def record(self, log_l: float) -> None:
        self.history_log_l.append(float(log_l))
        self.history_wall.append(time.time() - self.t0)

    def wrap(self, fn: Callable) -> Callable:
        """Decorate a log-likelihood callable so every call is recorded."""

        def wrapped(*args, **kwargs):
            log_l = fn(*args, **kwargs)
            self.record(log_l)
            return log_l

        return wrapped

    def finalise(
        self, max_log_l: float | None = None, tolerance: float = 1.0
    ) -> tuple[int | None, float | None]:
        """Return (evals_to_ml, time_to_ml) — the eval index and wall time at
        which the running max first came within ``tolerance`` nats of the
        final maximum. ``(None, None)`` if no evaluations were recorded."""
        if not self.history_log_l:
            return None, None
        if max_log_l is None:
            max_log_l = max(self.history_log_l)
        target = max_log_l - tolerance
        for i, log_l in enumerate(self.history_log_l):
            if log_l >= target:
                return i + 1, self.history_wall[i]
        return None, None

    @staticmethod
    def from_log_l_history(
        log_l_history: Sequence[float],
        total_sampling_time: float,
        tolerance: float = 1.0,
    ) -> tuple[int | None, float | None]:
        """Variant for samplers that run their likelihood inside JIT and only
        expose log L per dead/live point post hoc. ``time_to_ml`` is linearly
        interpolated from the total sampling time -- evaluations are assumed
        evenly distributed over the run, which is a reasonable approximation
        for nested sampling (each step is roughly the same cost)."""
        if not log_l_history:
            return None, None
        max_log_l = max(log_l_history)
        target = max_log_l - tolerance
        for i, log_l in enumerate(log_l_history):
            if log_l >= target:
                evals_to_ml = i + 1
                time_to_ml = total_sampling_time * (evals_to_ml / len(log_l_history))
                return evals_to_ml, time_to_ml
        return None, None


#: ``time_to_target_basis`` when every evaluation was timed as it happened.
TIME_BASIS_OBSERVED = "observed"

#: ``time_to_target_basis`` when the time was interpolated from the eval index.
TIME_BASIS_ESTIMATED = "estimated"

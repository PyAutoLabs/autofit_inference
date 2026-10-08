"""Per-cell measured step rates for HPC submit `--time` estimates.

Copied from ``autolens_inference/scripts/misc/wall/rates.py`` and **shipped empty**, as
autolens_inference shipped its own: this repo was born 2026-10-07 and inherits no rate
from any other project. Populated only from rates measured in this repo, on the cell the
row names; nothing is interpolated, extrapolated or carried across cells. Until a row
exists every submit declares ``source: unmeasured`` with ``probe-first: yes``.

Keys are ``(dataset, cell, instrument, device, precision, n_lanes, batch_size)``; the
``instrument`` slot holds the cell's leaf (the ``<model_type>``). Add a row with an
inline comment naming the RAL job and date, plus a ``PROVENANCE`` entry saying what it
must NOT be used for. `wall/check_submits.py` enforces the rule on every submit.
"""

from __future__ import annotations

# Key: (dataset, cell, instrument, device, precision, n_lanes, batch_size)
# Value: seconds per step, measured on that exact configuration, in this repo.
STEP_RATE: dict[tuple[str, str, str, str, str, int, int | None], float] = {}

#: One entry per key prefix in ``STEP_RATE``: what ran, on which job, and the
#: configuration the row is and is not valid for.
PROVENANCE: dict[str, str] = {}


class UnmeasuredCellError(KeyError):
    """No measured step rate exists for the requested configuration.

    Raised instead of returning a nearby cell's rate. Falling back across cells
    is the bug — see this module's docstring.
    """


def step_rate_for(
    dataset: str,
    cell: str,
    instrument: str,
    device: str,
    precision: str,
    n_lanes: int,
    batch_size: int | None = None,
) -> float:
    """Seconds per step for exactly this configuration.

    There is **no** nearest-neighbour fallback: an unmeasured configuration
    raises `UnmeasuredCellError` rather than silently answering with a rate
    measured on a different cell, lane count or batching.
    """
    key = (dataset, cell, instrument, device, precision, n_lanes, batch_size)
    try:
        return STEP_RATE[key]
    except KeyError:
        raise UnmeasuredCellError(
            f"no measured step rate for {key!r}. Do not substitute another cell's rate — "
            f"run one short arm on this cell and add the row to wall/rates.py, or declare "
            f"`source: unmeasured` with `probe-first: yes` in the submit's WALL-BASIS block."
        ) from None


def wall_estimate(rate_s_per_step: float, n_steps: int, compile_s: float = 0.0) -> float:
    """Estimated wall seconds for `n_steps` at `rate_s_per_step`, plus compile."""
    return rate_s_per_step * n_steps + compile_s

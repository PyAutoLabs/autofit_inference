"""The wave-1 pilot's expected-run manifest (Insight task ``gaussian_x3_search_wave1``).

Wave 1 (protocol §9) is 10 search seeds × the wave-1 menu × both datasets, on
``local_numpy_fp64`` and on ``local_jax_cpu_fp64`` where the search is JAX-native. The
menu is the registry in :mod:`searches._runner` (``SAMPLERS``): every registered search,
every named settings variant, every leg it supports. This module turns it into the list
of expected runs, so a missing row is visible as a **deferred** (never run) or a
**failed** (run, no usable row) attempt rather than silently absent — the runner
docstring's "expected-run manifest".

Standard library only (the exporter and the catalogue builder import it).

Status of a cell
----------------

- ``planned`` — in the menu and runnable by the harness.
- ``deferred`` — in the menu, not run, with a reason (:data:`DEFERRED`). A planned cell
  whose seeds did not all run within the pilot's compute budget is reported deferred for
  those seeds by the consumers (``budget_deferred``), with :data:`BUDGET_REASON`.
"""

from __future__ import annotations

from dataclasses import dataclass

from searches._runner import SAMPLERS, config_segment

PILOT_ID = "gaussian_x3_search_wave1"

#: Search seeds of the pilot (protocol §9).
SEEDS = tuple(range(10))

DATASETS = ("gaussian_x3_blend", "gaussian_x3_separated")

#: The two local CPU legs; JAX runs use the default warm compilation cache.
CONFIG_BY_BACKEND = {"numpy": "local_numpy_fp64", "jax_cpu": "local_jax_cpu_fp64"}
COMPILE_CACHE = "warm"

#: Cells in the menu that the pilot does not run, keyed by sampler (every dataset and
#: leg) — the reason is recorded on every expected run of the cell.
DEFERRED = {
    "nss": (
        "NSS is deferred until phase A3b (NSS onto Fitness): the B3 rule defers it on the "
        "blend, and the harness registers it deferred as a whole, so the separated control "
        "is deferred with it"
    ),
}

#: Why a planned run with no row is deferred (compute budget, never fabricated).
BUDGET_REASON = (
    "not run within the B3 pilot compute budget (one shared laptop, ~6 h wall beside the "
    "pending reference runs); wave 2 runs on RAL --partition=ral"
)


#: Planned cells whose remaining seeds were deliberately not run after the first
#: attempts showed a deterministic failure, keyed by ``(dataset, sampler)``: the reason
#: replaces :data:`BUDGET_REASON` for their un-run seeds (the attempts that ran stay).
CELL_REASONS: dict[tuple[str, str], str] = {
    ("gaussian_x3_blend", "blackjax_nuts_warm"): (
        "not run after seed 0: BlackJAXNUTS on the asserted blend fails deterministically "
        "at trace time (TracerBoolConversionError in the NUTS log_l: the ordered-centre "
        "assertion is evaluated as a Python bool), as the cold leg shows on every seed; "
        "each warm attempt would first spend ~20 min on its Nautilus provider"
    ),
}


def missing_reason(cell: Cell) -> str:
    """Why an expected run of ``cell`` has no row."""
    if cell.status == "deferred":
        return cell.reason or BUDGET_REASON
    return CELL_REASONS.get((cell.dataset, cell.sampler), BUDGET_REASON)


@dataclass(frozen=True)
class Cell:
    dataset: str
    sampler: str
    settings: str
    backend: str
    config_name: str
    config_id: str
    task: str
    status: str
    reason: str | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.dataset, self.sampler, self.settings, self.config_id)


def backends(spec) -> list[str]:
    out = []
    if spec.numpy_supported:
        out.append("numpy")
    if spec.jax_native:
        out.append("jax_cpu")
    return out


def cells() -> list[Cell]:
    """Every (dataset × search × settings × leg) cell of the wave-1 menu, in registry
    order (never sorted by any result)."""
    out = []
    for dataset in DATASETS:
        for name, spec in SAMPLERS.items():
            deferred = DEFERRED.get(name) or (spec.note if spec.status == "deferred" else None)
            for settings in spec.settings:
                for backend in backends(spec):
                    config_name = CONFIG_BY_BACKEND[backend]
                    config_id = config_segment(
                        config_name, COMPILE_CACHE if backend == "jax_cpu" else None
                    )
                    out.append(
                        Cell(
                            dataset=dataset,
                            sampler=name,
                            settings=settings,
                            backend=backend,
                            config_name=config_name,
                            config_id=config_id,
                            task=spec.task,
                            status="deferred" if deferred else "planned",
                            reason=deferred,
                        )
                    )
    return out


def expected_runs() -> list[tuple[Cell, int]]:
    return [(cell, seed) for cell in cells() for seed in SEEDS]


def row_key(row: dict) -> tuple[str, str, str, str]:
    """The :attr:`Cell.key` a result row belongs to."""
    return (
        row.get("dataset"),
        row.get("sampler"),
        row.get("settings_name"),
        row.get("config_id") or row.get("config_name"),
    )

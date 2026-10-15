"""The ``correlated_nd`` pilot's expected-run manifest, identity and attempt accounting.

Protocol ``wiki/project/protocol_correlated_nd.md`` (``correlated_nd@1``), §7 and §9: the
pilot runs **10 search seeds** (0–9) × **data realisation 1** × every cell of the
gaussian_x3 wave-2 search menu (the same searches, settings and backends, never retuned
for this family) × the four datasets, on RAL ``--partition=ral`` against the same private
PyAutoFit pin as gaussian_x3 wave 2 (:data:`PYAUTOFIT_PIN`). A cell is **deferred**, with
its reason, when it cannot answer its task by construction at that dimension
(:func:`deferral`, protocol §7):

1. *Live-point floor (nested searches).* Nested sampling's ln Z scatter is
   ``sqrt(H / n_live)`` (Skilling 2006), with ``H`` the exact prior-to-posterior
   information of the dataset (:data:`INFORMATION_NATS`, computed by
   ``models/correlated_nd_exact.information`` and pinned by a test). A leg with
   ``n_live < H`` has an expected ln Z scatter above the 1-nat tolerance of §4(c).
2. *Walker floor (ensemble chains).* Emcee and Zeus refuse fewer than ``2 D`` walkers;
   the menu's 50 walkers are an invalid configuration for ``D >= 26``.
3. *Warm-start provider.* A warm start whose provider leg is deferred is deferred with it.

The pilot ranks nothing (protocol §9). Standard library only (the planner and any future
exporter import it).

Identity: a pilot row belongs to the pilot only when its manifest key —
``(dataset, sampler, settings_name, config_id, data_seed, protocol_id, pyautofit_sha)``,
``_wave2.manifest_key`` — names a cell, realisation 1, ``correlated_nd@1`` and the pin, and
its ``model`` is ``correlated_nd``. Every expected run is ``row``, ``failed``, ``missing``
or ``deferred`` (the wave-2 accounting rules); a run stopped at the pilot's per-run cap
(:data:`PILOT_CAP_S`) is a failed attempt censored at the cap.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from searches import _protocol as protocol
from searches import _wave2 as wave2
from searches._runner import SAMPLERS, config_segment

MODEL_TYPE = "correlated_nd"
WAVE = 1
WAVE_ID = "correlated_nd_pilot"
PROTOCOL_ID = protocol.CND_PROTOCOL_ID

#: name -> dimension (``simulators/correlated_nd.py`` ``DATASETS``; a test pins them).
DIMENSION = {
    "corr_gaussian_d20": 20,
    "corr_gaussian_d50": 50,
    "corr_gaussian_d100": 100,
    "rosenbrock_d20": 20,
}
DATASETS = tuple(DIMENSION)

#: The exact information ``H = E_post[ln L] - ln Z`` (nats) of each dataset's default
#: realisation (``models/correlated_nd_exact.information``; a test recomputes them).
INFORMATION_NATS = {
    "corr_gaussian_d20": 53.13,
    "corr_gaussian_d50": 133.29,
    "corr_gaussian_d100": 267.63,
    "rosenbrock_d20": 133.29,
}

SEEDS = tuple(range(10))
DATA_SEEDS = (1,)

PYAUTOFIT_PIN = wave2.PYAUTOFIT_PIN
PYAUTONERVES_PIN = wave2.PYAUTONERVES_PIN
CONFIG_BY_BACKEND = wave2.CONFIG_BY_BACKEND
COMPILE_CACHE = wave2.COMPILE_CACHE
#: The gaussian_x3 wave-2 menu, unchanged: NSS runs its 50-step entry only.
SETTINGS = wave2.SETTINGS

#: PyAutoFit's ``DynestyDynamic(nlive_init=500)`` default at the pin (its ``default``
#: settings pass no kwargs).
DYNESTY_DYNAMIC_NLIVE_INIT = 500

#: Ensemble chains need at least this many walkers per dimension (emcee's red-blue move
#: and zeus both raise below ``2 * ndim``).
WALKERS_PER_DIMENSION = 2

#: The pilot's per-run operational cap on ``search.fit`` (protocol §7): a run stopped there
#: is a failed attempt, its wall a censored lower bound. Not a protocol timeout.
PILOT_CAP_S = 14400.0
RUN_OVERHEAD_S = wave2.RUN_OVERHEAD_S
KILL_GRACE_S = wave2.KILL_GRACE_S


def live_points(sampler: str, settings: str) -> int | None:
    """The live-point count of a nested leg, ``None`` for any other search."""
    spec = SAMPLERS[sampler]
    if spec.family != "nested":
        return None
    kwargs = spec.settings[settings]
    for name in ("n_live", "nlive", "nlive_init"):
        if name in kwargs:
            return int(kwargs[name])
    if spec.cls == "DynestyDynamic":
        return DYNESTY_DYNAMIC_NLIVE_INIT
    raise ValueError(f"no live-point count for {sampler}/{settings}")


def deferral(dataset: str, sampler: str, settings: str) -> str | None:
    """Why ``sampler``/``settings`` is not run on ``dataset`` (protocol §7), or ``None``."""
    spec = SAMPLERS[sampler]
    dimension = DIMENSION[dataset]
    information = INFORMATION_NATS[dataset]
    n_live = live_points(sampler, settings)
    if n_live is not None and n_live < information:
        return (
            f"deferred (live-point floor, protocol §7): n_live {n_live} < H {information:.1f} "
            f"nat at D = {dimension}, so the expected ln Z scatter sqrt(H/n_live) = "
            f"{math.sqrt(information / n_live):.2f} nat exceeds the 1-nat §4(c) tolerance"
        )
    walkers = spec.settings[settings].get("nwalkers")
    if spec.cls in ("Emcee", "Zeus") and walkers is not None:
        if walkers < WALKERS_PER_DIMENSION * dimension:
            return (
                f"deferred (walker floor, protocol §7): {spec.cls} refuses {walkers} walkers "
                f"at D = {dimension} (needs >= {WALKERS_PER_DIMENSION * dimension})"
            )
    warm_from = spec.extra.get("warm_from")
    if warm_from is not None:
        provider = deferral(dataset, *warm_from)
        if provider:
            return (
                f"deferred (warm-start provider, protocol §7): its provider "
                f"{warm_from[0]}/{warm_from[1]} is deferred at D = {dimension}"
            )
    return None


@dataclass(frozen=True)
class Cell:
    dataset: str
    sampler: str
    settings: str
    backend: str
    config_name: str
    config_id: str
    task: str
    status: str  # planned | deferred
    reason: str | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.dataset, self.sampler, self.settings, self.config_id)

    @property
    def leaf(self) -> str:
        return f"scripts/{self.dataset}/searches/{self.sampler}/{MODEL_TYPE}.py"


def cells() -> list[Cell]:
    """Every (dataset × search × menu settings × backend) cell, in registry order."""
    out = []
    for dataset in DATASETS:
        for name, spec in SAMPLERS.items():
            for settings in SETTINGS.get(name, tuple(spec.settings)):
                for backend in wave2.backends(spec):
                    config_name = CONFIG_BY_BACKEND[backend]
                    config_id = config_segment(
                        config_name, COMPILE_CACHE if backend == "jax_cpu" else None
                    )
                    reason = deferral(dataset, name, settings)
                    out.append(
                        Cell(
                            dataset=dataset,
                            sampler=name,
                            settings=settings,
                            backend=backend,
                            config_name=config_name,
                            config_id=config_id,
                            task=spec.task,
                            status="deferred" if reason else "planned",
                            reason=reason,
                        )
                    )
    return out


def refusal(dataset: str, sampler: str, settings: str, wave: int) -> str | None:
    """Why the runner must not run this ``correlated_nd`` leg, or ``None``."""
    if wave != WAVE:
        return (
            f"correlated_nd declares only its pilot (wave {WAVE}, {PROTOCOL_ID}); a scored "
            "wave waits for correlated_nd@2, frozen from the pilot (protocol §7)"
        )
    if dataset not in DIMENSION:
        return f"{dataset} is not a correlated_nd dataset ({', '.join(DATASETS)})"
    allowed = SETTINGS.get(sampler, tuple(SAMPLERS[sampler].settings))
    if settings not in allowed:
        return f"the correlated_nd pilot runs {sampler} only with {allowed}, not {settings!r}"
    return deferral(dataset, sampler, settings)


def expected_runs() -> list[tuple[Cell, int, int]]:
    return [(cell, d, seed) for cell in cells() for d in DATA_SEEDS for seed in SEEDS]


def cell_keys() -> set[tuple]:
    return {
        (*cell.key, data_seed, PROTOCOL_ID, PYAUTOFIT_PIN)
        for cell in cells()
        for data_seed in DATA_SEEDS
    }


def admit(row: dict) -> str | None:
    """``None`` when ``row`` is a pilot run of this family, else why it is refused."""
    if row.get("model") != MODEL_TYPE:
        return f"model {row.get('model')!r} is not {MODEL_TYPE}"
    if row.get("pilot") is not True or protocol.wave_of(row) != WAVE:
        return f"not a pilot row (wave {protocol.wave_of(row)}, pilot {row.get('pilot')})"
    if row.get("protocol_id") != PROTOCOL_ID:
        return f"protocol {row.get('protocol_id')!r} is not the pilot's {PROTOCOL_ID}"
    if row.get("pyautofit_commit") != PYAUTOFIT_PIN:
        return f"PyAutoFit commit {row.get('pyautofit_commit')!r} is not the pin {PYAUTOFIT_PIN}"
    if row.get("data_seed") not in DATA_SEEDS:
        return f"data_seed {row.get('data_seed')!r} is not one of {DATA_SEEDS}"
    if row.get("seed") not in SEEDS:
        return f"search seed {row.get('seed')!r} is outside 0–{len(SEEDS) - 1}"
    if row.get("resumed") is True:
        return "resumed from an earlier fit: the attempt's own wall was not observed"
    if wave2.manifest_key(row) not in cell_keys():
        return f"{wave2.manifest_key(row)[:4]} is not a cell of the correlated_nd pilot"
    return None


def operational_kill(row: dict) -> bool:
    """A ``Terminated`` row stopped before the pilot cap: not a censored attempt, so it is
    ``missing`` (to rerun)."""
    if "Terminated" not in str(row.get("status")):
        return False
    wall = row.get("total_wall_s")
    return wall is None or float(wall) < PILOT_CAP_S


def accounting(rows: list[dict]) -> dict:
    """Per leg: ``expected`` and ``row`` / ``failed`` / ``missing`` / ``deferred`` (summing
    to it), plus every refused row with why."""
    admitted: dict[tuple, dict] = {}
    refused = []
    for row in rows:
        why = admit(row)
        if why:
            refused.append({"run_id": row.get("run_id"), "reason": why})
            continue
        admitted[(*wave2.manifest_key(row)[:5], row.get("seed"))] = row
    legs = []
    totals = {"expected": 0, "row": 0, "failed": 0, "missing": 0, "deferred": 0}
    for cell in cells():
        for data_seed in DATA_SEEDS:
            leg = {
                "id": "/".join((*cell.key, f"data_seed{data_seed}")),
                "dataset": cell.dataset,
                "sampler": cell.sampler,
                "settings": cell.settings,
                "config_id": cell.config_id,
                "data_seed": data_seed,
                "task": cell.task,
                "cell_status": cell.status,
                "expected": len(SEEDS),
                "row": 0,
                "failed": [],
                "missing": [],
                "deferred": [],
                "reason": cell.reason,
            }
            for seed in SEEDS:
                row = admitted.get((*cell.key, data_seed, seed))
                if cell.status == "deferred":
                    leg["deferred"].append(seed)
                elif row is None or operational_kill(row):
                    leg["missing"].append(seed)
                elif row.get("status") == "complete" and row.get("completed") is True:
                    leg["row"] += 1
                else:
                    leg["failed"].append(seed)
            for state in ("failed", "missing", "deferred"):
                totals[state] += len(leg[state])
            totals["row"] += leg["row"]
            totals["expected"] += leg["expected"]
            legs.append(leg)
    return {"wave": WAVE_ID, "totals": totals, "legs": legs, "refused": refused}


@dataclass(frozen=True)
class Task:
    """One array task: one run (a pilot cell has no measured wall to block seeds by)."""

    cell: Cell
    data_seed: int
    seed: int

    @property
    def limit_s(self) -> float:
        return PILOT_CAP_S + RUN_OVERHEAD_S

    def line(self) -> str:
        return "\t".join(
            (
                self.cell.dataset,
                self.cell.sampler,
                self.cell.settings,
                self.cell.config_name,
                str(self.data_seed),
                str(int(math.ceil(self.limit_s))),
                str(self.seed),
            )
        )

    @property
    def budget_s(self) -> float:
        return self.limit_s + KILL_GRACE_S


def pilot_tasks() -> list[Task]:
    return [
        Task(cell, data_seed, seed)
        for cell in cells()
        if cell.status == "planned"
        for data_seed in DATA_SEEDS
        for seed in SEEDS
    ]


def summary() -> dict:
    all_cells = cells()
    planned = [c for c in all_cells if c.status == "planned"]
    return {
        "cells": len(all_cells),
        "planned": len(planned),
        "deferred": len(all_cells) - len(planned),
        "expected_runs": len(all_cells) * len(SEEDS) * len(DATA_SEEDS),
        "planned_runs": len(planned) * len(SEEDS) * len(DATA_SEEDS),
        "by_dataset": {d: sum(1 for c in planned if c.dataset == d) * len(SEEDS) for d in DATASETS},
    }


if __name__ == "__main__":  # pragma: no cover — a quick look at the plan
    print(json.dumps(summary(), indent=2))

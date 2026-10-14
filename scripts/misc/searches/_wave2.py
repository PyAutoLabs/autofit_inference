"""Scored wave 2's expected-run manifest, identity and attempt accounting (``gaussian_x3@2``).

Protocol §9: wave 2 is **scored** — 50 search seeds × 5 data realisations × every
non-deferred leg of the menu × both datasets, on RAL ``--partition=ral``, against one
PyAutoFit revision frozen after phases A2 and A4 (:data:`PYAUTOFIT_PIN`, a private checkout
under the RAL project root, never the shared ``/mnt/ral/jnightin/PyAuto`` mirror). Every
row is judged under ``gaussian_x3@2`` (frozen before the first scored run, protocol §7 and
§11 A2) against the reference of its own realisation.

Standard library only (the exporter, the catalogue and the submit planner import it).

Identity (review 06 item 7)
---------------------------

A scored row belongs to the wave only when its :func:`manifest_key` —
``(dataset, sampler, settings_name, config_id, data_seed, protocol_id, pyautofit_sha)`` —
names a cell of the menu, a realisation of :data:`DATA_SEEDS`, the wave's protocol and
the pinned library revision. :func:`admit` returns why a row is refused (a pilot row, the
wrong protocol, a different PyAutoFit commit, an unknown cell, a seed outside the wave);
a refused row is listed, never counted as an attempt of the wave.

Attempt accounting
------------------

Every expected run (:func:`expected_runs`) is exactly one of:

- ``row`` — an admitted row whose search completed (judged under ``@2``);
- ``failed`` — an admitted row that did not complete (crash, timeout ``Terminated``,
  non-finite answer): an attempt that failed, with its wall;
- ``missing`` — planned, no admitted row (not run yet, or a hard kill / OOM that left
  none). Once the wave is closed a missing run is a failed attempt censored at its
  timeout (protocol §7), never dropped;
- ``deferred`` — deliberately not run, with the reason (:data:`DEFERRED`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from searches import _protocol as protocol
from searches._runner import SAMPLERS, config_segment

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())

WAVE = 2
WAVE_ID = "gaussian_x3_search_wave2"
PROTOCOL_ID = protocol.PROTOCOL_IDS[WAVE]

#: Search seeds and data realisations (protocol §9: 50 × 5). The realisations are the
#: same five seeds on both datasets; each dataset's committed default (blend 1,
#: separated 2) is one of them.
SEEDS = tuple(range(50))
DATA_SEEDS = (1, 2, 3, 4, 5)

DATASETS = ("gaussian_x3_blend", "gaussian_x3_separated")

#: RAL CPU legs (``--partition=ral``); JAX runs use the warm compilation cache.
CONFIG_BY_BACKEND = {"numpy": "ral_numpy_fp64", "jax_cpu": "ral_jax_cpu_fp64"}
COMPILE_CACHE = "warm"

#: The library revision every scored row must carry (``pyautofit_commit``): PyAutoFit
#: ``main`` at the A4 merge (PyAutoFit#1687), the newest ``main`` at submission; no release
#: contains it yet. PyAutoNerves is pinned beside it at its ``main`` of the same day.
PYAUTOFIT_PIN = "df6d218f10cd75a7f0d79bb03dca5c131fd29d33"
PYAUTONERVES_PIN = "7f3055b7f3d1967f372bdb41ee732135bca0e68d"

#: Where the private pinned checkouts live, relative to the RAL project root
#: (``/mnt/ral/jnightin/autofit_inference/.pyauto_pin/{PyAutoFit,PyAutoNerves}``,
#: gitignored), PYTHONPATH-prefixed by every wave-2 submit.
PIN_DIR = ".pyauto_pin"

#: Settings run in wave 2 where they differ from the registry's full list. NSS runs a
#: wave-2 entry with 50 slice-MCMC steps (≈ 5 × ndim for the 10 free parameters): at
#: the default 5 steps the A3b witness lost the mode on seed 1 and ln Z by ~27 nat on
#: seed 2, at 50 steps seeds 0–2 recovered every centre with ln Z inside the reference
#: spread (PyAutoFit#1681). The pilot's ``n_live_200`` entry stays for the pilot's
#: deferred cells.
SETTINGS = {"nss": ("n_live_200_mcmc_steps_50",)}

#: Cells deliberately not run, keyed by ``(dataset, sampler)``, with the reason.
DEFERRED = {
    ("gaussian_x3_blend", "blackjax_nuts_warm"): (
        "deferred: BlackJAXNUTS on the asserted blend still fails deterministically at "
        "trace time at the wave-2 pin df6d218f1 (TracerBoolConversionError in the NUTS "
        "log_l: the ordered-centre assertion is evaluated as a Python bool; reproduced "
        "2026-10-14 on the cold leg), so every warm attempt would spend its Nautilus "
        "provider first and then fail; the cold leg runs and records the failure"
    ),
}

#: Search seeds of a probe-first cell's probe (data realisation 1): the cell has no pilot
#: wall, so its timeout is ``max(600 s, 10 × the probe's median)`` (protocol §11 A2) and
#: the rest of the cell is submitted once the probe has run.
PROBE_SEEDS = (0, 1, 2, 3, 4)
PROBE_DATA_SEED = 1
#: The per-run cap a probe runs under: an operational limit, not the protocol timeout.
PROBE_CAP_S = 43200.0

CALIBRATION = ROOT / "results" / "calibration" / "gaussian_x3_wave1.json"


@dataclass(frozen=True)
class Cell:
    dataset: str
    sampler: str
    settings: str
    backend: str
    config_name: str
    config_id: str
    task: str
    status: str  # planned | probe_first | deferred
    timeout_s: float | None
    median_pilot_wall_s: float | None
    reason: str | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.dataset, self.sampler, self.settings, self.config_id)

    @property
    def leaf(self) -> str:
        return f"scripts/{self.dataset}/searches/{self.sampler}/gaussian_x3.py"


def backends(spec) -> list[str]:
    out = []
    if spec.numpy_supported:
        out.append("numpy")
    if spec.jax_native:
        out.append("jax_cpu")
    return out


def calibration_timeouts(path: Path = CALIBRATION) -> dict:
    """``{"<sampler>/<settings>/<backend>": {"timeout_s", "median_pilot_wall_s", "n"}}``."""
    return (protocol.load_json(path) or {}).get("timeouts") or {}


def cells(timeouts: dict | None = None) -> list[Cell]:
    """Every (dataset × search × wave-2 settings × leg) cell, in registry order."""
    timeouts = calibration_timeouts() if timeouts is None else timeouts
    out = []
    for dataset in DATASETS:
        for name, spec in SAMPLERS.items():
            for settings in SETTINGS.get(name, tuple(spec.settings)):
                for backend in backends(spec):
                    config_name = CONFIG_BY_BACKEND[backend]
                    config_id = config_segment(
                        config_name, COMPILE_CACHE if backend == "jax_cpu" else None
                    )
                    entry = timeouts.get(f"{name}/{settings}/{backend}")
                    deferred = DEFERRED.get((dataset, name))
                    if deferred:
                        status, reason = "deferred", deferred
                    elif entry is None:
                        status = "probe_first"
                        reason = (
                            "probe-first: no pilot wall for this cell, so its timeout comes "
                            f"from a {len(PROBE_SEEDS)}-run probe (protocol §11 A2)"
                        )
                    else:
                        status, reason = "planned", None
                    out.append(
                        Cell(
                            dataset=dataset,
                            sampler=name,
                            settings=settings,
                            backend=backend,
                            config_name=config_name,
                            config_id=config_id,
                            task=spec.task,
                            status=status,
                            timeout_s=None if entry is None else float(entry["timeout_s"]),
                            median_pilot_wall_s=None
                            if entry is None
                            else float(entry["median_pilot_wall_s"]),
                            reason=reason,
                        )
                    )
    return out


def expected_runs(timeouts: dict | None = None) -> list[tuple[Cell, int, int]]:
    """``(cell, data_seed, seed)`` for every run the wave expects, deferred ones included
    (they are accounted as ``deferred``, never silently absent)."""
    return [
        (cell, data_seed, seed)
        for cell in cells(timeouts)
        for data_seed in DATA_SEEDS
        for seed in SEEDS
    ]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def manifest_key(row: dict) -> tuple:
    """``(dataset, sampler, settings_name, config_id, data_seed, protocol_id, pyautofit_sha)``."""
    return (
        row.get("dataset"),
        row.get("sampler"),
        row.get("settings_name"),
        row.get("config_id") or row.get("config_name"),
        row.get("data_seed"),
        row.get("protocol_id"),
        row.get("pyautofit_commit"),
    )


def cell_keys(timeouts: dict | None = None) -> set[tuple]:
    """Every manifest key the wave admits (one per cell × realisation)."""
    return {
        (*cell.key, data_seed, PROTOCOL_ID, PYAUTOFIT_PIN)
        for cell in cells(timeouts)
        for data_seed in DATA_SEEDS
    }


def admit(row: dict, timeouts: dict | None = None) -> str | None:
    """``None`` when ``row`` is a scored wave-2 run, else why it is refused."""
    if row.get("pilot") is True or protocol.wave_of(row) != WAVE:
        return f"not a wave-{WAVE} row (wave {protocol.wave_of(row)}, pilot {row.get('pilot')})"
    if row.get("protocol_id") != PROTOCOL_ID:
        return f"protocol {row.get('protocol_id')!r} is not the wave's {PROTOCOL_ID}"
    sha = row.get("pyautofit_commit")
    if sha != PYAUTOFIT_PIN:
        return f"PyAutoFit commit {sha!r} is not the wave pin {PYAUTOFIT_PIN}"
    if row.get("data_seed") not in DATA_SEEDS:
        return f"data_seed {row.get('data_seed')!r} is not one of {DATA_SEEDS}"
    if row.get("seed") not in SEEDS:
        return f"search seed {row.get('seed')!r} is outside 0–{len(SEEDS) - 1}"
    if manifest_key(row) not in cell_keys(timeouts):
        return f"{manifest_key(row)[:4]} is not a cell of the wave-2 menu"
    return None


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def _completed(row: dict) -> bool:
    return row.get("status") == "complete" and row.get("completed") is True


def accounting(rows: list[dict], timeouts: dict | None = None) -> dict:
    """Per leg (cell × data realisation): ``expected`` and the counts of ``row``,
    ``failed``, ``missing`` and ``deferred`` (they sum to ``expected``), with the
    seeds of every non-``row`` state and the reason for the missing and deferred ones;
    plus every refused row with why (never an attempt of the wave)."""
    admitted: dict[tuple, dict] = {}
    refused = []
    for row in rows:
        why = admit(row, timeouts)
        if why:
            refused.append({"run_id": row.get("run_id"), "reason": why})
            continue
        key = (*manifest_key(row)[:5], row.get("seed"))
        admitted[key] = row
    legs = []
    totals = {"expected": 0, "row": 0, "failed": 0, "missing": 0, "deferred": 0}
    for cell in cells(timeouts):
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
                "reason": None,
            }
            for seed in SEEDS:
                row = admitted.get((*cell.key, data_seed, seed))
                if cell.status == "deferred":
                    leg["deferred"].append(seed)
                elif row is None:
                    leg["missing"].append(seed)
                elif _completed(row):
                    leg["row"] += 1
                else:
                    leg["failed"].append(seed)
            if leg["deferred"] or leg["missing"]:
                leg["reason"] = cell.reason or (
                    "no admitted row: not run yet, or a hard kill left none; censored at "
                    "the cell timeout as a failed attempt once the wave closes (§7)"
                )
            for state in ("failed", "missing", "deferred"):
                totals[state] += len(leg[state])
            totals["row"] += leg["row"]
            totals["expected"] += leg["expected"]
            legs.append(leg)
    return {"wave": WAVE_ID, "totals": totals, "legs": legs, "refused": refused}


# ---------------------------------------------------------------------------
# Submit planning (hpc/batch_cpu/wave2/)
# ---------------------------------------------------------------------------

#: Seeds per array task: as many as fit ~2 h of median pilot wall, at most all 50.
BLOCK_TARGET_S = 7200.0

#: Seconds a run's process limit adds to its frozen timeout, for what the timeout (on
#: ``search.fit``'s ``total_wall_s``) does not cover: imports, the dataset load and the
#: admission-bar timing. Censoring at judging time compares ``total_wall_s`` with the
#: timeout itself.
RUN_OVERHEAD_S = 120.0


def block_size(cell: Cell, timeouts: dict | None = None) -> int:
    """Seeds per task for ``cell``: uniform over the settings of its search on its leg,
    sized by the slowest of them (an array task is sized by its slowest run)."""
    medians = [
        c.median_pilot_wall_s
        for c in cells(timeouts)
        if (c.dataset, c.sampler, c.backend) == (cell.dataset, cell.sampler, cell.backend)
        and c.median_pilot_wall_s
    ]
    if not medians:
        return 1
    return max(1, min(len(SEEDS), math.floor(BLOCK_TARGET_S / max(medians))))


@dataclass(frozen=True)
class Task:
    """One array task: the runs of one leg on one realisation, run in sequence, each under
    its own per-run limit (``timeout -s TERM``: the runner turns SIGTERM into a failure
    row with its wall)."""

    cell: Cell
    data_seed: int
    seeds: tuple[int, ...]
    limit_s: float

    def line(self) -> str:
        return "\t".join(
            (
                self.cell.sampler,
                self.cell.settings,
                self.cell.config_name,
                str(self.data_seed),
                str(int(math.ceil(self.limit_s))),
                ",".join(str(s) for s in self.seeds),
            )
        )

    @property
    def budget_s(self) -> float:
        return self.limit_s * len(self.seeds)


def scored_tasks(dataset: str, backend: str, timeouts: dict | None = None) -> list[Task]:
    """The scored array of one (dataset, backend): every planned cell × realisation,
    seeds blocked by :func:`block_size`, each run limited by the cell's frozen timeout."""
    out = []
    for cell in cells(timeouts):
        if cell.dataset != dataset or cell.backend != backend or cell.status != "planned":
            continue
        size = block_size(cell, timeouts)
        limit = cell.timeout_s + RUN_OVERHEAD_S
        for data_seed in DATA_SEEDS:
            for start in range(0, len(SEEDS), size):
                out.append(Task(cell, data_seed, SEEDS[start : start + size], limit))
    return out


def probe_tasks(timeouts: dict | None = None) -> list[Task]:
    """One task per probe run of every probe-first cell (both datasets)."""
    return [
        Task(cell, PROBE_DATA_SEED, (seed,), PROBE_CAP_S + RUN_OVERHEAD_S)
        for cell in cells(timeouts)
        if cell.status == "probe_first"
        for seed in PROBE_SEEDS
    ]


def summary(timeouts: dict | None = None) -> dict:
    all_cells = cells(timeouts)
    return {
        "cells": len(all_cells),
        "by_status": {
            s: sum(1 for c in all_cells if c.status == s)
            for s in ("planned", "probe_first", "deferred")
        },
        "expected_runs": len(all_cells) * len(DATA_SEEDS) * len(SEEDS),
    }


if __name__ == "__main__":  # pragma: no cover — a quick look at the plan
    print(json.dumps(summary(), indent=2))

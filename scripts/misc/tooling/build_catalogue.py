"""Build ``catalogue/search_catalogue.json``: every registered PyAutoFit search, by task.

The registry is PyAutoFit's own (phase A1): ``python -m autofit search-manifest --json``.
Its output is snapshotted to ``catalogue/search_manifest.json`` (``--refresh-manifest``
re-runs the command; it needs PyAutoFit importable), so the catalogue rebuilds and
``--check`` runs anywhere with the standard library alone (the lint job installs no
PyAuto library).

Every search in the manifest appears **exactly once**, keyed by its class name, under the
task it is asked to answer — point/MAP, posterior or evidence (protocol §1) — with
exactly one status:

- ``measured`` — at least one wave-1 row completed with a usable result;
- ``failed`` — attempted, but no attempt produced a usable result (crash, timeout,
  non-finite answer);
- ``deferred`` — in the wave-1 menu but not run, with the reason (A3b, compute budget);
- ``unsupported`` — registered in PyAutoFit but not benchmarked by this harness, with
  the reason (``Drawer`` is a sanity floor that answers no task, protocol §1).

**The wave-1 section ranks nothing.** Tasks are separate groups; within a task the searches keep the
manifest's order and each leg reports its own counts (attempts, verdicts, convergence,
the Wilson interval of its success rate, its walls). There is no rank, score or best
field, and no number is ever compared across tasks (``test_catalogue.py`` pins this).
Wave 1 is a pilot (protocol §9): its numbers calibrate the protocol, they do not order
the searches.

**The ``gaussian_x3@2`` section (B5b)** sits beside it and is scored (protocol §8, §9):
every expected run of the wave (``searches/_wave2.py``) is an attempt — an admitted row
judged under ``@2`` against its own realisation's reference, or, for a missing run of a
closed cell, a failed attempt censored at the cell's timeout — and each (dataset × search
× settings × leg) reports its success rate (Wilson 95 %) and expected wall per right
answer (bootstrap 95 %). Within one task and one dataset (never across either) the
fully-judged closed legs that clear the **reliability floor** (:data:`RANK_FLOOR`: success
rate >= 50 % over >= 20 attempts) are listed in ``ranking`` by expected wall per right
answer, then success rate; the closed legs below the floor are listed in ``unreliable``
with their numbers, never positioned; open legs are neither. The floor is a post-hoc
presentation convention adopted 2026-10-15 after seeing the wave-2 data (B5b-2, protocol
§8 note), not a pre-registered criterion: it changes no verdict. Only rows at the wave's
PyAutoFit pin are read; pilot rows never enter it.

    python scripts/misc/tooling/build_catalogue.py                     # write the catalogue
    python scripts/misc/tooling/build_catalogue.py --check             # verify the committed one
    python scripts/misc/tooling/build_catalogue.py --refresh-manifest  # re-snapshot the A1 registry
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(ROOT / "scripts" / "misc"), str(ROOT / "scripts" / "misc" / "tooling")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import export_inference_summary as exporter  # noqa: E402
from searches import _pilot as pilot  # noqa: E402
from searches import _protocol as protocol  # noqa: E402
from searches import _wave2 as wave2  # noqa: E402
from searches._runner import SAMPLERS  # noqa: E402

CATALOGUE_PATH = ROOT / "catalogue" / "search_catalogue.json"
MANIFEST_PATH = ROOT / "catalogue" / "search_manifest.json"
MANIFEST_COMMAND = (sys.executable, "-m", "autofit", "search-manifest", "--json")

STATUSES = ("measured", "unsupported", "deferred", "failed")
TASK_LABELS = {"point_map": "point/MAP", "posterior": "posterior", "evidence": "evidence"}
TASK_CRITERIA = {
    "point_map": "protocol §4(a): max_log_posterior >= ref MAP - 1 nat",
    "posterior": "protocol §4(b): relabelled medians, sigma-ratio band, ppc chi2, mode coverage",
    "evidence": "protocol §4(b) and §4(c): posterior accuracy and ln Z within 1 nat",
}
#: Where a manifest search the harness does not wire belongs (its manifest family).
FAMILY_TASK = {"mle": "point_map", "mcmc": "posterior", "nest": "evidence"}
UNSUPPORTED = {
    "Drawer": "not benchmarked: Drawer is a sanity floor that answers no task (protocol §1)",
}
BOOTSTRAP_RESAMPLES = 10_000
#: The reliability floor for a ``gaussian_x3@2`` ranking position (B5b-2 ruling 2): a
#: closed, fully-judged leg is ranked only with a success rate of at least
#: ``success_rate`` over at least ``attempts`` attempts; the rest of the closed legs are
#: listed as unreliable with their numbers. A **post-hoc presentation convention adopted
#: 2026-10-15 after seeing the wave-2 data** (protocol §8 note), never a pre-registered
#: criterion: it orders the catalogue and changes no verdict.
RANK_FLOOR = {"success_rate": 0.5, "attempts": 20}
RANK_FLOOR_NOTE = (
    "post-hoc presentation convention adopted 2026-10-15 after seeing the wave-2 data "
    "(B5b-2 ruling 2; protocol §8 note), not a pre-registered criterion: it orders the "
    "catalogue and changes no verdict"
)
#: The failure kind of a run stopped by SIGTERM (the pilot's operational wall cap; on
#: RAL, SLURM's --time).
BUDGET_KILL = "Terminated"
#: The pilot's operational caps (``timeout -s TERM``): the seed-0 probe batch, then the
#: main batch. Both are below the protocol §7 placeholder timeout (10x the median wall of
#: Nautilus n_live=100 on the same config).
PILOT_CAPS = "1800 s for the seed-0 probe, 3600 s after"


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text())


def refresh_manifest(path: Path = MANIFEST_PATH) -> dict:
    text = subprocess.check_output(MANIFEST_COMMAND, text=True, cwd=ROOT)
    manifest = json.loads(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def scan_rows(root: Path) -> list[tuple[str, dict]]:
    rows = []
    for path in sorted((root / "results" / "searches").rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        # The pilot catalogue (and the calibration built on it) reads wave-1 rows only;
        # scored wave-2 rows are catalogued under gaussian_x3@2 by their own stage (B5b).
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == 1
            and protocol.wave_of(payload) == 1
        ):
            rows.append((path.relative_to(root).as_posix(), payload))
    return rows


def usable(row: dict) -> bool:
    """A completed attempt with a finite answer (the runner marks a non-finite answer
    ``failed: non-finite result``)."""
    return row.get("status") == "complete" and row.get("completed") is True


def failure_kind(row: dict) -> str:
    """``failed: Terminated: signal …`` → ``Terminated``; ``failed: non-finite result (…)``
    → ``non-finite result``."""
    status = str(row.get("status"))
    if row.get("status") == "complete":
        return "no completion marker"
    parts = status.split(":")
    return parts[1].strip().split(" (")[0] if len(parts) > 1 else status


def bootstrap_wall_per_success(walls: list[float], successes: list[bool]) -> list | None:
    """95 % bootstrap interval of Σ wall / successes over seeds (protocol §8: 10 000
    resamples, seed 0); ``None`` when there is no success to divide by.

    A resample that draws zero successes has an unbounded wall per success: it is kept
    (as infinity) in the percentile, never discarded — discarding it would report the
    interval conditional on drawing a success. An unbounded endpoint is serialized as
    ``null`` (JSON has no infinity)."""
    if not walls or not any(successes):
        return None
    rng = random.Random(0)
    n = len(walls)
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        k = sum(successes[i] for i in idx)
        values.append(sum(walls[i] for i in idx) / k if k else math.inf)
    values.sort()
    ends = [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]
    return [None if math.isinf(v) else v for v in ends]


def leg_summary(cell: pilot.Cell, rows: list[dict], refs: dict, offsets) -> dict:
    """One (dataset × search × settings × leg) cell of the menu: counts, never a rank."""
    by_seed = {row.get("seed"): row for row in rows}
    verdicts = []
    for row in rows:
        verdict = protocol.verdict(row, exporter.select_reference(refs, row), offsets)
        verdicts.append((row, verdict))
    acceptance = Counter(v["acceptance"] for _, v in verdicts)
    convergence = Counter(v["convergence"] for _, v in verdicts)
    attempted = len(rows)
    failures = Counter(failure_kind(row) for row in rows if not usable(row))
    missing = [seed for seed in pilot.SEEDS if seed not in by_seed]
    # §8 wall per right answer: every attempt's wall plus any warm-start provider's. An
    # attempt with an unknown cost is missing, never 0: it suppresses the wall estimates.
    costs = [protocol.attempt_wall_s(row) for row, _ in verdicts]
    unknown = sum(c is None for c in costs)
    walls = [c for c in costs if c is not None]
    successes = [v["acceptance"] == "accepted" for _, v in verdicts]
    judged = acceptance.get("accepted", 0) + acceptance.get("rejected", 0)
    leg = {
        "dataset": cell.dataset,
        "sampler": cell.sampler,
        "settings": cell.settings,
        "backend": cell.backend,
        "config_id": cell.config_id,
        "expected": len(pilot.SEEDS),
        "attempted": attempted,
        "usable": sum(usable(r) for r in rows),
        "acceptance": dict(sorted(acceptance.items())),
        "convergence": dict(sorted(convergence.items())),
        "failures": dict(sorted(failures.items())),
        "deferred_seeds": missing,
        "deferred_reason": pilot.missing_reason(cell) if missing else None,
        "pyautofit_commits": sorted({r.get("pyautofit_commit") or "unknown" for r in rows}),
        "median_total_wall_s": statistics.median(walls) if walls and not unknown else None,
    }
    if unknown:
        leg["wall_cost_unknown"] = unknown
        leg["wall_reason"] = (
            f"{unknown} attempt(s) have an unknown elapsed cost: wall estimates suppressed"
        )
    if judged == attempted and attempted:
        k = acceptance.get("accepted", 0)
        lo, hi = protocol.wilson_interval(k, attempted)
        leg["success_rate"] = k / attempted
        leg["success_rate_wilson_95"] = [lo, hi]
        if unknown:
            leg["wall_per_success"] = None
            leg["wall_per_success_bootstrap_95"] = None
        else:
            leg["wall_per_success"] = protocol.wall_per_success(walls, successes)
            leg["wall_per_success_bootstrap_95"] = bootstrap_wall_per_success(walls, successes)
    else:
        leg["success_rate"] = None
        leg["success_rate_reason"] = (
            "not every attempt could be judged (a missing or incomplete reference leaves a "
            "verdict not_assessed)"
            if attempted
            else "no attempt"
        )
    return leg


# ---------------------------------------------------------------------------
# gaussian_x3@2 (scored wave 2, B5b)
# ---------------------------------------------------------------------------


def scan_wave2_rows(root: Path) -> list[tuple[str, dict]]:
    rows = []
    for path in sorted((root / "results" / "searches").rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == 1
            and protocol.wave_of(payload) == wave2.WAVE
        ):
            rows.append((path.relative_to(root).as_posix(), payload))
    return rows


def bootstrap_wall_per_success_v2(walls: list[float], successes: list[bool]) -> list | None:
    """:func:`bootstrap_wall_per_success` for the scored wave (same rule: 10 000
    resamples, seed 0, an unbounded resample kept as infinity), drawn with
    ``random.choices`` so 250-attempt legs stay fast. It resamples a leg's attempts
    (search seed × data realisation, the five realisations pooled), where §8 says "over
    seeds": with five realisations per seed a seed-only resample would ignore the
    realisation spread the scored wave was designed to include."""
    if not walls or not any(successes):
        return None
    rng = random.Random(0)
    n = len(walls)
    pairs = list(zip(walls, successes))
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        draw = rng.choices(pairs, k=n)
        k = sum(s for _, s in draw)
        values.append(sum(w for w, _ in draw) / k if k else math.inf)
    values.sort()
    ends = [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]
    return [None if math.isinf(v) else v for v in ends]


def censored_attempt(row: dict, cell) -> tuple[float | None, bool]:
    """``(cost, over_timeout)`` of one admitted attempt under §7: a run whose search.fit
    wall passed the cell's timeout hit it — a failed attempt whose consumer wall is set
    to the timeout (any warm-start provider's cost is kept on top).

    The comparison is on the consumer's ``total_wall_s``, as frozen with ``@2``
    (``_wave2.RUN_OVERHEAD_S``: "censoring at judging time compares ``total_wall_s`` with
    the timeout itself"), although a warm-start timeout was derived from the median
    *attempt* wall (provider included): the frozen rule is applied as written, and the
    difference is recorded here (B5b review; no scored row is near it — warm NUTS ~330 s
    against 11 502 s)."""
    cost = protocol.attempt_wall_s(row)
    wall = row.get("total_wall_s")
    if cell.timeout_s is None or wall is None or float(wall) <= cell.timeout_s:
        return cost, False
    provider = float(row.get("provider_wall_s") or 0.0) if row.get("warm_start") else 0.0
    return cell.timeout_s + provider, True


def wave2_leg(cell, rows: dict, refs: dict, offsets) -> dict:
    """One scored-wave cell over its 5 realisations × 50 seeds (``rows``: ``(data_seed,
    seed) -> admitted row``)."""
    # A pilot-timeout cell ran whole in the B5a scored arrays (all completed): closed. A
    # probe-sourced cell closes once every expected run has an admitted row (B5b-2 pulls
    # its remainder); until then it is open and never positioned. A missing run of a
    # closed cell is censored at the timeout per cell, i.e. the wave is treated as closed
    # cell by cell (protocol §9 says "once the wave closes"); its provider cost, if any,
    # is unknown and not added.
    closed = cell.status == "planned" and (
        cell.timeout_source == "pilot"
        or sum(not wave2.operational_kill(r, cell) for r in rows.values())
        == len(wave2.DATA_SEEDS) * len(wave2.SEEDS)
    )
    leg = {
        "id": "/".join(cell.key),
        "dataset": cell.dataset,
        "sampler": cell.sampler,
        "settings": cell.settings,
        "backend": cell.backend,
        "config_id": cell.config_id,
        "cell_status": cell.status,
        "timeout_s": cell.timeout_s,
        "timeout_source": cell.timeout_source,
        "expected": len(wave2.DATA_SEEDS) * len(wave2.SEEDS),
    }
    if cell.status == "deferred":
        leg.update({"state": "deferred", "reason": cell.reason})
        return leg
    acceptance, convergence, failures = Counter(), Counter(), Counter()
    walls, successes, unknown = [], [], 0
    censored = missing = op_kills = 0
    by_realisation = {}
    for data_seed in wave2.DATA_SEEDS:
        accepted_here = attempts_here = 0
        for seed in wave2.SEEDS:
            row = rows.get((data_seed, seed))
            if row is not None and wave2.operational_kill(row, cell):
                op_kills += 1
                row = None
            if row is None:
                if not closed:
                    continue
                # A closed cell's missing run is a failed attempt censored at its timeout.
                missing += 1
                attempts_here += 1
                acceptance["rejected"] += 1
                failures["no row (censored at the timeout)"] += 1
                walls.append(cell.timeout_s)
                successes.append(False)
                continue
            attempts_here += 1
            verdict = protocol.verdict(row, exporter.select_reference(refs, row), offsets)
            cost, over = censored_attempt(row, cell)
            ok = verdict["acceptance"] == "accepted" and not over
            if over:
                censored += 1
                failures["hit the timeout"] += 1
                acceptance["rejected"] += 1
            else:
                acceptance[verdict["acceptance"]] += 1
                if not usable(row):
                    failures[failure_kind(row)] += 1
            convergence[verdict["convergence"]] += 1
            accepted_here += ok
            if cost is None:
                unknown += 1
            else:
                walls.append(cost)
            successes.append(ok)
        by_realisation[f"data_seed{data_seed}"] = {
            "attempted": attempts_here,
            "accepted": accepted_here,
        }
    attempted = len(successes)
    k = sum(successes)
    leg.update(
        {
            "state": "closed" if closed else "open",
            "attempted": attempted,
            "acceptance": dict(sorted(acceptance.items())),
            "convergence": dict(sorted(convergence.items())),
            "failures": dict(sorted(failures.items())),
            "censored_at_timeout": censored,
            "missing_censored": missing,
            "operational_kills": op_kills,
            "by_realisation": by_realisation,
            "median_attempt_wall_s": statistics.median(walls) if walls and not unknown else None,
        }
    )
    if not closed:
        leg["reason"] = cell.reason or "open: the cell's remaining runs are not judged yet"
    if unknown:
        leg["wall_cost_unknown"] = unknown
    judged = acceptance.get("accepted", 0) + acceptance.get("rejected", 0)
    if attempted and judged == attempted:
        lo, hi = protocol.wilson_interval(k, attempted)
        leg["success_rate"] = k / attempted
        leg["success_rate_wilson_95"] = [lo, hi]
        if unknown:
            leg["wall_per_success"] = None
            leg["wall_per_success_bootstrap_95"] = None
        else:
            leg["wall_per_success"] = protocol.wall_per_success(walls, successes)
            leg["wall_per_success_bootstrap_95"] = bootstrap_wall_per_success_v2(walls, successes)
    else:
        leg["success_rate"] = None
        leg["success_rate_reason"] = (
            f"{attempted - judged} of {attempted} attempts not assessable (no complete "
            "reference of the row's identity, or no validated ln 3! offset)"
            if attempted
            else "no attempt"
        )
    return leg


def _ordering(leg: dict) -> tuple:
    return (leg["wall_per_success"]["value_s"], -leg["success_rate"], leg["id"])


def reliable(leg: dict) -> bool:
    """Whether a closed, fully-judged leg clears :data:`RANK_FLOOR`."""
    return (
        leg.get("success_rate") is not None
        and leg["success_rate"] >= RANK_FLOOR["success_rate"]
        and leg.get("attempted", 0) >= RANK_FLOOR["attempts"]
    )


def _unreliable_entry(leg: dict) -> dict:
    wps = leg.get("wall_per_success") or {}
    return {
        "id": leg["id"],
        "attempted": leg["attempted"],
        "accepted": leg["acceptance"].get("accepted", 0),
        "success_rate": leg["success_rate"],
        "success_rate_wilson_95": leg["success_rate_wilson_95"],
        "wall_per_success_s": wps.get("value_s"),
        "wall_per_success_unbounded": bool(wps.get("unbounded")),
        "wall_per_success_bootstrap_95": leg.get("wall_per_success_bootstrap_95"),
        "reason": (
            f"below the reliability floor: {leg['success_rate']:.0%} over "
            f"{leg['attempted']} attempts (floor >= {RANK_FLOOR['success_rate']:.0%} over "
            f">= {RANK_FLOOR['attempts']})"
        ),
    }


def wave2_section(root: Path, refs: dict, offsets) -> dict:
    pulled = scan_wave2_rows(root)
    timeouts = wave2.wave_timeouts()
    admitted: dict[tuple, dict] = {}
    refused = 0
    for _, row in pulled:
        if wave2.admit(row, timeouts) is not None:
            refused += 1
            continue
        admitted.setdefault(wave2.manifest_key(row)[:4], {})[
            (row.get("data_seed"), row.get("seed"))
        ] = row
    tasks = {
        task: {"label": TASK_LABELS[task], "judged_by": TASK_CRITERIA[task], "legs": []}
        for task in TASK_LABELS
    }
    for cell in wave2.cells(timeouts):
        tasks[cell.task]["legs"].append(wave2_leg(cell, admitted.get(cell.key, {}), refs, offsets))
    for group in tasks.values():
        ordering, unreliable = {}, {}
        for dataset in wave2.DATASETS:
            judged = [
                leg
                for leg in group["legs"]
                if leg["dataset"] == dataset
                and leg["state"] == "closed"
                and leg.get("success_rate") is not None
            ]
            legs = [
                leg
                for leg in judged
                if reliable(leg) and (leg.get("wall_per_success") or {}).get("value_s") is not None
            ]
            unreliable[dataset] = [
                _unreliable_entry(leg)
                for leg in sorted(
                    (leg for leg in judged if not reliable(leg)),
                    key=lambda leg: (-leg["success_rate"], leg["id"]),
                )
            ]
            ordered, previous = [], None
            for i, leg in enumerate(sorted(legs, key=_ordering)):
                ci = leg.get("wall_per_success_bootstrap_95") or [None, None]
                tied = bool(
                    previous
                    and previous[1] is not None
                    and ci[0] is not None
                    and ci[0] <= previous[1]
                )
                ordered.append(
                    {
                        "position": i + 1,
                        "id": leg["id"],
                        "attempted": leg["attempted"],
                        "accepted": leg["acceptance"].get("accepted", 0),
                        "success_rate": leg["success_rate"],
                        "success_rate_wilson_95": leg["success_rate_wilson_95"],
                        "wall_per_success_s": leg["wall_per_success"]["value_s"],
                        "wall_per_success_bootstrap_95": ci,
                        # The bootstrap 95 % interval overlaps the previous entry's: the
                        # two are not distinguished by this ordering.
                        "tied_with_previous": tied,
                    }
                )
                previous = ci
            ordering[dataset] = ordered
        group["ranking"] = ordering
        group["unreliable"] = unreliable
    commits = sorted({row.get("pyautofit_commit") or "unknown" for _, row in pulled})
    return {
        "protocol_id": wave2.PROTOCOL_ID,
        "campaign_task": wave2.WAVE_ID,
        "wave": "wave 2 (scored, protocol §9): ranks within a task and a dataset",
        "ranking": (
            "within one task and one dataset, never across: closed legs whose every attempt "
            "is judged and that clear the reliability floor (success rate >= 50 % over >= 20 "
            "attempts), by expected wall per right answer (ascending), then success rate; "
            "closed legs below the floor (zero-success legs included) are listed under "
            "unreliable with their numbers, never positioned; open legs are neither. "
            "Entries whose bootstrap intervals overlap the previous one are marked tied. "
            "The order and the floor are a presentation convention, not a verdict: §8 "
            "defines the headline numbers"
        ),
        "rank_floor": {**RANK_FLOOR, "note": RANK_FLOOR_NOTE},
        "pyautofit_pin": wave2.PYAUTOFIT_PIN,
        "pyautonerves_pin": wave2.PYAUTONERVES_PIN,
        "pyautofit_commits_pulled": commits,
        "accounting": "results/accounting/gaussian_x3_wave2.json",
        "censoring": (
            "protocol §7: an attempt that hit the timeout, crashed or wrote a non-finite "
            "answer is a failed attempt; in a closed leg a missing run is a failed attempt "
            "censored at the timeout; an operational kill (Terminated below the timeout) "
            "is not an attempt (rerun it)"
        ),
        "limitations": [
            "Pilot-derived timeouts are 10 x laptop medians and RAL ran 4-7 x faster than "
            "the laptop (B5a dry run): censoring is lenient on RAL for those cells",
            "Probe-first cells (Zeus on JAX, NSS 50-step) take 10 x their RAL probe median "
            "or the 600 s floor (results/calibration/gaussian_x3_wave2_probe.json); their "
            "remainder ran as RAL job 409930 (pulled in B5b-2)",
            *(
                [
                    f"{len(still_open)} probe-first leg(s) still open (rows not all pulled): "
                    + ", ".join(still_open)
                ]
                if (
                    still_open := [
                        leg["id"]
                        for group in tasks.values()
                        for leg in group["legs"]
                        if leg.get("timeout_source") == "probe" and leg["state"] == "open"
                    ]
                )
                else []
            ),
        ],
        "coverage": {
            "expected_runs": len(wave2.expected_runs(timeouts)),
            "pulled_rows": len(pulled),
            "admitted_rows": len(pulled) - refused,
            "refused_rows": refused,
        },
        "tasks": tasks,
    }


def search_status(cls: str, legs: list[dict], harness: list[str]) -> tuple[str, str]:
    if not harness:
        reason = UNSUPPORTED.get(cls, "registered in PyAutoFit but not wired into this harness")
        return "unsupported", reason
    attempted = sum(leg["attempted"] for leg in legs)
    usable_runs = sum(leg["usable"] for leg in legs)
    if usable_runs:
        return "measured", f"{usable_runs} usable of {attempted} attempted wave-1 runs"
    if attempted:
        kinds = Counter()
        for leg in legs:
            kinds.update(leg["failures"])
        detail = ", ".join(f"{k or 'failed'} ×{n}" for k, n in sorted(kinds.items()))
        if set(kinds) == {BUDGET_KILL}:
            # Killed by the pilot's own wall cap, below the §7 timeout: the search did
            # not fail by the protocol, the laptop budget ran out. The rows stay (each is
            # a failed attempt in its leg's counts); the search is not called failed.
            return "deferred", (
                f"{attempted} attempted, every one stopped at the pilot's operational wall "
                f"cap ({PILOT_CAPS}), below the protocol §7 timeout: budget-censored on a "
                "loaded laptop, rows kept; to be measured in wave 2 under the frozen timeout"
            )
        return "failed", f"{attempted} attempted, none usable ({detail})"
    reasons = sorted({leg["deferred_reason"] for leg in legs if leg["deferred_reason"]})
    return "deferred", "; ".join(reasons) or pilot.BUDGET_REASON


def build(root: Path = ROOT, manifest: dict | None = None) -> dict:
    manifest = manifest or load_manifest()
    refs, offsets = exporter.references(root)
    rows = scan_rows(root)
    by_cell: dict[tuple, list[dict]] = {}
    for _, row in rows:
        by_cell.setdefault(pilot.row_key(row), []).append(row)
    cells = pilot.cells()
    harness_by_class: dict[str, list[str]] = {}
    for name, spec in SAMPLERS.items():
        harness_by_class.setdefault(spec.cls, []).append(name)

    tasks = {
        task: {"label": TASK_LABELS[task], "judged_by": TASK_CRITERIA[task], "searches": {}}
        for task in TASK_LABELS
    }
    for entry in manifest["searches"]:
        cls = entry["name"]
        harness = harness_by_class.get(cls, [])
        task = SAMPLERS[harness[0]].task if harness else FAMILY_TASK.get(entry.get("family"))
        if task not in tasks:
            raise ValueError(f"{cls}: no task for manifest family {entry.get('family')!r}")
        legs = [
            leg_summary(cell, by_cell.get(cell.key, []), refs, offsets)
            for cell in cells
            if cell.sampler in harness
        ]
        status, reason = search_status(cls, legs, harness)
        if cls in pilot.SEARCH_NOTES:
            reason = f"{reason}; {pilot.SEARCH_NOTES[cls]}"
        caps = entry.get("capabilities") or {}
        tasks[task]["searches"][cls] = {
            "class_path": entry.get("class_path"),
            "family": entry.get("family"),
            "capabilities": {
                k: caps.get(k)
                for k in (
                    "jax_use",
                    "gradient",
                    "batched",
                    "posterior_kind",
                    "produces_evidence",
                    "status",
                )
            },
            "harness_samplers": harness,
            "status": status,
            "reason": reason,
            "legs": legs,
        }
    expected = pilot.expected_runs()
    rows_in_menu = sum(1 for _, row in rows if pilot.row_key(row) in {c.key for c in cells})
    commits = sorted({row.get("pyautofit_commit") or "unknown" for _, row in rows})
    return {
        "schema": "autofit-inference-search-catalogue",
        "version": 1,
        "project": exporter.PROJECT,
        "protocol_id": protocol.PROTOCOL_ID,
        "campaign_task": pilot.PILOT_ID,
        "wave": "wave 1 (pilot): ranks nothing (protocol §9)",
        "ranking": "none",
        "grouping": "by requested task; tasks are never compared",
        "manifest": {
            "source": "python -m autofit search-manifest --json",
            "snapshot": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "schema": manifest.get("schema"),
            "autofit_version": manifest.get("autofit_version"),
            "searches": len(manifest["searches"]),
        },
        "pilot_stopped": pilot.PILOT_STOPPED,
        "pyautofit_commits": commits,
        "coverage": {
            "expected_runs": len(expected),
            "rows": len(rows),
            "rows_in_menu": rows_in_menu,
            "statuses": dict(
                sorted(
                    Counter(
                        s["status"] for t in tasks.values() for s in t["searches"].values()
                    ).items()
                )
            ),
        },
        "tasks": tasks,
        wave2.PROTOCOL_ID: wave2_section(root, refs, offsets),
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2, sort_keys=False, allow_nan=False) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--refresh-manifest", action="store_true")
    args = parser.parse_args(argv)
    manifest = refresh_manifest() if args.refresh_manifest else load_manifest()
    content = render(build(ROOT, manifest))
    if args.check:
        if not CATALOGUE_PATH.exists() or CATALOGUE_PATH.read_text() != content:
            print("search catalogue is stale; run build_catalogue.py", file=sys.stderr)
            return 1
        print(f"{CATALOGUE_PATH.relative_to(ROOT)} is current")
        return 0
    CATALOGUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOGUE_PATH.write_text(content)
    print(f"wrote {CATALOGUE_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

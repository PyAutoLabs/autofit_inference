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

**Nothing is ranked.** Tasks are separate groups; within a task the searches keep the
manifest's order and each leg reports its own counts (attempts, verdicts, convergence,
the Wilson interval of its success rate, its walls). There is no rank, score or best
field, and no number is ever compared across tasks (``test_catalogue.py`` pins this).
Wave 1 is a pilot (protocol §9): its numbers calibrate the protocol, they do not order
the searches.

    python scripts/misc/tooling/build_catalogue.py                     # write the catalogue
    python scripts/misc/tooling/build_catalogue.py --check             # verify the committed one
    python scripts/misc/tooling/build_catalogue.py --refresh-manifest  # re-snapshot the A1 registry
"""

from __future__ import annotations

import argparse
import json
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
        if isinstance(payload, dict) and payload.get("schema_version") == 1:
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
    resamples, seed 0); ``None`` when there is no success to divide by."""
    if not walls or not any(successes):
        return None
    rng = random.Random(0)
    n = len(walls)
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        k = sum(successes[i] for i in idx)
        if k:
            values.append(sum(walls[i] for i in idx) / k)
    if not values:
        return None
    values.sort()
    return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]


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
    # §8 wall per right answer: every attempt's wall plus any warm-start provider's.
    walls = [
        float(row.get("total_wall_s") or 0.0) + float(row.get("provider_wall_s") or 0.0)
        for row, _ in verdicts
    ]
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
        "median_total_wall_s": statistics.median(walls) if walls else None,
    }
    if judged == attempted and attempted:
        k = acceptance.get("accepted", 0)
        lo, hi = protocol.wilson_interval(k, attempted)
        leg["success_rate"] = k / attempted
        leg["success_rate_wilson_95"] = [lo, hi]
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

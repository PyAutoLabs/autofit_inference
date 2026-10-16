"""Judge the ``correlated_nd`` pilot (RAL 410866) under ``correlated_nd@1``: admit or refuse
every pulled row, account for every expected run, and reconcile the run tree with ``sacct``.

Protocol ``wiki/project/protocol_correlated_nd.md`` §7 and §9. Every ``correlated_nd`` row
under ``results/searches/`` goes through :func:`_correlated_nd.admit` (the pilot identity:
``correlated_nd@1``, the PyAutoFit pin, a cell, realisation 1, a seed 0–9); a refused row
is listed with why. Every expected run (:func:`_correlated_nd.expected_runs`, 1040 = 930
planned + 110 deferred) is then exactly one of ``row`` / ``failed`` / ``missing`` /
``deferred`` / ``oom`` (:data:`_correlated_nd.ATTEMPT_CLASSES`). Each admitted row is judged
under ``@1`` against its own exact reference (§6), giving the per-leg attempt table: the
attempt classes, the acceptance and convergence counts, the failure kinds and the walls.
The pilot **ranks nothing** (§9): nothing here orders legs, within or across families.

The ``sacct`` extract (``results/accounting/correlated_nd_pilot_sacct.tsv``, one line per
array task, task N = manifest line N + 2) must agree with the rows: a task SLURM records
``OUT_OF_MEMORY`` is an ``oom`` run, a ``COMPLETED`` task left a row (``row`` or
``failed``), and no planned run is ``missing``.

It writes ``results/accounting/correlated_nd_pilot.json``; ``--check`` (lint) fails when
the committed record differs from what the committed rows give, or a problem is found.

    python scripts/misc/searches/judge_correlated_nd.py            # write + print
    python scripts/misc/searches/judge_correlated_nd.py --check    # lint gate
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(ROOT), str(ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from searches import _correlated_nd as cnd  # noqa: E402
from searches import _protocol as protocol  # noqa: E402
from searches import _wave2 as wave2  # noqa: E402

PATH = ROOT / "results" / "accounting" / "correlated_nd_pilot.json"
SACCT = ROOT / "results" / "accounting" / "correlated_nd_pilot_sacct.tsv"
MANIFEST = ROOT / "hpc" / "batch_cpu" / "correlated_nd" / "pilot.tsv"
JOB = "410866"

#: The pilot as planned (protocol §7, A1): 1040 expected, 930 run, 110 deferred.
EXPECTED = {"expected": 1040, "planned": 930, "deferred": 110}


def family_rows(root: Path = ROOT) -> list[tuple[str, dict]]:
    out = []
    for path in sorted((root / "results" / "searches").rglob("*.json")):
        payload = protocol.load_json(path)
        if isinstance(payload, dict) and protocol.is_family_row(payload):
            out.append((path.relative_to(root).as_posix(), payload))
    return out


def reference_for(root: Path, row: dict, cache: dict) -> dict | None:
    key = (row.get("dataset"), row.get("backend"), row.get("data_seed"))
    if key not in cache:
        cache[key] = protocol.load_json(protocol.reference_file(root, *key))
    return cache[key]


def failure_kind(row: dict) -> str:
    status = str(row.get("status"))
    parts = status.split(":")
    return parts[1].strip().split(" (")[0] if len(parts) > 1 else status


def read_sacct(path: Path = SACCT) -> list[dict]:
    lines = [ln for ln in Path(path).read_text().splitlines() if not ln.startswith("#")]
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line]


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def judge(root: Path = ROOT) -> dict:
    pulled = family_rows(root)
    rows = [row for _, row in pulled]
    acc = cnd.accounting(rows)
    admitted = {
        (*wave2.manifest_key(row)[:5], row.get("seed")): row
        for row in rows
        if cnd.admit(row) is None
    }
    refs: dict = {}
    legs = []
    for leg in acc["legs"]:
        key = (leg["dataset"], leg["sampler"], leg["settings"], leg["config_id"], 1)
        acceptance, convergence, failures = Counter(), Counter(), Counter()
        walls, process = [], []
        for seed in cnd.SEEDS:
            row = admitted.get((*key, seed))
            if row is None or seed in leg["deferred"] or seed in leg["missing"]:
                continue
            verdict = protocol.verdict(row, reference_for(root, row, refs))
            acceptance[verdict["acceptance"]] += 1
            convergence[verdict["convergence"]] += 1
            if row.get("status") != "complete":
                failures[failure_kind(row)] += 1
            elif protocol.attempt_wall_s(row) is not None:
                walls.append(protocol.attempt_wall_s(row))
            if row.get("process_elapsed_s") is not None:
                process.append(float(row["process_elapsed_s"]))
        attempts = leg["row"] + len(leg["failed"]) + len(leg["oom"])
        accepted = acceptance.get("accepted", 0)
        legs.append(
            {
                **{k: leg[k] for k in ("id", "dataset", "sampler", "settings", "config_id")},
                "task": leg["task"],
                "cell_status": leg["cell_status"],
                "expected": leg["expected"],
                **{state: leg[state] for state in cnd.ATTEMPT_CLASSES},
                "attempts": attempts,
                "acceptance": dict(sorted(acceptance.items())),
                "convergence": dict(sorted(convergence.items())),
                "failures": dict(sorted(failures.items())),
                "success_rate": accepted / attempts if attempts else None,
                "success_rate_wilson_95": list(protocol.wilson_interval(accepted, attempts))
                if attempts
                else None,
                "median_attempt_wall_s": _median(walls),
                "median_process_elapsed_s": _median(process),
                "reason": leg["reason"],
            }
        )
    totals = acc["totals"]
    problems = []
    if totals["expected"] != EXPECTED["expected"]:
        problems.append(f"expected {totals['expected']} != {EXPECTED['expected']}")
    if totals["deferred"] != EXPECTED["deferred"]:
        problems.append(f"deferred {totals['deferred']} != {EXPECTED['deferred']}")
    if sum(totals[s] for s in cnd.ATTEMPT_CLASSES) != totals["expected"]:
        problems.append(" + ".join(cnd.ATTEMPT_CLASSES) + " != expected")
    if totals["missing"]:
        problems.append(f"{totals['missing']} planned runs are missing")
    # sacct reconciliation: task N = manifest line N + 2.
    tasks = MANIFEST.read_text().splitlines()[1:]
    by_leg = {leg["id"]: leg for leg in acc["legs"]}
    states = Counter()
    sacct = (
        read_sacct(root / SACCT.relative_to(ROOT))
        if (root / SACCT.relative_to(ROOT)).exists()
        else []
    )
    if len(sacct) != EXPECTED["planned"]:
        problems.append(f"sacct has {len(sacct)} tasks, not {EXPECTED['planned']}")
    oom_tasks = []
    for record in sacct:
        dataset, sampler, settings, config, data_seed, _, seed = tasks[int(record["task"])].split(
            "\t"
        )
        config_id = next(
            c.config_id
            for c in cnd.cells()
            if (c.dataset, c.sampler, c.settings, c.config_name)
            == (dataset, sampler, settings, config)
        )
        leg = by_leg["/".join((dataset, sampler, settings, config_id, f"data_seed{data_seed}"))]
        seed = int(seed)
        cls = next(
            (s for s in ("oom", "missing", "deferred", "failed") if seed in leg[s]),
            "row",
        )
        states[(record["state"], cls)] += 1
        if record["state"] == "OUT_OF_MEMORY":
            oom_tasks.append(
                {
                    "task": int(record["task"]),
                    "run": f"{leg['id']}/seed{seed}",
                    "elapsed_s": int(record["elapsed_s"]),
                    "max_rss_kb": int(record["max_rss_kb"]) if record["max_rss_kb"] else None,
                    "req_mem": record["req_mem"],
                }
            )
            if cls != "oom":
                problems.append(f"task {record['task']} is OUT_OF_MEMORY but {cls}")
        elif record["state"] == "COMPLETED" and cls not in ("row", "failed"):
            problems.append(f"task {record['task']} COMPLETED but {cls}")
        elif record["state"] not in ("COMPLETED", "OUT_OF_MEMORY"):
            problems.append(f"task {record['task']} sacct state {record['state']}")
    refused_reasons = Counter(r["reason"].split(":")[0].split(" (")[0] for r in acc["refused"])
    acceptance = Counter()
    for leg in legs:
        acceptance.update(leg["acceptance"])
    return {
        "schema": "autofit-inference-wave-accounting",
        "version": 1,
        "wave": cnd.WAVE_ID,
        "protocol_id": cnd.PROTOCOL_ID,
        "ranking": "none: the pilot ranks nothing (protocol correlated_nd §9)",
        "pyautofit_pin": cnd.PYAUTOFIT_PIN,
        "pyautonerves_pin": cnd.PYAUTONERVES_PIN,
        "slurm_job": JOB,
        "pulled_rows": len(pulled),
        "admitted_rows": len(pulled) - len(acc["refused"]),
        "refused_rows": len(acc["refused"]),
        "refused_by_reason": dict(sorted(refused_reasons.items())),
        "totals": totals,
        "acceptance": dict(sorted(acceptance.items())),
        "expected_split": EXPECTED,
        "sacct_vs_rows": {f"{s} -> {c}": n for (s, c), n in sorted(states.items())},
        "oom_tasks": oom_tasks,
        "problems": problems,
        "refused": acc["refused"],
        "legs": legs,
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, allow_nan=False) + "\n"


def report(doc: dict) -> None:
    print(
        f"{doc['wave']} @ {doc['pyautofit_pin'][:12]}: {doc['pulled_rows']} pulled, "
        f"{doc['admitted_rows']} admitted, {doc['refused_rows']} refused"
    )
    print(f"  totals {doc['totals']}")
    print(f"  acceptance (@1) {doc['acceptance']}")
    print(f"  sacct vs rows {doc['sacct_vs_rows']}")
    for line in doc["problems"]:
        print(f"  PROBLEM: {line}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    doc = judge(ROOT)
    content = render(doc)
    if args.check:
        if not PATH.exists() or PATH.read_text() != content:
            print(f"{PATH.relative_to(ROOT)} is stale; run judge_correlated_nd.py", file=sys.stderr)
            return 1
        if doc["problems"]:
            print("; ".join(doc["problems"]), file=sys.stderr)
            return 1
        print(f"{PATH.relative_to(ROOT)} is current ({doc['admitted_rows']} admitted rows)")
        return 0
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(content)
    report(doc)
    return 1 if doc["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Judge scored wave 2's attempt accounting (``gaussian_x3@2``): admit or refuse every
pulled row, and account for every expected run.

Every ``wave: 2`` row under ``results/searches/`` goes through :func:`_wave2.admit` (the
manifest identity: the wave's protocol, the pinned PyAutoFit revision
:data:`_wave2.PYAUTOFIT_PIN`, a menu cell, a realisation and a seed of the wave); a refused
row is listed with why and is never an attempt of the wave. Every expected run
(:func:`_wave2.expected_runs`) is then exactly one of ``row`` / ``failed`` / ``missing`` /
``deferred`` (:func:`_wave2.accounting`), with operational kills (a ``Terminated`` row
below its cell's protocol timeout) counted ``missing``, never ``failed``.

This is accounting, not acceptance: the per-row ``@2`` verdicts are computed by the
catalogue and the exporter against each row's own-realisation reference.

It writes ``results/accounting/gaussian_x3_wave2.json`` and checks the totals against
the wave as planned (:data:`EXPECTED`); ``--check`` (lint) fails when the committed record
differs from what the committed rows give.

    python scripts/misc/searches/judge_wave2.py            # write + print
    python scripts/misc/searches/judge_wave2.py --check    # lint gate
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
if str(ROOT / "scripts" / "misc") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "misc"))

from searches import _protocol as protocol  # noqa: E402
from searches import _wave2 as wave2  # noqa: E402

PATH = ROOT / "results" / "accounting" / "gaussian_x3_wave2.json"

#: The wave as B5a planned it (``wiki/project/state.md``): 11 750 runs in the scored
#: arrays (pilot-derived timeouts), 4 probe-first cells of 250 runs (20 probe runs, then
#: 980 more once their timeouts are written) and 250 deferred (blend warm NUTS).
EXPECTED = {"expected": 13000, "scored": 11750, "probe": 20, "remainder": 980, "deferred": 250}


def wave2_rows(root: Path = ROOT) -> list[tuple[str, dict]]:
    out = []
    for path in sorted((root / "results" / "searches").rglob("*.json")):
        payload = protocol.load_json(path)
        if isinstance(payload, dict) and protocol.wave_of(payload) == wave2.WAVE:
            out.append((path.relative_to(root).as_posix(), payload))
    return out


def _is_probe_run(data_seed: int, seed: int) -> bool:
    return data_seed == wave2.PROBE_DATA_SEED and seed in wave2.PROBE_SEEDS


def judge(root: Path = ROOT) -> dict:
    pulled = wave2_rows(root)
    rows = [row for _, row in pulled]
    timeouts = wave2.wave_timeouts()
    cells = {c.key: c for c in wave2.cells(timeouts)}
    acc = wave2.accounting(rows, timeouts)
    kills = []
    for rel, row in pulled:
        if wave2.admit(row, timeouts) is not None:
            continue
        cell = cells[wave2.manifest_key(row)[:4]]
        if wave2.operational_kill(row, cell):
            kills.append(
                {
                    "path": rel,
                    "total_wall_s": row.get("total_wall_s"),
                    "timeout_s": cell.timeout_s,
                    "status": row.get("status"),
                }
            )
    # The planned split: the scored arrays, the probe and its remainder, the deferred.
    split = {k: Counter() for k in ("scored", "probe", "remainder", "deferred")}  # state -> count
    for leg in acc["legs"]:
        cell = cells[(leg["dataset"], leg["sampler"], leg["settings"], leg["config_id"])]
        for seed in wave2.SEEDS:
            if seed in leg["deferred"]:
                group, state = "deferred", "deferred"
            else:
                if cell.timeout_source == "probe" or cell.status == "probe_first":
                    group = "probe" if _is_probe_run(leg["data_seed"], seed) else "remainder"
                else:
                    group = "scored"
                state = (
                    "failed"
                    if seed in leg["failed"]
                    else "missing"
                    if seed in leg["missing"]
                    else "row"
                )
            split[group][state] += 1
    by_group = {
        group: {"expected": sum(c.values()), **dict(sorted(c.items()))}
        for group, c in split.items()
    }
    problems = [
        f"{group}: expected {EXPECTED[group]}, accounted {by_group[group]['expected']}"
        for group in ("scored", "probe", "remainder", "deferred")
        if by_group[group]["expected"] != EXPECTED[group]
    ]
    if acc["totals"]["expected"] != EXPECTED["expected"]:
        problems.append(f"total expected {acc['totals']['expected']} != {EXPECTED['expected']}")
    totals = acc["totals"]
    if sum(totals[s] for s in ("row", "failed", "missing", "deferred")) != totals["expected"]:
        problems.append("row + failed + missing + deferred != expected")
    refused_reasons = Counter(r["reason"].split(":")[0].split(" (")[0] for r in acc["refused"])
    return {
        "schema": "autofit-inference-wave-accounting",
        "version": 1,
        "wave": wave2.WAVE_ID,
        "protocol_id": wave2.PROTOCOL_ID,
        "pyautofit_pin": wave2.PYAUTOFIT_PIN,
        "pyautonerves_pin": wave2.PYAUTONERVES_PIN,
        "pulled_rows": len(pulled),
        "admitted_rows": len(pulled) - len(acc["refused"]),
        "refused_rows": len(acc["refused"]),
        "refused_by_reason": dict(sorted(refused_reasons.items())),
        "totals": totals,
        "by_group": by_group,
        "expected_split": EXPECTED,
        "problems": problems,
        "operational_kills": kills,
        "refused": acc["refused"],
        "legs": acc["legs"],
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, allow_nan=False) + "\n"


def report(doc: dict) -> None:
    print(
        f"{doc['wave']} @ {doc['pyautofit_pin'][:12]}: {doc['pulled_rows']} pulled, "
        f"{doc['admitted_rows']} admitted, {doc['refused_rows']} refused"
    )
    print(f"  totals {doc['totals']}")
    for group, counts in doc["by_group"].items():
        print(f"  {group:9s} {counts}")
    print(f"  operational kills: {len(doc['operational_kills'])}")
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
            print(f"{PATH.relative_to(ROOT)} is stale; run judge_wave2.py", file=sys.stderr)
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

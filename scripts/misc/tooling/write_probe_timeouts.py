"""Write ``results/calibration/gaussian_x3_wave2_probe.json``: the probe-first cells' ``@2``
timeouts from their 5-run probes (protocol §11 A2, ``_protocol.TIMEOUT_RULE_V2``).

A probe-first cell had no pilot wall, so B5a ran a probe of it — data realisation
:data:`_wave2.PROBE_DATA_SEED`, search seeds :data:`_wave2.PROBE_SEEDS`, on each dataset —
under the operational cap :data:`_wave2.PROBE_CAP_S`. Its timeout is then the frozen rule
over those runs, exactly as the wave-1 calibration applied it to the pilot
(``calibrate_pilot.py``): ``max(600 s, 10 × the median attempt wall of the completed
probe runs)``, per (search × settings × backend) and pooled over both datasets. Every
probe row is cited with its wall and status.

``_wave2.cells()`` reads this record after the wave-1 calibration, so it only fills a cell
the pilot left without a timeout; ``plan_wave2.py`` then generates the cell's remaining
runs (``submit_wave2_remainder``).

    python scripts/misc/tooling/write_probe_timeouts.py           # write the record
    python scripts/misc/tooling/write_probe_timeouts.py --check   # verify the committed one
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(ROOT / "scripts" / "misc"), str(ROOT / "scripts" / "misc" / "tooling")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import calibrate_pilot  # noqa: E402
from searches import _protocol as protocol  # noqa: E402
from searches import _wave2 as wave2  # noqa: E402

PATH = wave2.PROBE_CALIBRATION


def wave2_rows(root: Path = ROOT) -> list[tuple[str, dict]]:
    """Every scored-wave row under ``results/searches`` (``wave: 2``), path-sorted."""
    out = []
    for path in sorted((root / "results" / "searches").rglob("*.json")):
        payload = protocol.load_json(path)
        if isinstance(payload, dict) and protocol.wave_of(payload) == wave2.WAVE:
            out.append((path.relative_to(root).as_posix(), payload))
    return out


def usable(row: dict) -> bool:
    return row.get("status") == "complete" and row.get("completed") is True


def build(root: Path = ROOT) -> dict:
    # The probe-first cells as B5a submitted them: the wave-1 timeouts alone.
    pre = wave2.cells(wave2.calibration_timeouts())
    probe_cells = {c.key: c for c in pre if c.status == "probe_first"}
    cited, walls = [], defaultdict(list)
    for rel, row in wave2_rows(root):
        if wave2.admit(row, wave2.calibration_timeouts()) is not None:
            continue
        cell = probe_cells.get(wave2.manifest_key(row)[:4])
        if cell is None or row.get("data_seed") != wave2.PROBE_DATA_SEED:
            continue
        if row.get("seed") not in wave2.PROBE_SEEDS:
            continue
        cost = protocol.attempt_wall_s(row)
        key = f"{cell.sampler}/{cell.settings}/{cell.backend}"
        ok = usable(row) and cost is not None
        cited.append(
            {
                "path": rel,
                "cell": key,
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "status": row.get("status"),
                "attempt_wall_s": cost,
                "counted": ok,
                "slurm_array_job_id": row.get("slurm_array_job_id"),
            }
        )
        if ok:
            walls[key].append(cost)
    expected = len(probe_cells) * len(wave2.PROBE_SEEDS)
    keys = sorted({f"{c.sampler}/{c.settings}/{c.backend}" for c in probe_cells.values()})
    timeouts, unresolved = {}, []
    for key in keys:
        values = walls.get(key, [])
        if not values:
            unresolved.append(key)
            continue
        median = statistics.median(values)
        timeouts[key] = {
            "median_probe_wall_s": median,
            "n": len(values),
            "timeout_s": max(
                calibrate_pilot.TIMEOUT_FLOOR_S, calibrate_pilot.TIMEOUT_FACTOR * median
            ),
        }
    doc = {
        "schema": "autofit-inference-probe-calibration",
        "version": 1,
        "protocol_id": wave2.PROTOCOL_ID,
        "campaign_task": wave2.WAVE_ID,
        "pyautofit_pin": wave2.PYAUTOFIT_PIN,
        "timeout_rule": protocol.TIMEOUT_RULE_V2,
        "rule_applied": (
            f"max({calibrate_pilot.TIMEOUT_FLOOR_S:.0f} s, "
            f"{calibrate_pilot.TIMEOUT_FACTOR:.0f} x median attempt wall of the completed "
            "probe runs), per (search x settings x backend), pooled over both datasets "
            "(as calibrate_pilot.py pools the pilot)"
        ),
        "probe": {
            "data_seed": wave2.PROBE_DATA_SEED,
            "seeds": list(wave2.PROBE_SEEDS),
            "cap_s": wave2.PROBE_CAP_S,
            "cells": [
                f"{c.dataset}/{c.sampler}/{c.settings}/{c.config_id}" for c in probe_cells.values()
            ],
            "expected_rows": expected,
            "rows": len(cited),
        },
        "timeouts": timeouts,
        "unresolved": unresolved,
        "rows": cited,
        "limitations": [
            "The probe ran on RAL (2 CPUs per task), whereas every pilot-derived timeout is "
            "10 x a laptop median and RAL measured 4-7 x faster than the laptop (B5a dry "
            "run): these probe-derived timeouts are tighter relative to their cells' "
            "walls than the pilot-derived ones, which are lenient on RAL",
        ],
    }
    return doc


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2, allow_nan=False) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    doc = build(ROOT)
    content = render(doc)
    if args.check:
        if not PATH.exists() or PATH.read_text() != content:
            print(
                f"{PATH.relative_to(ROOT)} is stale; run write_probe_timeouts.py", file=sys.stderr
            )
            return 1
        print(f"{PATH.relative_to(ROOT)} is current")
        return 0
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(content)
    print(
        f"wrote {PATH.relative_to(ROOT)}: {doc['probe']['rows']} of "
        f"{doc['probe']['expected_rows']} probe rows"
    )
    for key, entry in doc["timeouts"].items():
        print(
            f"  {key}: median {entry['median_probe_wall_s']:.1f} s over {entry['n']} -> "
            f"timeout {entry['timeout_s']:.0f} s"
        )
    for key in doc["unresolved"]:
        print(f"  {key}: UNRESOLVED (no completed probe run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

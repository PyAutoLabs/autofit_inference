"""Calibrate ``correlated_nd@1``'s PLACEHOLDERs, timeouts, censoring and memory requests
from its pilot (RAL 410866), for the ``correlated_nd@2`` freeze.

Protocol ``wiki/project/protocol_correlated_nd.md`` §7 (the sibling of
``calibrate_pilot.py``, which calibrates ``gaussian_x3`` and is not touched): every
PLACEHOLDER — the marginal and whitened σ-ratio bands, the nested ESS bar, the chain R-hat
and bulk-ESS bars — plus the per-cell timeout and the censoring rule are calibrated on the
pilot by the ``gaussian_x3`` rules where they apply, then frozen as ``correlated_nd@2``
before any scored row. This script computes the evidence and applies the rules below; it
writes ``results/calibration/correlated_nd_wave1.json``, and the frozen values are copied
by hand into ``_protocol.FROZEN_CND2`` and protocol §7/§11 A2 (a human-readable diff; a
test pins the two together). It also derives this family's wall-rates table
(``wall/rates.py`` ``CORRELATED_ND_STEP_RATE``) and ``--check`` fails when it differs.
Standard library only.

The rules (the ``gaussian_x3`` §7 rules carried over; protocol §7 names them)
---------------------------------------------------------------------------

**Calibration set, per task** (posterior, evidence): admitted pilot rows of the task that
completed, ``converged`` under ``@1`` §5, and pass §4(b)1 (every marginal median within 1
reference σ), the median half of §4(b)3 (every whitened median within 1) and, on the
evidence task, §4(c) (ln Z within 1 nat) — they found the right posterior by every
criterion that does not involve σ; what scatters is the finite-sample estimate of σ. (The
``gaussian_x3`` set was nested-only and used mode coverage; this family has no modes, and
the protocol's rule is per task. §4(c) is required because a collapsed banana posterior —
Nautilus ``n_live_200``, σ ratios 0.04–0.13 — passes the medians against the banana's wide
reference σ and would otherwise drag the lower band edge to 0; decided in B5c-2a after
seeing the pilot, and flagged.)

- **σ-ratio bands** (marginal, and whitened, each on its own row statistics), **per
  task** ("a band may move only on ≥ 20 completed pilot rows of its task", §7) —
  ``lo = floor_0.05(q0.5 of each row's minimum ratio)``,
  ``hi = ceil_0.05(q99.5 of each row's maximum ratio)`` over the task's calibration rows,
  widened if needed to contain the §6 item 2 exact-draw tolerance ``[0.98, 1.02]`` and
  every committed reference's 200 000-draw ratios, so the exact draws can never fail it.
  A task with fewer than :data:`MIN_CALIBRATION_ROWS` calibration rows keeps the ``@1``
  band, the shortfall recorded. ``@1`` had one band for both tasks; ``@2`` keeps a band
  per task (each task is judged only within itself, §1).
- **ESS and R-hat bars** — the ``@1`` values are standard (Kish ESS 500; R-hat 1.01 and
  bulk ESS 400: Vehtari et al. 2021) and are kept; the pilot distributions are recorded,
  and a bar would change only if no completed row of its family could ever meet it at
  pilot settings (recorded, never silently loosened).
- **Timeout** per (dataset × search × settings × backend): ``max(600 s, 10 × the median
  pilot attempt wall)`` (``total_wall_s`` plus any warm-start provider) over the cell's
  completed rows; a planned cell with no completed row is ``probe_first``.
- **Cap rule** — a cell none of whose pilot attempts ended on its own (every one stopped
  at the 4 h process cap, or killed out of memory before it) is deferred from the scored
  wave with that reason; a warm start whose provider is so deferred is deferred with it
  (§7 rule 3).
- **Censoring** — as ``gaussian_x3@2``: an expected run that hit its timeout, crashed,
  wrote a non-finite answer or left no row is an attempted failure censored at the
  timeout; an ``oom`` run (SLURM ``OUT_OF_MEMORY``) is an operational failure of the
  request, rerun once at twice the cell's request, and a second ``oom`` is a failed
  attempt censored at its wall.
- **Memory** per cell class (dataset × search × settings, both backends pooled; A2):
  ``peak`` = the largest sampled ``sacct`` MaxRSS of the class's pilot tasks, or the
  8 G request itself when any task was killed out of memory (the sampled RSS undercounts
  the peak at the kill); ``--mem = max(8 G, ceil(`` :data:`MEMORY_HEADROOM` ``× peak))``.

    python scripts/misc/tooling/calibrate_correlated_nd.py           # write the record
    python scripts/misc/tooling/calibrate_correlated_nd.py --check   # verify it (lint)
    python scripts/misc/tooling/calibrate_correlated_nd.py --rates   # print the rates table
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (
    str(ROOT),
    str(ROOT / "scripts" / "misc"),
    str(ROOT / "scripts" / "misc" / "tooling"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from calibrate_pilot import ceil_to, describe, floor_to, quantile  # noqa: E402
from searches import _correlated_nd as cnd  # noqa: E402
from searches import _protocol as protocol  # noqa: E402
from searches import judge_correlated_nd as judge  # noqa: E402
from wall import rates  # noqa: E402

OUT = ROOT / "results" / "calibration" / "correlated_nd_wave1.json"
MIN_CALIBRATION_ROWS = 20
TIMEOUT_FLOOR_S = 600.0
TIMEOUT_FACTOR = 10.0
#: §6 item 2: 200 000 exact draws pass with every σ ratio within 1 ± 0.02.
EXACT_DRAW_TOLERANCE = 0.02
#: A2 memory headroom over the measured peak (decide-and-flag, B5c-2a).
MEMORY_HEADROOM = 2.0
PILOT_MEM_GB = 8
TASKS = ("posterior", "evidence")


def _ratios(row: dict, ref: dict) -> list[float]:
    out = []
    for key, ref_stats in (ref.get("posterior_relabelled") or {}).items():
        stats = (row.get("posterior_relabelled") or {}).get(key) or {}
        if ref_stats.get("sigma") and stats.get("sigma") is not None:
            out.append(stats["sigma"] / ref_stats["sigma"])
    return out


def _passes(verdict: dict, criterion: str) -> bool:
    return any(c["detail"].startswith(criterion) and c["ok"] is True for c in verdict["criteria"])


def _passes_medians(verdict: dict) -> bool:
    return _passes(verdict, "(b)1 ")


def exact_draw_ratios(root: Path) -> dict:
    """The committed references' 200 000-draw σ ratios (marginal and whitened)."""
    marginal, whitened = [], []
    for dataset in cnd.DATASETS:
        for backend in ("numpy", "jax_cpu"):
            ref = protocol.load_json(protocol.reference_file(root, dataset, backend, 1)) or {}
            draws = (ref.get("agreement") or {}).get("draws") or {}
            off = draws.get("worst_sigma_ratio_offset")
            if off is not None:
                marginal += [1 - off, 1 + off]
            w = draws.get("whitened") or {}
            whitened += [w[k] for k in ("sigma_ratio_min", "sigma_ratio_max") if k in w]
    return {"marginal": marginal, "whitened": whitened}


def band(row_min: dict, row_max: dict, exact: list[float], at_one) -> tuple[dict, dict]:
    """One σ-ratio band **per task** from that task's row statistics (see the module
    rules): ``({task: [lo, hi]}, {task: note})``. A task with fewer than
    :data:`MIN_CALIBRATION_ROWS` calibration rows keeps the ``@1`` band."""
    bands, notes = {}, {}
    floor_lo = min([1 - EXACT_DRAW_TOLERANCE, *exact])
    floor_hi = max([1 + EXACT_DRAW_TOLERANCE, *exact])
    for task in TASKS:
        n = len(row_min[task])
        if n < MIN_CALIBRATION_ROWS:
            bands[task] = list(at_one)
            notes[task] = f"kept at @1: {n} calibration rows (< {MIN_CALIBRATION_ROWS})"
            continue
        lo = floor_to(quantile(row_min[task], 0.005), 0.05)
        hi = ceil_to(quantile(row_max[task], 0.995), 0.05)
        lo, hi = min(lo, floor_to(floor_lo, 0.05)), max(hi, ceil_to(floor_hi, 0.05))
        bands[task] = [round(lo, 2), round(hi, 2)]
        notes[task] = (
            f"calibrated on {n} rows; contains the exact-draw tolerance "
            f"[{1 - EXACT_DRAW_TOLERANCE}, {1 + EXACT_DRAW_TOLERANCE}] and every reference's "
            "200 000-draw ratio"
        )
    return bands, notes


def memory(root: Path) -> dict:
    """Per cell class (dataset/sampler/settings): sacct peaks and the A2 request."""
    tasks = judge.MANIFEST.read_text().splitlines()[1:]
    by_class: dict[str, dict] = {}
    for record in judge.read_sacct(root / judge.SACCT.relative_to(ROOT)):
        dataset, sampler, settings, *_ = tasks[int(record["task"])].split("\t")
        entry = by_class.setdefault(
            f"{dataset}/{sampler}/{settings}",
            {"tasks": 0, "sampled": 0, "max_rss_gb": None, "oom": 0},
        )
        entry["tasks"] += 1
        if record["max_rss_kb"]:
            gb = int(record["max_rss_kb"]) / 1024**2
            entry["sampled"] += 1
            entry["max_rss_gb"] = round(max(entry["max_rss_gb"] or 0.0, gb), 2)
        entry["oom"] += record["state"] == "OUT_OF_MEMORY"
    for entry in by_class.values():
        peak = entry["max_rss_gb"] or 0.0
        if entry["oom"]:
            peak = max(peak, float(PILOT_MEM_GB))
        entry["peak_gb"] = round(peak, 2)
        entry["mem_gb"] = max(PILOT_MEM_GB, math.ceil(MEMORY_HEADROOM * peak - 1e-9))
    return dict(sorted(by_class.items()))


def pilot_walls(rows: list[dict]) -> dict[tuple, list[float]]:
    """Admitted rows' attempt walls per (dataset, sampler, settings, backend), over every
    attempt that ran to its own end (never one stopped by the cap: ``Terminated``; never an
    ``oom`` kill) — the ``wall/rates.py`` basis."""
    walls: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if cnd.admit(row) is not None or "Terminated" in str(row.get("status")):
            continue
        if cnd.out_of_memory(row):
            continue
        cost = protocol.attempt_wall_s(row)
        if cost is not None:
            key = (row["dataset"], row["sampler"], row["settings_name"], row["backend"])
            walls[key].append(cost)
    return walls


def derived_rates(rows: list[dict]) -> dict[tuple, float]:
    """``wall/rates.py`` ``CORRELATED_ND_STEP_RATE`` from the pilot: per (dataset, cell,
    backend) the median wall of one run of the cell's slowest settings (RAL, 2 CPUs)."""
    out: dict[tuple, float] = {}
    for (dataset, sampler, _settings, backend), values in pilot_walls(rows).items():
        key = (dataset, f"searches/{sampler}", cnd.MODEL_TYPE, f"ral_{backend}", "fp64", 1, None)
        out[key] = max(out.get(key, 0.0), statistics.median(values))
    return dict(sorted(out.items()))


def calibrate(root: Path = ROOT) -> dict:
    pulled = [row for _, row in judge.family_rows(root)]
    rows = [row for row in pulled if cnd.admit(row) is None]
    refs: dict = {}
    row_min = {t: [] for t in TASKS}
    row_max = {t: [] for t in TASKS}
    w_min = {t: [] for t in TASKS}
    w_max = {t: [] for t in TASKS}
    calib_ids = {t: [] for t in TASKS}
    nested_ess, rhat, ess_bulk = [], [], []
    walls: dict[tuple, list[float]] = defaultdict(list)
    ended: dict[tuple, int] = defaultdict(int)
    attempts: dict[tuple, int] = defaultdict(int)
    for row in rows:
        cell = (row["dataset"], row["sampler"], row["settings_name"], row["backend"])
        attempts[cell] += 1
        capped = "Terminated" in str(row.get("status")) and not cnd.operational_kill(row)
        if not capped and not cnd.out_of_memory(row):
            ended[cell] += 1
        if row.get("status") != "complete" or row.get("completed") is not True:
            continue
        cost = protocol.attempt_wall_s(row)
        if cost is not None:
            walls[cell].append(cost)
        if row.get("family") == "nested" and row.get("ess_kish") is not None:
            nested_ess.append(float(row["ess_kish"]))
        if row.get("rhat_max") is not None:
            rhat.append(float(row["rhat_max"]))
        if row.get("ess_bulk_min") is not None:
            ess_bulk.append(float(row["ess_bulk_min"]))
        task = row.get("task")
        if task not in TASKS:
            continue
        ref = judge.reference_for(root, row, refs)
        verdict = protocol.verdict(row, ref)
        whitened = row.get("whitened") or {}
        if verdict["convergence"] != "converged" or not _passes_medians(verdict):
            continue
        if (whitened.get("worst_median") or math.inf) > protocol.THRESHOLDS_CND1[
            "whitened_median_tolerance"
        ]:
            continue
        if task == "evidence" and not _passes(verdict, "(c)"):
            continue
        ratios = _ratios(row, ref)
        if not ratios or whitened.get("sigma_ratio_min") is None:
            continue
        calib_ids[task].append(row["run_id"])
        row_min[task].append(min(ratios))
        row_max[task].append(max(ratios))
        w_min[task].append(float(whitened["sigma_ratio_min"]))
        w_max[task].append(float(whitened["sigma_ratio_max"]))

    t = protocol.THRESHOLDS_CND1
    exact = exact_draw_ratios(root)
    frozen, notes = {}, {}
    frozen["sigma_ratio_band"], notes["sigma_ratio_band"] = band(
        row_min, row_max, exact["marginal"], t["sigma_ratio_band"]
    )
    frozen["whitened_sigma_ratio_band"], notes["whitened_sigma_ratio_band"] = band(
        w_min, w_max, exact["whitened"], t["whitened_sigma_ratio_band"]
    )
    for key in ("nested_min_ess", "chain_max_rhat", "chain_min_ess"):
        frozen[key] = t[key]
    notes["nested_min_ess"] = (
        "kept at @1 (standard)"
        if nested_ess and max(nested_ess) >= t["nested_min_ess"]
        else "kept at @1; no completed nested row reaches the bar at pilot settings"
    )
    chain_ok = [
        r <= t["chain_max_rhat"] and e >= t["chain_min_ess"]
        for r, e in ((row.get("rhat_max"), row.get("ess_bulk_min")) for row in rows)
        if r is not None and e is not None
    ]
    notes["chain_max_rhat"] = notes["chain_min_ess"] = (
        f"kept at @1 (standard); {sum(chain_ok)} of {len(chain_ok)} completed chain rows "
        "with diagnostics meet both bars"
    )

    # Timeouts and the cap rule, per planned cell of the pilot.
    timeouts, probe_first, scored_deferred = {}, {}, {}
    for cell in cnd.cells():
        key = (cell.dataset, cell.sampler, cell.settings, cell.backend)
        name = "/".join(key)
        if cell.status == "deferred":
            scored_deferred[name] = cell.reason
            continue
        a2 = cnd.A2_DEFERRED.get(key[:3])
        if a2:
            scored_deferred[name] = a2
            continue
        if attempts[key] and not ended[key]:
            scored_deferred[name] = (
                f"deferred (cap rule, protocol §7): none of the {attempts[key]} pilot attempts "
                "ended on its own — each was stopped at the 4 h process cap or killed out of "
                "memory before it"
            )
            continue
        values = walls.get(key)
        if values:
            median = statistics.median(values)
            timeouts[name] = {
                "median_pilot_wall_s": median,
                "n": len(values),
                "timeout_s": max(TIMEOUT_FLOOR_S, TIMEOUT_FACTOR * median),
            }
        else:
            probe_first[name] = (
                "no completed pilot row (the runner's chain diagnostics failed after the "
                "search, fixed in B5c-2a): the timeout comes from a probe in the scored wave"
            )
    # A warm start whose provider is deferred by the cap rule is deferred with it (rule 3).
    for name in list(timeouts) + list(probe_first):
        dataset, sampler, settings, backend = name.split("/")
        warm = cnd.SAMPLERS[sampler].extra.get("warm_from")
        if not warm:
            continue
        providers = [n for n in scored_deferred if n.startswith(f"{dataset}/{warm[0]}/{warm[1]}/")]
        if providers:
            timeouts.pop(name, None)
            probe_first.pop(name, None)
            scored_deferred[name] = (
                f"deferred (warm-start provider, protocol §7 rule 3): its provider "
                f"{warm[0]}/{warm[1]} is deferred at {dataset}"
            )
    for name, reason in list(scored_deferred.items()):
        dataset, sampler, settings, backend = name.split("/")
        warm = cnd.SAMPLERS[sampler].extra.get("warm_from")
        if warm and "cap rule" in reason:
            scored_deferred[name] = (
                f"{reason}; and its provider {warm[0]}/{warm[1]} is deferred at {dataset} "
                "(protocol §7 rule 3)"
            )

    mem = memory(root)
    raised = {
        name: entry["mem_gb"]
        for name, entry in mem.items()
        if entry["mem_gb"] > PILOT_MEM_GB
        and any(
            n.startswith(name + "/") and n not in scored_deferred for n in (*timeouts, *probe_first)
        )
    }
    acc = judge.judge(root)
    return {
        "schema": "autofit-inference-calibration",
        "version": 1,
        "family": cnd.MODEL_TYPE,
        "protocol_from": cnd.PROTOCOL_ID,
        "protocol_to": "correlated_nd@2",
        "campaign_task": cnd.WAVE_ID,
        "slurm_job": judge.JOB,
        "pyautofit_pin": cnd.PYAUTOFIT_PIN,
        "rows": len(pulled),
        "admitted_rows": len(rows),
        "accounting": acc["totals"],
        "calibration_set": {
            "rule": "per task: completed, converged under @1 §5, passes §4(b)1, the "
            "whitened-median half of §4(b)3 and, on the evidence task, §4(c)",
            **{
                task: {
                    "n": len(calib_ids[task]),
                    "row_min_sigma_ratio": describe(row_min[task]),
                    "row_max_sigma_ratio": describe(row_max[task]),
                    "row_min_whitened_sigma_ratio": describe(w_min[task]),
                    "row_max_whitened_sigma_ratio": describe(w_max[task]),
                }
                for task in TASKS
            },
        },
        "exact_draw_ratios": {k: describe(v) for k, v in exact.items()},
        "diagnostics": {
            "nested_kish_ess": describe(nested_ess),
            "chain_rhat_max": describe(rhat),
            "chain_ess_bulk_min": describe(ess_bulk),
        },
        "frozen": frozen,
        "notes": notes,
        "timeout_rule": (
            f"max({TIMEOUT_FLOOR_S:.0f} s, {TIMEOUT_FACTOR:.0f} x median pilot attempt wall) "
            "per dataset x search x settings x backend"
        ),
        "timeouts": timeouts,
        "probe_first": probe_first,
        "scored_deferred": dict(sorted(scored_deferred.items())),
        "censoring": (
            "an expected run that hit its timeout, crashed, wrote a non-finite answer or left "
            "no row is an attempted failure, its total_wall_s set to the timeout when it ran "
            "past it, never dropped; an oom run (SLURM OUT_OF_MEMORY) is rerun once at twice "
            "its cell's --mem, and a second oom is a failed attempt censored at its wall"
        ),
        "memory_rule": (
            f"per cell class (dataset/search/settings, both backends): --mem = max("
            f"{PILOT_MEM_GB} G, ceil({MEMORY_HEADROOM:g} x peak)), peak = the largest sampled "
            f"sacct MaxRSS, or {PILOT_MEM_GB} G when any pilot task was OUT_OF_MEMORY"
        ),
        "memory_headroom": MEMORY_HEADROOM,
        "memory_raised_gb": raised,
        "memory": mem,
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, allow_nan=False) + "\n"


def rates_literal(table: dict) -> str:
    lines = ["CORRELATED_ND_STEP_RATE: dict[tuple[str, str, str, str, str, int, None], float] = {"]
    for key, value in table.items():
        lines.append(f"    {key!r}: {value!r},")
    lines.append("}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--rates", action="store_true", help="print the rates literal")
    args = parser.parse_args(argv)
    rows = [row for _, row in judge.family_rows(ROOT)]
    table = derived_rates(rows)
    if args.rates:
        print(rates_literal(table))
        return 0
    content = render(calibrate())
    if args.check:
        ok = True
        if not OUT.exists() or OUT.read_text() != content:
            print(
                "correlated_nd calibration is stale; run calibrate_correlated_nd.py",
                file=sys.stderr,
            )
            ok = False
        if rates.CORRELATED_ND_STEP_RATE != table:
            print(
                "wall/rates.py CORRELATED_ND_STEP_RATE differs from the pilot walls",
                file=sys.stderr,
            )
            ok = False
        if ok:
            print(f"{OUT.relative_to(ROOT)} and the correlated_nd rates are current")
        return 0 if ok else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(content)
    print(f"wrote {OUT.relative_to(ROOT)}")
    doc = json.loads(content)
    print(json.dumps({k: doc[k] for k in ("frozen", "notes", "memory_raised_gb")}, indent=1))
    print(
        f"timeouts {len(doc['timeouts'])}, probe_first {len(doc['probe_first'])}, "
        f"scored_deferred {len(doc['scored_deferred'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Calibrate the protocol's PLACEHOLDERs from the wave-1 pilot and the reference scatter.

Protocol ``gaussian_x3@1`` §7 (D16 iii): every PLACEHOLDER — the σ-ratio band, the ppc
tolerance, the mode radius, the ESS and R-hat bars — plus the per-search timeout and
the censoring rule are calibrated on the **reference runs' seed scatter and the wave-1
pilot**, then frozen as ``gaussian_x3@2`` before any wave-2 row. This script computes the
evidence and applies the pre-declared rules below; it writes
``results/calibration/gaussian_x3_wave1.json``, and the frozen values are copied by hand
into ``_protocol.FROZEN_V2`` and protocol §7/§11 A2 (a human-readable diff, not a
silent import). Standard library only.

The rules (declared here, before the numbers are looked at in the amendment)
-----------------------------------------------------------------------------

**Calibration set.** Rows whose task is posterior or evidence, that completed, whose
search family is ``nested`` (Nautilus, DynestyStatic, DynestyDynamic: the estimators
whose σ noise at a given budget is best understood), that pass §4(b)1 (medians) and
§4(b)4 (mode coverage) under ``@1``, and that ``converged`` under §5. They found the
right mode; what scatters is the sampler's finite-ESS estimate of σ and of the ppc curve.

- **σ-ratio band** — ``lo = floor_0.05(q0.5 of each row's minimum ratio)``,
  ``hi = ceil_0.05(q99.5 of each row's maximum ratio)``, widened if needed to contain
  every reference run's ratio to its pooled reference; with fewer than
  :data:`MIN_CALIBRATION_ROWS` rows the ``@1`` band is kept and the shortfall recorded.
- **ppc tolerance** — ``ceil(q99.5 of ppc_χ² − ref_ppc_χ²)`` over the same set, at least
  the reference runs' own maximum excess, never below 1; the ``@1`` value is kept on too
  few rows.
- **mode radius** — kept unless it violates either bound: at least 10× the largest
  reference centre σ, at most half the smallest truth centre separation.
- **nested ESS, chain R-hat and bulk ESS bars** — the ``@1`` values are standard
  (Kish ESS 500: σ to ~3 %; R-hat 1.01 and bulk ESS 400: Vehtari et al. 2021); the pilot
  distributions are recorded beside them, and a bar is only changed if no completed row
  of its family could ever meet it at pilot settings (recorded, not silently loosened).
- **timeout** — per (search × settings × backend): ``max(600 s, 10 × the median pilot
  total wall of that cell over both datasets)``; a cell with no completed pilot row has
  no timeout and must probe first (WALL gate ``source: unmeasured``).
- **censoring** — an expected run whose attempt hit the timeout (a ``failed:
  Terminated`` row), crashed, wrote a non-finite answer or left no row is attempted and
  failed, with ``total_wall_s`` set to the timeout when it ran past it; it is never
  dropped from the success-rate denominator or the wall-per-right-answer sum.

    python scripts/misc/tooling/calibrate_pilot.py           # write the calibration record
    python scripts/misc/tooling/calibrate_pilot.py --check   # verify the committed one
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
for _path in (str(ROOT / "scripts" / "misc"), str(ROOT / "scripts" / "misc" / "tooling")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_catalogue as catalogue  # noqa: E402
import export_inference_summary as exporter  # noqa: E402
from searches import _pilot as pilot  # noqa: E402
from searches import _protocol as protocol  # noqa: E402

OUT = ROOT / "results" / "calibration" / "gaussian_x3_wave1.json"
MIN_CALIBRATION_ROWS = 20
TIMEOUT_FLOOR_S = 600.0
TIMEOUT_FACTOR = 10.0
#: Truth centres per dataset (``dataset/<name>/truth.json``), for the mode-radius bound.
TRUTH_FILE = "truth.json"


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "q0.5": quantile(values, 0.005),
        "q2.5": quantile(values, 0.025),
        "median": statistics.median(values),
        "q97.5": quantile(values, 0.975),
        "q99.5": quantile(values, 0.995),
        "max": max(values),
    }


def sigma_ratios(run: dict, ref: dict) -> list[float]:
    out = []
    for key, ref_stats in (ref.get("posterior_relabelled") or {}).items():
        stats = (run.get("posterior_relabelled") or {}).get(key) or {}
        if ref_stats.get("sigma") and stats.get("sigma") is not None:
            out.append(stats["sigma"] / ref_stats["sigma"])
    return out


def reference_scatter(root: Path) -> dict:
    out = {}
    for path in sorted((root / "results" / "reference").glob("*/*/reference.json")):
        ref = json.loads(path.read_text())
        if ref.get("status") != "complete":
            out[f"{ref.get('dataset')}/{ref.get('backend')}"] = {"status": ref.get("status")}
            continue
        ratios, ppc = [], []
        for name in ref.get("included_runs") or []:
            run_path = path.parent / f"{name}.json"
            if not run_path.exists():
                continue
            run = json.loads(run_path.read_text())
            ratios += sigma_ratios(run, ref)
            if run.get("ppc_chi2") is not None and ref.get("ppc_chi2") is not None:
                ppc.append(run["ppc_chi2"] - ref["ppc_chi2"])
        centre_sigmas = [
            s["sigma"]
            for k, s in (ref.get("posterior_relabelled") or {}).items()
            if k.endswith(".centre")
        ]
        out[f"{ref['dataset']}/{ref['backend']}"] = {
            "status": "complete",
            "included_runs": ref.get("included_runs"),
            "sigma_ratio": describe(ratios),
            "ppc_excess": describe(ppc),
            "log_evidence_spread": (ref.get("agreement_included") or {}).get("log_evidence_spread"),
            "max_centre_sigma": max(centre_sigmas) if centre_sigmas else None,
        }
    return out


def floor_to(x: float, step: float) -> float:
    return math.floor(x / step + 1e-9) * step


def ceil_to(x: float, step: float) -> float:
    return math.ceil(x / step - 1e-9) * step


def calibrate(root: Path = ROOT) -> dict:
    refs, offsets = exporter.references(root)
    rows = [row for _, row in catalogue.scan_rows(root)]
    scatter = reference_scatter(root)

    calib_rows, row_min, row_max, ppc_excess = [], [], [], []
    ess_by_family: dict[str, list[float]] = defaultdict(list)
    rhat, ess_bulk = [], []
    walls: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        cell = (row.get("sampler"), row.get("settings_name"), row.get("backend"))
        if catalogue.usable(row) and row.get("total_wall_s") is not None:
            walls[cell].append(
                float(row["total_wall_s"]) + float(row.get("provider_wall_s") or 0.0)
            )
        if not catalogue.usable(row):
            continue
        if row.get("family") == "nested" and row.get("ess_kish") is not None:
            ess_by_family["nested"].append(float(row["ess_kish"]))
        if row.get("rhat_max") is not None:
            rhat.append(float(row["rhat_max"]))
        if row.get("ess_bulk_min") is not None:
            ess_bulk.append(float(row["ess_bulk_min"]))
        if row.get("task") not in ("posterior", "evidence") or row.get("family") != "nested":
            continue
        ref = exporter.select_reference(refs, row)
        verdict = protocol.verdict(row, ref, offsets)
        crit = {c["detail"][:5]: c["ok"] for c in verdict["criteria"]}
        if verdict["convergence"] != "converged":
            continue
        if crit.get("(b)1 ") is not True or crit.get("(b)4 ") is not True:
            continue
        ratios = sigma_ratios(row, ref)
        if not ratios:
            continue
        calib_rows.append(row["run_id"])
        row_min.append(min(ratios))
        row_max.append(max(ratios))
        if row.get("ppc_chi2") is not None and ref.get("ppc_chi2") is not None:
            ppc_excess.append(row["ppc_chi2"] - ref["ppc_chi2"])

    t = protocol.THRESHOLDS
    frozen: dict = {}
    notes: dict = {}
    ref_ratios = [
        v
        for s in scatter.values()
        if s.get("status") == "complete"
        for v in (s["sigma_ratio"].get("min"), s["sigma_ratio"].get("max"))
        if v is not None
    ]
    if len(calib_rows) >= MIN_CALIBRATION_ROWS:
        lo = floor_to(quantile(row_min, 0.005), 0.05)
        hi = ceil_to(quantile(row_max, 0.995), 0.05)
        if ref_ratios:
            lo, hi = (
                min(lo, floor_to(min(ref_ratios), 0.05)),
                max(hi, ceil_to(max(ref_ratios), 0.05)),
            )
        frozen["sigma_ratio_band"] = [round(lo, 2), round(hi, 2)]
        ref_ppc = [
            s["ppc_excess"]["max"]
            for s in scatter.values()
            if s.get("status") == "complete" and s["ppc_excess"].get("n")
        ]
        tol = max([1.0, math.ceil(quantile(ppc_excess, 0.995) or 0.0)] + ref_ppc)
        frozen["ppc_chi2_tolerance"] = float(math.ceil(tol))
    else:
        frozen["sigma_ratio_band"] = list(t["sigma_ratio_band"])
        frozen["ppc_chi2_tolerance"] = t["ppc_chi2_tolerance"]
        notes["sigma_ratio_band"] = notes["ppc_chi2_tolerance"] = (
            f"kept at @1: only {len(calib_rows)} calibration rows (< {MIN_CALIBRATION_ROWS})"
        )

    centre_sigma = [
        s["max_centre_sigma"] for s in scatter.values() if s.get("max_centre_sigma") is not None
    ]
    separations = []
    for dataset in pilot.DATASETS:
        truth = protocol.load_json(root / "dataset" / dataset / TRUTH_FILE) or {}
        centres = sorted(
            v
            for k, v in (truth.get("parameters") or truth).items()
            if isinstance(v, (int, float)) and k.endswith("centre")
        )
        separations += [b - a for a, b in zip(centres, centres[1:])]
    radius = t["mode_radius_px"]
    lower = 10 * max(centre_sigma) if centre_sigma else None
    upper = min(separations) / 2 if separations else None
    if lower is not None and radius < lower:
        radius = ceil_to(lower, 0.5)
    if upper is not None and radius > upper:
        radius = floor_to(upper, 0.5)
    frozen["mode_radius_px"] = radius
    notes["mode_radius_px"] = (
        f"bounds: >= 10 x max reference centre sigma ({lower}) and <= half the smallest "
        f"truth centre separation ({upper})"
    )

    for key in ("nested_min_ess", "chain_max_rhat", "chain_min_ess"):
        frozen[key] = t[key]
    nested = ess_by_family.get("nested", [])
    if nested and max(nested) < t["nested_min_ess"]:
        notes["nested_min_ess"] = "no completed nested row reaches the bar at pilot settings"
    if ess_bulk and max(ess_bulk) < t["chain_min_ess"]:
        notes["chain_min_ess"] = "no completed chain row reaches the bar at pilot settings"

    timeouts = {}
    for (sampler, settings, backend), values in sorted(walls.items()):
        timeouts[f"{sampler}/{settings}/{backend}"] = {
            "median_pilot_wall_s": statistics.median(values),
            "n": len(values),
            "timeout_s": max(TIMEOUT_FLOOR_S, TIMEOUT_FACTOR * statistics.median(values)),
        }
    measured_cells = {(c.sampler, c.settings, c.backend) for c in pilot.cells()}
    probe_first = sorted(
        f"{s}/{st}/{b}" for (s, st, b) in measured_cells if f"{s}/{st}/{b}" not in timeouts
    )
    return {
        "schema": "autofit-inference-calibration",
        "version": 1,
        "protocol_from": protocol.PROTOCOL_ID,
        "protocol_to": "gaussian_x3@2",
        "campaign_task": pilot.PILOT_ID,
        "rows": len(rows),
        "calibration_set": {
            "rule": "nested, completed, converged, passes (b)1 and (b)4 under @1",
            "n": len(calib_rows),
            "row_min_sigma_ratio": describe(row_min),
            "row_max_sigma_ratio": describe(row_max),
            "ppc_excess": describe(ppc_excess),
        },
        "reference_scatter": scatter,
        "diagnostics": {
            "nested_kish_ess": describe(nested),
            "chain_rhat_max": describe(rhat),
            "chain_ess_bulk_min": describe(ess_bulk),
        },
        "frozen": frozen,
        "notes": notes,
        "timeouts": timeouts,
        "timeout_rule": f"max({TIMEOUT_FLOOR_S:.0f} s, {TIMEOUT_FACTOR:.0f} x median pilot total wall)",
        "probe_first": probe_first,
        "censoring": (
            "an expected run that hit the timeout, crashed, wrote a non-finite answer or left "
            "no row is attempted and failed, its total_wall_s set to the timeout when it ran "
            "past it; never dropped from the success rate or the wall-per-right-answer sum"
        ),
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2, allow_nan=False) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    content = render(calibrate())
    if args.check:
        if not OUT.exists() or OUT.read_text() != content:
            print("calibration record is stale; run calibrate_pilot.py", file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(ROOT)} is current")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(content)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(json.dumps(json.loads(content)["frozen"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

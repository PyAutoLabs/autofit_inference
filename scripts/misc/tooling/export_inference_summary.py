"""Publish the project-owned ``inference-summary@1`` exchange, with protocol verdicts.

Copied from ``autolens_inference/scripts/misc/tooling/export_inference_summary.py`` and
parametrised (``PROJECT``). Standard library only. The one change in substance: the
``scientific`` block is **producer-asserted** from protocol ``gaussian_x3@1``
(``scripts/misc/searches/_protocol.py``, also stdlib only) — ``convergence`` and
``acceptance`` with ``protocol_id`` and ``reason`` — instead of a hard-coded
``not_assessed``. A row with no complete reference of its identity (dataset, backend,
data seed, assertion mechanism) stays ``not_assessed`` with that reason.

Scored wave 2 (B5b) is exported in the same ``inference-summary@1`` schema: each
record names its own protocol (``protocol_id``, also ``scientific.protocol_id``:
``gaussian_x3@1`` for a pilot row, ``gaussian_x3@2`` for a scored row) and the top level
lists every protocol present (``protocols``); the top-level ``protocol_id`` stays the
pilot's. A wave-2 record carries only the row's scalar diagnostics — the posterior
blocks (``posterior``, ``posterior_relabelled``, ``truths``, ``truth_delta_sigma``) stay
in the committed row its ``evidence_paths`` names. Only rows admitted by
``_wave2.admit`` (the wave's protocol and PyAutoFit pin) are records; a refused wave-2
row is listed under ``coverage.excluded`` with why.

**The committed summary withholds the wave-2 records (B5b, autofit_inference#10).** All
11 770 of them validate against the Insight ``inference-summary@1`` contract, but the
file is then ~84 MB (63 MB compact): above GitHub's 50 MB warning, near its 100 MB
limit, and copied again into every Insight snapshot. Until a human chooses the form
(``WAVE2_RECORDS``), ``dashboard/summary.json`` carries the wave-1 records plus the
wave-2 attempt accounting (``coverage.expected_by_protocol``); ``--wave2-records`` writes
the full export elsewhere for inspection. The scored results themselves are the
``gaussian_x3@2`` section of ``catalogue/search_catalogue.json``.

Scans ``results/searches/**/*.json`` (never ``results/reference/``, which holds the
reference posteriors the rows are judged against). Git dates describe publication
provenance only, never measurement freshness; ``measured_at`` comes from each row.

    python scripts/misc/tooling/export_inference_summary.py            # write dashboard/summary.json
    python scripts/misc/tooling/export_inference_summary.py --check    # verify the committed one
    # CI witness: judge rows written to a scratch root, require an accepted row
    python scripts/misc/tooling/export_inference_summary.py --rows-root /tmp/w \
        --output /tmp/w/summary.json --require-accepted
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "misc"))
from searches import _pilot as pilot  # noqa: E402
from searches import _protocol as protocol  # noqa: E402
from searches import _wave2 as wave2  # noqa: E402

PROJECT = "autofit_inference"

#: Paths whose latest commit is the producer revision.
PRODUCER_PATHS = (
    "results",
    "scripts/misc/tooling/export_inference_summary.py",
    "scripts/misc/searches/_protocol.py",
    "scripts/misc/searches/_wave2.py",
    "wiki/project/protocol_gaussian_x3.md",
)

DEFINITIONS = {
    "setup_s": "Not measured separately; unknown, not total minus sampling",
    "sampling_s": "wall_s: sampler clock from samples_info.json; includes sampler overhead",
    "total_s": "total_wall_s: elapsed search.fit call; excludes the admission-bar probe and dataset/model setup",
    "compile_s": "First jitted likelihood call of a separate warm-up probe, in a cold- or warm-cache run (compile_cache); not added to total_s; null on numpy",
}
#: A warm-start row's ``total_s`` is its whole attempt cost (protocol §8): the provider
#: search's fit plus the consumer's, combined by the producer (the reader never sums).
WARM_TOTAL_DEFINITION = (
    "Attempt cost: warm-start provider search.fit (provider_wall_s) + consumer search.fit "
    "(consumer_total_s); excludes the admission-bar probe and dataset/model setup; null "
    "when either part is unknown"
)
DIAGNOSTICS = (
    "log_evidence",
    "log_evidence_err",
    "log_evidence_err_note",
    "max_log_likelihood",
    "max_log_posterior",
    "posterior",
    "posterior_relabelled",
    "modes_found",
    "label_permutations_found",
    "ess_kish",
    "ess_per_s",
    "ppc_chi2",
    "truth_delta_sigma",
    "likelihood_evals",
    "evals_to_target",
    "time_to_target_s",
    "time_to_target_basis",
    "evals_to_target_basis",
    "evals_to_target_note",
    "termination",
    "target_log_likelihood",
    "per_call_s",
    "per_call_basis",
    "likelihood_share",
    "likelihood_share_basis",
    "truths",
    "log_likelihood_at_truth",
    "rhat_max",
    "ess_bulk_min",
)
#: Diagnostics a scored-wave record leaves in its row (``evidence_paths``): the per-
#: parameter blocks are ~60 % of a record, and 11 770 records must stay one file.
#: Whether the committed summary carries the scored-wave records (see the module
#: docstring: withheld on size until a human chooses the form, autofit_inference#10).
WAVE2_RECORDS = False
WAVE2_WITHHELD_REASON = (
    "gaussian_x3@2 records withheld from the committed summary on size (11 770 records = "
    "~84 MB, above GitHub's 50 MB warning), pending a human choice of form "
    "(autofit_inference#10); they validate against inference-summary@1 when exported with "
    "--wave2-records; the scored results are in catalogue/search_catalogue.json "
    "(gaussian_x3@2 section)"
)
WAVE2_ROW_ONLY_DIAGNOSTICS = ("posterior", "posterior_relabelled", "truths", "truth_delta_sigma")
CONFIG = (
    "config_name",
    "where",
    "task",
    "family",
    "settings_name",
    "settings",
    "priors",
    "assertion_mechanism",
    "free_parameters",
    "data_seed",
    "seed_effective",
    "compile_cache",
    "config_id",
    "cores",
    "use_jax",
    "test_mode",
    "pilot",
    "resumed",
)


def safe_path(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return False
    p = PurePosixPath(value)
    return not p.is_absolute() and all(s not in ("", ".", "..") for s in value.split("/"))


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite metric")
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def references(root: Path) -> tuple[dict, dict | None]:
    """``{reference_key: reference}``, keyed by the full identity (dataset, backend,
    data_seed, assertion_mechanism), and the measured offsets."""
    refs = {}
    paths = [
        *(root / "results" / "reference").glob("*/*/reference.json"),
        *(root / "results" / "reference").glob("*/*/data_seed*/reference.json"),
    ]
    for path in sorted(paths):
        ref = protocol.load_json(path)
        if isinstance(ref, dict):
            refs[protocol.reference_key(ref)] = ref
    return refs, protocol.load_json(protocol.offsets_file(root))


def select_reference(refs: dict, row: dict) -> dict | None:
    """The reference with the row's full identity; failing that, one for the same
    dataset and backend, so the verdict names the identity mismatch rather than
    claiming there is no reference. Never a reference for another dataset/backend."""
    exact = refs.get(protocol.reference_key(row))
    if exact is not None:
        return exact
    for ref in refs.values():
        if (ref.get("dataset"), ref.get("backend")) == (row.get("dataset"), row.get("backend")):
            return ref
    return None


def record(row: dict, rel: str, refs: dict, offsets) -> dict:
    status = row.get("status") or ("complete" if row.get("completed") is True else "unknown")
    raw_path = row.get("output_path")
    output_path = f"output/{raw_path}" if safe_path(raw_path) else None
    scored = protocol.wave_of(row) == wave2.WAVE
    values = {
        k: row[k]
        for k in DIAGNOSTICS
        if k in row and row[k] is not None and not (scored and k in WAVE2_ROW_ONLY_DIAGNOSTICS)
    }
    reference = select_reference(refs, row)
    verdict = protocol.verdict(row, reference, offsets)
    warm = bool(row.get("warm_start"))
    result = {
        "id": row.get("run_id") or rel,
        "parent_run_id": None,
        "stage": None,
        "target": row.get("target"),
        "dataset": {
            "class": row.get("dataset_class"),
            "instrument": None,
            "id": f"{row.get('dataset')}/data_seed{row.get('data_seed')}"
            if row.get("dataset") and row.get("data_seed") is not None
            else None,
        },
        "model": row.get("model"),
        "pipeline": None,
        "sampler": row.get("sampler"),
        "configuration": {k: row[k] for k in CONFIG if k in row},
        "backend": row.get("backend"),
        "hardware": {
            k: row[k]
            for k in (
                "device",
                "host",
                "cores",
                "slurm_job_id",
                "slurm_array_job_id",
                "slurm_array_task_id",
            )
            if k in row
        },
        "precision": row.get("precision"),
        "seed": row.get("seed"),
        "dependency_revisions": {
            k: v for k, v in (row.get("library_revisions") or {}).items() if isinstance(v, str)
        },
        "library_version": row.get("version"),
        "execution": {
            "status": status,
            "completed": row.get("completed") if isinstance(row.get("completed"), bool) else None,
            "reason": status if str(status).startswith("failed") else None,
        },
        "scientific": {
            "convergence": verdict["convergence"],
            "acceptance": verdict["acceptance"],
            "protocol_id": verdict["protocol_id"],
            "reason": f"acceptance: {verdict['acceptance_reason']} | convergence: "
            f"{verdict['convergence_reason']}",
            "criteria": verdict["criteria"],
            "reference_limitations": verdict["reference_limitations"],
            "map_diagnostic": verdict["map_diagnostic"],
            "placeholders": verdict["placeholders"],
            "asserted_by": "producer",
        },
        "timings": {
            "setup_s": None,
            "sampling_s": row.get("wall_s"),
            "total_s": protocol.attempt_wall_s(row) if warm else row.get("total_wall_s"),
            "compile_s": row.get("compile_s"),
            "definitions": (
                {**DEFINITIONS, "total_s": WARM_TOTAL_DEFINITION} if warm else dict(DEFINITIONS)
            ),
        },
        # Additive @1 fields (review 06 finding 4): warm-start provenance and the split.
        "warm_start": row.get("warm_start") if warm else None,
        "provider_wall_s": row.get("provider_wall_s") if warm else None,
        "consumer_total_s": row.get("total_wall_s"),
        "diagnostics": {
            "status": "available" if values else "missing",
            "values": values,
        },
        "comparison": {
            "group": None,
            "declared_target": row.get("target"),
            "note": "Declared target is not a verified comparison group; no ranking (wave 1 is a pilot)"
            if not scored
            else "Declared target is not a verified comparison group; gaussian_x3@2 legs are "
            "ranked in catalogue/search_catalogue.json, within one task and one dataset",
        },
        "samples": {
            "availability": "unknown",
            "path": output_path,
            "access": "project-storage-not-published",
            "note": "Committed results do not verify current sample availability",
        },
        "evidence_paths": [rel],
        "measured_at": row.get("measured_at"),
        "archived": False,
        "unknown_reasons": {},
    }
    if scored:
        result["protocol_id"] = verdict["protocol_id"]
        result["diagnostics"]["row_only"] = [k for k in WAVE2_ROW_ONLY_DIAGNOSTICS if k in row]
    for key in ("target", "model", "sampler", "backend", "precision", "seed", "measured_at"):
        if result[key] is None:
            result["unknown_reasons"][key] = "Not recorded in source result"
    if result["dataset"]["id"] is None:
        result["unknown_reasons"]["dataset.id"] = "Dataset identity not recorded"
    if not result["dependency_revisions"]:
        result["unknown_reasons"]["dependency_revisions"] = "Dependency revisions not recorded"
    if result["execution"]["completed"] is None:
        result["unknown_reasons"]["execution.completed"] = "Completion marker not recorded"
    if warm and result["timings"]["total_s"] is None:
        result["unknown_reasons"]["timings.total_s"] = (
            "Warm-start attempt cost unknown: provider or consumer wall not recorded"
        )
    for key in ("setup_s", "sampling_s", "total_s", "compile_s"):
        value = result["timings"][key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (float, int)) or value < 0
        ):
            raise ValueError(f"invalid timing {key}")
    return result


def expected_coverage(rows: list[dict]) -> dict:
    """The wave-1 expected-run manifest (``scripts/misc/searches/_pilot.py``) against the
    rows present: every expected run with no row is listed as deferred with its reason, so
    the attempt count §8 divides by never silently shrinks."""
    seen = {(pilot.row_key(row), row.get("seed")) for row in rows}
    deferred: dict[tuple, dict] = {}
    for cell, seed in pilot.expected_runs():
        if (cell.key, seed) in seen:
            continue
        entry = deferred.setdefault(
            cell.key,
            {
                "id": "/".join(cell.key),
                "task": cell.task,
                "seeds": [],
                "reason": pilot.missing_reason(cell),
            },
        )
        entry["seeds"].append(seed)
    expected = pilot.expected_runs()
    return {
        "runs": len(expected),
        "manifest": f"scripts/misc/searches/_pilot.py ({pilot.PILOT_ID})",
        "with_row": len(expected) - sum(len(d["seeds"]) for d in deferred.values()),
        "deferred_runs": sum(len(d["seeds"]) for d in deferred.values()),
        "deferred": list(deferred.values()),
    }


def wave2_coverage(rows: list[dict], timeouts: dict) -> dict:
    """The scored wave's attempt accounting (``_wave2.accounting``) in brief: the totals
    and every leg with a non-row state, with its seeds' counts and reason."""
    acc = wave2.accounting(rows, timeouts)
    legs = [
        {
            "id": leg["id"],
            "task": leg["task"],
            "failed": len(leg["failed"]),
            "missing": len(leg["missing"]),
            "deferred": len(leg["deferred"]),
            "reason": leg["reason"],
        }
        for leg in acc["legs"]
        if leg["failed"] or leg["missing"] or leg["deferred"]
    ]
    return {
        "manifest": f"scripts/misc/searches/_wave2.py ({wave2.WAVE_ID})",
        "pyautofit_pin": wave2.PYAUTOFIT_PIN,
        "accounting": "results/accounting/gaussian_x3_wave2.json",
        **acc["totals"],
        "legs_not_all_rows": legs,
    }


def build(
    root: Path,
    revision: str | None,
    generated_at: str,
    rows_root: Path | None = None,
    wave2_records: bool | None = None,
):
    timestamp = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Generation timestamp must include timezone")
    generated_at = timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
    rows_root = rows_root or root
    wave2_records = WAVE2_RECORDS if wave2_records is None else wave2_records
    refs, offsets = references(root)
    timeouts = wave2.wave_timeouts()
    records, excluded, raw_rows, wave2_rows = [], [], [], []
    for path in sorted((rows_root / "results" / "searches").rglob("*.json")):
        rel = path.relative_to(rows_root).as_posix()
        try:
            if path.is_symlink():
                raise ValueError("unsafe result path")
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("result must be an object")
            finite(payload)
            if payload.get("schema_version") != 1:
                raise ValueError("unsupported result schema_version")
            wave = protocol.wave_of(payload)
            if wave == wave2.WAVE:
                why = wave2.admit(payload, timeouts)
                if why is not None:
                    excluded.append({"path": rel, "reason": why, "status": "refused"})
                    continue
                wave2_rows.append(payload)
                if wave2_records:
                    records.append(record(payload, rel, refs, offsets))
                continue
            if wave != 1:
                raise ValueError(f"no protocol for wave {wave!r}")
            records.append(record(payload, rel, refs, offsets))
            raw_rows.append(payload)
        except (ValueError, TypeError, OSError) as exc:
            excluded.append({"path": rel, "reason": str(exc), "status": "invalid"})
    counts = Counter(str(r["execution"]["status"]).split(":", 1)[0] for r in records)
    verdicts = Counter(r["scientific"]["acceptance"] for r in records)
    by_protocol = Counter(r["scientific"]["protocol_id"] for r in records)
    protocols = sorted({*by_protocol, *([wave2.PROTOCOL_ID] if wave2_rows else [])})
    measured = sorted(r["measured_at"] for r in records if r["measured_at"])
    reference_status = {
        f"{ref.get('dataset')}/{ref.get('backend')}": {
            "status": ref.get("status"),
            "data_seed": ref.get("data_seed"),
            "assertion_mechanism": ref.get("assertion_mechanism"),
        }
        for _, ref in sorted(refs.items(), key=lambda item: json.dumps(item[0]))
    }
    doc = {
        "schema": "inference-summary",
        "version": 1,
        "project": PROJECT,
        "scope": "inference",
        "producer_revision": revision,
        "generated_at": generated_at,
        "generation_basis": "Source commit timestamp; not a measurement or check-in time",
        "evidence_updated_at": measured[-1] if measured else None,
        "valid_until": None,
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_id_note": "The wave-1 pilot's protocol; every record names its own "
        "(scientific.protocol_id; a gaussian_x3@2 record also at the record's protocol_id), "
        "and protocols lists every protocol this summary covers, in records or coverage",
        "protocols": protocols,
        "references": reference_status,
        "records": records,
        "coverage": {
            "expected": expected_coverage(raw_rows),
            "expected_by_protocol": {
                protocol.PROTOCOL_ID: f"coverage.expected ({pilot.PILOT_ID})",
                wave2.PROTOCOL_ID: {
                    **wave2_coverage(wave2_rows, timeouts),
                    "records": "exported" if wave2_records else "withheld",
                    **({} if wave2_records else {"records_reason": WAVE2_WITHHELD_REASON}),
                },
            },
            "observed": {
                "runs": len(records),
                "records": len(records),
                **{f"records_{pid}": n for pid, n in sorted(by_protocol.items())},
                **{f"execution_{key}": value for key, value in sorted(counts.items())},
                **{f"acceptance_{key}": value for key, value in sorted(verdicts.items())},
            },
            "excluded": excluded,
        },
        "comparisons": [],
        "comparison_policy": {
            "id": "gaussian_x3-task-groups-v1",
            "ranking": "none",
            "note": "Grouped by task (point/MAP, posterior, evidence); wave 1 is a pilot that ranks nothing; never best-seed. gaussian_x3@2 legs are ranked within one task and one dataset in catalogue/search_catalogue.json, not here",
        },
        "limitations": [
            "Verdicts are producer-asserted under protocol gaussian_x3@1, whose PLACEHOLDER thresholds are calibrated in the wave-1 pilot and then frozen",
            "Rows are judged only against a reference with the same dataset, backend, data seed and assertion mechanism; log evidences are compared only within one backend and one assertion mechanism",
            "Nested-search convergence needs the search's own termination condition: Nautilus is read from its sampler; other nested searches are not_assessed until PyAutoFit exposes it (phase A3)",
            "Completion is execution, not convergence; convergence is reported separately from acceptance",
            "A warm-start record's total_s is its whole attempt cost (provider_wall_s + consumer_total_s, combined by the producer, protocol §8); cold records' total_s is the search.fit call alone",
            "Sample archives remain in project storage; current availability unknown",
            "gaussian_x3@2 (scored wave 2) rows are judged under the frozen @2 thresholds against their own data realisation's reference, at the pinned PyAutoFit revision; as records their posterior blocks stay in the committed row (evidence_paths)",
            "gaussian_x3@2 timeouts are 10 x laptop pilot medians while RAL ran 4-7 x faster, so censoring is lenient on RAL; probe-first cells (Zeus on JAX, NSS 50-step) hold only their 5-run probes until B5b-2",
            *([] if wave2_records or not wave2_rows else [WAVE2_WITHHELD_REASON]),
        ],
    }
    if revision is None:
        doc["producer_revision_reason"] = "Not generated from a git checkout"
    if doc["evidence_updated_at"] is None:
        doc["evidence_updated_at_reason"] = "No result rows yet (wave 1 is phase B3)"
    return doc


def git_revision(root: Path) -> tuple[str | None, str]:
    try:
        revision = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", *PRODUCER_PATHS], cwd=root, text=True
        ).strip()
        date = subprocess.check_output(
            ["git", "show", "-s", "--format=%cI", revision], cwd=root, text=True
        ).strip()
        return revision or None, date
    except (subprocess.CalledProcessError, OSError):
        return None, datetime.now(UTC).isoformat()


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default="dashboard/summary.json")
    parser.add_argument(
        "--rows-root",
        type=Path,
        default=None,
        help="read results/searches from here instead of the repo (CI witness)",
    )
    parser.add_argument(
        "--wave2-records",
        action="store_true",
        help="export the gaussian_x3@2 records too (never to the committed summary path)",
    )
    parser.add_argument(
        "--require-accepted",
        action="store_true",
        help="exit 1 unless every record is accepted (and there is at least one)",
    )
    args = parser.parse_args(argv)
    revision, date = git_revision(ROOT)
    target = Path(args.output)
    if not target.is_absolute():
        target = ROOT / target
    if args.wave2_records and target.resolve() == (ROOT / "dashboard" / "summary.json").resolve():
        parser.error("--wave2-records is withheld from dashboard/summary.json; pass --output")
    doc = build(ROOT, revision, date, args.rows_root, wave2_records=args.wave2_records or None)
    content = render(doc)
    if args.check:
        if not target.exists() or target.read_text() != content:
            print("inference summary is stale; regenerate", file=sys.stderr)
            return 1
        print(f"{target.relative_to(ROOT)} is current ({len(doc['records'])} records)")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        print(f"wrote {target} ({len(doc['records'])} records)")
    for r in doc["records"]:
        print(f"  {r['id']}: {r['scientific']['acceptance']} / {r['scientific']['convergence']}")
        print(f"    {r['scientific']['reason']}")
    if args.require_accepted:
        if not doc["records"] or any(
            r["scientific"]["acceptance"] != "accepted" for r in doc["records"]
        ):
            print("FAIL: not every record is accepted", file=sys.stderr)
            return 1
        print("witness: every record accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

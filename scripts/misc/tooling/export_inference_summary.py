"""Publish the project-owned ``inference-summary@1`` exchange, with protocol verdicts.

Copied from ``autolens_inference/scripts/misc/tooling/export_inference_summary.py`` and
parametrised (``PROJECT``). Standard library only. The one change in substance: the
``scientific`` block is **producer-asserted** from protocol ``gaussian_x3@1``
(``scripts/misc/searches/_protocol.py``, also stdlib only) — ``convergence`` and
``acceptance`` with ``protocol_id`` and ``reason`` — instead of a hard-coded
``not_assessed``. A row with no complete reference of its identity (dataset, backend,
data seed, assertion mechanism) stays ``not_assessed`` with that reason.

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

PROJECT = "autofit_inference"

#: Paths whose latest commit is the producer revision.
PRODUCER_PATHS = (
    "results",
    "scripts/misc/tooling/export_inference_summary.py",
    "scripts/misc/searches/_protocol.py",
    "wiki/project/protocol_gaussian_x3.md",
)

DEFINITIONS = {
    "setup_s": "Not measured separately; unknown, not total minus sampling",
    "sampling_s": "wall_s: sampler clock from samples_info.json; includes sampler overhead",
    "total_s": "total_wall_s: elapsed search.fit call; excludes the admission-bar probe and dataset/model setup",
    "compile_s": "First jitted likelihood call of a separate warm-up probe, in a cold- or warm-cache run (compile_cache); not added to total_s; null on numpy",
}
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
    for path in sorted((root / "results" / "reference").glob("*/*/reference.json")):
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
    values = {k: row[k] for k in DIAGNOSTICS if k in row and row[k] is not None}
    reference = select_reference(refs, row)
    verdict = protocol.verdict(row, reference, offsets)
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
            "total_s": row.get("total_wall_s"),
            "compile_s": row.get("compile_s"),
            "definitions": dict(DEFINITIONS),
        },
        "diagnostics": {
            "status": "available" if values else "missing",
            "values": values,
        },
        "comparison": {
            "group": None,
            "declared_target": row.get("target"),
            "note": "Declared target is not a verified comparison group; no ranking (wave 1 is a pilot)",
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
    for key in ("target", "model", "sampler", "backend", "precision", "seed", "measured_at"):
        if result[key] is None:
            result["unknown_reasons"][key] = "Not recorded in source result"
    if result["dataset"]["id"] is None:
        result["unknown_reasons"]["dataset.id"] = "Dataset identity not recorded"
    if not result["dependency_revisions"]:
        result["unknown_reasons"]["dependency_revisions"] = "Dependency revisions not recorded"
    if result["execution"]["completed"] is None:
        result["unknown_reasons"]["execution.completed"] = "Completion marker not recorded"
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


def build(root: Path, revision: str | None, generated_at: str, rows_root: Path | None = None):
    timestamp = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Generation timestamp must include timezone")
    generated_at = timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
    rows_root = rows_root or root
    refs, offsets = references(root)
    records, excluded, raw_rows = [], [], []
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
            records.append(record(payload, rel, refs, offsets))
            raw_rows.append(payload)
        except (ValueError, TypeError, OSError) as exc:
            excluded.append({"path": rel, "reason": str(exc), "status": "invalid"})
    counts = Counter(str(r["execution"]["status"]).split(":", 1)[0] for r in records)
    verdicts = Counter(r["scientific"]["acceptance"] for r in records)
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
        "references": reference_status,
        "records": records,
        "coverage": {
            "expected": expected_coverage(raw_rows),
            "observed": {
                "runs": len(records),
                "records": len(records),
                **{f"execution_{key}": value for key, value in sorted(counts.items())},
                **{f"acceptance_{key}": value for key, value in sorted(verdicts.items())},
            },
            "excluded": excluded,
        },
        "comparisons": [],
        "comparison_policy": {
            "id": "gaussian_x3-task-groups-v1",
            "ranking": "none",
            "note": "Grouped by task (point/MAP, posterior, evidence); wave 1 is a pilot that ranks nothing; never best-seed",
        },
        "limitations": [
            "Verdicts are producer-asserted under protocol gaussian_x3@1, whose PLACEHOLDER thresholds are calibrated in the wave-1 pilot and then frozen",
            "Rows are judged only against a reference with the same dataset, backend, data seed and assertion mechanism; log evidences are compared only within one backend and one assertion mechanism",
            "Nested-search convergence needs the search's own termination condition: Nautilus is read from its sampler; other nested searches are not_assessed until PyAutoFit exposes it (phase A3)",
            "Completion is execution, not convergence; convergence is reported separately from acceptance",
            "Sample archives remain in project storage; current availability unknown",
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
        "--require-accepted",
        action="store_true",
        help="exit 1 unless every record is accepted (and there is at least one)",
    )
    args = parser.parse_args(argv)
    revision, date = git_revision(ROOT)
    doc = build(ROOT, revision, date, args.rows_root)
    content = render(doc)
    target = Path(args.output)
    if not target.is_absolute():
        target = ROOT / target
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

"""build_readme.py — refresh the auto-generated tables in README.md from ``results/``.

Copied in shape from ``autolens_inference/scripts/misc/tooling/build_readme.py`` (the
sentinel machinery and the ``--check`` gate are the same; the table specs are this
repo's). Standard library only.

    python scripts/misc/tooling/build_readme.py          # rewrite README tables in place
    python scripts/misc/tooling/build_readme.py --check  # exit non-zero if a rewrite is pending

Each table region is delimited by sentinel comments::

    <!-- BEGIN auto-table:searches -->
    | ... |
    <!-- END auto-table:searches -->

Only the content between BEGIN and END is rewritten. An unknown sentinel is left intact
and warned about rather than blanked.

Regions
-------

``catalogue``
    Every search in PyAutoFit's registry, from ``catalogue/search_catalogue.json``
    (``build_catalogue.py``), one table per task — point/MAP, posterior, evidence — with
    its status (measured, unsupported, deferred, failed) and reason. Tasks are never
    ranked or compared.
``references``
    The reference posteriors under ``results/reference/<dataset>/<backend>/``: status,
    the runs included, the normalised log evidence, the agreement statistics and the
    MAP log posterior.
``searches``
    The pilot's per-cell summaries (from the catalogue), one table per task: attempts,
    verdicts, convergence, the Wilson interval and walls. Never per best seed.

Every region renders a one-line empty state, so ``--check`` is a real gate from the
first commit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(REPO_ROOT / "scripts" / "misc"))
sys.path.insert(0, str(REPO_ROOT))

from searches import _protocol as protocol  # noqa: E402

RESULTS_ROOT = REPO_ROOT / "results"
SEARCHES_ROOT = RESULTS_ROOT / "searches"
REFERENCE_ROOT = RESULTS_ROOT / "reference"

SENTINEL_RE = re.compile(
    r"(<!-- BEGIN auto-table:(?P<name>[a-z0-9_\-]+) -->)"
    r".*?"
    r"(<!-- END auto-table:(?P=name) -->)",
    re.DOTALL,
)

TASK_LABELS = {"point_map": "point/MAP", "posterior": "posterior", "evidence": "evidence"}
DASH = "—"


def _fmt(value, spec: str = ".3g") -> str:
    if value is None or isinstance(value, bool):
        return DASH
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _scan(root: Path, pattern: str) -> list[tuple[Path, dict]]:
    rows = []
    if not root.is_dir():
        return rows
    for path in sorted(root.rglob(pattern)):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            rows.append((path, payload))
    return rows


def _references() -> tuple[dict, dict | None]:
    refs = {}
    for _, ref in _scan(REFERENCE_ROOT, "reference.json"):
        refs[(ref.get("dataset"), ref.get("backend"))] = ref
    return refs, protocol.load_json(protocol.offsets_file(REPO_ROOT))


def _catalogue() -> dict | None:
    return protocol.load_json(REPO_ROOT / "catalogue" / "search_catalogue.json")


def _counts(leg: dict) -> str:
    acc = leg.get("acceptance") or {}
    return f"{acc.get('accepted', 0)}/{leg.get('attempted', 0)}"


def render_catalogue() -> str:
    """One table per task, from ``catalogue/search_catalogue.json`` (built by
    ``build_catalogue.py`` from PyAutoFit's search manifest). Tasks never share a table,
    and rows keep the manifest's order: nothing is ranked."""
    doc = _catalogue()
    if not doc:
        return "\n_No catalogue yet — run `scripts/misc/tooling/build_catalogue.py`._\n"
    lines = [""]
    for task, group in doc["tasks"].items():
        lines += [
            f"**{group['label']}** — judged by {group['judged_by']}.",
            "",
            "| Search class | Status | Accepted / attempted (blend: numpy, jax) | "
            "Accepted / attempted (separated: numpy, jax) | Reason |",
            "|---|---|---|---|---|",
        ]
        for cls, entry in group["searches"].items():
            cells = []
            for dataset in ("gaussian_x3_blend", "gaussian_x3_separated"):
                parts = []
                for backend in ("numpy", "jax_cpu"):
                    legs = [
                        leg
                        for leg in entry["legs"]
                        if leg["dataset"] == dataset and leg["backend"] == backend
                    ]
                    if not legs:
                        parts.append(DASH)
                        continue
                    parts.append(
                        " · ".join(
                            f"{_counts(leg)}"
                            + (
                                f" (`{leg['sampler']}` `{leg['settings']}`)"
                                if len(legs) > 1
                                else ""
                            )
                            for leg in legs
                        )
                    )
                cells.append("; ".join(parts))
            lines.append(
                f"| `{cls}` | {entry['status']} | {cells[0]} | {cells[1]} | {entry['reason']} |"
            )
        lines.append("")
    cov = doc.get("coverage") or {}
    lines.append(
        f"Wave-1 coverage: {cov.get('rows_in_menu', 0)} rows of {cov.get('expected_runs', 0)} "
        f"expected runs; PyAutoFit {', '.join(f'`{c[:12]}`' for c in doc.get('pyautofit_commits') or [])}. "
        "Accepted counts are judged by protocol `gaussian_x3@1`; an attempt that could not be "
        "judged (no complete reference) is not counted as accepted."
    )
    return "\n".join(lines) + "\n"


def render_references() -> str:
    refs, offsets = _references()
    if not refs:
        return "\n_No reference posterior yet._\n"
    lines = [
        "",
        "| Dataset | Backend | Status | Included runs | ln Z (normalised) | ln Z spread | "
        "Worst median offset (σ) | MAP ln P |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (dataset, backend), ref in sorted(refs.items()):
        agreement = ref.get("agreement_included") or {}
        included = ref.get("included_runs") or []
        lines.append(
            f"| `{dataset}` | `{backend}` | {ref.get('status')} | {len(included)} | "
            f"{_fmt(ref.get('log_evidence_normalised'), '.3f')} | "
            f"{_fmt(agreement.get('log_evidence_spread'), '.3f')} | "
            f"{_fmt(agreement.get('median_agreement_sigma'), '.3f')} | "
            f"{_fmt((ref.get('map') or {}).get('log_posterior'), '.3f')} |"
        )
    validated = []
    for sampler, backends in sorted(((offsets or {}).get("offsets") or {}).items()):
        for backend, entry in sorted(backends.items()):
            validated.append(
                f"`{sampler}`/`{backend}`: offset {_fmt(entry.get('offset'), '.3f')} "
                f"({'validated' if entry.get('validated') else 'NOT validated'}, measured "
                f"{_fmt(entry.get('difference'), '.3f')})"
            )
    lines.append("")
    lines.append(
        "ln 3! convention (constant likelihood): " + ("; ".join(validated) or "pending") + "."
    )
    return "\n".join(lines) + "\n"


def render_searches() -> str:
    """Per-cell pilot summaries, one table per task (never per best seed, never ranked):
    the counts and walls of every (dataset × search × settings × leg) cell that has rows."""
    doc = _catalogue()
    rows = _scan(SEARCHES_ROOT, "*.json")
    if not rows or not doc:
        return "\n_No search rows yet — wave 1 (the pilot) runs in phase B3._\n"
    lines = [""]
    for task, group in doc["tasks"].items():
        legs = [
            (cls, leg)
            for cls, entry in group["searches"].items()
            for leg in entry["legs"]
            if leg["attempted"]
        ]
        if not legs:
            continue
        lines += [
            f"**{group['label']}**",
            "",
            "| Dataset | Search | Settings | Config | Attempted | Accepted | Rejected | "
            "Not assessed | Converged | Success rate (Wilson 95 %) | Median wall (s) | "
            "Wall per right answer (s) | Deferred seeds |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, leg in legs:
            acc, conv = leg.get("acceptance") or {}, leg.get("convergence") or {}
            rate = leg.get("success_rate")
            if rate is None:
                rate_text = DASH
            else:
                lo, hi = leg["success_rate_wilson_95"]
                rate_text = f"{rate:.0%} [{lo:.0%}, {hi:.0%}]"
            wps = leg.get("wall_per_success") or {}
            if wps.get("unbounded"):
                wps_text = f"unbounded (≥ {_fmt(wps.get('lower_bound_s'), '.0f')})"
            else:
                wps_text = _fmt(wps.get("value_s"), ".0f")
            lines.append(
                f"| `{leg['dataset']}` | `{leg['sampler']}` | `{leg['settings']}` | "
                f"`{leg['config_id']}` | {leg['attempted']} | {acc.get('accepted', 0)} | "
                f"{acc.get('rejected', 0)} | {acc.get('not_assessed', 0)} | "
                f"{conv.get('converged', 0)} | {rate_text} | "
                f"{_fmt(leg.get('median_total_wall_s'), '.1f')} | {wps_text} | "
                f"{len(leg.get('deferred_seeds') or []) or DASH} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


RENDERERS = {
    "catalogue": render_catalogue,
    "references": render_references,
    "searches": render_searches,
}

TARGET_READMES = [REPO_ROOT / "README.md"]


def _rewrite_file(path: Path, renderers: dict) -> tuple[str, str, list[str]]:
    original = path.read_text()
    unknown: list[str] = []

    def replace(match: re.Match) -> str:
        name = match.group("name")
        renderer = renderers.get(name)
        if renderer is None:
            unknown.append(name)
            return match.group(0)
        return f"{match.group(1)}{renderer()}{match.group(3)}"

    return original, SENTINEL_RE.sub(replace, original), unknown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a rewrite is pending")
    args = parser.parse_args(argv)
    any_changed = False
    for target in TARGET_READMES:
        original, rewritten, unknown = _rewrite_file(target, RENDERERS)
        for name in unknown:
            print(f"WARNING: unknown sentinel '{name}' in {target.name} — left intact")
        if rewritten == original:
            print(f"  unchanged {target.relative_to(REPO_ROOT)}")
            continue
        any_changed = True
        if args.check:
            print(f"  WOULD rewrite {target.relative_to(REPO_ROOT)}")
        else:
            target.write_text(rewritten)
            print(f"  rewrote   {target.relative_to(REPO_ROOT)}")
    if args.check and any_changed:
        print(
            "ERROR: `build_readme.py --check` found pending changes. Run "
            "`python scripts/misc/tooling/build_readme.py` and commit the result.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

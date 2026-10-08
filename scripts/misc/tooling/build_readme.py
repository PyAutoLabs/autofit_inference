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
    Every registered search (``scripts/misc/searches/_runner.py::SAMPLERS``) grouped by
    task — point/MAP, posterior, evidence — with its status: ``measured`` once a row
    exists, otherwise ``registered`` or ``deferred``. Tasks are never ranked.
``references``
    The reference posteriors under ``results/reference/<dataset>/<backend>/``: status,
    the runs included, the normalised log evidence, the agreement statistics and the
    MAP log posterior.
``searches``
    Single-search rows from ``results/searches/**/*.json``, grouped by task and dataset,
    with the protocol verdicts the exporter emits.

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
from searches._runner import SAMPLERS  # noqa: E402

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


def render_catalogue() -> str:
    measured = {row.get("sampler") for _, row in _scan(SEARCHES_ROOT, "*.json")}
    lines = [
        "",
        "| Task | Search | PyAutoFit class | Status | Settings | Note |",
        "|---|---|---|---|---|---|",
    ]
    for task in ("point_map", "posterior", "evidence"):
        for name, spec in SAMPLERS.items():
            if spec.task != task:
                continue
            status = (
                "deferred"
                if spec.status == "deferred"
                else ("measured" if name in measured else "registered")
            )
            settings = ", ".join(f"`{s}`" for s in spec.settings)
            lines.append(
                f"| {TASK_LABELS[task]} | `{name}` | `{spec.cls}` | {status} | {settings} | "
                f"{spec.note or DASH} |"
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
    rows = _scan(SEARCHES_ROOT, "*.json")
    if not rows:
        return "\n_No search rows yet — wave 1 (the pilot) runs in phase B3._\n"
    refs, offsets = _references()
    lines = [
        "",
        "| Task | Dataset | Search | Settings | Config | Seed | Acceptance | Convergence | "
        "Wall (s) | Evals | ln Z | max ln P | Modes | per-call (s) | Share |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def key(item):
        row = item[1]
        return (
            ("point_map", "posterior", "evidence").index(row.get("task", "evidence")),
            row.get("dataset", ""),
            row.get("sampler", ""),
            row.get("settings_name", ""),
            row.get("config_name", ""),
            row.get("seed", 0),
        )

    for _, row in sorted(rows, key=key):
        verdict = protocol.verdict(row, refs.get((row.get("dataset"), row.get("backend"))), offsets)
        lines.append(
            f"| {TASK_LABELS.get(row.get('task'), DASH)} | `{row.get('dataset')}` | "
            f"`{row.get('sampler')}` | `{row.get('settings_name')}` | `{row.get('config_name')}` | "
            f"{row.get('seed')} | {verdict['acceptance']} | {verdict['convergence']} | "
            f"{_fmt(row.get('wall_s'), '.1f')} | {row.get('likelihood_evals') or DASH} | "
            f"{_fmt(row.get('log_evidence'), '.3f')} | {_fmt(row.get('max_log_posterior'), '.3f')} | "
            f"{row.get('modes_found') if row.get('modes_found') is not None else DASH} | "
            f"{_fmt(row.get('per_call_s'))} | {_fmt(row.get('likelihood_share'), '.2%')} |"
        )
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

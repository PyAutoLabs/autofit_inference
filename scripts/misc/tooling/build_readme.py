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
``correlated_nd``
    The second family (B5c, protocol ``correlated_nd@1``): per dataset its dimension, the
    exact ln Z and information of the committed realisation, each backend's exact
    reference status, and the pilot's planned / deferred / pulled cells, from the
    catalogue's ``correlated_nd@1`` section. Never compared with ``gaussian_x3``.
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
        refs[(ref.get("dataset"), ref.get("backend"), ref.get("data_seed") or 0)] = ref
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
    if doc.get("pilot_stopped"):
        lines.append("")
        lines.append(f"**Pilot stopped.** {doc['pilot_stopped']}")
    return "\n".join(lines) + "\n"


def render_references() -> str:
    refs, offsets = _references()
    if not refs:
        return "\n_No reference posterior yet._\n"
    lines = [
        "",
        "| Dataset | Backend | Realisation | Status | Included runs | ln Z (normalised) | "
        "ln Z spread | Worst median offset (σ) | MAP ln P |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (dataset, backend, data_seed), ref in sorted(refs.items()):
        if protocol.is_family_row(ref):
            continue  # exact references: the correlated_nd region renders them
        agreement = ref.get("agreement_included") or {}
        included = ref.get("included_runs") or []
        lines.append(
            f"| `{dataset}` | `{backend}` | {data_seed or DASH} | {ref.get('status')} | "
            f"{len(included)} | "
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


def render_correlated_nd() -> str:
    """The second family's datasets, exact references and pilot coverage (never ranked,
    never compared with gaussian_x3)."""
    doc = _catalogue() or {}
    section = doc.get(protocol.CND_PROTOCOL_ID)
    if not section:
        return "\n_No correlated_nd section in the catalogue yet._\n"
    refs, _ = _references()
    counts: dict[str, dict] = {}
    for group in section["tasks"].values():
        for dataset, legs in group["datasets"].items():
            c = counts.setdefault(dataset, {"planned": 0, "deferred": 0, "rows": 0, "failed": 0})
            for leg in legs:
                c["planned" if leg["cell_status"] == "planned" else "deferred"] += 1
                c["rows"] += leg["rows"]
                c["failed"] += leg["failed"]
    lines = [
        "",
        "| Dataset | D | Exact ln Z | H (nat) | Exact references | Planned cells | "
        "Deferred cells | Pilot rows (failed) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for dataset, info in section["datasets"].items():
        backends = []
        ln_z = h = None
        for (name, backend, data_seed), ref in sorted(refs.items()):
            if name == dataset and data_seed == 1:
                backends.append(f"`{backend}` {ref.get('status')}")
                ln_z, h = ref.get("log_evidence"), ref.get("information_nats")
        c = counts.get(dataset, {})
        lines.append(
            f"| `{dataset}` | {info['dimension']} | {_fmt(ln_z, '.3f')} | {_fmt(h, '.1f')} | "
            f"{', '.join(backends) or DASH} | {c.get('planned', 0)} | {c.get('deferred', 0)} | "
            f"{c.get('rows', 0)} ({c.get('failed', 0)}) |"
        )
    cov = section["coverage"]
    lines += [
        "",
        f"Pilot: {cov['expected']} expected runs — {cov['row']} rows, {cov['failed']} failed, "
        f"{cov['missing']} not yet pulled, {cov['deferred']} deferred with reasons "
        f"(`catalogue/search_catalogue.json`, section `{section['protocol_id']}`). "
        f"{section['wave'][0].upper()}{section['wave'][1:]}. Cross-family ranking: "
        f"{section['cross_family']}.",
    ]
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


def render_wave2() -> str:
    """The scored wave's headline per task (``gaussian_x3@2`` section of the catalogue):
    one table per task and dataset: the ranked legs in the catalogue's order (expected wall
    per right answer, then success rate, among legs that clear the reliability floor), then
    the closed legs below the floor marked unreliable, then the legs not judged. Every row
    shows its attempts and its success rate."""
    doc = _catalogue()
    section = (doc or {}).get("gaussian_x3@2")
    if not section or not section["coverage"]["admitted_rows"]:
        return "\n_No scored wave-2 row yet._\n"
    cov = section["coverage"]
    floor = section["rank_floor"]
    lines = [
        "",
        f"`{section['protocol_id']}` at PyAutoFit `{section['pyautofit_pin'][:12]}`: "
        f"{cov['admitted_rows']} admitted rows of {cov['pulled_rows']} pulled "
        f"({cov['refused_rows']} refused), {cov['expected_runs']} expected runs. "
        f"{section['censoring'][0].upper()}{section['censoring'][1:]}.",
        "",
        "**Order (#):** by expected wall per right answer, then success rate, among closed "
        f"legs that clear the reliability floor (success rate ≥ "
        f"{floor['success_rate']:.0%} over ≥ {floor['attempts']} attempts); `=n` marks a leg "
        "whose bootstrap interval overlaps the one above it. Closed legs below the floor "
        "(zero-success legs included) follow, marked **unreliable** with their numbers and "
        "no position; open or unjudged legs come last. The floor is a post-hoc presentation "
        "convention adopted 2026-10-15 after seeing the wave-2 data (protocol §8 note), not "
        "a pre-registered criterion: it changes no verdict.",
        "",
    ]
    for task, group in section["tasks"].items():
        by_id = {leg["id"]: leg for leg in group["legs"]}
        for dataset, ordered in group["ranking"].items():
            legs = [by_id[e["id"]] for e in ordered]
            weak = [by_id[e["id"]] for e in group["unreliable"][dataset]]
            placed = {leg["id"] for leg in legs + weak}
            rest = [
                leg
                for leg in group["legs"]
                if leg["dataset"] == dataset and leg["id"] not in placed
            ]
            lines += [
                f"**{group['label']}** — `{dataset}`",
                "",
                "| # | Search | Settings | Config | Attempted | Accepted | Success rate "
                "(Wilson 95 %) | Wall per right answer (s, bootstrap 95 %) | Median attempt "
                "wall (s) | Timeout (s) | State |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            ties = {e["id"]: e.get("tied_with_previous") for e in ordered}
            for position, leg in (
                [(i + 1, leg) for i, leg in enumerate(legs)]
                + [("unreliable", leg) for leg in weak]
                + [(None, leg) for leg in rest]
            ):
                if isinstance(position, int) and ties.get(leg["id"]):
                    position = f"={position}"
                acc = leg.get("acceptance") or {}
                rate = leg.get("success_rate")
                if rate is None:
                    rate_text = DASH
                else:
                    lo, hi = leg["success_rate_wilson_95"]
                    rate_text = f"{rate:.0%} [{lo:.0%}, {hi:.0%}]"
                wps = leg.get("wall_per_success") or {}
                if wps.get("unbounded"):
                    wps_text = f"unbounded (≥ {_fmt(wps.get('lower_bound_s'), '.0f')})"
                elif wps.get("value_s") is not None:
                    ci = leg.get("wall_per_success_bootstrap_95") or [None, None]
                    wps_text = (
                        f"{_fmt(wps['value_s'], '.0f')} "
                        f"[{_fmt(ci[0], '.0f')}, {_fmt(ci[1], '.0f') if ci[1] is not None else '∞'}]"
                    )
                else:
                    wps_text = DASH
                state = leg["state"]
                lines.append(
                    f"| {position or DASH} | `{leg['sampler']}` | `{leg['settings']}` | "
                    f"`{leg['config_id']}` | {leg.get('attempted', 0)} | "
                    f"{acc.get('accepted', 0)} | {rate_text} | {wps_text} | "
                    f"{_fmt(leg.get('median_attempt_wall_s'), '.1f')} | "
                    f"{_fmt(leg.get('timeout_s'), '.0f')} | {state} |"
                )
            lines.append("")
    lines += [f"- {line}" for line in section["limitations"]]
    return "\n".join(lines) + "\n"


RENDERERS = {
    "wave2": render_wave2,
    "catalogue": render_catalogue,
    "references": render_references,
    "correlated_nd": render_correlated_nd,
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

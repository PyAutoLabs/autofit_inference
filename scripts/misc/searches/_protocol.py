"""Protocol ``gaussian_x3@1`` verdicts — standard library only.

Implements §4 (acceptance) and §5 (convergence) of
``wiki/project/protocol_gaussian_x3.md`` as pure functions of a result row, the
committed reference with the same identity (:data:`REFERENCE_IDENTITY`: dataset,
backend, data seed, assertion mechanism), and the measured ``ln 3!`` offsets. Imported by the runner (to print a verdict) and by the stdlib-only exporter
(to emit ``scientific.convergence`` / ``scientific.acceptance``), so it must never import
numpy or autofit.

Every threshold lives in :data:`THRESHOLDS`, with the protocol's PLACEHOLDER marking
carried in :data:`PLACEHOLDERS`. Changing one is a protocol version bump (§ change rule).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

PROTOCOL_ID = "gaussian_x3@1"

LN_3_FACTORIAL = math.log(6.0)

THRESHOLDS = {
    # §4(a)
    "map_tolerance_nat": 1.0,
    # §4(b)1
    "median_tolerance_sigma": 1.0,
    # §4(b)2 — PLACEHOLDER
    "sigma_ratio_band": (0.5, 2.0),
    # §4(b)3 — PLACEHOLDER
    "ppc_chi2_tolerance": 10.0,
    # §4(b)4
    "mode_coverage_min_weight": 0.05,
    "mode_radius_px": 5.0,  # PLACEHOLDER, mirrors _posterior.MODE_RADIUS_PX
    # §4(c)
    "evidence_tolerance_nat": 1.0,
    # §5 — PLACEHOLDERS
    "nested_min_ess": 500.0,
    "chain_max_rhat": 1.01,
    "chain_min_ess": 400.0,
    # §3 offset validation window
    "offset_window_nat": 0.1,
}

PLACEHOLDERS = (
    "sigma_ratio_band",
    "ppc_chi2_tolerance",
    "mode_radius_px",
    "nested_min_ess",
    "chain_max_rhat",
    "chain_min_ess",
)

TASKS = ("point_map", "posterior", "evidence")

#: What makes a reference applicable to a row (§4: same dataset and backend; §4(c): same
#: assertion mechanism; and the same noise realisation, ``data_seed``). A row whose
#: identity differs in any field is ``not_assessed``, never judged against it.
REFERENCE_IDENTITY = ("dataset", "backend", "data_seed", "assertion_mechanism")


def reference_key(record: dict) -> tuple:
    """The :data:`REFERENCE_IDENTITY` tuple of a row or a reference."""
    return tuple(record.get(field) for field in REFERENCE_IDENTITY)


def identity_mismatch(row: dict, reference: dict) -> str | None:
    """``None`` when ``reference`` applies to ``row``, else why it does not."""
    for field in REFERENCE_IDENTITY:
        got, ref = row.get(field), reference.get(field)
        if got is None or ref is None:
            return f"reference identity incomplete: {field} row {got!r} vs reference {ref!r}"
        if got != ref:
            return f"reference identity mismatch: {field} row {got!r} vs reference {ref!r}"
    return None


def _num(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def reference_file(root: Path, dataset: str, backend: str) -> Path:
    return Path(root) / "results" / "reference" / dataset / backend / "reference.json"


def offsets_file(root: Path) -> Path:
    return Path(root) / "results" / "reference" / "constant_likelihood.json"


def evidence_offset(offsets: dict | None, sampler: str, backend: str) -> float | None:
    """The measured ``ln Z_assert − ln Z_exch`` for this sampler and backend
    (``0`` or ``−ln 3!``), or ``None`` when it was not measured or failed validation."""
    if not offsets:
        return None
    entry = (offsets.get("offsets") or {}).get(sampler, {}).get(backend)
    if not isinstance(entry, dict) or entry.get("validated") is not True:
        return None
    return _num(entry.get("offset"))


def normalised_log_evidence(log_evidence, offset):
    log_z, off = _num(log_evidence), _num(offset)
    if log_z is None or off is None:
        return None
    return log_z - off


# ---------------------------------------------------------------------------
# §4 acceptance
# ---------------------------------------------------------------------------


def criterion_map(row: dict, reference: dict) -> tuple[bool | None, str]:
    got = _num(row.get("max_log_posterior"))
    ref = _num((reference.get("map") or {}).get("log_posterior"))
    if got is None or ref is None:
        return None, "(a) max_log_posterior or ref MAP missing"
    ok = got >= ref - THRESHOLDS["map_tolerance_nat"]
    return ok, f"(a) max_log_posterior {got:.3f} vs ref MAP {ref:.3f}"


def criteria_posterior(row: dict, reference: dict) -> list[tuple[bool | None, str]]:
    out: list[tuple[bool | None, str]] = []
    run = row.get("posterior_relabelled") or {}
    ref = reference.get("posterior_relabelled") or {}
    if not ref or not run:
        return [(None, "(b) relabelled posterior missing on row or reference")]
    worst_median, worst_key = 0.0, None
    lo, hi = THRESHOLDS["sigma_ratio_band"]
    ratio_fail = []
    for key, ref_stats in ref.items():
        run_stats = run.get(key) or {}
        r_med, r_sig = _num(ref_stats.get("median")), _num(ref_stats.get("sigma"))
        g_med, g_sig = _num(run_stats.get("median")), _num(run_stats.get("sigma"))
        if None in (r_med, r_sig, g_med, g_sig) or r_sig <= 0:
            return [(None, f"(b) {key} statistics missing")]
        delta = abs(g_med - r_med) / r_sig
        if delta > worst_median:
            worst_median, worst_key = delta, key
        ratio = g_sig / r_sig
        if not lo <= ratio <= hi:
            ratio_fail.append(f"{key} {ratio:.2f}")
    out.append(
        (
            worst_median <= THRESHOLDS["median_tolerance_sigma"],
            f"(b)1 worst |median-ref|/ref_sigma {worst_median:.2f} ({worst_key})",
        )
    )
    out.append(
        (
            not ratio_fail,
            "(b)2 sigma ratio in band" if not ratio_fail else f"(b)2 sigma ratio {ratio_fail}",
        )
    )
    g_chi2, r_chi2 = _num(row.get("ppc_chi2")), _num(reference.get("ppc_chi2"))
    if g_chi2 is None or r_chi2 is None:
        out.append((None, "(b)3 ppc_chi2 missing"))
    else:
        out.append(
            (
                g_chi2 - r_chi2 <= THRESHOLDS["ppc_chi2_tolerance"],
                f"(b)3 ppc_chi2 {g_chi2:.1f} vs ref {r_chi2:.1f}",
            )
        )
    out.append(criterion_modes(row, reference))
    return out


def criterion_modes(row: dict, reference: dict) -> tuple[bool | None, str]:
    ref_modes = [
        m
        for m in reference.get("modes") or []
        if (_num(m.get("weight")) or 0.0) >= THRESHOLDS["mode_coverage_min_weight"]
    ]
    run_modes = row.get("modes")
    if run_modes is None:
        return None, "(b)4 run modes missing"
    radius = THRESHOLDS["mode_radius_px"]
    missed = 0
    for ref_mode in ref_modes:
        centres = ref_mode.get("centres") or []
        found = any(
            len(m.get("centres") or []) == len(centres)
            and all(abs(a - b) <= radius for a, b in zip(m["centres"], centres))
            for m in run_modes
        )
        missed += not found
    return missed == 0, f"(b)4 reference modes covered {len(ref_modes) - missed}/{len(ref_modes)}"


def criterion_evidence(row: dict, reference: dict, offsets: dict | None) -> tuple[bool | None, str]:
    offset = evidence_offset(offsets, row.get("sampler", ""), row.get("backend", ""))
    got = normalised_log_evidence(row.get("log_evidence"), offset)
    ref = _num(reference.get("log_evidence_normalised"))
    if offset is None:
        return None, f"(c) no validated ln 3! offset for {row.get('sampler')}/{row.get('backend')}"
    if got is None or ref is None:
        return None, "(c) log evidence missing on row or reference"
    ok = abs(got - ref) <= THRESHOLDS["evidence_tolerance_nat"]
    return ok, f"(c) ln Z_norm {got:.3f} vs ref {ref:.3f}"


def acceptance(row: dict, reference: dict | None, offsets: dict | None = None) -> dict:
    """``{"acceptance", "reason", "criteria", "reference_limitations"}`` —
    ``accepted`` / ``rejected`` / ``not_assessed`` per protocol §4.

    Every verdict reached against a reference carries that reference's
    ``limitations`` (§6: a disagreement among the reference runs is a limitation of
    every verdict that uses the reference), in ``reference_limitations`` and appended
    to ``reason``.
    """
    task = row.get("task")
    if task not in TASKS:
        return _judged("not_assessed", f"unknown task {task!r}", [])
    if not reference or reference.get("status") != "complete":
        status = (reference or {}).get("status", "missing")
        return _judged(
            "not_assessed",
            f"no complete reference for {row.get('dataset')}/{row.get('backend')} "
            f"(reference status: {status})",
            [],
        )
    mismatch = identity_mismatch(row, reference)
    if mismatch:
        return _judged("not_assessed", mismatch, [])
    limitations = [str(line) for line in reference.get("limitations") or []]
    if row.get("status") != "complete" or row.get("completed") is not True:
        return _judged("rejected", f"not completed: {row.get('status')}", [], limitations)
    if task == "point_map":
        criteria = [criterion_map(row, reference)]
    else:
        criteria = criteria_posterior(row, reference)
        if task == "evidence":
            criteria.append(criterion_evidence(row, reference, offsets))
    failed = [text for ok, text in criteria if ok is False]
    unknown = [text for ok, text in criteria if ok is None]
    if failed:
        verdict, reason = "rejected", "failed: " + "; ".join(failed)
    elif unknown:
        verdict, reason = "not_assessed", "not assessable: " + "; ".join(unknown)
    else:
        verdict, reason = "accepted", "all criteria met: " + "; ".join(t for _, t in criteria)
    return _judged(verdict, reason, criteria, limitations)


def _judged(verdict: str, reason: str, criteria, limitations=()) -> dict:
    limitations = list(limitations)
    if limitations:
        reason = f"{reason} [reference limitations: {'; '.join(limitations)}]"
    return {
        "acceptance": verdict,
        "reason": reason,
        "criteria": [{"ok": ok, "detail": text} for ok, text in criteria],
        "reference_limitations": limitations,
    }


# ---------------------------------------------------------------------------
# §5 convergence
# ---------------------------------------------------------------------------


def convergence(row: dict) -> dict:
    task = row.get("task")
    family = row.get("family")
    if task == "point_map":
        return {"convergence": "not_assessed", "reason": "point/MAP: (a) is the test"}
    if row.get("completed") is not True:
        return {"convergence": "not_converged", "reason": f"not completed: {row.get('status')}"}
    if family == "nested":
        ess = _num(row.get("ess_kish"))
        if ess is None:
            return {"convergence": "not_assessed", "reason": "no ESS recorded"}
        ok = ess >= THRESHOLDS["nested_min_ess"]
        return {
            "convergence": "converged" if ok else "not_converged",
            "reason": f"terminated; Kish ESS {ess:.0f} vs {THRESHOLDS['nested_min_ess']:.0f}",
        }
    rhat, ess = _num(row.get("rhat_max")), _num(row.get("ess_bulk_min"))
    if rhat is None or ess is None:
        return {"convergence": "not_assessed", "reason": "no chain diagnostics recorded"}
    ok = rhat <= THRESHOLDS["chain_max_rhat"] and ess >= THRESHOLDS["chain_min_ess"]
    return {
        "convergence": "converged" if ok else "not_converged",
        "reason": f"R-hat {rhat:.3f}, bulk ESS {ess:.0f}",
    }


def verdict(row: dict, reference: dict | None, offsets: dict | None = None) -> dict:
    """The producer-asserted ``scientific`` block for one row."""
    acc = acceptance(row, reference, offsets)
    conv = convergence(row)
    return {
        "protocol_id": PROTOCOL_ID,
        "acceptance": acc["acceptance"],
        "acceptance_reason": acc["reason"],
        "convergence": conv["convergence"],
        "convergence_reason": conv["reason"],
        "criteria": acc["criteria"],
        "reference_limitations": acc["reference_limitations"],
        "placeholders": list(PLACEHOLDERS),
    }


# ---------------------------------------------------------------------------
# §8 headline helpers
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def wall_per_success(walls: list[float], successes: list[bool]) -> dict:
    """Expected wall per right answer (§8): Σ wall over all attempts / successes; with
    zero successes, unbounded plus the Wilson-upper lower bound."""
    n = len(walls)
    k = sum(bool(s) for s in successes)
    total = sum(walls)
    lo, hi = wilson_interval(k, n)
    if n == 0:
        return {"value_s": None, "lower_bound_s": None, "n": 0, "successes": 0}
    if k == 0:
        mean = total / n
        return {
            "value_s": None,
            "unbounded": True,
            "lower_bound_s": mean / hi if hi > 0 else None,
            "n": n,
            "successes": 0,
        }
    return {"value_s": total / k, "unbounded": False, "n": n, "successes": k}

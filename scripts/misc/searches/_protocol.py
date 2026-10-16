"""Protocol ``gaussian_x3`` verdicts (``@1`` for the wave-1 pilot, ``@2`` for scored wave 2) —
standard library only.

Implements §4 (acceptance) and §5 (convergence) of
``wiki/project/protocol_gaussian_x3.md`` as pure functions of a result row, the
committed reference with the same identity (:data:`REFERENCE_IDENTITY`: dataset,
backend, data seed, assertion mechanism), and the measured ``ln 3!`` offsets. Imported by the runner (to print a verdict) and by the stdlib-only exporter
(to emit ``scientific.convergence`` / ``scientific.acceptance``), so it must never import
numpy or autofit.

Every threshold lives in :data:`THRESHOLDS` (``@1``, with its PLACEHOLDER marking carried in
:data:`PLACEHOLDERS`) and :data:`THRESHOLDS_V2` (``@1`` with :data:`FROZEN_V2` applied: the
calibrated values frozen from ``results/calibration/gaussian_x3_wave1.json`` before any
scored run, protocol §7 and §11 A2). Changing one is a protocol version bump (§ change
rule). A row is judged by the protocol of the wave it belongs to (:func:`protocol_for`):
the 223 pilot rows stay ``@1`` with their verdicts byte-identical; scored wave-2 rows are
``@2``, and every ``@2`` verdict string starts with the id.

**The second family (B5c).** Rows of the ``correlated_nd`` model (``row["model"]``) are
judged by their own protocol, ``correlated_nd@1`` (``wiki/project/protocol_correlated_nd.md``;
:data:`FAMILY_PROTOCOL_IDS`, :data:`THRESHOLDS_CND1`) against an **exact** reference (no
reference search): §4(a) and §4(c) as above with no evidence offset (no assertions), and
§4(b) replaces the posterior-predictive and mode criteria with a whitened-coordinate
check (§4(b)3 of that protocol, read from the row's ``whitened`` block). Every
``correlated_nd@1`` verdict string starts with the id. Nothing here ranks or compares
across families.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

#: The protocol that judges each wave (§9): wave 1 is the pilot, wave 2 the scored wave.
PROTOCOL_IDS = {1: "gaussian_x3@1", 2: "gaussian_x3@2"}

#: The wave-1 (pilot) id. Kept under this name for the wave-1 consumers (the calibration
#: record, the pilot catalogue and summary), whose output must not move; resolve a row's
#: id with :func:`protocol_for`, never with this constant.
PROTOCOL_ID = PROTOCOL_IDS[1]

#: The protocol new artefacts (reference runs, scored rows) are made under from the
#: ``@2`` freeze commit on. §6 (references) is unchanged by ``@2``.
CURRENT_PROTOCOL_ID = PROTOCOL_IDS[2]

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

#: ``gaussian_x3@2`` — the six PLACEHOLDERs frozen from the calibration record
#: ``results/calibration/gaussian_x3_wave1.json`` (``frozen``; 17 calibration rows, so the
#: σ-ratio band and ppc tolerance keep their ``@1`` values by the pre-declared < 20 rule;
#: the mode radius moves 5.0 → 5.5 px to clear 10 × the largest reference centre σ,
#: 5.32 px). Copied by hand, a human-readable diff (``calibrate_pilot.py`` docstring); a
#: test pins it to the record. ``mode_radius_px`` here is the §4(b)4 coverage radius; the
#: §3 clustering radius that writes ``modes`` onto rows and references stays
#: ``_posterior.MODE_RADIUS_PX`` (5 px) so every row and reference is clustered alike
#: (protocol §11 A2).
FROZEN_V2 = {
    "sigma_ratio_band": (0.5, 2.0),
    "ppc_chi2_tolerance": 10.0,
    "mode_radius_px": 5.5,
    "nested_min_ess": 500.0,
    "chain_max_rhat": 1.01,
    "chain_min_ess": 400.0,
}

#: The ``@2`` timeout and censoring rules (protocol §7, §11 A2): per (search × settings ×
#: backend), ``max(600 s, 10 × the median pilot total wall)`` from the calibration record's
#: ``timeouts``; a cell with no pilot wall (``probe_first``) takes the same rule over its
#: first five wave-2 runs (data realisation 1, seeds 0–4) before the rest of it is
#: submitted. An expected run that hit the timeout, crashed, wrote a non-finite answer or
#: left no row is an attempted failure censored at the timeout; never dropped.
TIMEOUT_RULE_V2 = (
    "max(600 s, 10 x median pilot total wall); probe-first cells: same rule over a 5-run probe"
)

THRESHOLDS_V2 = {**THRESHOLDS, **FROZEN_V2}

#: The second family's protocol per wave (B5c): only its pilot (wave 1) is declared; the
#: scored wave waits for ``correlated_nd@2``, frozen from the pilot (protocol §7).
FAMILY_PROTOCOL_IDS = {"correlated_nd": {1: "correlated_nd@1"}}
CND_PROTOCOL_ID = FAMILY_PROTOCOL_IDS["correlated_nd"][1]

#: ``correlated_nd@1`` (protocol ``wiki/project/protocol_correlated_nd.md`` §4, §5). The
#: values are gaussian_x3@2's frozen ones where the criterion is the same; the
#: PLACEHOLDERs (:data:`PLACEHOLDERS_CND1`) are calibrated on the pilot and frozen as
#: ``correlated_nd@2`` before any scored row.
THRESHOLDS_CND1 = {
    # §4(a)
    "map_tolerance_nat": 1.0,
    # §4(b)1, §4(b)2 — every parameter's marginal
    "median_tolerance_sigma": 1.0,
    "sigma_ratio_band": (0.5, 2.0),  # PLACEHOLDER
    # §4(b)3 — every exact-whitened coordinate (unit-normal units)
    "whitened_median_tolerance": 1.0,
    "whitened_sigma_ratio_band": (0.5, 2.0),  # PLACEHOLDER
    # §4(c)
    "evidence_tolerance_nat": 1.0,
    # §5 — PLACEHOLDERS
    "nested_min_ess": 500.0,
    "chain_max_rhat": 1.01,
    "chain_min_ess": 400.0,
}

PLACEHOLDERS_CND1 = (
    "sigma_ratio_band",
    "whitened_sigma_ratio_band",
    "nested_min_ess",
    "chain_max_rhat",
    "chain_min_ess",
)

THRESHOLDS_BY_PROTOCOL = {
    PROTOCOL_IDS[1]: THRESHOLDS,
    PROTOCOL_IDS[2]: THRESHOLDS_V2,
    CND_PROTOCOL_ID: THRESHOLDS_CND1,
}

#: What is still a PLACEHOLDER under each protocol: nothing, once frozen.
PLACEHOLDERS_BY_PROTOCOL = {
    PROTOCOL_IDS[1]: PLACEHOLDERS,
    PROTOCOL_IDS[2]: (),
    CND_PROTOCOL_ID: PLACEHOLDERS_CND1,
}


def is_family_row(row: dict) -> bool:
    """A row (or reference) of a family other than gaussian_x3 (judged by its own protocol)."""
    return row.get("model") in FAMILY_PROTOCOL_IDS


def protocol_id_for(model_type: str, wave: int) -> str | None:
    """The protocol a new row of ``model_type`` in ``wave`` is stamped with."""
    if model_type in FAMILY_PROTOCOL_IDS:
        return FAMILY_PROTOCOL_IDS[model_type].get(wave)
    return PROTOCOL_IDS[wave]


def wave_of(row: dict) -> int:
    """The wave a row belongs to: its ``wave`` field, else 1 (every row written before
    the ``@2`` freeze is a wave-1 pilot row and carries ``pilot: true``)."""
    wave = row.get("wave")
    if isinstance(wave, int) and not isinstance(wave, bool):
        return wave
    return 1


def protocol_for(row: dict) -> str | None:
    """The protocol id that judges ``row`` — the one of its wave, never a later one (the
    change rule: a verdict stays under the protocol in force when its wave was declared).
    ``None`` for a wave with no protocol, or a row stamped with another wave's id."""
    if is_family_row(row):
        pid = FAMILY_PROTOCOL_IDS[row["model"]].get(wave_of(row))
    else:
        pid = PROTOCOL_IDS.get(wave_of(row))
    stamped = row.get("protocol_id")
    if pid is None or (stamped is not None and stamped != pid):
        return None
    return pid


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


#: Each dataset's committed default realisation (``simulators/gaussian_x3.py``
#: ``DATASETS``; a test pins the two together). Its reference stays at the B2/B3 path;
#: every other realisation's lives under ``data_seed<d>/`` beside it.
DEFAULT_DATA_SEED = {"gaussian_x3_blend": 1, "gaussian_x3_separated": 2}

#: The ``correlated_nd`` datasets' committed default realisation
#: (``simulators/correlated_nd.py`` ``DEFAULT_DATA_SEED``; a test pins them together).
FAMILY_DEFAULT_DATA_SEED = {
    "corr_gaussian_d20": 1,
    "corr_gaussian_d50": 1,
    "corr_gaussian_d100": 1,
    "rosenbrock_d20": 1,
}


def reference_dir(root: Path, dataset: str, backend: str, data_seed: int | None = None) -> Path:
    """``results/reference/<dataset>/<backend>/`` for the dataset's default realisation,
    ``…/<backend>/data_seed<d>/`` for any other (scored wave 2, protocol §9: five
    realisations, each judged only against its own reference — :data:`REFERENCE_IDENTITY`)."""
    base = Path(root) / "results" / "reference" / dataset / backend
    default = DEFAULT_DATA_SEED.get(dataset, FAMILY_DEFAULT_DATA_SEED.get(dataset))
    if data_seed is None or data_seed == default:
        return base
    return base / f"data_seed{int(data_seed)}"


def reference_file(root: Path, dataset: str, backend: str, data_seed: int | None = None) -> Path:
    return reference_dir(root, dataset, backend, data_seed) / "reference.json"


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


def criterion_map(row: dict, reference: dict, t: dict = THRESHOLDS) -> tuple[bool | None, str]:
    got = _num(row.get("max_log_posterior"))
    ref = _num((reference.get("map") or {}).get("log_posterior"))
    if got is None or ref is None:
        return None, "(a) max_log_posterior or ref MAP missing"
    ok = got >= ref - t["map_tolerance_nat"]
    return ok, f"(a) max_log_posterior {got:.3f} vs ref MAP {ref:.3f}"


def criteria_posterior(
    row: dict, reference: dict, t: dict = THRESHOLDS
) -> list[tuple[bool | None, str]]:
    out = criteria_marginals(row, reference, t)
    if len(out) == 1:
        return out
    g_chi2, r_chi2 = _num(row.get("ppc_chi2")), _num(reference.get("ppc_chi2"))
    if g_chi2 is None or r_chi2 is None:
        out.append((None, "(b)3 ppc_chi2 missing"))
    else:
        out.append(
            (
                g_chi2 - r_chi2 <= t["ppc_chi2_tolerance"],
                f"(b)3 ppc_chi2 {g_chi2:.1f} vs ref {r_chi2:.1f}",
            )
        )
    out.append(criterion_modes(row, reference, t))
    return out


def criteria_marginals(
    row: dict, reference: dict, t: dict = THRESHOLDS
) -> list[tuple[bool | None, str]]:
    """§4(b)1 and §4(b)2 (shared by both families): every parameter's relabelled median
    within ``median_tolerance_sigma`` reference sigma, and its sigma ratio in the band. A
    single ``None`` entry when the statistics are missing."""
    out: list[tuple[bool | None, str]] = []
    run = row.get("posterior_relabelled") or {}
    ref = reference.get("posterior_relabelled") or {}
    if not ref or not run:
        return [(None, "(b) relabelled posterior missing on row or reference")]
    worst_median, worst_key = 0.0, None
    lo, hi = t["sigma_ratio_band"]
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
            worst_median <= t["median_tolerance_sigma"],
            f"(b)1 worst |median-ref|/ref_sigma {worst_median:.2f} ({worst_key})",
        )
    )
    out.append(
        (
            not ratio_fail,
            "(b)2 sigma ratio in band" if not ratio_fail else f"(b)2 sigma ratio {ratio_fail}",
        )
    )
    return out


def criterion_whitened(row: dict, t: dict = THRESHOLDS_CND1) -> tuple[bool | None, str]:
    """``correlated_nd`` §4(b)3: in the exact whitening map's coordinates (each ``N(0, 1)``
    under the true posterior) every coordinate's ``|median|`` is within
    ``whitened_median_tolerance`` unit sigma and its sigma ratio lies in the band. Read from
    the row's ``whitened`` block (the runner computes it from the samples)."""
    stats = row.get("whitened") or {}
    worst = _num(stats.get("worst_median"))
    lo_ratio, hi_ratio = _num(stats.get("sigma_ratio_min")), _num(stats.get("sigma_ratio_max"))
    if None in (worst, lo_ratio, hi_ratio):
        return None, "(b)3 whitened statistics missing"
    lo, hi = t["whitened_sigma_ratio_band"]
    ok = worst <= t["whitened_median_tolerance"] and lo <= lo_ratio and hi_ratio <= hi
    return ok, (
        f"(b)3 whitened worst |median| {worst:.2f} ({stats.get('worst_median_key')}), "
        f"sigma ratio [{lo_ratio:.2f} ({stats.get('sigma_ratio_min_key')}), "
        f"{hi_ratio:.2f} ({stats.get('sigma_ratio_max_key')})]"
    )


def criterion_modes(row: dict, reference: dict, t: dict = THRESHOLDS) -> tuple[bool | None, str]:
    ref_modes = [
        m
        for m in reference.get("modes") or []
        if (_num(m.get("weight")) or 0.0) >= t["mode_coverage_min_weight"]
    ]
    run_modes = row.get("modes")
    if run_modes is None:
        return None, "(b)4 run modes missing"
    radius = t["mode_radius_px"]
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


#: The separated control's mechanism (D15 option (b), amendment A1): disjoint centre
#: priors and no assertions exclude no prior volume, so the offset is 0 by construction;
#: likewise ``none``, the ``correlated_nd`` family (no assertions, no label symmetry).
NO_EXCLUDED_VOLUME = ("disjoint_priors", "none")


def run_evidence_offset(run: dict, offsets: dict | None) -> float | None:
    """The §3 offset for a row or reference run: ``0`` for a mechanism that excludes no
    prior volume (:data:`NO_EXCLUDED_VOLUME`), else the measured sampler/backend offset.
    The reference builder and :func:`criterion_evidence` must both use this, so a row and
    its reference are normalised by the same rule."""
    if run.get("assertion_mechanism") in NO_EXCLUDED_VOLUME:
        return 0.0
    return evidence_offset(offsets, run.get("sampler", ""), run.get("backend", ""))


def criterion_evidence(
    row: dict, reference: dict, offsets: dict | None, t: dict = THRESHOLDS
) -> tuple[bool | None, str]:
    offset = run_evidence_offset(row, offsets)
    got = normalised_log_evidence(row.get("log_evidence"), offset)
    ref = _num(reference.get("log_evidence_normalised"))
    if offset is None:
        return None, f"(c) no validated ln 3! offset for {row.get('sampler')}/{row.get('backend')}"
    if got is None or ref is None:
        return None, "(c) log evidence missing on row or reference"
    ok = abs(got - ref) <= t["evidence_tolerance_nat"]
    return ok, f"(c) ln Z_norm {got:.3f} vs ref {ref:.3f}"


def acceptance(
    row: dict, reference: dict | None, offsets: dict | None = None, t: dict = THRESHOLDS
) -> dict:
    """``{"acceptance", "reason", "criteria", "reference_limitations", "map_diagnostic"}`` —
    ``accepted`` / ``rejected`` / ``not_assessed`` per protocol §4.

    Every verdict reached against a reference carries that reference's
    ``limitations`` (§6: a disagreement among the reference runs is a limitation of
    every verdict that uses the reference), in ``reference_limitations`` and appended
    to ``reason``. ``map_diagnostic`` records §4(a) on every completed row: the
    criterion for point/MAP, a diagnostic only (never a condition) for the others.
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
    # §4(a) on every row: the criterion for point/MAP, a diagnostic for the others.
    map_ok, map_text = criterion_map(row, reference, t)
    map_diagnostic = {"ok": map_ok, "detail": map_text, "criterion": task == "point_map"}
    if task == "point_map":
        criteria = [criterion_map(row, reference, t)]
    elif is_family_row(row):
        criteria = criteria_marginals(row, reference, t)
        if len(criteria) > 1:
            criteria.append(criterion_whitened(row, t))
        if task == "evidence":
            criteria.append(criterion_evidence(row, reference, offsets, t))
    else:
        criteria = criteria_posterior(row, reference, t)
        if task == "evidence":
            criteria.append(criterion_evidence(row, reference, offsets, t))
    failed = [text for ok, text in criteria if ok is False]
    unknown = [text for ok, text in criteria if ok is None]
    if failed:
        verdict, reason = "rejected", "failed: " + "; ".join(failed)
    elif unknown:
        verdict, reason = "not_assessed", "not assessable: " + "; ".join(unknown)
    else:
        verdict, reason = "accepted", "all criteria met: " + "; ".join(t for _, t in criteria)
    return _judged(verdict, reason, criteria, limitations, map_diagnostic)


def _judged(verdict: str, reason: str, criteria, limitations=(), map_diagnostic=None) -> dict:
    limitations = list(limitations)
    if limitations:
        reason = f"{reason} [reference limitations: {'; '.join(limitations)}]"
    return {
        "acceptance": verdict,
        "reason": reason,
        "criteria": [{"ok": ok, "detail": text} for ok, text in criteria],
        "reference_limitations": limitations,
        "map_diagnostic": map_diagnostic,
    }


# ---------------------------------------------------------------------------
# §5 convergence
# ---------------------------------------------------------------------------


def convergence(row: dict, t: dict = THRESHOLDS) -> dict:
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
        ess_text = f"Kish ESS {ess:.0f} vs {t['nested_min_ess']:.0f}"
        termination = row.get("termination") or {}
        met = termination.get("met")
        if met is None:
            why = termination.get("reason") or "termination condition not recorded on the row"
            return {"convergence": "not_assessed", "reason": f"{why}; {ess_text}"}
        if met is not True:
            return {
                "convergence": "not_converged",
                "reason": f"own termination criterion not met ({_termination_text(termination)}); "
                f"{ess_text}",
            }
        ok = ess >= t["nested_min_ess"]
        return {
            "convergence": "converged" if ok else "not_converged",
            "reason": f"terminated ({_termination_text(termination)}); {ess_text}",
        }
    invalid = (row.get("chain_diagnostics") or {}).get("invalid_parameters")
    if invalid:
        return {
            "convergence": "not_converged",
            "reason": "no valid R-hat/bulk ESS for every free parameter (§5); non-finite for: "
            + ", ".join(invalid),
        }
    rhat, ess = _num(row.get("rhat_max")), _num(row.get("ess_bulk_min"))
    if rhat is None or ess is None:
        return {"convergence": "not_assessed", "reason": "no chain diagnostics recorded"}
    ok = rhat <= t["chain_max_rhat"] and ess >= t["chain_min_ess"]
    return {
        "convergence": "converged" if ok else "not_converged",
        "reason": f"R-hat {rhat:.3f}, bulk ESS {ess:.0f}",
    }


def _termination_text(termination: dict) -> str:
    fields = ("explored", "shells_full", "n_eff", "n_eff_target", "n_like", "n_like_max")
    parts = [f"{k} {termination[k]}" for k in fields if k in termination]
    return ", ".join(parts) or "met"


#: References completed after the wave-1 pilot that do **not** judge its ``@1`` rows
#: (B5b ruling 1): the pilot ranks nothing, so re-judging its rows against a reference
#: finished later would be a reassessment (review 06 item 8), a separate human-chosen
#: task. Keyed by :func:`reference_key`; those pilot rows stay ``not_assessed`` with this
#: reason. Scored (``@2``) rows are judged against the same reference as usual.
PILOT_NOT_REJUDGED = {
    ("gaussian_x3_blend", "jax_cpu", 1, "xp_where_penalty"): (
        "reference completed after the pilot (B5b)"
    ),
}


def verdict(row: dict, reference: dict | None, offsets: dict | None = None) -> dict:
    """The producer-asserted ``scientific`` block for one row, under the protocol of the
    row's wave (:func:`protocol_for`). ``@1`` verdicts are byte-identical to the pilot's;
    every ``@2`` reason starts with ``gaussian_x3@2:``. A row whose stamped id disagrees
    with its wave is ``not_assessed``, never judged under the wrong thresholds."""
    pid = protocol_for(row)
    if pid is None:
        why = f"no protocol for wave {row.get('wave')!r} with stamped id {row.get('protocol_id')!r}"
        return {
            "protocol_id": row.get("protocol_id"),
            "acceptance": "not_assessed",
            "acceptance_reason": why,
            "convergence": "not_assessed",
            "convergence_reason": why,
            "criteria": [],
            "reference_limitations": [],
            "map_diagnostic": None,
            "placeholders": [],
        }
    t = THRESHOLDS_BY_PROTOCOL[pid]
    held = (
        PILOT_NOT_REJUDGED.get(reference_key(reference))
        if pid == PROTOCOL_IDS[1] and reference and reference_key(reference) == reference_key(row)
        else None
    )
    acc = _judged("not_assessed", held, []) if held else acceptance(row, reference, offsets, t)
    conv = convergence(row, t)
    prefix = "" if pid == PROTOCOL_IDS[1] else f"{pid}: "  # every family id is prefixed
    return {
        "protocol_id": pid,
        "acceptance": acc["acceptance"],
        "acceptance_reason": prefix + acc["reason"],
        "convergence": conv["convergence"],
        "convergence_reason": prefix + conv["reason"],
        "criteria": acc["criteria"],
        "reference_limitations": acc["reference_limitations"],
        "map_diagnostic": acc["map_diagnostic"],
        "placeholders": list(PLACEHOLDERS_BY_PROTOCOL[pid]),
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


def attempt_wall_s(row: dict) -> float | None:
    """One attempt's elapsed cost (§8): the consumer's ``total_wall_s`` plus, for a
    warm start, the provider's ``provider_wall_s``. ``None`` when any part is unknown —
    an unknown cost is missing, never 0. A warm start whose provider failed never
    started its consumer, so the consumer's share is a known 0."""
    warm = bool(row.get("warm_start"))
    provider = row.get("provider_wall_s") if warm else 0.0
    if provider is None:
        return None
    consumer = row.get("total_wall_s")
    if consumer is None:
        if warm and row.get("provider_completed") is False:
            consumer = 0.0
        else:
            return None
    return float(consumer) + float(provider)


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

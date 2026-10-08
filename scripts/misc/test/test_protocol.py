"""Protocol gaussian_x3@1 verdict logic (§4, §5, §8)."""

import copy
import math

import pytest
from searches import _protocol as protocol

POST = {"a": {"median": 10.0, "sigma": 1.0}, "b": {"median": 0.0, "sigma": 2.0}}
REFERENCE = {
    "status": "complete",
    "dataset": "gaussian_x3_blend",
    "backend": "numpy",
    "data_seed": 1,
    "assertion_mechanism": "raise_resample",
    "posterior_relabelled": POST,
    "ppc_chi2": 60.0,
    "modes": [{"centres": [25.0, 45.0, 60.0], "weight": 1.0}],
    "log_evidence_normalised": 100.0,
    "map": {"log_posterior": 50.0},
}
OFFSETS = {"offsets": {"nautilus": {"numpy": {"offset": -math.log(6), "validated": True}}}}


def row(**overrides):
    base = {
        "task": "evidence",
        "family": "nested",
        "sampler": "nautilus",
        "dataset": "gaussian_x3_blend",
        "backend": "numpy",
        "data_seed": 1,
        "assertion_mechanism": "raise_resample",
        "status": "complete",
        "completed": True,
        "posterior_relabelled": copy.deepcopy(POST),
        "ppc_chi2": 61.0,
        "modes": [{"centres": [25.2, 44.9, 60.3], "weight": 1.0}],
        "log_evidence": 100.0 - math.log(6),
        "max_log_posterior": 49.5,
        "ess_kish": 1000.0,
    }
    base.update(overrides)
    return base


def test_accepted_when_every_criterion_holds():
    out = protocol.verdict(row(), REFERENCE, OFFSETS)
    assert out["acceptance"] == "accepted", out
    assert out["convergence"] == "converged"
    assert out["protocol_id"] == "gaussian_x3@1"


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"posterior_relabelled": {"a": {"median": 11.5, "sigma": 1.0}, "b": POST["b"]}}, "(b)1"),
        ({"posterior_relabelled": {"a": {"median": 10.0, "sigma": 3.0}, "b": POST["b"]}}, "(b)2"),
        ({"ppc_chi2": 75.0}, "(b)3"),
        ({"modes": [{"centres": [25.0, 45.0, 80.0], "weight": 1.0}]}, "(b)4"),
        ({"log_evidence": 98.0 - math.log(6)}, "(c)"),
    ],
)
def test_each_failed_criterion_rejects_and_is_named(overrides, needle):
    out = protocol.acceptance(row(**overrides), REFERENCE, OFFSETS)
    assert out["acceptance"] == "rejected"
    assert needle in out["reason"]


def test_point_map_judged_on_map_only():
    ok = protocol.acceptance(row(task="point_map", family="optimizer", ppc_chi2=999), REFERENCE)
    assert ok["acceptance"] == "accepted"
    bad = protocol.acceptance(row(task="point_map", max_log_posterior=48.0), REFERENCE)
    assert bad["acceptance"] == "rejected" and "(a)" in bad["reason"]
    assert protocol.convergence(row(task="point_map"))["convergence"] == "not_assessed"


def test_posterior_task_ignores_evidence():
    out = protocol.acceptance(row(task="posterior", log_evidence=None), REFERENCE)
    assert out["acceptance"] == "accepted"


def test_not_assessed_without_reference_or_offset():
    assert protocol.acceptance(row(), None)["acceptance"] == "not_assessed"
    pending = dict(REFERENCE, status="pending")
    assert protocol.acceptance(row(), pending)["acceptance"] == "not_assessed"
    other_backend = dict(REFERENCE, backend="jax_cpu")
    assert protocol.acceptance(row(), other_backend)["acceptance"] == "not_assessed"
    unvalidated = {"offsets": {"nautilus": {"numpy": {"offset": None, "validated": False}}}}
    out = protocol.acceptance(row(), REFERENCE, unvalidated)
    assert out["acceptance"] == "not_assessed" and "ln 3!" in out["reason"]


@pytest.mark.parametrize(
    "field, value", [("data_seed", 777), ("assertion_mechanism", "no_assertions")]
)
def test_reference_identity_mismatch_is_not_assessed(field, value):
    """Review finding 3: a different noise realisation or assertion mechanism is never
    judged against the reference, even when every statistic would pass."""
    out = protocol.acceptance(row(**{field: value}), REFERENCE, OFFSETS)
    assert out["acceptance"] == "not_assessed"
    assert "identity mismatch" in out["reason"] and field in out["reason"]
    missing = protocol.acceptance(row(**{field: None}), REFERENCE, OFFSETS)
    assert missing["acceptance"] == "not_assessed" and "incomplete" in missing["reason"]
    legacy = {k: v for k, v in REFERENCE.items() if k != field}
    assert protocol.acceptance(row(), legacy, OFFSETS)["acceptance"] == "not_assessed"


def test_reference_limitations_accompany_every_verdict():
    """Review finding 5 / protocol §6: a reference built from an agreeing subset carries
    the disagreement into every verdict that uses it."""
    note = "the six runs disagree (ln Z spread 2.33); the reference is the nautilus runs only"
    limited = dict(REFERENCE, limitations=[note])
    for overrides in ({}, {"ppc_chi2": 75.0}, {"status": "failed: X", "completed": False}):
        out = protocol.verdict(row(**overrides), limited, OFFSETS)
        assert out["reference_limitations"] == [note]
        assert note in out["acceptance_reason"]
    assert protocol.verdict(row(), REFERENCE, OFFSETS)["reference_limitations"] == []


def test_incomplete_run_is_rejected_and_not_converged():
    bad = row(status="failed: RuntimeError: boom", completed=False)
    assert protocol.acceptance(bad, REFERENCE, OFFSETS)["acceptance"] == "rejected"
    assert protocol.convergence(bad)["convergence"] == "not_converged"


def test_convergence_is_separate_from_acceptance():
    low_ess = row(ess_kish=10.0)
    assert protocol.verdict(low_ess, REFERENCE, OFFSETS)["acceptance"] == "accepted"
    assert protocol.convergence(low_ess)["convergence"] == "not_converged"
    chain = row(task="posterior", family="chain")
    assert protocol.convergence(chain)["convergence"] == "not_assessed"
    good_chain = row(task="posterior", family="chain", rhat_max=1.005, ess_bulk_min=900)
    assert protocol.convergence(good_chain)["convergence"] == "converged"


def test_wilson_and_wall_per_success():
    lo, hi = protocol.wilson_interval(10, 10)
    assert abs(hi - 1.0) < 1e-12 and 0.6 < lo < 0.75
    lo, hi = protocol.wilson_interval(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.32
    out = protocol.wall_per_success([10.0, 10.0, 20.0], [True, False, True])
    assert out["value_s"] == 20.0 and out["successes"] == 2
    zero = protocol.wall_per_success([10.0] * 10, [False] * 10)
    assert zero["value_s"] is None and zero["unbounded"] and zero["lower_bound_s"] > 10.0


def test_placeholders_are_named_thresholds():
    assert set(protocol.PLACEHOLDERS) <= set(protocol.THRESHOLDS)
    assert protocol.THRESHOLDS["sigma_ratio_band"] == (0.5, 2.0)

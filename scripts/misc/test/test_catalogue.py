"""The search catalogue: every registered search exactly once, one status, no ranking."""

import copy
import functools
import json
import re

import build_catalogue as catalogue
import build_readme
from searches import _pilot as pilot
from searches._runner import SAMPLERS


@functools.lru_cache(maxsize=1)
def _built() -> dict:
    return catalogue.build(catalogue.ROOT, catalogue.load_manifest())


def built() -> dict:
    """The catalogue as built from the committed rows (built once per session: ~12 000
    wave-2 rows make a build take ~15 s)."""
    return copy.deepcopy(_built())


RANK_WORDS = re.compile(r"(^|_)(rank|ranking_position|score|best|winner|order_by)(_|$)")


def _keys(value, out=None):
    out = set() if out is None else out
    if isinstance(value, dict):
        for key, item in value.items():
            out.add(key)
            _keys(item, out)
    elif isinstance(value, list):
        for item in value:
            _keys(item, out)
    return out


def test_every_manifest_search_appears_exactly_once_with_one_status():
    manifest = catalogue.load_manifest()
    doc = built()
    seen = [cls for group in doc["tasks"].values() for cls in group["searches"]]
    assert sorted(seen) == sorted(s["name"] for s in manifest["searches"])
    assert len(seen) == len(set(seen))
    for group in doc["tasks"].values():
        for cls, entry in group["searches"].items():
            assert entry["status"] in catalogue.STATUSES, cls
            assert entry["reason"], cls


def test_tasks_are_separate_groups_and_nothing_is_ranked():
    doc = built()
    # The wave-1 (pilot) part ranks nothing; the scored gaussian_x3@2 section is checked
    # by test_wave2_section_ranks_within_one_task_and_one_dataset.
    doc.pop("gaussian_x3@2")
    assert list(doc["tasks"]) == ["point_map", "posterior", "evidence"]
    assert doc["ranking"] == "none"
    for task, group in doc["tasks"].items():
        for cls, entry in group["searches"].items():
            for name in entry["harness_samplers"]:
                assert SAMPLERS[name].task == task, (cls, name)
    offenders = {k for k in _keys(doc) if RANK_WORDS.search(k)}
    assert not offenders, offenders
    # within a task the manifest's order is kept, never a result order
    manifest_order = [s["name"] for s in catalogue.load_manifest()["searches"]]
    for group in doc["tasks"].values():
        names = list(group["searches"])
        assert names == [n for n in manifest_order if n in names]


def test_readme_never_puts_two_tasks_in_one_table():
    text = build_readme.render_catalogue() + build_readme.render_searches()
    labels = ("**point/MAP**", "**posterior**", "**evidence**")
    tables = [b for b in re.split(r"\n\s*\n", text) if b.lstrip().startswith("|")]
    assert tables
    for table in tables:
        assert not any(label in table for label in labels)
    # each task heading is followed by its own table(s) only
    sections = re.split(r"(?=\*\*(?:point/MAP|posterior|evidence)\*\*)", text)
    for section in sections:
        assert sum(label in section for label in labels) <= 1


def test_unwired_and_deferred_searches_get_reasons():
    doc = built()
    drawer = doc["tasks"]["point_map"]["searches"]["Drawer"]
    assert drawer["status"] == "unsupported" and "sanity floor" in drawer["reason"]
    nss = doc["tasks"]["evidence"]["searches"]["NSS"]
    assert nss["status"] == "deferred" and "A3b" in nss["reason"]


def test_status_rules():
    cell = next(c for c in pilot.cells() if c.sampler == "lbfgs")
    failed = {"status": "failed: non-finite result (max_log_likelihood='inf')", "completed": True}
    leg = catalogue.leg_summary(cell, [dict(failed, seed=0)], {}, None)
    assert leg["failures"] == {"non-finite result": 1}
    assert catalogue.search_status("LBFGS", [leg], ["lbfgs"])[0] == "failed"
    empty = catalogue.leg_summary(cell, [], {}, None)
    assert catalogue.search_status("LBFGS", [empty], ["lbfgs"])[0] == "deferred"
    assert empty["deferred_seeds"] == list(pilot.SEEDS)
    assert catalogue.search_status("Drawer", [], [])[0] == "unsupported"


def test_committed_catalogue_is_current():
    committed = json.loads(catalogue.CATALOGUE_PATH.read_text())
    assert committed == json.loads(catalogue.render(built()))


def test_pilot_manifest_covers_the_menu():
    cells = pilot.cells()
    assert {c.dataset for c in cells} == set(pilot.DATASETS)
    assert all(c.status == "deferred" for c in cells if c.sampler == "nss")
    for c in cells:
        spec = SAMPLERS[c.sampler]
        legs = (["numpy"] if spec.numpy_supported else []) + (
            ["jax_cpu"] if spec.jax_native else []
        )
        assert c.backend in legs
    assert len(pilot.expected_runs()) == len(cells) * len(pilot.SEEDS)
    assert len(pilot.SEEDS) == 10


def test_bootstrap_keeps_zero_success_resamples_as_unbounded():
    # One success in two attempts: a resample draws zero successes with probability
    # 1/4, which is above the 2.5 % tail, so the upper endpoint is unbounded (null).
    interval = catalogue.bootstrap_wall_per_success([10.0, 10.0], [True, False])
    assert interval == [10.0, None]
    json.dumps(interval, allow_nan=False)  # no JSON infinity
    # Every resample has a success: both endpoints stay finite.
    assert catalogue.bootstrap_wall_per_success([10.0, 10.0], [True, True]) == [10.0, 10.0]
    assert catalogue.bootstrap_wall_per_success([10.0], [False]) is None


def test_leg_summary_reports_unknown_cost_as_missing_never_zero():
    """Review 06 finding 3: a row with no recorded cost suppresses the wall estimates."""
    cell = next(c for c in pilot.cells() if c.sampler == "lbfgs" and c.backend == "numpy")
    base = {
        "dataset": cell.dataset,
        "sampler": cell.sampler,
        "settings_name": cell.settings,
        "config_id": cell.config_id,
        "task": "point_map",
        "family": "optimizer",
    }
    rows = [
        {**base, "seed": 0, "status": "complete", "completed": True, "total_wall_s": 10.0},
        {**base, "seed": 1, "status": "failed: Boom: x", "completed": False, "total_wall_s": None},
    ]
    leg = catalogue.leg_summary(cell, rows, {}, None)
    assert leg["median_total_wall_s"] is None
    assert leg["wall_cost_unknown"] == 1
    assert "unknown" in leg["wall_reason"]


def test_wave2_section_ranks_within_one_task_and_one_dataset():
    doc = built()
    section = doc["gaussian_x3@2"]
    assert section["protocol_id"] == "gaussian_x3@2"
    assert section["pyautofit_pin"].startswith("df6d218f1")
    assert list(section["tasks"]) == ["point_map", "posterior", "evidence"]
    for task, group in section["tasks"].items():
        ids = {leg["id"]: leg for leg in group["legs"]}
        for leg in group["legs"]:
            assert SAMPLERS[leg["sampler"]].task == task
        for dataset, ordered in group["ranking"].items():
            assert [e["position"] for e in ordered] == list(range(1, len(ordered) + 1))
            for entry in ordered:
                leg = ids[entry["id"]]
                # only closed, fully judged legs of this dataset are ordered
                assert leg["dataset"] == dataset and leg["state"] == "closed"
                assert leg["attempted"] == leg["expected"]
                assert leg["success_rate"] is not None


def test_wave2_censoring_rules():
    from searches import _wave2 as wave2

    cell = next(
        c
        for c in wave2.cells()
        if c.sampler == "lbfgs" and c.backend == "numpy" and c.dataset == "gaussian_x3_blend"
    )
    over = {"total_wall_s": cell.timeout_s + 50.0, "status": "complete", "completed": True}
    assert catalogue.censored_attempt(over, cell) == (cell.timeout_s, True)
    under = {"total_wall_s": 10.0, "status": "complete", "completed": True}
    assert catalogue.censored_attempt(under, cell) == (10.0, False)
    # A closed leg with no rows at all: every expected run is a failed attempt at the timeout.
    leg = catalogue.wave2_leg(cell, {}, {}, None)
    assert leg["attempted"] == 250 and leg["missing_censored"] == 250
    assert leg["success_rate"] == 0.0 and leg["wall_per_success"]["unbounded"] is True


def test_wave2_leg_excludes_operational_kills_and_censors_warm_starts():
    from searches import _wave2 as wave2

    cell = next(
        c
        for c in wave2.cells()
        if c.sampler == "lbfgs" and c.backend == "numpy" and c.dataset == "gaussian_x3_blend"
    )
    kill = {
        "status": "failed: Terminated: signal SIGTERM",
        "completed": False,
        "total_wall_s": 5.0,
    }
    leg = catalogue.wave2_leg(cell, {(1, 0): kill}, {}, None)
    # the killed run is not an attempt; in a closed leg it is censored as a missing run
    assert leg["operational_kills"] == 1 and leg["missing_censored"] == 250
    warm = {
        "warm_start": {"provider": "nautilus"},
        "provider_wall_s": 40.0,
        "total_wall_s": cell.timeout_s + 1.0,
    }
    assert catalogue.censored_attempt(warm, cell) == (cell.timeout_s + 40.0, True)


def test_probe_sourced_leg_closes_only_when_every_run_has_a_row():
    from searches import _wave2 as wave2

    cell = next(c for c in wave2.cells() if c.timeout_source == "probe")
    row = {"status": "complete", "completed": True, "total_wall_s": 1.0}
    some = {(1, s): row for s in wave2.PROBE_SEEDS}
    assert catalogue.wave2_leg(cell, some, {}, None)["state"] == "open"
    every = {(d, s): row for d in wave2.DATA_SEEDS for s in wave2.SEEDS}
    assert catalogue.wave2_leg(cell, every, {}, None)["state"] == "closed"

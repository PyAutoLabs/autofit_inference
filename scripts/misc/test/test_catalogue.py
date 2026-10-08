"""The search catalogue: every registered search exactly once, one status, no ranking."""

import json
import re

import build_catalogue as catalogue
import build_readme
from searches import _pilot as pilot
from searches._runner import SAMPLERS

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
    doc = catalogue.build(catalogue.ROOT, manifest)
    seen = [cls for group in doc["tasks"].values() for cls in group["searches"]]
    assert sorted(seen) == sorted(s["name"] for s in manifest["searches"])
    assert len(seen) == len(set(seen))
    for group in doc["tasks"].values():
        for cls, entry in group["searches"].items():
            assert entry["status"] in catalogue.STATUSES, cls
            assert entry["reason"], cls


def test_tasks_are_separate_groups_and_nothing_is_ranked():
    doc = catalogue.build(catalogue.ROOT, catalogue.load_manifest())
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
    doc = catalogue.build(catalogue.ROOT, catalogue.load_manifest())
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
    assert committed == json.loads(catalogue.render(catalogue.build()))


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

"""Exporter fixture, Insight validation, README gate and the WALL-BASIS gate."""

import json
import os
import sys
from pathlib import Path as _Path

import build_readme
import export_inference_summary as exporter
import pytest

ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "ruff.toml").exists())
from wall import check_submits

FIXTURES = ROOT / "scripts" / "misc" / "test" / "fixtures"
FIXTURE_REVISION = "0" * 40
FIXTURE_TIME = "2026-10-08T10:00:00+00:00"


def fixture_doc():
    return exporter.build(FIXTURES / "root", FIXTURE_REVISION, FIXTURE_TIME)


def _export_files(base, index_name):
    """``{relative path: committed text}`` of an ``@3`` export rooted at ``base``."""
    out = {"": (base / index_name).read_text()}
    for path in sorted((base / exporter.RECORDS_DIR).rglob("*.json")):
        out[path.relative_to(base).as_posix()] = path.read_text()
    return out


def test_fixture_summary_is_current():
    committed = _export_files(FIXTURES, "summary_fixture.json")
    assert exporter.files(fixture_doc()) == committed, (
        "regenerate the fixture: exporter.build(FIXTURES / 'root', ...) -> summary_fixture.json "
        "+ records/"
    )


def test_index_carries_shards_not_records_and_every_shard_is_small():
    doc = fixture_doc()
    index, shards = exporter.split(doc)
    assert "records" not in index and index["version"] == 3
    assert sum(e["records"] for e in index["record_shards"]) == len(doc["records"])
    assert {e["path"] for e in index["record_shards"]} == set(shards)
    for entry in index["record_shards"]:
        shard = json.loads(shards[entry["path"]])
        assert shard["shard"] == entry["id"] and shard["protocol_id"] == entry["protocol_id"]
        assert all(r["scientific"]["protocol_id"] == entry["protocol_id"] for r in shard["records"])
    # the committed export: no file approaches GitHub's 50 MB warning
    base = ROOT / "dashboard"
    for path in [base / "summary.json", *(base / exporter.RECORDS_DIR).rglob("*.json")]:
        assert path.stat().st_size < 10 * 1024 * 1024, path


def test_fixture_verdicts_are_producer_asserted():
    records = {r["seed"]: r["scientific"] for r in fixture_doc()["records"]}
    assert records[0]["acceptance"] == "accepted"
    assert records[1]["acceptance"] == "rejected" and "(b)1" in records[1]["reason"]
    for sci in records.values():
        assert sci["protocol_id"] == "gaussian_x3@1" and sci["asserted_by"] == "producer"


@pytest.mark.parametrize(
    "field, value", [("data_seed", 777), ("assertion_mechanism", "no_assertions")]
)
def test_exporter_never_judges_a_row_against_another_identity(tmp_path, field, value):
    source = (
        FIXTURES
        / "root/results/searches/gaussian_x3_blend/data_seed1/nautilus/n_live_100"
        / "local_numpy_fp64/search_seed0.json"
    )
    row = json.loads(source.read_text())
    row[field] = value
    target = tmp_path / "results" / "searches" / "row.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(row))
    doc = exporter.build(FIXTURES / "root", FIXTURE_REVISION, FIXTURE_TIME, rows_root=tmp_path)
    (record,) = doc["records"]
    assert record["scientific"]["acceptance"] == "not_assessed"
    assert "identity mismatch" in record["scientific"]["reason"]


def test_fixture_validates_against_pyautoinsight():
    candidates = [ROOT.parent / "PyAutoInsight", ROOT.parents[1] / "organs" / "PyAutoInsight"]
    if os.environ.get("PYAUTO_INSIGHT"):
        candidates.insert(0, _Path(os.environ["PYAUTO_INSIGHT"]))
    for candidate in candidates:
        if (candidate / "insight" / "summary.py").exists():
            insight_root = candidate
            break
    else:
        pytest.skip("PyAutoInsight not checked out beside this repo")
    sys.path.insert(0, str(insight_root))
    from insight import summary

    texts = _export_files(FIXTURES, "summary_fixture.json")
    index = json.loads(texts[""])
    assert index["project"] == "autofit_inference" and index["version"] == 3
    if hasattr(summary, "load_sharded"):
        # A reader with inference-summary@3: the shards assembled and validated natively.
        doc, errors = summary.load_sharded(index, lambda path: texts[path].encode())
        assert errors == []
        assert summary.classify(doc, ("inference-summary", 3)) == ("ok", [])
    # Every @3 record is an @1 record: the assembled list validates under @1 too (and
    # this is all an @1-only reader, before the @3 reader is deployed, can check).
    v1 = {k: v for k, v in index.items() if k != "record_shards"}
    v1["version"] = 1
    v1["records"] = [r for rel, text in texts.items() if rel for r in json.loads(text)["records"]]
    assert summary.classify(v1, ("inference-summary", 1)) == ("ok", [])


def test_committed_summary_is_current():
    path = ROOT / "dashboard" / "summary.json"
    revision, date = exporter.git_revision(ROOT)
    if revision is None:
        pytest.skip("not a git checkout")
    assert _export_files(path.parent, path.name) == exporter.files(
        exporter.build(ROOT, revision, date)
    )


def test_readme_tables_are_current():
    for target in build_readme.TARGET_READMES:
        original, rewritten, unknown = build_readme._rewrite_file(target, build_readme.RENDERERS)
        assert not unknown
        assert original == rewritten, "run scripts/misc/tooling/build_readme.py"


def test_batch_template_passes_the_wall_gate_and_stays_on_ral():
    template = (ROOT / "hpc" / "batch_cpu" / "template").read_text()
    assert check_submits.check_text(template, "template") == []
    assert "#SBATCH --partition=ral\n" in template
    assert not (ROOT / "hpc" / "batch_gpu").exists()


def test_warm_start_export_carries_provider_and_combined_total():
    """Review 06 finding 4: a warm-start row exports its provider provenance and cost,
    and total_s is the combined attempt cost, its definition saying so."""
    rel = (
        "results/searches/gaussian_x3_separated/data_seed2/blackjax_nuts_warm/"
        "warmup_200_samples_500/local_jax_cpu_fp64_cache_warm/search_seed0.json"
    )
    row = json.loads((ROOT / rel).read_text())
    refs, offsets = exporter.references(ROOT)
    rec = exporter.record(row, rel, refs, offsets)
    assert rec["warm_start"]["provider"] == "nautilus"
    assert rec["provider_wall_s"] == pytest.approx(1076.4696580, abs=1e-6)
    assert rec["timings"]["total_s"] == pytest.approx(1150.1580484, abs=1e-6)
    assert rec["consumer_total_s"] == pytest.approx(73.6883904, abs=1e-6)
    assert "provider" in rec["timings"]["definitions"]["total_s"]
    cold = json.loads(
        (
            FIXTURES
            / "root/results/searches/gaussian_x3_blend/data_seed1/nautilus/n_live_100"
            / "local_numpy_fp64/search_seed0.json"
        ).read_text()
    )
    cold_rec = exporter.record(cold, "results/x.json", refs, offsets)
    assert cold_rec["warm_start"] is None and cold_rec["provider_wall_s"] is None
    assert cold_rec["timings"]["total_s"] == cold["total_wall_s"]

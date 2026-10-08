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


def test_fixture_summary_is_current():
    committed = (FIXTURES / "summary_fixture.json").read_text()
    assert exporter.render(fixture_doc()) == committed, "regenerate summary_fixture.json"


def test_fixture_verdicts_are_producer_asserted():
    records = {r["seed"]: r["scientific"] for r in fixture_doc()["records"]}
    assert records[0]["acceptance"] == "accepted"
    assert records[1]["acceptance"] == "rejected" and "(b)1" in records[1]["reason"]
    for sci in records.values():
        assert sci["protocol_id"] == "gaussian_x3@1" and sci["asserted_by"] == "producer"


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

    doc = json.loads((FIXTURES / "summary_fixture.json").read_text())
    assert summary.classify(doc, ("inference-summary", 1)) == ("ok", [])
    assert doc["project"] == "autofit_inference"


def test_committed_summary_is_current():
    path = ROOT / "dashboard" / "summary.json"
    revision, date = exporter.git_revision(ROOT)
    if revision is None:
        pytest.skip("not a git checkout")
    assert path.read_text() == exporter.render(exporter.build(ROOT, revision, date))


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

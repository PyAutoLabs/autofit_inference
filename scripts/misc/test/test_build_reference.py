"""Reference builder failure paths (protocol §6): it never certifies a reference that
the agreement rule or the raw samples do not support."""

import json
import math
import sys
from pathlib import Path as _Path

import numpy as np

ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts" / "misc" / "reference"))
import build_reference  # noqa: E402
from searches import _posterior as post  # noqa: E402
from searches import _protocol as protocol  # noqa: E402

KEYS = [
    "g0.centre",
    "g0.normalization",
    "g0.sigma",
    "g1.centre",
    "g1.normalization",
    "g1.sigma",
    "g2.centre",
    "g2.normalization",
    "g2.sigma",
    "background.level",
]
TRUTH = [25.0, 30.0, 3.0, 45.0, 50.0, 6.0, 60.0, 40.0, 10.0, 0.02]
DATASET, BACKEND = "gaussian_x3_blend", "numpy"


def write_run(root, sampler, seed, log_evidence, write_samples=True):
    rng = np.random.default_rng(100 * len(sampler) + seed)
    matrix = np.asarray(TRUTH) + rng.normal(0.0, 0.1, size=(2000, len(KEYS)))
    weights = np.ones(len(matrix))
    output_path = f"reference/{DATASET}/local_numpy_fp64/{sampler}_seed{seed}/hash"
    if write_samples:
        files = root / "output" / output_path / "files"
        files.mkdir(parents=True)
        lines = [",".join(KEYS + ["weight"])]
        lines += [",".join(repr(float(v)) for v in [*row, w]) for row, w in zip(matrix, weights)]
        (files / "samples.csv").write_text("\n".join(lines) + "\n")
    summary = post.summarise_samples(matrix, weights, KEYS, np.zeros(2000), np.zeros(2000))
    run = {
        "kind": "reference_run",
        "dataset": DATASET,
        "data_seed": 1,
        "backend": BACKEND,
        "sampler": sampler,
        "seed": seed,
        "assertion_mechanism": "raise_resample",
        "log_evidence": log_evidence,
        "ppc_chi2": 60.0,
        "truths": dict(zip(KEYS, TRUTH)),
        "output_path": output_path,
        **summary,
    }
    directory = root / "results" / "reference" / DATASET / BACKEND
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{sampler}_seed{seed}.json").write_text(json.dumps(run))


def make_root(tmp_path, nautilus_logz, dynesty_logz, missing=()):
    offsets = {
        "offsets": {
            "nautilus": {BACKEND: {"offset": -math.log(6), "validated": True}},
            "dynesty_static": {BACKEND: {"offset": 0.0, "validated": True}},
        }
    }
    (tmp_path / "results" / "reference").mkdir(parents=True)
    (tmp_path / "results" / "reference" / "constant_likelihood.json").write_text(
        json.dumps(offsets)
    )
    for sampler, values in (("nautilus", nautilus_logz), ("dynesty_static", dynesty_logz)):
        for seed, value in enumerate(values):
            write_run(tmp_path, sampler, seed, value, (sampler, seed) not in missing)
    return tmp_path


def test_agreeing_runs_give_a_complete_reference(tmp_path):
    n = 100.0 - math.log(6)
    root = make_root(tmp_path, [n, n + 0.05, n + 0.1], [100.0, 100.02, 100.04])
    out = build_reference.build(DATASET, BACKEND, root)
    assert out["status"] == "complete", out["limitations"]
    assert len(out["included_runs"]) == 6
    assert abs(out["modes"][0]["weight"] - 1.0) < 1e-9


def test_no_agreeing_family_is_disagreeing_and_judges_nothing(tmp_path):
    root = make_root(tmp_path, [100.0, 200.0, 300.0], [100.0, 150.0, 250.0])
    out = build_reference.build(DATASET, BACKEND, root)
    assert out["status"] == "disagreeing"
    assert not out["family_agreement"]["nautilus"]["logz_ok"]
    assert not out["family_agreement"]["dynesty_static"]["logz_ok"]
    assert "posterior_relabelled" not in out and "log_evidence_normalised" not in out
    row = {"task": "evidence", "dataset": DATASET, "backend": BACKEND}
    verdict = protocol.acceptance(row, out)
    assert verdict["acceptance"] == "not_assessed" and "disagreeing" in verdict["reason"]


def test_one_agreeing_family_is_complete_with_the_limitation(tmp_path):
    n = 100.0 - math.log(6)
    root = make_root(tmp_path, [n, n + 0.01, n + 0.02], [100.0, 102.0, 104.0])
    out = build_reference.build(DATASET, BACKEND, root)
    assert out["status"] == "complete"
    assert out["included_runs"] == ["nautilus_seed0", "nautilus_seed1", "nautilus_seed2"]
    assert any("nautilus runs only" in line for line in out["limitations"])


def test_missing_raw_samples_is_never_complete(tmp_path):
    n = 100.0 - math.log(6)
    root = make_root(tmp_path, [n, n, n], [100.0, 100.0, 100.0], missing={("dynesty_static", 1)})
    out = build_reference.build(DATASET, BACKEND, root)
    assert out["status"] == "samples_missing"
    assert out["missing_samples"] == ["dynesty_static_seed1"]
    assert "posterior_relabelled" not in out
    row = {"task": "evidence", "dataset": DATASET, "backend": BACKEND}
    assert protocol.acceptance(row, out)["acceptance"] == "not_assessed"


def test_disjoint_prior_references_take_no_ln_3_factorial_offset(tmp_path):
    """The separated control excludes no prior volume (amendment A1): its reference must
    be normalised with offset 0 for every sampler, exactly as its rows are, else every
    evidence row would be judged against a reference shifted by ln 3!."""
    root = make_root(tmp_path, [100.0, 100.01, 100.02], [100.0, 100.05, 100.1])
    directory = root / "results" / "reference" / DATASET / BACKEND
    for path in directory.glob("*_seed*.json"):
        run = json.loads(path.read_text())
        run["assertion_mechanism"] = "disjoint_priors"
        path.write_text(json.dumps(run))
    out = build_reference.build(DATASET, BACKEND, root)
    assert out["status"] == "complete", out["limitations"]
    assert abs(out["log_evidence_normalised"] - out["log_evidence"]) < 1e-12
    row = {
        "sampler": "dynesty_static",
        "backend": BACKEND,
        "assertion_mechanism": "disjoint_priors",
        "log_evidence": out["log_evidence"],
    }
    ok, _ = protocol.criterion_evidence(row, out, None)
    assert ok is True

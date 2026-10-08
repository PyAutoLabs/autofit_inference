"""Runner pure helpers, the sampler registry and the leaves' literal declarations."""

import ast
from pathlib import Path
from pathlib import Path as _Path

ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "ruff.toml").exists())
from searches import _runner as runner


def test_run_id_and_results_path_share_the_identity():
    rid = runner.run_id(
        "gaussian_x3", "gaussian_x3_blend", 1, "nautilus", "n_live_100", "local_numpy_fp64", 3
    )
    assert (
        rid == "gaussian_x3/gaussian_x3_blend/data_seed1/nautilus/n_live_100/local_numpy_fp64/seed3"
    )
    path = runner.results_path(
        Path("/r"), "gaussian_x3_blend", 1, "nautilus", "n_live_100", "local_numpy_fp64", 3
    )
    assert path.as_posix().endswith(
        "results/searches/gaussian_x3_blend/data_seed1/nautilus/n_live_100/local_numpy_fp64/search_seed3.json"
    )
    prefix = runner.output_path_prefix(
        "gaussian_x3_blend", 1, "nautilus", "n_live_100", "local_numpy_fp64", 3
    )
    assert "local_numpy_fp64" in prefix.parts and "seed_3" in prefix.parts


def test_truth_delta_sigma_and_likelihood_share():
    post = {"a": {"median": 2.0, "sigma": 0.5}, "b": {"median": 1.0, "sigma": 0.0}}
    assert runner.truth_delta_sigma_from(post, {"a": 1.0, "b": 1.0, "c": 3.0}) == {"a": 2.0}
    assert runner.likelihood_share(1e-3, 1000, 10.0) == 0.1
    assert runner.likelihood_share(None, 1000, 10.0) is None
    assert runner.likelihood_share(1e-3, 1000, 0.0) is None


def test_registry_covers_every_task_and_default_settings_exist():
    tasks = {spec.task for spec in runner.SAMPLERS.values()}
    assert tasks == {"point_map", "posterior", "evidence"}
    for name, spec in runner.SAMPLERS.items():
        assert spec.default in spec.settings, name
    assert runner.SAMPLERS["nss"].status == "deferred"
    assert runner.SAMPLERS["nautilus"].settings["n_live_100"] == {"n_live": 100}


def _declared(leaf: Path):
    for node in ast.walk(ast.parse(leaf.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run_search":
            return {
                k.arg: k.value.value
                for k in node.keywords
                if isinstance(k.value, ast.Constant) and isinstance(k.value.value, str)
            }
    return None


def test_every_leaf_declares_its_cell_with_literal_kwargs():
    leaves = sorted((ROOT / "scripts").glob("*/searches/*/*.py"))
    assert leaves, "no leaves found"
    for leaf in leaves:
        declared = _declared(leaf)
        assert declared is not None, leaf
        assert {"sampler", "dataset_class", "model_type"} <= declared.keys(), leaf
        dataset, _, sampler, model = leaf.relative_to(ROOT / "scripts").with_suffix("").parts
        assert declared == {"sampler": sampler, "dataset_class": dataset, "model_type": model}
        assert sampler in runner.SAMPLERS

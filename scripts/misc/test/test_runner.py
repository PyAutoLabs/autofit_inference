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


def test_nested_termination_reads_nautilus_state_and_flags_budget_stops():
    from types import SimpleNamespace

    search = SimpleNamespace(n_eff=500, n_shell=1, f_live=0.01, n_like_max=float("inf"))

    def sampler(**kw):
        state = {"n_eff": 900.0, "explored": True, "shell_n": [3, 4], "n_like": 10, "f_live": 0.005}
        state.update(kw)
        return SimpleNamespace(**state)

    ok = runner.nested_termination("nautilus", search, sampler())
    assert ok["observable"] and ok["met"] is True and ok["n_like_max"] is None
    budget = runner.nested_termination("nautilus", search, sampler(n_eff=100.0))
    assert budget["met"] is False
    unexplored = runner.nested_termination("nautilus", search, sampler(explored=False))
    assert unexplored["met"] is False
    for name, internal in (("nautilus", None), ("dynesty_static", sampler())):
        out = runner.nested_termination(name, search, internal)
        assert out["met"] is None and out["reason"] == runner.TERMINATION_NOT_EXPOSED


def _blend_fitness(model, dataset="gaussian_x3_blend"):
    from autofit.non_linear.fitness import Fitness
    from models import gaussian_x3 as gx3

    data, noise_map, truth = gx3.load_dataset(ROOT / "dataset" / dataset)
    fitness = Fitness(
        model=model,
        analysis=gx3.AnalysisGaussianX3(data=data, noise_map=noise_map),
        paths=None,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=runner.RESAMPLE_SENTINEL,
    )
    return fitness, gx3.truth_dict(truth)


def test_admission_vector_reaches_the_likelihood_and_medians_do_not():
    """Review finding 7: the timed vector must reach the likelihood, not the sentinel."""
    import pytest

    pytest.importorskip("autofit")
    import numpy as np
    from models import gaussian_x3 as gx3

    model = gx3.build_model()
    fitness, truths = _blend_fitness(model)
    vector = runner.admission_vector(gx3.parameter_keys(model), truths)
    assert fitness.call(np.asarray(vector)) > runner.RESAMPLE_SENTINEL / 10
    medians = np.asarray(model.physical_values_from_prior_medians, dtype=float)
    assert fitness.call(medians) == runner.RESAMPLE_SENTINEL

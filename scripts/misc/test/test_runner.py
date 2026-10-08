"""Runner pure helpers, the sampler registry and the leaves' literal declarations."""

import ast
import json
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


def test_cold_and_warm_runs_never_share_an_identity():
    """Review finding 8: the cold and warm JAX runs are separate rows and outputs."""
    cold = runner.config_segment("local_jax_cpu_fp64", "cold")
    warm = runner.config_segment("local_jax_cpu_fp64", "warm")
    assert cold != warm and runner.config_segment("local_numpy_fp64") == "local_numpy_fp64"
    args = ("gaussian_x3_blend", 1, "nautilus", "n_live_100")
    assert runner.output_path_prefix(*args, cold, 0) != runner.output_path_prefix(*args, warm, 0)
    assert runner.results_path(Path("/r"), *args, cold, 0) != runner.results_path(
        Path("/r"), *args, warm, 0
    )
    assert runner.run_id("gaussian_x3", *args, cold, 0) != runner.run_id(
        "gaussian_x3", *args, warm, 0
    )


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
    # A real Sampler reports f_live None after exploration (found on the local witness).
    after = runner.nested_termination("nautilus", search, sampler(f_live=None))
    assert after["observable"] and after["met"] is True and after["f_live"] is None
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

    for dataset in ("gaussian_x3_blend", "gaussian_x3_separated"):
        separated = gx3.build_model(dataset=dataset)
        fitness, truths = _blend_fitness(separated, dataset)
        vector = runner.admission_vector(gx3.parameter_keys(separated), truths)
        assert fitness.call(np.asarray(vector)) > runner.RESAMPLE_SENTINEL / 10
    model = gx3.build_model()
    fitness, _ = _blend_fitness(model)
    medians = np.asarray(model.physical_values_from_prior_medians, dtype=float)
    assert fitness.call(medians) == runner.RESAMPLE_SENTINEL


def test_a_failed_attempt_keeps_its_elapsed_wall(tmp_path, monkeypatch):
    """Review finding 9: a crash inside search.fit still writes a row with its cost."""
    import json
    import time

    import pytest

    pytest.importorskip("autofit")

    class Boom(RuntimeError):
        pass

    class FailingSearch:
        paths = None

        def fit(self, model, analysis):
            time.sleep(0.2)
            raise Boom("simulated crash")

    monkeypatch.setattr(runner, "build_search", lambda *a, **k: FailingSearch())
    monkeypatch.setenv("PYAUTO_TEST_MODE", "0")
    argv = [
        "--config-name",
        "local_numpy_fp64",
        "--seed",
        "0",
        "--output-dir",
        str(tmp_path / "output"),
        "--results-root",
        str(tmp_path / "rows"),
    ]
    with pytest.raises(Boom):
        runner.run_search("nautilus", "gaussian_x3_blend", "gaussian_x3", argv=argv)
    (path,) = (tmp_path / "rows" / "results" / "searches").rglob("*.json")
    row = json.loads(path.read_text())
    assert row["status"].startswith("failed: Boom")
    assert row["total_wall_s"] is not None and row["total_wall_s"] >= 0.2
    assert row["scientific"]["acceptance"] != "accepted"


def test_sigterm_writes_a_failure_row_with_its_wall(tmp_path, monkeypatch):
    """The SLURM time limit sends SIGTERM first: the row and its cost survive."""
    import json
    import os
    import signal
    import time

    import pytest

    pytest.importorskip("autofit")

    class TimedOut:
        paths = None

        def fit(self, model, analysis):
            time.sleep(0.1)
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)

    monkeypatch.setattr(runner, "build_search", lambda *a, **k: TimedOut())
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")
    argv = ["--seed", "0", "--output-dir", str(tmp_path / "o"), "--results-root", str(tmp_path)]
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(runner.Terminated):
        runner.run_search("nautilus", "gaussian_x3_blend", "gaussian_x3", argv=argv)
    assert signal.getsignal(signal.SIGTERM) is before
    (path,) = (tmp_path / "results" / "searches").rglob("*.json")
    row = json.loads(path.read_text())
    assert row["status"].startswith("failed: Terminated") and 0.1 <= row["total_wall_s"] < 5


def test_separated_control_is_disjoint_priors_without_assertions():
    """Review finding 10 / D15 option (b): the control is identified by its priors."""
    import pytest

    pytest.importorskip("autofit")
    from models import gaussian_x3 as gx3

    blend = gx3.build_model(dataset="gaussian_x3_blend")
    separated = gx3.build_model(dataset="gaussian_x3_separated")
    assert len(blend.assertions) == 2 and len(separated.assertions) == 0
    edges = [
        (getattr(separated, n).centre.lower_limit, getattr(separated, n).centre.upper_limit)
        for n in gx3.COMPONENTS
    ]
    assert edges == list(gx3.DISJOINT_CENTRE_PRIORS)
    assert all(a[1] <= b[0] for a, b in zip(edges, edges[1:]))
    _, _, truth = gx3.load_dataset(ROOT / "dataset" / "gaussian_x3_separated")
    for name, (lo, hi) in zip(gx3.COMPONENTS, edges):
        assert lo < truth["parameters"][f"{name}.centre"] < hi
    assert gx3.assertion_mechanism("gaussian_x3_separated", use_jax=True) == "disjoint_priors"
    assert gx3.assertion_mechanism("gaussian_x3_blend", use_jax=True) == "xp_where_penalty"
    assert gx3.priors_record("gaussian_x3_separated")["g2.centre"] == "U(65, 100)"


def test_a_non_finite_answer_is_a_failed_attempt_never_a_dropped_row():
    from searches._runner import sanitise_nonfinite

    row = {
        "status": "complete",
        "max_log_likelihood": float("inf"),
        "max_log_posterior": float("inf"),
        "posterior": {"g0.centre": {"sigma": float("nan")}},
    }
    out = sanitise_nonfinite(row)
    assert out["status"].startswith("failed: non-finite result")
    assert out["max_log_likelihood"] is None and out["posterior"]["g0.centre"]["sigma"] is None
    assert set(out["nonfinite_fields"]) == {
        "max_log_likelihood",
        "max_log_posterior",
        "posterior.g0.centre.sigma",
    }
    json.dumps(out, allow_nan=False)
    clean = {"status": "complete", "max_log_likelihood": 1.0}
    assert sanitise_nonfinite(clean) is clean


def test_warm_nuts_names_its_provider():
    from searches._runner import SAMPLERS

    spec = SAMPLERS["blackjax_nuts_warm"]
    provider, settings = spec.extra["warm_from"]
    assert settings in SAMPLERS[provider].settings
    assert spec.cls == SAMPLERS["blackjax_nuts"].cls and spec.task == "posterior"

"""The ``correlated_nd`` family (B5c): generator, exact references, manifest and verdicts.

Exactness is tested against independent routes — brute-force grid integration on small
synthetic datasets of the same two kinds, exact iid draws, and quadrature refinement —
never against the module's own formula alone. Seed determinism is tested by regenerating
every committed dataset from its seeds.
"""

import json
import math

import numpy as np
import pytest
from conftest import ROOT
from models import correlated_nd_exact as exact
from searches import _correlated_nd as cnd
from searches import _posterior as post
from searches import _protocol as protocol
from simulators import correlated_nd as sim

DATASETS = sorted(sim.DATASETS)


def _committed(name):
    return exact.load(ROOT / "dataset" / name)


# --- generator: seeds, determinism, structure ----------------------------------------


def _float_array(value):
    """``value`` as a float ndarray if it is a (nested) list holding a float, else None."""
    if not isinstance(value, list):
        return None
    try:
        array = np.asarray(value)
    except ValueError:  # ragged
        return None
    return array if array.dtype == np.float64 else None


def _assert_same_json(regenerated, committed, where):
    """Floats agree to rtol 1e-12; every other value (keys, strings, ints, seeds) is exact.

    QR/LAPACK builds differ in the last bit or two across machines, so float arrays are
    compared numerically and normwise (atol = 1e-12 x the array's largest magnitude: an
    orthogonal matrix's near-zero entries carry the matrix's absolute error, not their
    own); the structure and metadata must still be identical.
    """
    array = _float_array(committed)
    if array is not None:
        other = _float_array(regenerated)
        assert other is not None and other.shape == array.shape, where
        scale = float(np.max(np.abs(array))) if array.size else 0.0
        assert np.allclose(other, array, rtol=1e-12, atol=1e-12 * scale), where
    elif isinstance(committed, float) or isinstance(regenerated, float):
        assert isinstance(committed, (int, float)) and isinstance(regenerated, (int, float))
        assert not isinstance(committed, bool) and not isinstance(regenerated, bool), where
        assert np.allclose(regenerated, committed, rtol=1e-12, atol=0), where
    elif isinstance(committed, dict):
        assert isinstance(regenerated, dict) and list(regenerated) == list(committed), where
        for key in committed:
            _assert_same_json(regenerated[key], committed[key], f"{where}.{key}")
    elif isinstance(committed, list):
        assert isinstance(regenerated, list) and len(regenerated) == len(committed), where
        for i, (r, c) in enumerate(zip(regenerated, committed)):
            _assert_same_json(r, c, f"{where}[{i}]")
    else:
        assert type(regenerated) is type(committed) and regenerated == committed, where


@pytest.mark.parametrize("name", DATASETS)
def test_committed_dataset_regenerates_byte_for_byte(name, tmp_path):
    """Byte-identical where the platform allows; floats to rtol 1e-12 where it does not."""
    sim.simulate(name, output_root=tmp_path)
    for filename in ("data.json", "model.json", "priors.json", "truth.json"):
        committed = (ROOT / "dataset" / name / filename).read_text()
        regenerated = (tmp_path / "dataset" / name / filename).read_text()
        if regenerated != committed:
            _assert_same_json(json.loads(regenerated), json.loads(committed), filename)


def test_data_seed_changes_only_the_data():
    struct = sim.structure("corr_gaussian_d20")
    y1 = sim.simulate_data("corr_gaussian_d20", 1, struct)
    y2 = sim.simulate_data("corr_gaussian_d20", 2, struct)
    assert not np.allclose(y1, y2)
    assert np.array_equal(sim.simulate_data("corr_gaussian_d20", 1), y1)
    again = sim.structure("corr_gaussian_d20")
    assert np.array_equal(again["eigenvectors"], struct["eigenvectors"])
    assert np.array_equal(again["lower"], struct["lower"])


def test_datasets_are_independent_draws():
    a = sim.simulate_data("corr_gaussian_d20", 1)
    b = sim.simulate_data("rosenbrock_d20", 1)
    assert not np.allclose(a[1:] - np.mean(a[1:]), b[1:] - np.mean(b[1:]))


@pytest.mark.parametrize("name", [n for n in DATASETS if sim.DATASETS[n][0] == "gaussian"])
def test_gaussian_structure_is_orthogonal_with_condition_number_1e3(name):
    ds = _committed(name)
    q = ds.eigenvectors
    assert np.allclose(q.T @ q, np.eye(ds.dimension), atol=1e-12)
    assert ds.eigenvalues.max() / ds.eigenvalues.min() == pytest.approx(1e3, rel=1e-12)
    model = json.loads((ROOT / "dataset" / name / "model.json").read_text())
    sigma = q @ np.diag(ds.eigenvalues) @ q.T
    assert np.allclose(np.sqrt(np.diag(sigma)), model["marginal_sigma"], rtol=1e-12)


def test_refused_realisation(monkeypatch):
    monkeypatch.setattr(sim, "MAX_DEVIATION_SIGMA", 0.1)
    with pytest.raises(sim.RefusedRealisation):
        sim.simulate_data("corr_gaussian_d20", 1)


@pytest.mark.parametrize("name", DATASETS)
def test_data_inside_the_bounded_deviation(name):
    ds = _committed(name)
    u = exact.whiten(ds, ds.truth[None, :])[0]  # the truth in the data's whitened frame
    assert np.max(np.abs(u)) <= sim.MAX_DEVIATION_SIGMA


# --- exactness against independent routes --------------------------------------------


def _toy_gaussian():
    rng = np.random.default_rng(5)
    q = sim.haar_orthogonal(rng, 2)
    lam = np.array([0.3, 3.0])
    y = np.array([0.4, -0.7])
    sigma = np.sqrt((q**2) @ lam)
    return exact.Dataset(
        name="toy",
        kind="gaussian",
        dimension=2,
        data_seed=1,
        keys=("x000", "x001"),
        y=y,
        lower=y - 4.0 * sigma,  # deliberately tight: truncation is visible but bounded
        upper=y + 5.0 * sigma,
        truth=y,
        eigenvalues=lam,
        eigenvectors=q,
    )


def _toy_banana():
    return exact.Dataset(
        name="toy",
        kind="rosenbrock",
        dimension=2,
        data_seed=1,
        keys=("x000", "x001"),
        y=np.array([0.8, 0.1]),
        lower=np.array([-8.0, -2.0]),
        upper=np.array([10.0, 110.0]),
        truth=np.array([1.0, 1.0]),
        sigma1=math.sqrt(10.0),
        sigma2=math.sqrt(0.1),
    )


def _grid_log_evidence(ds, n=3000):
    """Brute-force: Z = (1/V) * integral of L over the box on a fine 2-D grid."""
    xs = [np.linspace(lo, hi, n) for lo, hi in zip(ds.lower, ds.upper)]
    g0, g1 = np.meshgrid(*xs, indexing="ij")
    log_l = exact.log_likelihood(ds, np.column_stack([g0.ravel(), g1.ravel()])).reshape(g0.shape)
    integral = np.trapezoid(np.trapezoid(np.exp(log_l), xs[1], axis=1), xs[0])
    return math.log(integral) - ds.log_volume, xs, np.exp(log_l)


def test_gaussian_ln_z_matches_brute_force_within_its_truncation_bound():
    ds = _toy_gaussian()
    log_z, how = exact.log_evidence(ds)
    grid, _, _ = _grid_log_evidence(ds)
    # The toy box truncates measurably; the exact ln Z is -ln V and the bound covers the gap.
    assert how["truncation_bound"] > 1e-6
    assert abs(log_z - grid) <= how["log_evidence_error_bound"] + 1e-6


def test_banana_ln_z_and_marginals_match_brute_force():
    pytest.importorskip("scipy")
    ds = _toy_banana()
    log_z, how = exact.log_evidence(ds)
    grid, xs, density = _grid_log_evidence(ds, n=4000)
    assert log_z == pytest.approx(grid, abs=2e-6)
    assert how["box_mass"] < 1.0  # truncated on purpose: the integral, not a bound, is used
    marginal = exact.marginals(ds)
    for axis, key in enumerate(ds.keys):
        pdf = np.trapezoid(density, xs[1 - axis], axis=1 - axis)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(xs[axis]))])
        cdf /= cdf[-1]
        median = np.interp(0.5, cdf, xs[axis])
        q16, q84 = np.interp([0.16, 0.84], cdf, xs[axis])
        assert marginal[key]["median"] == pytest.approx(median, abs=0.02)
        assert marginal[key]["sigma"] == pytest.approx((q84 - q16) / 2.0, rel=5e-3)


def test_banana_quadrature_is_converged():
    pytest.importorskip("scipy")
    ds = _committed("rosenbrock_d20")
    base = exact.banana_mass(ds)
    fine = exact.X1_PANELS
    try:
        exact.X1_PANELS = 2 * fine
        assert exact.banana_mass(ds) == pytest.approx(base, abs=1e-13)
    finally:
        exact.X1_PANELS = fine


@pytest.mark.parametrize("name", DATASETS)
def test_exact_draws_reproduce_the_exact_marginals_and_whiten_to_unit_normal(name):
    pytest.importorskip("scipy")
    ds = _committed(name)
    draws = exact.sample(ds, 40_000, seed=3)
    assert np.all((draws >= ds.lower) & (draws <= ds.upper))
    stats = post.posterior_stats(draws, np.ones(len(draws)), list(ds.keys))
    ref = exact.marginals(ds)
    for key in ds.keys:
        assert abs(stats[key]["median"] - ref[key]["median"]) / ref[key]["sigma"] < 0.05
        assert stats[key]["sigma"] / ref[key]["sigma"] == pytest.approx(1.0, abs=0.05)
    white = exact.whitened_stats(ds, draws, np.ones(len(draws)))
    assert white["worst_median"] < 0.05
    assert 0.95 < white["sigma_ratio_min"] <= white["sigma_ratio_max"] < 1.05


@pytest.mark.parametrize("name", DATASETS)
def test_information_floor_values_are_the_exact_ones(name):
    pytest.importorskip("scipy")
    ds = _committed(name)
    log_z, _ = exact.log_evidence(ds)
    assert cnd.INFORMATION_NATS[name] == round(exact.information(ds, log_z), 2)


def test_max_log_likelihood_is_attained_at_the_mode():
    for name in DATASETS:
        ds = _committed(name)
        at_mode = exact.log_likelihood(ds, exact.mode(ds)[None, :])[0]
        assert at_mode == pytest.approx(exact.max_log_likelihood(ds), abs=1e-12)
        assert np.all((exact.mode(ds) >= ds.lower) & (exact.mode(ds) <= ds.upper))


# --- the committed exact references --------------------------------------------------


@pytest.mark.parametrize("name", DATASETS)
@pytest.mark.parametrize("backend", ("numpy", "jax_cpu"))
def test_committed_reference_is_complete_exact_and_agrees(name, backend):
    path = protocol.reference_file(ROOT, name, backend, 1)
    assert path == ROOT / "results/reference" / name / backend / "reference.json"
    ref = json.loads(path.read_text())
    assert ref["status"] == "complete" and ref["exact"] is True
    assert protocol.reference_key(ref) == (name, backend, 1, "none")
    assert ref["protocol_id"] == "correlated_nd@1" and ref["model"] == "correlated_nd"
    agreement = ref["agreement"]
    assert agreement["failures"] == []
    assert agreement["backend_identity"]["max_abs_delta_nat"] <= 1e-8
    assert agreement["control_verdicts"] == {"posterior": "accepted", "evidence": "accepted"}
    assert ref["library_revisions"]["autofit"] == cnd.PYAUTOFIT_PIN
    ds = _committed(name)
    if ds.kind == "gaussian" or pytest.importorskip("scipy"):
        log_z, _ = exact.log_evidence(ds)
        assert ref["log_evidence"] == pytest.approx(log_z, abs=1e-12)
    assert ref["max_log_likelihood"] == pytest.approx(exact.max_log_likelihood(ds), abs=1e-12)


# --- the manifest: cells, deferrals, accounting --------------------------------------


def test_manifest_matches_the_simulator_and_the_cli():
    from _autofit_inference_cli import CORRELATED_ND_DATASETS

    assert cnd.DIMENSION == {name: dim for name, (_, dim, _) in sim.DATASETS.items()}
    assert set(CORRELATED_ND_DATASETS) == set(cnd.DATASETS)
    assert protocol.FAMILY_DEFAULT_DATA_SEED == {n: sim.DEFAULT_DATA_SEED for n in sim.DATASETS}
    assert cnd.DATA_SEEDS == (sim.DEFAULT_DATA_SEED,)


def test_pilot_size_and_deferrals_are_the_pre_registered_ones():
    summary = cnd.summary()
    assert summary["expected_runs"] == 1040 and summary["planned_runs"] == 930
    assert summary["by_dataset"] == {
        "corr_gaussian_d20": 260,
        "corr_gaussian_d50": 220,
        "corr_gaussian_d100": 190,
        "rosenbrock_d20": 260,
    }
    deferred = {(c.dataset, c.sampler, c.settings) for c in cnd.cells() if c.status == "deferred"}
    assert ("corr_gaussian_d100", "nautilus", "n_live_400") not in deferred
    assert ("corr_gaussian_d100", "dynesty_dynamic", "default") not in deferred
    assert ("corr_gaussian_d50", "nautilus", "n_live_200") not in deferred
    # Nautilus is exempt from the live-point floor (protocol §11 A1).
    assert ("corr_gaussian_d100", "nautilus", "n_live_100") not in deferred
    assert ("corr_gaussian_d100", "dynesty_static", "nlive_200") in deferred
    assert ("corr_gaussian_d20", "emcee", "walkers_50_steps_2000") not in deferred
    assert ("corr_gaussian_d50", "zeus", "walkers_50_steps_2000") in deferred
    assert ("rosenbrock_d20", "blackjax_nuts_warm", "warmup_200_samples_500") not in deferred
    for cell in cnd.cells():
        assert (cell.status == "deferred") == bool(cell.reason)


def test_the_live_point_rule_is_the_expected_ln_z_scatter():
    reason = cnd.deferral("corr_gaussian_d100", "nss", "n_live_200_mcmc_steps_50")
    assert "n_live 200 < H 267.6" in reason and "1.16 nat" in reason
    assert cnd.live_points("dynesty_dynamic", "default") == 500
    assert cnd.live_points("lbfgs", "default") is None


def test_runner_refuses_deferred_cells_other_waves_and_other_settings():
    from searches import _runner as runner

    assert (
        runner.family_refusal("correlated_nd", "corr_gaussian_d20", "lbfgs", "default", 1) is None
    )
    assert "walker floor" in runner.family_refusal(
        "correlated_nd", "corr_gaussian_d50", "emcee", "walkers_50_steps_2000", 1
    )
    assert "pilot" in runner.family_refusal(
        "correlated_nd", "corr_gaussian_d20", "lbfgs", "default", 2
    )
    assert "only with" in runner.family_refusal(
        "correlated_nd", "corr_gaussian_d20", "nss", "n_live_200", 1
    )
    assert "correlated_nd" in runner.MODEL_TYPES


def _pilot_row(cell, seed=0, **overrides):
    row = {
        "model": "correlated_nd",
        "pilot": True,
        "wave": 1,
        "protocol_id": "correlated_nd@1",
        "pyautofit_commit": cnd.PYAUTOFIT_PIN,
        "dataset": cell.dataset,
        "sampler": cell.sampler,
        "settings_name": cell.settings,
        "config_id": cell.config_id,
        "data_seed": 1,
        "seed": seed,
        "status": "complete",
        "completed": True,
        "run_id": f"{cell.dataset}/{cell.sampler}/{seed}",
    }
    row.update(overrides)
    return row


def test_accounting_sums_and_refuses():
    planned = [c for c in cnd.cells() if c.status == "planned"]
    rows = [
        _pilot_row(planned[0], 0),
        _pilot_row(planned[0], 1, status="failed: RuntimeError: x", completed=False),
        _pilot_row(planned[0], 2, status="failed: Terminated: signal", process_elapsed_s=10.0),
        # A warm-start-shaped capped run: search.fit's own wall is short (the provider took
        # the time), but the process reached the cap, so it is a censored failure.
        _pilot_row(
            planned[0],
            3,
            status="failed: Terminated",
            total_wall_s=30.0,
            provider_wall_s=cnd.PILOT_CAP_S - 100.0,
            process_elapsed_s=cnd.PILOT_CAP_S + 60.0,
        ),
        _pilot_row(planned[0], 4, pyautofit_commit="deadbeef"),
        _pilot_row(planned[0], 5, model="gaussian_x3"),
    ]
    out = cnd.accounting(rows)
    totals = out["totals"]
    assert totals["expected"] == 1040 and totals["deferred"] == 110
    assert totals["row"] + totals["failed"] + totals["missing"] + totals["deferred"] == 1040
    leg = out["legs"][0]
    assert leg["row"] == 1 and leg["failed"] == [1, 3]
    assert leg["missing"] == [2, 4, 5, 6, 7, 8, 9]  # the operational kill is missing
    assert len(out["refused"]) == 2


# --- correlated_nd@1 verdicts --------------------------------------------------------


def _reference(name="corr_gaussian_d20", backend="numpy"):
    return json.loads(protocol.reference_file(ROOT, name, backend, 1).read_text())


def _judged_row(task, reference, **overrides):
    row = {
        "model": "correlated_nd",
        "wave": 1,
        "protocol_id": "correlated_nd@1",
        "task": task,
        "family": {"evidence": "nested", "posterior": "chain"}.get(task, "optimizer"),
        "status": "complete",
        "completed": True,
        "sampler": "nautilus",
        "dataset": reference["dataset"],
        "backend": reference["backend"],
        "data_seed": 1,
        "assertion_mechanism": "none",
        "posterior_relabelled": reference["posterior_relabelled"],
        "whitened": {"worst_median": 0.1, "sigma_ratio_min": 0.9, "sigma_ratio_max": 1.1},
        "log_evidence": reference["log_evidence"] + 0.5,
        "max_log_posterior": reference["map"]["log_posterior"] - 0.5,
        "ess_kish": 800.0,
        "termination": {"met": True},
    }
    row.update(overrides)
    return row


def test_family_rows_are_judged_by_their_own_protocol():
    ref = _reference()
    verdict = protocol.verdict(_judged_row("evidence", ref), ref)
    assert verdict["protocol_id"] == "correlated_nd@1"
    assert verdict["acceptance"] == "accepted", verdict["acceptance_reason"]
    assert verdict["acceptance_reason"].startswith("correlated_nd@1: ")
    assert "(b)3 whitened" in verdict["acceptance_reason"]
    assert "ppc" not in verdict["acceptance_reason"] and "(b)4" not in verdict["acceptance_reason"]
    assert set(verdict["placeholders"]) == set(protocol.PLACEHOLDERS_CND1)
    assert verdict["convergence"] == "converged"


def test_family_criteria_reject_what_they_should():
    ref = _reference()
    bad_z = protocol.verdict(
        _judged_row("evidence", ref, log_evidence=ref["log_evidence"] + 2), ref
    )
    assert bad_z["acceptance"] == "rejected" and "(c)" in bad_z["acceptance_reason"]
    squashed = {"worst_median": 0.1, "sigma_ratio_min": 0.3, "sigma_ratio_max": 1.0}
    bad_w = protocol.verdict(_judged_row("posterior", ref, whitened=squashed), ref)
    assert bad_w["acceptance"] == "rejected" and "(b)3" in bad_w["acceptance_reason"]
    no_w = protocol.verdict(_judged_row("posterior", ref, whitened=None), ref)
    assert no_w["acceptance"] == "not_assessed"
    bad_map = protocol.verdict(
        _judged_row("point_map", ref, max_log_posterior=ref["map"]["log_posterior"] - 1.5), ref
    )
    assert bad_map["acceptance"] == "rejected"
    good_map = protocol.verdict(_judged_row("point_map", ref), ref)
    assert good_map["acceptance"] == "accepted"


def test_family_rows_never_judged_against_another_backend_or_family():
    ref = _reference(backend="jax_cpu")
    wrong = protocol.verdict(_judged_row("posterior", _reference()), ref)
    assert wrong["acceptance"] == "not_assessed" and "backend" in wrong["acceptance_reason"]
    stamped = _judged_row("posterior", _reference(), protocol_id="gaussian_x3@1")
    assert protocol.verdict(stamped, _reference())["acceptance"] == "not_assessed"
    assert protocol.protocol_for({"model": "correlated_nd", "wave": 2}) is None
    assert protocol.protocol_id_for("correlated_nd", 1) == "correlated_nd@1"
    assert protocol.protocol_id_for("gaussian_x3", 2) == "gaussian_x3@2"


def test_summarise_without_relabelling_has_no_modes():
    draws = np.random.default_rng(0).normal(size=(500, 3))
    out = post.summarise_samples(draws, np.ones(500), ["x000", "x001", "x002"], relabel=False)
    assert out["posterior_relabelled"] is out["posterior"]
    assert out["modes"] is None and out["modes_found"] is None
    assert out["label_permutations_found"] is None


def test_family_chain_convergence_uses_the_cnd1_bars():
    ref = _reference()
    row = _judged_row("posterior", ref, rhat_max=1.005, ess_bulk_min=450.0)
    assert protocol.verdict(row, ref)["convergence"] == "converged"
    row = _judged_row("posterior", ref, rhat_max=1.02, ess_bulk_min=450.0)
    assert protocol.verdict(row, ref)["convergence"] == "not_converged"


def test_runner_shaped_rows_are_admitted():
    """The manifest key the runner writes (config_segment, warm cache on JAX) is a cell key."""
    from searches._runner import config_segment

    for cell in cnd.cells():
        if cell.status != "planned":
            continue
        cache = cnd.COMPILE_CACHE if cell.backend == "jax_cpu" else None
        config_id = config_segment(cell.config_name, cache)
        row = _pilot_row(cell, 9, config_id=config_id, config_name=cell.config_name)
        assert cnd.admit(row) is None, (cell, cnd.admit(row))


def test_cap_stub_row_is_a_censored_admitted_failure(tmp_path):
    import cnd_cap_stub as stub

    argv = [
        "--dataset", "corr_gaussian_d100", "--sampler", "blackjax_nuts", "--settings",
        "warmup_200_samples_500", "--config", "ral_jax_cpu_fp64", "--data-seed", "1",
        "--seed", "4", "--limit", "14520", "--root", str(tmp_path),
    ]  # fmt: skip
    assert stub.main([*argv, "--exit-code", "1"]) == 0  # not a timeout: nothing written
    assert not list(tmp_path.rglob("*.json"))
    assert stub.main([*argv, "--exit-code", "137"]) == 0
    (path,) = tmp_path.rglob("search_seed4.json")
    row = json.loads(path.read_text())
    assert cnd.admit(row) is None
    assert not cnd.operational_kill(row)
    leg = next(
        leg
        for leg in cnd.accounting([row])["legs"]
        if leg["id"].startswith("corr_gaussian_d100/blackjax_nuts/")
    )
    assert leg["failed"] == [4]
    first = path.read_text()
    assert stub.main([*argv, "--exit-code", "124"]) == 0  # an existing row is never replaced
    assert path.read_text() == first

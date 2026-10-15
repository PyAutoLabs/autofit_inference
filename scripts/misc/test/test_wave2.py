"""Scored wave 2 (B5a): the ``@2`` freeze, versioned judging, the wave manifest identity,
per-realisation references, attempt accounting and the generated RAL plan."""

import json
import math
import re

import plan_wave2
import pytest
from searches import _pilot as pilot
from searches import _protocol as protocol
from searches import _runner as runner
from searches import _wave2 as wave2
from simulators import gaussian_x3 as simulator
from test_protocol import OFFSETS, REFERENCE, row
from wall import check_submits, rates

from _autofit_inference_cli import parse_inference_cli

ROOT = wave2.ROOT
CALIBRATION = json.loads((ROOT / "results/calibration/gaussian_x3_wave1.json").read_text())


# --- the freeze ------------------------------------------------------------------------


def test_frozen_v2_is_the_calibration_record():
    frozen = {k: list(v) if isinstance(v, tuple) else v for k, v in protocol.FROZEN_V2.items()}
    assert frozen == CALIBRATION["frozen"]
    assert set(protocol.FROZEN_V2) == set(protocol.PLACEHOLDERS)
    assert protocol.PLACEHOLDERS_BY_PROTOCOL["gaussian_x3@2"] == ()
    # @1 is untouched: its thresholds still carry the pilot's placeholders.
    assert protocol.THRESHOLDS["mode_radius_px"] == 5.0
    assert protocol.THRESHOLDS_V2["mode_radius_px"] == 5.5


def test_protocol_resolves_per_wave():
    assert protocol.protocol_for({"pilot": True, "protocol_id": "gaussian_x3@1"}) == "gaussian_x3@1"
    assert protocol.protocol_for({}) == "gaussian_x3@1"
    assert protocol.protocol_for({"wave": 2, "protocol_id": "gaussian_x3@2"}) == "gaussian_x3@2"
    # A row stamped with another wave's id is never judged under either.
    assert protocol.protocol_for({"wave": 2, "protocol_id": "gaussian_x3@1"}) is None
    assert protocol.protocol_for({"wave": 3}) is None


def test_a_pilot_row_is_judged_under_v1_exactly_as_before():
    path = next(
        (ROOT / "results/searches/gaussian_x3_separated/data_seed2/nautilus").rglob(
            "local_*/search_seed0.json"
        )
    )
    pilot_row = json.loads(path.read_text())
    assert pilot_row["pilot"] is True and "wave" not in pilot_row
    reference = protocol.load_json(
        protocol.reference_file(ROOT, pilot_row["dataset"], pilot_row["backend"], 2)
    )
    offsets = protocol.load_json(protocol.offsets_file(ROOT))
    out = protocol.verdict(pilot_row, reference, offsets)
    assert out["protocol_id"] == "gaussian_x3@1"
    assert not out["acceptance_reason"].startswith("gaussian_x3@")
    assert out["placeholders"] == list(protocol.PLACEHOLDERS)
    v1 = protocol.acceptance(pilot_row, reference, offsets, protocol.THRESHOLDS)
    assert out["acceptance"] == v1["acceptance"] and out["acceptance_reason"] == v1["reason"]


def test_a_scored_row_is_judged_under_v2_and_says_so():
    # A run mode 5.3 px from the reference mode: outside @1's 5 px coverage radius,
    # inside @2's frozen 5.5 px.
    shifted = [{"centres": [25.0 + 5.3, 45.0, 60.0], "weight": 1.0}]
    pilot_row = row(modes=shifted, pilot=True, protocol_id="gaussian_x3@1")
    scored = row(modes=shifted, wave=2, protocol_id="gaussian_x3@2", pilot=False)
    v1 = protocol.verdict(pilot_row, REFERENCE, OFFSETS)
    v2 = protocol.verdict(scored, REFERENCE, OFFSETS)
    assert v1["protocol_id"] == "gaussian_x3@1" and v1["acceptance"] == "rejected"
    assert "(b)4" in v1["acceptance_reason"]
    assert v2["protocol_id"] == "gaussian_x3@2" and v2["acceptance"] == "accepted", v2
    assert v2["acceptance_reason"].startswith("gaussian_x3@2: ")
    assert v2["convergence_reason"].startswith("gaussian_x3@2: ")
    assert v2["placeholders"] == []
    mislabelled = protocol.verdict(row(wave=2, protocol_id="gaussian_x3@1"), REFERENCE, OFFSETS)
    assert mislabelled["acceptance"] == "not_assessed"


# --- per-realisation references --------------------------------------------------------


def test_default_realisations_match_the_simulator():
    assert protocol.DEFAULT_DATA_SEED == {
        name: default for name, (_, default) in simulator.DATASETS.items()
    }


def test_each_realisation_has_its_own_reference_path():
    base = ROOT / "results/reference/gaussian_x3_blend/numpy"
    assert protocol.reference_file(ROOT, "gaussian_x3_blend", "numpy", 1) == base / "reference.json"
    assert protocol.reference_file(ROOT, "gaussian_x3_blend", "numpy", None) == (
        base / "reference.json"
    )
    assert protocol.reference_file(ROOT, "gaussian_x3_blend", "numpy", 3) == (
        base / "data_seed3" / "reference.json"
    )
    sep = ROOT / "results/reference/gaussian_x3_separated/jax_cpu"
    assert protocol.reference_dir(ROOT, "gaussian_x3_separated", "jax_cpu", 2) == sep
    assert protocol.reference_dir(ROOT, "gaussian_x3_separated", "jax_cpu", 1) == sep / "data_seed1"


def test_a_row_is_never_judged_against_another_realisation():
    other = dict(REFERENCE, data_seed=3)
    out = protocol.verdict(row(wave=2, protocol_id="gaussian_x3@2"), other, OFFSETS)
    assert out["acceptance"] == "not_assessed"
    assert "data_seed" in out["acceptance_reason"]


def test_every_realisation_is_committed_and_reproducible():
    for dataset in wave2.DATASETS:
        for data_seed in wave2.DATA_SEEDS:
            path = simulator.dataset_dir(dataset, data_seed, ROOT)
            truth = json.loads((path / "truth.json").read_text())
            assert truth["data_seed"] == data_seed, path


# --- the manifest identity --------------------------------------------------------------


def scored_row(**overrides):
    base = {
        "dataset": "gaussian_x3_blend",
        "sampler": "nautilus",
        "settings_name": "n_live_100",
        "config_id": "ral_numpy_fp64",
        "data_seed": 3,
        "seed": 7,
        "wave": 2,
        "pilot": False,
        "protocol_id": "gaussian_x3@2",
        "pyautofit_commit": wave2.PYAUTOFIT_PIN,
        "status": "complete",
        "completed": True,
        "run_id": "x",
    }
    base.update(overrides)
    return base


def test_manifest_key_carries_realisation_protocol_and_library():
    assert wave2.manifest_key(scored_row()) == (
        "gaussian_x3_blend",
        "nautilus",
        "n_live_100",
        "ral_numpy_fp64",
        3,
        "gaussian_x3@2",
        wave2.PYAUTOFIT_PIN,
    )
    assert wave2.admit(scored_row()) is None


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"pyautofit_commit": "0dbf258c4f5e" + "0" * 28}, "not the wave pin"),
        ({"pyautofit_commit": None}, "not the wave pin"),
        ({"protocol_id": "gaussian_x3@1"}, "protocol"),
        ({"wave": None, "pilot": True}, "not a wave-2 row"),
        ({"data_seed": 6}, "data_seed"),
        ({"seed": 50}, "search seed"),
        ({"settings_name": "n_live_800"}, "not a cell"),
        (
            {
                "sampler": "nss",
                "settings_name": "n_live_200",
                "config_id": "ral_jax_cpu_fp64_cache_warm",
            },
            "not a cell",
        ),
    ],
)
def test_a_row_outside_the_wave_identity_is_refused(overrides, needle):
    why = wave2.admit(scored_row(**overrides))
    assert why is not None and needle in why, why


def test_the_runner_refuses_a_wave2_run_on_another_library():
    assert runner.wave_refusal(1, {"autofit": "abc"}, wave2.PYAUTOFIT_PIN) is None
    assert runner.wave_refusal(2, {"autofit": wave2.PYAUTOFIT_PIN}, wave2.PYAUTOFIT_PIN) is None
    why = runner.wave_refusal(2, {"autofit": "0dbf258c4f5e"}, wave2.PYAUTOFIT_PIN)
    assert why and wave2.PYAUTOFIT_PIN in why and "0dbf258c4f5e" in why


def test_wave2_seeds_every_search():
    for spec in runner.SAMPLERS.values():
        assert runner.seed_kwarg_for(spec, 2) == "seed"
    # The pilot's provenance is unchanged: Dynesty took no seed in wave 1.
    assert runner.seed_kwarg_for(runner.SAMPLERS["dynesty_static"], 1) is None


def test_cli_wave_flag():
    assert parse_inference_cli([]).wave == 1
    assert parse_inference_cli(["--wave", "2"]).wave == 2


# --- the menu and attempt accounting ----------------------------------------------------


def test_menu_shape_and_statuses():
    # The menu as B5a submitted it: the wave-1 timeouts alone.
    cells = wave2.cells(wave2.calibration_timeouts())
    by_status = {
        s: [c for c in cells if c.status == s] for s in ("planned", "probe_first", "deferred")
    }
    assert {(c.dataset, c.sampler) for c in by_status["deferred"]} == {
        ("gaussian_x3_blend", "blackjax_nuts_warm")
    }
    assert {(c.sampler, c.settings, c.backend) for c in by_status["probe_first"]} == {
        ("nss", "n_live_200_mcmc_steps_50", "jax_cpu"),
        ("zeus", "walkers_50_steps_2000", "jax_cpu"),
    }
    assert all(c.timeout_s is not None for c in by_status["planned"])
    assert len(wave2.expected_runs()) == len(cells) * 5 * 50
    # NSS runs only its wave-2 entry, with at least 50 MCMC steps.
    nss = [c for c in cells if c.sampler == "nss"]
    assert {c.settings for c in nss} == {"n_live_200_mcmc_steps_50"}
    assert runner.SAMPLERS["nss"].settings["n_live_200_mcmc_steps_50"]["num_mcmc_steps"] >= 50


def test_the_pilot_manifest_is_unchanged_by_wave2():
    assert len(pilot.expected_runs()) == 520
    assert not any(c.settings == "n_live_200_mcmc_steps_50" for c in pilot.cells())


def test_accounting_separates_row_failed_missing_and_deferred():
    rows = [
        scored_row(data_seed=1, seed=0),
        scored_row(data_seed=1, seed=1, status="failed: FloatingPointError: x"),
        scored_row(data_seed=1, seed=2, pyautofit_commit="f" * 40),
    ]
    out = wave2.accounting(rows)
    leg = next(
        g
        for g in out["legs"]
        if g["id"] == "gaussian_x3_blend/nautilus/n_live_100/ral_numpy_fp64/data_seed1"
    )
    assert leg["row"] == 1 and leg["failed"] == [1]
    assert 2 in leg["missing"] and len(leg["missing"]) == 48 and leg["deferred"] == []
    assert leg["reason"]
    deferred = next(
        g
        for g in out["legs"]
        if g["sampler"] == "blackjax_nuts_warm" and g["dataset"] == "gaussian_x3_blend"
    )
    assert len(deferred["deferred"]) == 50 and deferred["missing"] == []
    assert "TracerBoolConversionError" in deferred["reason"]
    assert out["refused"] == [{"run_id": "x", "reason": wave2.admit(rows[2])}]
    totals = out["totals"]
    assert (
        totals["row"] + totals["failed"] + totals["missing"] + totals["deferred"]
        == totals["expected"]
    )
    assert totals["expected"] == len(wave2.expected_runs())


def test_an_operational_kill_is_missing_and_a_resumed_row_is_refused():
    cell = next(
        c
        for c in wave2.cells()
        if c.key[:3] == ("gaussian_x3_blend", "nautilus", "n_live_100") and c.backend == "numpy"
    )
    early = scored_row(
        data_seed=1, seed=0, status="failed: Terminated: signal SIGTERM", total_wall_s=10.0
    )
    late = scored_row(
        data_seed=1,
        seed=1,
        status="failed: Terminated: signal SIGTERM",
        total_wall_s=cell.timeout_s + 1,
    )
    assert wave2.operational_kill(early, cell) and not wave2.operational_kill(late, cell)
    leg = next(
        g
        for g in wave2.accounting([early, late])["legs"]
        if g["id"] == "gaussian_x3_blend/nautilus/n_live_100/ral_numpy_fp64/data_seed1"
    )
    assert 0 in leg["missing"] and leg["failed"] == [1]
    assert "resumed" in wave2.admit(scored_row(resumed=True))


def test_wave2_rows_never_enter_the_pilot_tooling(tmp_path):
    import build_catalogue as catalogue

    path = tmp_path / "results/searches/x/search_seed0.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, **scored_row()}))
    assert catalogue.scan_rows(tmp_path) == []


# --- the generated RAL plan -------------------------------------------------------------


def test_rates_table_is_the_pilot_walls():
    assert rates.STEP_RATE == plan_wave2.derived_rates()
    assert set(rates.PROVENANCE) == {key[3] for key in rates.STEP_RATE}


def test_committed_plan_is_current_and_passes_the_wall_gate():
    files = plan_wave2.plan()
    for path, content in files.items():
        assert path.exists() and path.read_text() == content, path
        if path.name.startswith("submit_"):
            assert check_submits.check_text(content, path.name) == [], path


def test_every_submit_stays_on_ral_and_within_slurm_limits():
    for path in sorted((ROOT / "hpc/batch_cpu").glob("submit_wave2_*")):
        text = path.read_text()
        partitions = re.findall(r"^#SBATCH --partition=(\S+)", text, re.M)
        assert partitions == ["ral"], path
        assert "gpu" not in partitions[0]
        assert "--gres" not in text
        array = re.search(r"^#SBATCH --array=0-(\d+)%(\d+)$", text, re.M)
        assert array and int(array.group(1)) <= 1000, path  # RAL MaxArraySize 1001
        assert (
            "AF_PIN/PyAutoFit" in text
            and "/mnt/ral/jnightin/PyAuto/" not in text.split("#SBATCH")[-1]
        )


def test_scored_tasks_cover_every_planned_run_once():
    """The B5a scored arrays (pilot-derived timeouts only) cover every pilot-planned run
    once; with the probe record written, the probe plus the remainder cover every run
    of the probe-sourced cells once, and nothing overlaps."""
    seen = set()
    for dataset in wave2.DATASETS:
        for backend in wave2.CONFIG_BY_BACKEND:
            for task in wave2.scored_tasks(dataset, backend):
                assert task.limit_s == task.cell.timeout_s + wave2.RUN_OVERHEAD_S
                for seed in task.seeds:
                    key = (*task.cell.key, task.data_seed, seed)
                    assert key not in seen
                    seen.add(key)
    planned = {
        (*c.key, d, s)
        for c, d, s in wave2.expected_runs()
        if c.status == "planned" and c.timeout_source == "pilot"
    }
    assert seen == planned
    assert len(seen) == 11750
    probes = {(*t.cell.key, t.data_seed, t.seeds[0]) for t in wave2.probe_tasks()}
    assert len(probes) == 4 * len(wave2.PROBE_SEEDS)
    assert math.isclose(wave2.PROBE_CAP_S, 43200.0)
    rest = set()
    for task in wave2.remainder_tasks():
        assert task.limit_s == task.cell.timeout_s + wave2.RUN_OVERHEAD_S
        assert task.cell.timeout_source == "probe"
        for seed in task.seeds:
            key = (*task.cell.key, task.data_seed, seed)
            assert key not in rest and key not in probes and key not in seen
            rest.add(key)
    sourced = {(*c.key, d, s) for c, d, s in wave2.expected_runs() if c.timeout_source == "probe"}
    if sourced:
        assert rest | probes == sourced and len(rest) == 980


def test_probe_timeouts_fill_only_cells_the_pilot_left_open(tmp_path):
    pilot_t = wave2.calibration_timeouts()
    probe = tmp_path / "probe.json"
    probe.write_text(
        json.dumps(
            {
                "timeouts": {
                    "zeus/walkers_50_steps_2000/jax_cpu": {
                        "median_probe_wall_s": 100.0,
                        "n": 10,
                        "timeout_s": 1000.0,
                    },
                    # A probe entry never overrides a pilot-derived timeout.
                    "emcee/walkers_50_steps_2000/numpy": {
                        "median_probe_wall_s": 1.0,
                        "n": 10,
                        "timeout_s": 600.0,
                    },
                }
            }
        )
    )
    merged = wave2.wave_timeouts(probe=probe)
    assert (
        merged["emcee/walkers_50_steps_2000/numpy"] == pilot_t["emcee/walkers_50_steps_2000/numpy"]
    )
    cells = wave2.cells(merged)
    zeus = [c for c in cells if c.sampler == "zeus" and c.backend == "jax_cpu"]
    assert zeus and all(c.status == "planned" and c.timeout_source == "probe" for c in zeus)
    assert all(c.timeout_s == 1000.0 for c in zeus)
    nss = [c for c in cells if c.sampler == "nss"]
    assert all(c.status == "probe_first" for c in nss)
    # The B5a arrays never move: the probe-sourced cell stays out of the scored tasks.
    for dataset in wave2.DATASETS:
        assert not any(
            t.cell.sampler == "zeus" for t in wave2.scored_tasks(dataset, "jax_cpu", merged)
        )
    assert {t.cell.sampler for t in wave2.probe_tasks(merged)} == {"zeus", "nss"}
    assert {t.cell.sampler for t in wave2.remainder_tasks(merged)} == {"zeus"}


def test_pilot_rows_are_not_rejudged_against_a_later_reference():
    reference = dict(
        REFERENCE,
        dataset="gaussian_x3_blend",
        backend="jax_cpu",
        data_seed=1,
        assertion_mechanism="xp_where_penalty",
    )
    pilot_row = row(
        dataset="gaussian_x3_blend",
        backend="jax_cpu",
        data_seed=1,
        assertion_mechanism="xp_where_penalty",
        pilot=True,
        protocol_id="gaussian_x3@1",
    )
    out = protocol.verdict(pilot_row, reference, OFFSETS)
    assert out["acceptance"] == "not_assessed"
    assert out["acceptance_reason"] == "reference completed after the pilot (B5b)"
    scored = dict(pilot_row, pilot=False, wave=2, protocol_id="gaussian_x3@2")
    out = protocol.verdict(scored, reference, OFFSETS)
    assert "after the pilot" not in out["acceptance_reason"]
    assert out["acceptance_reason"].startswith("gaussian_x3@2: ")


def test_template_documents_the_pin():
    template = (ROOT / "hpc/batch_cpu/template").read_text()
    assert "AF_PIN/PyAutoFit" in template and ".pyauto_pin" in template
    assert template.count("#SBATCH --partition=ral\n") == 1

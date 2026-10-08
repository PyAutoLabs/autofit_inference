"""The single-search driver: one search, one dataset, one config, one seed, one row.

``run_search(sampler=, dataset_class=, model_type=)`` is generalised from
``autolens_inference/scripts/misc/searches/_point_runner.py::run_point_search``. Every
leaf ``scripts/<dataset_class>/searches/<sampler>/<model_type>.py`` is a few lines that
call it with **literal** keyword arguments: the Brain samplers faculty's AST parser
(``_declared_cell``) reads the cell identity from that call, not from the path.

What a row records (schema v1)
------------------------------

The autolens_inference row (identity, provenance, clocks, ``posterior``,
``truth_delta_sigma``, the admission bar) plus the protocol fields of
``wiki/project/protocol_gaussian_x3.md``:

- ``posterior_relabelled``, ``modes``, ``modes_found``, ``label_permutations_found``
  (convention D15(c));
- ``max_log_likelihood`` **and** ``max_log_posterior`` (never interchanged), with the
  vectors that attain them;
- ``ppc_chi2`` (the §4(b)3 posterior-predictive check) and ``ess_kish`` / ``ess_per_s``;
- ``evals_to_target`` / ``time_to_target_s`` against the shared target
  ``ref_max_log_likelihood − 1`` with ``time_to_target_basis`` ``observed`` (numpy:
  MLTracker times every call) or ``estimated`` (JAX: interpolated from the eval index);
- ``evals_to_target_basis`` (``estimated`` under JAX: a sample index in the search's
  posterior sample order, not an evaluation history);
- ``compile_s`` with ``compile_cache`` ``cold`` / ``warm`` (separate runs, whose
  ``_cache_<cold|warm>`` suffix on the config is part of the run ID, the PyAutoFit
  ``path_prefix`` and the result file path, so they never resume or overwrite each
  other), and
  ``per_call_s`` / ``likelihood_share`` flagged ``estimated``;
- ``run_id`` — ``gaussian_x3/<dataset>/data_seed<d>/<sampler>/<settings>/<config_id>/seed<s>``
  (``config_id`` = ``config_segment(config_name, compile_cache)``),
  the stable identity autofit_profiling shares — and ``pyautofit_commit``.

Hazards carried over from autolens_inference
--------------------------------------------

- The backend environment is exported before autofit is imported, so argv is parsed
  first and the modelling stack is imported inside the driver.
- ``config_name`` and ``seed`` live in the PyAutoFit ``path_prefix``, so two legs that
  differ only in config never resume each other's fit.
- A completed row is written in a ``finally`` block, so a crashed search still leaves a
  ``status: failed: …`` row (a failure is data, §8).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _autofit_inference_cli import (  # noqa: E402
    auto_simulate_if_missing,
    device_info_dict,
    library_revisions,
    parse_inference_cli,
    set_backend_env,
)

#: Version of the ``search_seed<n>.json`` schema written by :func:`run_search`.
SCHEMA_VERSION = 1

#: Admission-bar timing: warm-up calls discarded, then this many timed calls.
WARMUP_CALLS = 10
TIMED_CALLS = 200

#: Equal-weight draws for the posterior-predictive check (protocol §4(b)3).
PPC_DRAWS = 500

MODEL_TYPES = ("gaussian_x3",)

EVALS_TO_TARGET_JAX_NOTE = (
    "estimated: the index of the first sample reaching the target in the search's "
    "posterior sample order (Nautilus: posterior()), not an evaluation history, so it "
    "is not an observed evaluation count"
)

LOG_EVIDENCE_ERR_NOTE = "no sampler-agnostic log Z error is recorded; compare seeds instead"


@dataclass(frozen=True)
class SamplerSpec:
    """One registered search: its PyAutoFit class, task, family and named settings.

    ``settings`` maps a settings name to constructor kwargs; ``default`` names the one a
    leaf runs without ``--settings``. ``seed_kwarg`` is the constructor argument that
    seeds the search, or ``None`` when PyAutoFit exposes none (the search-level seed is
    phase A4). ``jax_native`` says whether a ``jax_cpu`` leg is meaningful;
    ``numpy_supported`` is ``False`` for the JAX-only searches PyAutoFit refuses to run on a
    numpy Analysis (MultiStart*, BlackJAX NUTS/SMC, NSS).
    Settings for searches not yet run are provisional and are settled when B3 first runs
    them (from ``autofit_workspace_test/scripts/searches/*``).
    """

    cls: str
    task: str
    family: str
    settings: dict
    default: str
    seed_kwarg: str | None
    jax_native: bool
    vectorised: bool = False
    numpy_supported: bool = True
    status: str = "measured"
    note: str = ""
    extra: dict = field(default_factory=dict)


SAMPLERS: dict[str, SamplerSpec] = {
    "nautilus": SamplerSpec(
        cls="Nautilus",
        task="evidence",
        family="nested",
        settings={
            "n_live_100": {"n_live": 100},
            "n_live_200": {"n_live": 200},
            "n_live_400": {"n_live": 400},
        },
        default="n_live_100",
        seed_kwarg="seed",
        jax_native=True,
        vectorised=True,
    ),
    "dynesty_static": SamplerSpec(
        cls="DynestyStatic",
        task="evidence",
        family="nested",
        settings={"nlive_200": {"nlive": 200}},
        default="nlive_200",
        seed_kwarg=None,
        jax_native=True,
    ),
    "dynesty_dynamic": SamplerSpec(
        cls="DynestyDynamic",
        task="evidence",
        family="nested",
        settings={"default": {}},
        default="default",
        seed_kwarg=None,
        jax_native=True,
    ),
    "nss": SamplerSpec(
        cls="NSS",
        task="evidence",
        family="nested",
        settings={"n_live_200": {"n_live": 200}},
        default="n_live_200",
        seed_kwarg="seed",
        jax_native=True,
        numpy_supported=False,
        status="deferred",
        note="NSS on the blend is deferred until phase A3b (NSS onto Fitness)",
    ),
    "emcee": SamplerSpec(
        cls="Emcee",
        task="posterior",
        family="chain",
        settings={"walkers_50_steps_2000": {"nwalkers": 50, "nsteps": 2000}},
        default="walkers_50_steps_2000",
        seed_kwarg=None,
        jax_native=True,
    ),
    "zeus": SamplerSpec(
        cls="Zeus",
        task="posterior",
        family="chain",
        settings={"walkers_50_steps_2000": {"nwalkers": 50, "nsteps": 2000}},
        default="walkers_50_steps_2000",
        seed_kwarg=None,
        jax_native=True,
    ),
    "blackjax_nuts": SamplerSpec(
        cls="BlackJAXNUTS",
        task="posterior",
        family="chain",
        settings={"warmup_200_samples_500": {"num_warmup": 200, "num_samples": 500}},
        default="warmup_200_samples_500",
        seed_kwarg="seed",
        jax_native=True,
        numpy_supported=False,
    ),
    "smc": SamplerSpec(
        cls="SMC",
        task="posterior",
        family="chain",
        settings={"default": {}},
        default="default",
        seed_kwarg="seed",
        jax_native=True,
        numpy_supported=False,
    ),
    "lbfgs": SamplerSpec(
        cls="LBFGS",
        task="point_map",
        family="optimizer",
        settings={"default": {}},
        default="default",
        seed_kwarg=None,
        jax_native=True,
    ),
    "bfgs": SamplerSpec(
        cls="BFGS",
        task="point_map",
        family="optimizer",
        settings={"default": {}},
        default="default",
        seed_kwarg=None,
        jax_native=True,
    ),
    "multi_start_adam": SamplerSpec(
        cls="MultiStartAdam",
        task="point_map",
        family="optimizer",
        settings={"starts_48_steps_300": {"n_starts": 48, "n_steps": 300}},
        default="starts_48_steps_300",
        seed_kwarg="seed",
        jax_native=True,
        numpy_supported=False,
    ),
    "multi_start_prodigy": SamplerSpec(
        cls="MultiStartProdigy",
        task="point_map",
        family="optimizer",
        settings={"starts_48_steps_300": {"n_starts": 48, "n_steps": 300}},
        default="starts_48_steps_300",
        seed_kwarg="seed",
        jax_native=True,
        numpy_supported=False,
    ),
    "multi_start_adabelief": SamplerSpec(
        cls="MultiStartADABelief",
        task="point_map",
        family="optimizer",
        settings={"starts_48_steps_300": {"n_starts": 48, "n_steps": 300}},
        default="starts_48_steps_300",
        seed_kwarg="seed",
        jax_native=True,
        numpy_supported=False,
    ),
    "multi_start_lion": SamplerSpec(
        cls="MultiStartLion",
        task="point_map",
        family="optimizer",
        settings={"starts_48_steps_300": {"n_starts": 48, "n_steps": 300}},
        default="starts_48_steps_300",
        seed_kwarg="seed",
        jax_native=True,
        numpy_supported=False,
    ),
}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in scripts/misc/test/test_runner.py)
# ---------------------------------------------------------------------------


def config_segment(config_name: str, compile_cache: str | None = None) -> str:
    """The config part of a row's identity: ``config_name``, plus ``_cache_<cold|warm>``
    on a JAX leg. The cold and warm runs of §8 are separate runs, so they must never
    share a PyAutoFit output (the second would resume the first's completed fit and
    record a load time) or a result file (the second would overwrite the first). On
    numpy there is no compilation cache and ``compile_cache`` is ``None``."""
    return config_name if compile_cache is None else f"{config_name}_cache_{compile_cache}"


def run_id(
    model_type: str,
    dataset: str,
    data_seed: int,
    sampler: str,
    settings: str,
    config_name: str,
    seed: int,
) -> str:
    """The stable run identity, shared with autofit_profiling."""
    return (
        f"{model_type}/{dataset}/data_seed{data_seed}/{sampler}/{settings}/{config_name}/seed{seed}"
    )


def results_path(
    root: Path,
    dataset: str,
    data_seed: int,
    sampler: str,
    settings: str,
    config_name: str,
    seed: int,
) -> Path:
    """``results/searches/<dataset>/data_seed<d>/<sampler>/<settings>/<config>/search_seed<s>.json``."""
    return (
        Path(root)
        / "results"
        / "searches"
        / dataset
        / f"data_seed{data_seed}"
        / sampler
        / settings
        / config_name
        / f"search_seed{seed}.json"
    )


def output_path_prefix(
    dataset: str, data_seed: int, sampler: str, settings: str, config_name: str, seed: int
) -> Path:
    """The PyAutoFit ``path_prefix`` — config name and seed in the PATH."""
    return (
        Path("searches")
        / dataset
        / f"data_seed{data_seed}"
        / sampler
        / settings
        / config_name
        / f"seed_{seed}"
    )


def truth_delta_sigma_from(posterior: dict, truths: dict) -> dict:
    """``(median − truth) / sigma`` per parameter present in both (a key intersection)."""
    out = {}
    for key, truth in truths.items():
        stats = posterior.get(key)
        if not stats or not stats.get("sigma"):
            continue
        out[key] = (stats["median"] - truth) / stats["sigma"]
    return out


def likelihood_share(per_call_s, likelihood_evals, wall_s) -> float | None:
    """``per_call_s * likelihood_evals / wall_s``, never clipped (an estimate, §8)."""
    try:
        per_call, evals, wall = float(per_call_s), int(likelihood_evals), float(wall_s)
    except (TypeError, ValueError):
        return None
    if wall <= 0:
        return None
    return per_call * evals / wall


def timing_summary(samples: list[float]) -> dict:
    if not samples:
        return {"median_s": None, "p16_s": None, "p84_s": None, "min_s": None, "n": 0}
    ordered = sorted(samples)
    n = len(ordered)

    def quantile(q: float) -> float:
        return ordered[min(n - 1, max(0, round(q * (n - 1))))]

    return {
        "median_s": float(statistics.median(ordered)),
        "p16_s": float(quantile(0.16)),
        "p84_s": float(quantile(0.84)),
        "min_s": float(ordered[0]),
        "n": n,
    }


def time_calls(fn, arg, *, warmup: int = WARMUP_CALLS, n: int = TIMED_CALLS, block=None) -> dict:
    sync = block if block is not None else (lambda value: value)
    for _ in range(warmup):
        sync(fn(arg))
    walls = []
    for _ in range(n):
        start = time.perf_counter()
        sync(fn(arg))
        walls.append(time.perf_counter() - start)
    summary = timing_summary(walls)
    summary["warmup"] = warmup
    return summary


#: Fitness's resample sentinel; a timed call returning it measured the rejection path.
RESAMPLE_SENTINEL = -1.0e99


def admission_vector(keys, truths: dict) -> list[float]:
    """The vector the admission bar times: the generating truth, in model order.

    Never the prior medians: all three centre medians are 50, which violates the
    ordered-centre assertions, so numpy ``Fitness`` short-circuits to the resample
    sentinel (``autofit/non_linear/fitness.py``, the ``FitException`` branch) and the
    timing measures the rejection path, not a likelihood evaluation. The truth satisfies
    the assertions (and the separated control's disjoint priors) by construction.
    """
    return [float(truths[key]) for key in keys]


TERMINATION_NOT_EXPOSED = "termination condition not exposed by PyAutoFit (phase A3)"


def nested_termination(sampler: str, search, internal) -> dict:
    """The nested search's own termination condition, read back after the fit (§5).

    Nautilus: PyAutoFit's ``call_search`` also stops when the likelihood-call budget
    ``n_like_max`` is spent, so completion alone does not mean convergence. The
    sampler object (``result.search_internal``) keeps the state nautilus's own
    ``run`` tests — ``explored`` (the exploration phase ended at ``f_live``), every
    shell holding ``n_shell`` points, and ``n_eff`` against the target — so the
    condition is recomputed from it. Any other nested search, or a Nautilus run whose
    sampler is not available (e.g. resumed with ``search_internal`` removed), reports
    ``met: None`` with :data:`TERMINATION_NOT_EXPOSED`.
    """
    if sampler == "nautilus" and internal is not None:
        try:
            import numpy as np

            n_eff = float(internal.n_eff)
            n_eff_target = float(search.n_eff)
            n_shell = int(search.n_shell)
            explored = bool(internal.explored)
            shells_full = bool(np.all(np.asarray(internal.shell_n) >= n_shell))
            n_like = int(internal.n_like)
            n_like_max = search.n_like_max
            met = explored and shells_full and n_eff >= n_eff_target
            return {
                "observable": True,
                "met": bool(met),
                "criterion": "nautilus: explored (f_live <= target), every shell >= n_shell "
                "points, n_eff >= target",
                "explored": explored,
                "shells_full": shells_full,
                "n_eff": n_eff,
                "n_eff_target": n_eff_target,
                "f_live": float(internal.f_live),
                "f_live_target": float(search.f_live),
                "n_like": n_like,
                "n_like_max": None
                if n_like_max is None or n_like_max == float("inf")
                else float(n_like_max),
            }
        except (AttributeError, TypeError, ValueError) as exc:
            return {
                "observable": False,
                "met": None,
                "reason": f"{TERMINATION_NOT_EXPOSED}: {type(exc).__name__}",
            }
    return {"observable": False, "met": None, "reason": TERMINATION_NOT_EXPOSED}


def build_search(af, spec: SamplerSpec, settings: dict, *, path_prefix, name, seed: int):
    kwargs = dict(settings)
    if spec.seed_kwarg is not None:
        kwargs[spec.seed_kwarg] = seed
    return getattr(af, spec.cls)(path_prefix=path_prefix, name=name, **kwargs)


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def run_search(
    sampler: str = "nautilus",
    dataset_class: str = "gaussian_x3_blend",
    model_type: str = "gaussian_x3",
    argv=None,
) -> int:
    """Fit ``model_type`` on ``dataset_class`` with ``sampler`` for one config and seed.

    Returns the process exit code: 0 on a written row (whatever its verdict), 2 on a
    refused combination (unknown sampler, a deferred search, a JAX leg of a search that
    is not JAX-native, a ``--dataset`` that disagrees with the leaf).
    """
    cli = parse_inference_cli(argv, default_dataset=dataset_class)
    if sampler not in SAMPLERS:
        print(
            f"ERROR: unknown sampler {sampler!r}; registered: {sorted(SAMPLERS)}", file=sys.stderr
        )
        return 2
    if model_type not in MODEL_TYPES:
        print(f"ERROR: unknown model_type {model_type!r}", file=sys.stderr)
        return 2
    if cli.dataset != dataset_class:
        print(f"ERROR: this leaf fits {dataset_class}, not {cli.dataset}", file=sys.stderr)
        return 2
    spec = SAMPLERS[sampler]
    if spec.status == "deferred":
        print(f"ERROR: {sampler} is deferred: {spec.note}", file=sys.stderr)
        return 2
    if cli.backend == "jax_cpu" and not spec.jax_native:
        print(f"ERROR: {sampler} has no JAX-native leg", file=sys.stderr)
        return 2
    if cli.backend == "numpy" and not spec.numpy_supported:
        print(f"ERROR: {sampler} is JAX-only in PyAutoFit; run a jax_cpu config", file=sys.stderr)
        return 2
    settings_name = cli.settings or spec.default
    if settings_name not in spec.settings:
        print(
            f"ERROR: {sampler} has no settings {settings_name!r}; choose from "
            f"{sorted(spec.settings)}",
            file=sys.stderr,
        )
        return 2
    settings = spec.settings[settings_name]
    env = set_backend_env(cli.backend, cli.precision, cli.cores, cli.compile_cache)

    # --- imports: everything below here sees the backend environment above ---
    import autofit as af
    from autonerves.test_mode import is_test_mode, with_test_mode_segment

    if os.environ.get("AUTOFIT_INFERENCE_SMOKE") == "1":
        print(f"[smoke] {sampler}/{dataset_class}: imports + module setup OK; exiting.")
        return 0

    import numpy as np
    from autofit.non_linear.fitness import Fitness
    from models import gaussian_x3 as gx3

    from searches import _posterior as post
    from searches import _protocol as protocol
    from searches._metrics import TIME_BASIS_ESTIMATED, TIME_BASIS_OBSERVED, MLTracker

    use_jax = cli.backend == "jax_cpu"
    seed = cli.seed
    root = _ROOT
    np.random.seed(seed)

    dataset_dir = auto_simulate_if_missing(dataset_class, cli.data_seed, root)
    data, noise_map, truth_record = gx3.load_dataset(dataset_dir)
    data_seed = int(truth_record["data_seed"])
    truths = gx3.truth_dict(truth_record)

    output_root = cli.output_dir or Path(os.environ.get("PYAUTO_OUTPUT_DIR") or (root / "output"))
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    af.conf.instance.push(new_path=root / "config", output_path=output_root)

    reference = protocol.load_json(protocol.reference_file(root, dataset_class, cli.backend))
    offsets = protocol.load_json(protocol.offsets_file(root))
    target_log_l = None
    if reference and reference.get("max_log_likelihood") is not None:
        target_log_l = float(reference["max_log_likelihood"])

    model = gx3.build_model()
    keys = gx3.parameter_keys(model)
    tracker = MLTracker()
    analysis = gx3.TrackedAnalysisGaussianX3(
        data=data, noise_map=noise_map, use_jax=use_jax, tracker=None if use_jax else tracker
    )

    compile_cache = cli.compile_cache if use_jax else None
    config_id = config_segment(cli.config_name, compile_cache)
    rid = run_id(model_type, dataset_class, data_seed, sampler, settings_name, config_id, seed)
    path_prefix = output_path_prefix(
        dataset_class, data_seed, sampler, settings_name, config_id, seed
    )
    search_name = sampler
    search = build_search(
        af, spec, settings, path_prefix=str(path_prefix), name=search_name, seed=seed
    )
    search_root = with_test_mode_segment(output_root) / path_prefix

    print("=" * 78)
    print(f"{rid}")
    print("=" * 78)
    print(f"  backend {cli.backend} precision {cli.precision} cores {cli.cores} env {env}")
    print(f"  output {output_root}  test mode {is_test_mode()}")

    timing: dict = {}
    log_likelihood_at_truth = None

    def admission_bar() -> None:
        n_batch = int(settings.get("n_batch", getattr(search, "n_batch", 1) or 1))
        fitness = Fitness(
            model=model,
            analysis=gx3.AnalysisGaussianX3(data=data, noise_map=noise_map, use_jax=use_jax),
            paths=None,
            fom_is_log_likelihood=True,
            resample_figure_of_merit=RESAMPLE_SENTINEL,
            use_jax_vmap=use_jax and spec.vectorised,
            batch_size=n_batch,
        )
        vector = np.asarray(admission_vector(keys, truths), dtype=float)
        timing["vector"] = vector.tolist()
        timing["vector_source"] = "truth (satisfies the assertions; never the prior medians)"
        truth = vector.tolist()
        if use_jax:
            import jax

            single = jax.jit(fitness.call)
            start = time.perf_counter()
            jax.block_until_ready(single(vector))
            timing["compile_s"] = time.perf_counter() - start
            timing["single"] = time_calls(single, vector, block=jax.block_until_ready)
            if spec.vectorised:
                batch = np.tile(vector, (n_batch, 1))
                start = time.perf_counter()
                jax.block_until_ready(fitness._vmap(batch))
                timing["compile_batched_s"] = time.perf_counter() - start
                timing["batched"] = time_calls(fitness._vmap, batch, block=jax.block_until_ready)
                timing["n_batch"] = n_batch
            truth_value = float(single(np.asarray(truth, dtype=float)))
        else:
            timing["single"] = time_calls(fitness.call, vector)
            truth_value = float(fitness.call(truth))
        if truth_value <= RESAMPLE_SENTINEL / 10:
            raise RuntimeError("admission vector hit the resample sentinel: timing is invalid")
        nonlocal log_likelihood_at_truth
        log_likelihood_at_truth = truth_value

    status = "complete"
    total_wall_s = None
    resumed = any((search_root / search_name).glob("*/files/samples_info.json"))
    summary: dict = {}
    log_l_history: list[float] = []
    ppc_chi2 = None
    log_evidence = None
    termination = None
    try:
        if is_test_mode():
            timing["note"] = "admission-bar timing skipped under PYAUTO_TEST_MODE"
        else:
            admission_bar()
        tracker.t0 = time.time()
        start = time.perf_counter()
        result = search.fit(model=model, analysis=analysis)
        total_wall_s = time.perf_counter() - start

        samples = result.samples
        matrix = np.asarray(samples.parameter_lists, dtype=float)
        weights = np.asarray(samples.weight_list, dtype=float)
        if not np.any(weights > 0):
            weights = np.ones(len(matrix))
        log_l_history = [float(v) for v in samples.log_likelihood_list]
        summary = post.summarise_samples(
            matrix,
            weights,
            keys,
            np.asarray(samples.log_likelihood_list, dtype=float),
            np.asarray(samples.log_posterior_list, dtype=float),
        )
        try:
            log_evidence = float(samples.log_evidence)
        except Exception:
            log_evidence = None
        if spec.family == "nested":
            try:
                internal = result.search_internal
            except Exception:
                internal = None
            termination = nested_termination(sampler, search, internal)
        draws = post.relabel_matrix(post.equal_weight_draws(matrix, weights, PPC_DRAWS, seed), keys)
        xvalues = np.arange(data.shape[0], dtype=float)
        curves = np.array(
            [
                gx3.model_data_from(
                    model.instance_from_vector(vector=list(row), ignore_assertions=True),
                    xvalues,
                )
                for row in draws
            ]
        )
        ppc_chi2 = float(np.sum(((data - np.median(curves, axis=0)) / noise_map) ** 2))
    except BaseException as exc:
        status = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            output_path = Path(search.paths.output_path)
        except Exception:
            output_path = search_root / search_name
        info = _read_json(output_path / "files" / "samples_info.json") or {}
        wall_s = _as_float(info.get("time"))  # samples_info writes it as a string
        likelihood_evals = info.get("total_samples")
        # --- time to the shared target (protocol §8) ------------------------
        evals_to_target = time_to_target = None
        evals_note = None
        if use_jax:
            basis = TIME_BASIS_ESTIMATED
            evals_note = EVALS_TO_TARGET_JAX_NOTE
            # Interpolated from the eval index over the sampler's clock: an estimate.
            # from_log_l_history targets its own max, so pass the history clipped to the
            # shared target instead (first index reaching ref_max_logL - 1).
            if target_log_l is not None and log_l_history and wall_s:
                for index, value in enumerate(log_l_history):
                    if value >= target_log_l - 1.0:
                        evals_to_target = index + 1
                        time_to_target = float(wall_s) * evals_to_target / len(log_l_history)
                        break
        else:
            basis = TIME_BASIS_OBSERVED
            if target_log_l is not None:
                evals_to_target, time_to_target = tracker.finalise(
                    max_log_l=target_log_l, tolerance=1.0
                )
        single = timing.get("single") or {}
        batched = timing.get("batched") or {}
        if batched.get("median_s") is not None:
            per_call_s = batched["median_s"] / timing["n_batch"]
            per_call_basis = "batched_vmap"
        else:
            per_call_s = single.get("median_s")
            per_call_basis = "jit_single" if use_jax else "numpy_single"
        likelihood_s = (
            per_call_s * likelihood_evals
            if per_call_s is not None and likelihood_evals is not None
            else None
        )
        posterior = summary.get("posterior", {})
        ess = summary.get("ess_kish")
        row = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": protocol.PROTOCOL_ID,
            "run_id": rid,
            "target": f"{dataset_class}/data_seed{data_seed}/seed{seed}",
            "pilot": True,
            "sampler": sampler,
            "sampler_class": spec.cls,
            "task": spec.task,
            "family": spec.family,
            "settings_name": settings_name,
            "settings": settings,
            "model": model_type,
            "model_description": gx3.MODEL_DESCRIPTION,
            "priors": gx3.PRIORS_RECORD,
            "assertion_mechanism": "xp_where_penalty" if use_jax else "raise_resample",
            "config_name": cli.config_name,
            "where": cli.where,
            "backend": cli.backend,
            "precision": cli.precision,
            "dataset": dataset_class,
            "dataset_class": dataset_class,
            "data_seed": data_seed,
            "seed": seed,
            "seed_effective": spec.seed_kwarg is not None,
            "version": af.__version__,
            "library_revisions": (revisions := library_revisions()),
            "pyautofit_commit": revisions.get("autofit"),
            "device": device_info_dict(cli.backend),
            "host": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "cores": cli.cores,
            "use_jax": use_jax,
            "test_mode": bool(is_test_mode()),
            "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": status,
            "completed": (output_path / ".completed").exists(),
            "resumed": bool(resumed),
            "output_path": _relative(output_path, output_root),
            "free_parameters": model.prior_count,
            # --- clocks ---------------------------------------------------------
            "wall_s": wall_s,
            "total_wall_s": total_wall_s,
            "compile_s": timing.get("compile_s"),
            "compile_cache": compile_cache,
            "config_id": config_id,
            "likelihood_evals": likelihood_evals,
            # --- results --------------------------------------------------------
            "log_evidence": log_evidence,
            "log_evidence_err": None,
            "log_evidence_err_note": LOG_EVIDENCE_ERR_NOTE,
            "max_log_likelihood": summary.get("max_log_likelihood"),
            "max_log_likelihood_vector": summary.get("max_log_likelihood_vector"),
            "max_log_posterior": summary.get("max_log_posterior"),
            "max_log_posterior_vector": summary.get("max_log_posterior_vector"),
            "posterior": posterior,
            "posterior_relabelled": summary.get("posterior_relabelled", {}),
            "modes": summary.get("modes"),
            "modes_found": summary.get("modes_found"),
            "label_permutations_found": summary.get("label_permutations_found"),
            "n_samples": summary.get("n_samples"),
            "ess_kish": ess,
            "ess_per_s": ess / wall_s if ess is not None and wall_s else None,
            "termination": termination,
            "ppc_chi2": ppc_chi2,
            "truths": truths,
            "truth_delta_sigma": truth_delta_sigma_from(
                summary.get("posterior_relabelled", {}), truths
            ),
            "log_likelihood_at_truth": log_likelihood_at_truth,
            # --- time to the shared target -------------------------------------
            "target_log_likelihood": target_log_l - 1.0 if target_log_l is not None else None,
            "evals_to_target": evals_to_target,
            "time_to_target_s": time_to_target,
            "time_to_target_basis": basis,
            "evals_to_target_basis": basis,
            "evals_to_target_note": evals_note,
            # --- the admission bar (estimates) ---------------------------------
            "per_call_s": per_call_s,
            "per_call_basis": per_call_basis if per_call_s is not None else None,
            "per_call_single_s": single.get("median_s"),
            "likelihood_s": likelihood_s,
            "overhead_s": wall_s - likelihood_s if likelihood_s is not None and wall_s else None,
            "likelihood_share": likelihood_share(per_call_s, likelihood_evals, wall_s),
            "likelihood_share_basis": "estimated",
            "timing": timing,
        }
        row["scientific"] = protocol.verdict(row, reference, offsets)
        json_path = results_path(
            cli.results_root or root,
            dataset_class,
            data_seed,
            sampler,
            settings_name,
            config_id,
            seed,
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(row, indent=2) + "\n")
        print(
            f"\n  {status}: wall {wall_s}s, evals {likelihood_evals}, log Z {log_evidence}, "
            f"modes {summary.get('modes_found')}"
        )
        print(
            f"  verdict: acceptance {row['scientific']['acceptance']} "
            f"({row['scientific']['acceptance_reason']}); convergence "
            f"{row['scientific']['convergence']}"
        )
        print(f"  Result row: {json_path}")
    return 0


def _relative(path: Path, root: Path) -> str | None:
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return None

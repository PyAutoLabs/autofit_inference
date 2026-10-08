"""The MAP reference that point/MAP searches are judged against (protocol §6, §4(a)).

Two routes to the maximum of the **log posterior** (PyAutoFit's MLE searches add the log
prior to the log likelihood; a LogUniform prior contributes ``−ln x``), polished, and the
higher one kept:

1. a long ``MultiStartAdam`` from the prior, then an ``LBFGS`` polish from its best point.
   ``MultiStartAdam`` is JAX-native only (PyAutoFit refuses a numpy Analysis), so this
   route always runs its MultiStart on a JAX Analysis; the polish and every recorded
   value use the requested backend's Analysis;
2. an ``LBFGS`` polish from the reference posterior's best (max log posterior) sample.

Writes ``results/reference/<dataset>/<backend>/map_reference.json``. Both
``log_likelihood`` and ``log_posterior`` of the MAP point are recorded, never
interchanged.

    python scripts/misc/reference/run_map_reference.py --config-name local_numpy_fp64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _autofit_inference_cli import (  # noqa: E402
    DATASETS,
    auto_simulate_if_missing,
    device_info_dict,
    library_revisions,
    set_backend_env,
    validate_cli,
)

MULTISTART = {"n_starts": 256, "n_steps": 2000, "seed": 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config-name", default="local_numpy_fp64")
    parser.add_argument("--dataset", choices=DATASETS, default="gaussian_x3_blend")
    args = parser.parse_args(argv)
    where, backend, precision = validate_cli(args.config_name, None, None)
    # The MultiStart route needs JAX even on the numpy leg, so PYAUTO_DISABLE_JAX is not
    # exported here; the numpy leg's Analyses are built with use_jax=False instead.
    set_backend_env("jax_cpu", precision, cores=1)

    import autofit as af
    import numpy as np
    from autofit.non_linear.fitness import Fitness
    from models import gaussian_x3 as gx3

    data, noise_map, truth_record = gx3.load_dataset(auto_simulate_if_missing(args.dataset))
    model = gx3.build_model(dataset=args.dataset)
    keys = gx3.parameter_keys(model)
    analysis = gx3.AnalysisGaussianX3(data=data, noise_map=noise_map, use_jax=backend == "jax_cpu")
    output_root = (_ROOT / "output").resolve()
    af.conf.instance.push(new_path=_ROOT / "config", output_path=output_root)
    prefix = Path("reference") / args.dataset / args.config_name / "map"

    reference_dir = _ROOT / "results" / "reference" / args.dataset / backend
    runs = [json.loads(p.read_text()) for p in sorted(reference_dir.glob("*_seed*.json"))]
    if not runs:
        print(f"ERROR: no reference runs under {reference_dir}", file=sys.stderr)
        return 2
    best_run = max(runs, key=lambda r: r["max_log_posterior"])

    fitness = Fitness(
        model=model,
        analysis=gx3.AnalysisGaussianX3(data=data, noise_map=noise_map, use_jax=False),
        paths=None,
        fom_is_log_likelihood=False,
        resample_figure_of_merit=-1.0e99,
    )
    likelihood = Fitness(
        model=model,
        analysis=gx3.AnalysisGaussianX3(data=data, noise_map=noise_map, use_jax=False),
        paths=None,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )

    def evaluate(vector) -> dict:
        vector = [float(v) for v in vector]
        return {
            "vector": vector,
            "log_posterior": float(fitness.call(vector)),
            "log_likelihood": float(likelihood.call(vector)),
        }

    def polish(start, name) -> dict:
        initializer = af.InitializerParamStartPoints(
            parameter_dict=dict(zip(model.priors_ordered_by_id, start))
        )
        search = af.LBFGS(path_prefix=prefix, name=name, initializer=initializer)
        result = search.fit(model=model, analysis=analysis)
        samples = result.samples
        return evaluate(samples.parameter_lists[samples.max_log_posterior_index])

    routes = {}
    start = time.perf_counter()
    multistart = af.MultiStartAdam(path_prefix=prefix, name="multistart_adam", **MULTISTART)
    jax_analysis = gx3.AnalysisGaussianX3(data=data, noise_map=noise_map, use_jax=True)
    ms_result = multistart.fit(model=model, analysis=jax_analysis)
    ms_best = ms_result.samples.parameter_lists[ms_result.samples.max_log_posterior_index]
    routes["multistart"] = evaluate(ms_best)
    routes["multistart_polished"] = polish(ms_best, "lbfgs_from_multistart")
    routes["reference_best_sample"] = evaluate(best_run["max_log_posterior_vector"])
    routes["reference_polished"] = polish(
        best_run["max_log_posterior_vector"], "lbfgs_from_reference"
    )
    wall = time.perf_counter() - start
    best_route = max(routes, key=lambda k: routes[k]["log_posterior"])
    payload = {
        "schema_version": 1,
        "kind": "map_reference",
        "dataset": args.dataset,
        "backend": backend,
        "config_name": args.config_name,
        "where": where,
        "multistart_settings": MULTISTART,
        "multistart_backend": "jax_cpu",
        "reference_best_run": f"{best_run['sampler']}_seed{best_run['seed']}",
        "routes": routes,
        "map": {**routes[best_route], "route": best_route, "keys": keys},
        "route_spread_log_posterior": max(r["log_posterior"] for r in routes.values())
        - min(routes[k]["log_posterior"] for k in ("multistart_polished", "reference_polished")),
        "wall_s": wall,
        "version": af.__version__,
        "library_revisions": library_revisions(),
        "device": device_info_dict(backend),
        "host": os.uname().nodename,
        "truth_log_posterior": evaluate([gx3.truth_dict(truth_record)[k] for k in keys])[
            "log_posterior"
        ],
    }
    out = reference_dir / "map_reference.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    for key, route in routes.items():
        print(f"  {key:24s} logP {route['log_posterior']:.4f}  logL {route['log_likelihood']:.4f}")
    print(f"  MAP = {best_route}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

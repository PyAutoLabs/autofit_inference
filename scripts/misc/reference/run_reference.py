"""One long reference run: the posterior every wave-1 row is judged against.

``protocol_gaussian_x3.md`` §Reference pre-registers, **per backend** (numpy and JAX),
three long ``Nautilus`` (``n_live=2000``) and three long ``DynestyStatic``
(``nlive=1000``) runs, seeds 0–2. This script runs one of them and writes
``results/reference/<dataset>/<backend>/<sampler>_seed<n>.json``: the evidence, the
labelled and relabelled marginals, the modes, the maximum log likelihood and maximum
log posterior samples (never interchanged), the posterior-predictive χ² and the
provenance. ``build_reference.py`` combines the six into ``reference.json``.

DynestyStatic takes no seed through PyAutoFit (``DynestyStatic`` has no ``seed``
argument and does not forward ``rstate``), so its ``--seed`` only labels the run and
seeds NumPy's global generator; the row says ``seed_effective: false``. The
search-level seed is phase A4 of the search-extensibility epic.

Usage::

    python scripts/misc/reference/run_reference.py --sampler nautilus --seed 0 \
        --config-name local_numpy_fp64
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

REFERENCE_SETTINGS = {
    "nautilus": {"n_live": 2000},
    "dynesty_static": {"nlive": 1000},
}

#: Equal-weight draws for the posterior-predictive check.
PPC_DRAWS = 500


def reference_path(root: Path, dataset: str, backend: str, sampler: str, seed: int) -> Path:
    return root / "results" / "reference" / dataset / backend / f"{sampler}_seed{seed}.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sampler", choices=sorted(REFERENCE_SETTINGS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config-name", default="local_numpy_fp64")
    parser.add_argument("--dataset", choices=DATASETS, default="gaussian_x3_blend")
    args = parser.parse_args(argv)

    where, backend, precision = validate_cli(args.config_name, None, None)
    env = set_backend_env(backend, precision, cores=1)

    import autofit as af
    import numpy as np
    from autonerves.test_mode import is_test_mode
    from models import gaussian_x3 as gx3
    from searches import _posterior as post

    np.random.seed(args.seed)
    dataset_dir = auto_simulate_if_missing(args.dataset)
    data, noise_map, truth_record = gx3.load_dataset(dataset_dir)
    model = gx3.build_model()
    keys = gx3.parameter_keys(model)
    assert tuple(keys) == gx3.PARAMETER_KEYS, keys
    analysis = gx3.AnalysisGaussianX3(data=data, noise_map=noise_map, use_jax=backend == "jax_cpu")

    output_root = (_ROOT / "output").resolve()
    af.conf.instance.push(new_path=_ROOT / "config", output_path=output_root)
    path_prefix = Path("reference") / args.dataset / args.config_name
    name = f"{args.sampler}_seed{args.seed}"
    settings = REFERENCE_SETTINGS[args.sampler]
    if args.sampler == "nautilus":
        search = af.Nautilus(
            path_prefix=path_prefix,
            name=name,
            n_live=2000,
            seed=args.seed,
            number_of_cores=1,
        )
        seed_effective = True
    else:
        search = af.DynestyStatic(
            path_prefix=path_prefix,
            name=name,
            nlive=1000,
            number_of_cores=1,
        )
        seed_effective = False

    print(f"reference {args.sampler} seed {args.seed} {args.config_name} {args.dataset} env={env}")
    start = time.perf_counter()
    result = search.fit(model=model, analysis=analysis)
    total_wall_s = time.perf_counter() - start

    samples = result.samples
    matrix = np.asarray(samples.parameter_lists, dtype=float)
    weights = np.asarray(samples.weight_list, dtype=float)
    log_l = np.asarray(samples.log_likelihood_list, dtype=float)
    log_p = np.asarray(samples.log_posterior_list, dtype=float)
    summary = post.summarise_samples(matrix, weights, keys, log_l, log_p)

    draws = post.equal_weight_draws(matrix, weights, PPC_DRAWS, seed=args.seed)
    xvalues = np.arange(data.shape[0], dtype=float)
    curves = np.array(
        [
            gx3.model_data_from(model.instance_from_vector(vector=list(row)), xvalues)
            for row in post.relabel_matrix(draws, keys)
        ]
    )
    ppc_curve = np.median(curves, axis=0)
    ppc_chi2 = float(np.sum(((data - ppc_curve) / noise_map) ** 2))

    info_path = Path(search.paths.output_path) / "files" / "samples_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    payload = {
        "schema_version": 1,
        "kind": "reference_run",
        "protocol_id": "gaussian_x3@1",
        "dataset": args.dataset,
        "data_seed": truth_record.get("data_seed"),
        "model": gx3.MODEL_ID,
        "model_description": gx3.MODEL_DESCRIPTION,
        "priors": gx3.PRIORS_RECORD,
        "assertion_mechanism": "xp_where_penalty" if backend == "jax_cpu" else "raise_resample",
        "sampler": args.sampler,
        "settings": settings,
        "seed": args.seed,
        "seed_effective": seed_effective,
        "config_name": args.config_name,
        "where": where,
        "backend": backend,
        "precision": precision,
        "log_evidence": float(samples.log_evidence),
        "likelihood_evals": info.get("total_samples"),
        "wall_s": float(info["time"]) if info.get("time") is not None else None,
        "total_wall_s": total_wall_s,
        "ppc_chi2": ppc_chi2,
        "ppc_draws": PPC_DRAWS,
        "pixels": int(data.shape[0]),
        "truths": gx3.truth_dict(truth_record),
        **summary,
        "version": af.__version__,
        "library_revisions": library_revisions(),
        "device": device_info_dict(backend),
        "host": os.uname().nodename,
        "test_mode": bool(is_test_mode()),
        "output_path": str(Path(search.paths.output_path).resolve().relative_to(output_root)),
    }
    out = reference_path(_ROOT, args.dataset, backend, args.sampler, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"  logZ {payload['log_evidence']:.3f}  maxlogL {summary['max_log_likelihood']:.3f}  "
        f"modes {summary['modes_found']}  ppc_chi2 {ppc_chi2:.1f}  wall {total_wall_s:.0f}s"
    )
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

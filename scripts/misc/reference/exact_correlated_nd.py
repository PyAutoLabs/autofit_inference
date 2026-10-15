"""Write the exact ``correlated_nd`` reference for one dataset realisation and backend.

Protocol ``wiki/project/protocol_correlated_nd.md`` §6: the family's references are
**exact by construction** (``models/correlated_nd_exact.py``) — no reference search is run.
One ``reference.json`` is written per dataset × backend × realisation under
``results/reference/<dataset>/<backend>/`` (``data_seed<d>/`` for a non-default
realisation), carrying the identity :data:`searches._protocol.REFERENCE_IDENTITY`
(dataset, backend, data_seed, assertion mechanism ``none``) so a row is judged only
against the reference of its own backend and realisation.

The numbers are backend-independent; what is per backend is the **§6 agreement test**,
run before the file is written ``status: complete``:

1. *Backend identity.* The backend's own figure of merit — PyAutoFit ``Fitness`` over the
   family's Analysis, the path every search calls (``jax.jit`` on JAX) — at the exact mode
   and at :data:`IDENTITY_POINTS` exact posterior draws equals the analytic ``ln L`` to
   :data:`IDENTITY_TOLERANCE_NAT`.
2. *Exact draws reproduce the reference.* :data:`DRAWS` iid exact posterior draws, summarised
   by the same ``_posterior`` code the runner uses: every median within
   :data:`DRAW_MEDIAN_TOLERANCE_SIGMA` reference sigma, every sigma ratio and every whitened
   sigma ratio within ``1 ± DRAW_SIGMA_TOLERANCE``, every whitened median within
   :data:`DRAW_MEDIAN_TOLERANCE_SIGMA`.
3. *Truncation.* The fraction of unconstrained exact draws inside the box agrees with the
   box mass the ln Z uses (within 5 binomial sigma, or exactly 1 when no draw leaves it).
4. *Control verdict.* A synthetic row built from the draws (posterior and evidence tasks,
   with the exact ln Z) is ``accepted`` by ``correlated_nd@1``.

A failed check writes ``status: failed`` with the reasons, so nothing is judged against it.

    python scripts/misc/reference/exact_correlated_nd.py --dataset corr_gaussian_d20 --backend numpy
    python scripts/misc/reference/exact_correlated_nd.py --all      # every dataset x backend
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(ROOT), str(ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _autofit_inference_cli import (  # noqa: E402
    CORRELATED_ND_DATASETS,
    library_revisions,
    set_backend_env,
)

BACKENDS = ("numpy", "jax_cpu")
IDENTITY_POINTS = 64
IDENTITY_TOLERANCE_NAT = 1.0e-8
DRAWS = 200_000
DRAW_SEED = 0
DRAW_MEDIAN_TOLERANCE_SIGMA = 0.02
DRAW_SIGMA_TOLERANCE = 0.02
SCHEMA_VERSION = 1


def reference_payload(dataset: str, backend: str, data_seed: int | None = None) -> dict:
    """Build (and self-test) the exact reference; imports autofit under ``backend``."""
    set_backend_env(backend, "fp64", 1)
    import numpy as np
    from autofit.non_linear.fitness import Fitness
    from models import correlated_nd as fam
    from models import correlated_nd_exact as exact
    from searches import _posterior as post
    from searches import _protocol as protocol
    from simulators import correlated_nd as sim

    use_jax = backend == "jax_cpu"
    ds = exact.load(sim.dataset_dir(dataset, data_seed, ROOT))
    log_z, how = exact.log_evidence(ds)
    marginals = exact.marginals(ds)
    mode = exact.mode(ds)
    max_log_l = exact.max_log_likelihood(ds)
    failures: list[str] = []

    # 1. Backend identity through the searches' own Fitness path.
    model = fam.build_model(ds)
    keys = fam.parameter_keys(model)
    assert tuple(keys) == ds.keys, (keys[:3], ds.keys[:3])
    fitness = Fitness(
        model=model,
        analysis=fam.AnalysisCorrelatedND(dataset=ds, use_jax=use_jax),
        paths=None,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )
    call = fitness.call
    if use_jax:
        import jax

        call = jax.jit(fitness.call)
    points = np.vstack([mode, exact.sample(ds, IDENTITY_POINTS - 1, seed=DRAW_SEED + 1)])
    backend_log_l = np.asarray([float(call(np.asarray(p, dtype=float))) for p in points])
    analytic_log_l = exact.log_likelihood(ds, points)
    identity_delta = float(np.max(np.abs(backend_log_l - analytic_log_l)))
    if not identity_delta <= IDENTITY_TOLERANCE_NAT:
        failures.append(f"backend identity: max |ln L - analytic| {identity_delta:.3g} nat")

    # 2. Exact draws through the runner's statistics.
    draws = exact.sample(ds, DRAWS, seed=DRAW_SEED)
    weights = np.ones(len(draws))
    stats = post.posterior_stats(draws, weights, keys)
    worst_median, worst_sigma = 0.0, 0.0
    for key in keys:
        ref = marginals[key]
        worst_median = max(worst_median, abs(stats[key]["median"] - ref["median"]) / ref["sigma"])
        worst_sigma = max(worst_sigma, abs(stats[key]["sigma"] / ref["sigma"] - 1.0))
    whitened = exact.whitened_stats(ds, draws, weights)
    whitened_sigma = max(
        abs(whitened["sigma_ratio_min"] - 1.0), abs(whitened["sigma_ratio_max"] - 1.0)
    )
    if worst_median > DRAW_MEDIAN_TOLERANCE_SIGMA:
        failures.append(f"draws: worst median offset {worst_median:.4f} sigma")
    if worst_sigma > DRAW_SIGMA_TOLERANCE:
        failures.append(f"draws: worst sigma ratio offset {worst_sigma:.4f}")
    if whitened["worst_median"] > DRAW_MEDIAN_TOLERANCE_SIGMA:
        failures.append(f"draws: worst whitened median {whitened['worst_median']:.4f}")
    if whitened_sigma > DRAW_SIGMA_TOLERANCE:
        failures.append(f"draws: worst whitened sigma ratio offset {whitened_sigma:.4f}")

    # 3. Truncation: unconstrained exact draws inside the box vs the box mass.
    unconstrained = _unconstrained_draws(ds, DRAWS, np)
    inside = float(
        np.mean(np.all((unconstrained >= ds.lower) & (unconstrained <= ds.upper), axis=1))
    )
    mass = how.get("box_mass", 1.0 - how["truncation_bound"])
    binomial = math.sqrt(max(mass * (1.0 - mass), 1.0 / DRAWS**2) / DRAWS)
    truncation_ok = abs(inside - mass) <= 5.0 * binomial or inside == 1.0
    if not truncation_ok:
        failures.append(f"truncation: inside fraction {inside} vs box mass {mass}")

    reference = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not failures else "failed",
        "exact": True,
        "model": fam.MODEL_ID,
        "protocol_id": protocol.CND_PROTOCOL_ID,
        "dataset": dataset,
        "backend": backend,
        "data_seed": ds.data_seed,
        "assertion_mechanism": fam.ASSERTION_MECHANISM,
        "kind": ds.kind,
        "dimension": ds.dimension,
        "log_evidence": log_z,
        "log_evidence_normalised": log_z,
        "log_evidence_offset": 0.0,
        "log_evidence_how": how,
        "log_volume": ds.log_volume,
        "information_nats": exact.information(ds, log_z),
        "max_log_likelihood": max_log_l,
        "map": {
            "log_posterior": max_log_l,
            "vector": [float(v) for v in mode],
            "method": "exact mode (inside the box); PyAutoFit's uniform prior adds 0 there",
        },
        "posterior": marginals,
        "posterior_relabelled": marginals,
        "relabelled_note": "no label symmetry: the relabelled posterior is the posterior",
        "modes": None,
        "ppc_chi2": None,
        "whitened": {"worst_median": 0.0, "sigma_ratio_min": 1.0, "sigma_ratio_max": 1.0},
        "limitations": [],
        "agreement": {
            "test": "protocol correlated_nd §6 (exact reference self-test)",
            "backend_identity": {
                "points": int(len(points)),
                "max_abs_delta_nat": identity_delta,
                "tolerance_nat": IDENTITY_TOLERANCE_NAT,
                "path": "autofit Fitness.call" + (" under jax.jit" if use_jax else ""),
            },
            "draws": {
                "n": DRAWS,
                "seed": DRAW_SEED,
                "worst_median_offset_sigma": worst_median,
                "worst_sigma_ratio_offset": worst_sigma,
                "whitened": whitened,
                "tolerance_median_sigma": DRAW_MEDIAN_TOLERANCE_SIGMA,
                "tolerance_sigma": DRAW_SIGMA_TOLERANCE,
            },
            "truncation": {
                "unconstrained_draws_inside": inside,
                "box_mass": mass,
                "ok": bool(truncation_ok),
            },
            "failures": failures,
        },
        "library_revisions": library_revisions(),
        "written_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # 4. Control verdict: the exact draws, as a row, must be accepted (posterior, evidence).
    control = {}
    for task in ("posterior", "evidence"):
        row = _control_row(reference, task, stats, whitened, log_z)
        verdict = protocol.acceptance(row, reference, None, protocol.THRESHOLDS_CND1)
        control[task] = verdict["acceptance"]
        if verdict["acceptance"] != "accepted":
            failures.append(f"control {task} verdict {verdict['acceptance']}: {verdict['reason']}")
    reference["agreement"]["control_verdicts"] = control
    if failures:
        reference["status"] = "failed"
    return reference


def _unconstrained_draws(ds, n: int, np):
    rng = np.random.default_rng([ds.dimension, ds.data_seed, 99])
    z = rng.standard_normal((n, ds.dimension))
    if ds.kind == "gaussian":
        return ds.y + (z * np.sqrt(ds.eigenvalues)) @ ds.eigenvectors.T
    theta = np.empty_like(z)
    theta[:, 0] = ds.y[0] + ds.sigma1 * z[:, 0]
    theta[:, 1:] = theta[:, [0]] ** 2 + ds.y[1:] + ds.sigma2 * z[:, 1:]
    return theta


def _control_row(reference: dict, task: str, stats: dict, whitened: dict, log_z: float) -> dict:
    return {
        "model": reference["model"],
        "task": task,
        "status": "complete",
        "completed": True,
        "sampler": "exact_draws",
        "dataset": reference["dataset"],
        "backend": reference["backend"],
        "data_seed": reference["data_seed"],
        "assertion_mechanism": reference["assertion_mechanism"],
        "posterior_relabelled": stats,
        "whitened": whitened,
        "log_evidence": log_z,
        "max_log_posterior": reference["map"]["log_posterior"],
    }


def write(dataset: str, backend: str, data_seed: int | None = None) -> Path:
    from searches import _protocol as protocol

    payload = reference_payload(dataset, backend, data_seed)
    path = protocol.reference_file(ROOT, dataset, backend, payload["data_seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"  {payload['status']}: {path.relative_to(ROOT)}  ln Z {payload['log_evidence']:.6f}")
    if payload["agreement"]["failures"]:
        print("    " + "; ".join(payload["agreement"]["failures"]))
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=CORRELATED_ND_DATASETS)
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="every dataset x backend")
    args = parser.parse_args(argv)
    if args.all:
        # One process per backend: the backend environment must be set before autofit loads.
        status = 0
        for dataset in CORRELATED_ND_DATASETS:
            for backend in BACKENDS:
                command = [sys.executable, __file__, "--dataset", dataset, "--backend", backend]
                if args.data_seed is not None:
                    command += ["--data-seed", str(args.data_seed)]
                status |= subprocess.run(command, env=dict(os.environ)).returncode
        return status
    if not (args.dataset and args.backend):
        parser.error("--dataset and --backend (or --all)")
    path = write(args.dataset, args.backend, args.data_seed)
    return 0 if json.loads(path.read_text())["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

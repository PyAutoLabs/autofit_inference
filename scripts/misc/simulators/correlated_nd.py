"""
Simulator: the ``correlated_nd`` datasets
=========================================

The second toy-likelihood family of the search benchmark (protocol
``wiki/project/protocol_correlated_nd.md``, ``correlated_nd@1``): high-dimensional,
ill-conditioned targets whose posterior and log evidence are known **by construction**,
so no reference search is ever run.

Four datasets, one structure each (fixed by its ``structure_seed``); a ``data_seed``
draws one noise realisation of the observation vector ``y``:

- ``corr_gaussian_d20``, ``corr_gaussian_d50``, ``corr_gaussian_d100`` — a correlated
  Gaussian likelihood ``L(theta) = N(y; theta, Sigma)`` in ``D`` = 20, 50, 100 dimensions.
  ``Sigma = Q diag(lambda) Q^T`` with ``Q`` a seeded Haar-random orthogonal matrix (QR of
  a standard-normal matrix, signs fixed by ``diag(R) > 0``) and ``lambda`` log-spaced
  from 0.1 to 100, so the condition number is exactly ``10^3``. The truth ``theta*`` is
  seeded in ``U(-10, 10)`` per axis and ``y = theta* + Q (sqrt(lambda) * z)``,
  ``z ~ N(0, I)``. Under the uniform prior box the posterior is ``N(y, Sigma)``.
- ``rosenbrock_d20`` — the hybrid Rosenbrock "banana" of Pagani, Wiegand & Nadarajah
  (2022, Scand. J. Stat. 49, 657) with ``n1 = 2`` (one shared coordinate) and
  ``n2 = 19`` blocks, ``a = 1/20``, ``b = 100/20``, ``mu = 1``: ``x1 | y ~ N(y1, 1/(2a))``
  and ``x_j | x1, y ~ N(x1^2 + y_j, 1/(2b))`` for ``j = 2..20``. The map
  ``(x1, x_j) -> (x1, x_j - x1^2)`` has unit Jacobian, so the density is normalised
  exactly and can be sampled directly. The truth is ``x1* = 1``, ``x_j* = 1``; the data
  are ``y1 = 1 + sigma1 z0`` and ``y_j = sigma2 z_j``.

The uniform prior box is fixed by the structure (never by the data): per axis it extends
12–16 marginal sigma (seeded) either side of the truth for the Gaussians; for the banana
``x1`` extends 10–12 ``sigma1`` either side of 1 and every ``x_j`` covers
``[-c sigma2, max(x1)^2 + c sigma2]`` with ``c`` seeded in 12–16. A realisation whose
data sit more than :data:`MAX_DEVIATION_SIGMA` sigma from the truth on any axis is
refused (probability below 1e-4 at D = 100), so the truncation of the posterior by the
box is bounded (Gaussians, by a union bound, below 1e-25 on the committed realisations) or integrated exactly
(banana) — the exact reference (``scripts/misc/reference/exact_correlated_nd.py``)
records which.

Files per dataset directory (``dataset/<name>/`` for the default realisation,
``dataset/<name>/data_seed<d>/`` for any other): ``data.json`` (the observation vector
``y``), ``model.json`` (the structure: eigenvalues and eigenvectors, or the banana
constants), ``priors.json`` (the box), ``truth.json`` (truth, seeds, generator).

Pure numpy and reproducible: ``np.random.default_rng([structure_seed])`` for the
structure, ``np.random.default_rng([structure_seed, data_seed])`` for the data.

Usage
-----

    python scripts/misc/simulators/correlated_nd.py                         # all four, data_seed 1
    python scripts/misc/simulators/correlated_nd.py --name corr_gaussian_d50 --data-seed 2
"""

from __future__ import annotations

import json
import math
from pathlib import Path as _Path

import numpy as np


def _repo_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autofit_inference root (ruff.toml) not found")


_REPO_ROOT = _repo_root()

FAMILY = "correlated_nd"

#: name -> (kind, dimension, structure_seed). Every dataset's default realisation is
#: data_seed 1 (:data:`DEFAULT_DATA_SEED`).
DATASETS = {
    "corr_gaussian_d20": ("gaussian", 20, 2020),
    "corr_gaussian_d50": ("gaussian", 50, 2050),
    "corr_gaussian_d100": ("gaussian", 100, 2100),
    "rosenbrock_d20": ("rosenbrock", 20, 3020),
}
DEFAULT_DATA_SEED = 1

#: Gaussian structure (protocol §2): eigenvalues log-spaced over this range, condition
#: number ``EIGEN_RANGE[1] / EIGEN_RANGE[0]`` = 10^3 exactly.
EIGEN_RANGE = (0.1, 100.0)
CONDITION_NUMBER = EIGEN_RANGE[1] / EIGEN_RANGE[0]
TRUTH_RANGE = (-10.0, 10.0)
#: Gaussian prior box: ``truth -/+ margin * marginal sigma``, margin seeded per side.
GAUSSIAN_MARGIN_SIGMA = (12.0, 16.0)

#: Hybrid Rosenbrock constants (Pagani et al. 2022): ``a``, ``b``, ``mu``, n1, n2.
ROSENBROCK_A = 1.0 / 20.0
ROSENBROCK_B = 100.0 / 20.0
ROSENBROCK_MU = 1.0
ROSENBROCK_N1 = 2
SIGMA1 = math.sqrt(1.0 / (2.0 * ROSENBROCK_A))
SIGMA2 = math.sqrt(1.0 / (2.0 * ROSENBROCK_B))
#: Banana prior box: x1 margin (in sigma1) and x_j margin (in sigma2), seeded per side.
ROSENBROCK_X1_MARGIN_SIGMA = (10.0, 12.0)
ROSENBROCK_XJ_MARGIN_SIGMA = (12.0, 16.0)

#: A realisation whose data lie further than this many sigma from the truth on any
#: (whitened) axis is refused, which bounds the box truncation (module docstring).
MAX_DEVIATION_SIGMA = 5.0

GENERATOR = (
    "structure: numpy.random.default_rng([structure_seed]); "
    "data: numpy.random.default_rng([structure_seed, data_seed]).standard_normal(D)"
)


def parameter_keys(dimension: int) -> list[str]:
    """``x000``, ``x001``, … — zero-padded so the key order is the vector order."""
    return [f"x{i:03d}" for i in range(dimension)]


def haar_orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    """A Haar-random orthogonal matrix: QR of a standard-normal matrix with the signs of
    ``diag(R)`` folded into ``Q`` (Mezzadri 2007), so it is unique given the draw."""
    q, r = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    return q * np.sign(np.diag(r))


def structure(name: str) -> dict:
    """The data-independent part of a dataset: truth, prior box, eigen-structure or the
    banana constants. Pure function of the dataset's ``structure_seed``."""
    kind, dimension, structure_seed = DATASETS[name]
    rng = np.random.default_rng([structure_seed])
    keys = parameter_keys(dimension)
    if kind == "gaussian":
        eigenvectors = haar_orthogonal(rng, dimension)
        eigenvalues = np.logspace(math.log10(EIGEN_RANGE[0]), math.log10(EIGEN_RANGE[1]), dimension)
        marginal_sigma = np.sqrt((eigenvectors**2) @ eigenvalues)
        truth = rng.uniform(*TRUTH_RANGE, dimension)
        below = rng.uniform(*GAUSSIAN_MARGIN_SIGMA, dimension)
        above = rng.uniform(*GAUSSIAN_MARGIN_SIGMA, dimension)
        lower = truth - below * marginal_sigma
        upper = truth + above * marginal_sigma
        return {
            "kind": kind,
            "dimension": dimension,
            "structure_seed": structure_seed,
            "keys": keys,
            "truth": truth,
            "lower": lower,
            "upper": upper,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "marginal_sigma": marginal_sigma,
        }
    if kind == "rosenbrock":
        truth = np.full(dimension, ROSENBROCK_MU**2)
        truth[0] = ROSENBROCK_MU
        x1_below, x1_above = rng.uniform(*ROSENBROCK_X1_MARGIN_SIGMA, 2)
        x1_lower = ROSENBROCK_MU - x1_below * SIGMA1
        x1_upper = ROSENBROCK_MU + x1_above * SIGMA1
        x1_sq_max = max(x1_lower**2, x1_upper**2)
        xj_below = rng.uniform(*ROSENBROCK_XJ_MARGIN_SIGMA, dimension - 1)
        xj_above = rng.uniform(*ROSENBROCK_XJ_MARGIN_SIGMA, dimension - 1)
        lower = np.concatenate([[x1_lower], -xj_below * SIGMA2])
        upper = np.concatenate([[x1_upper], x1_sq_max + xj_above * SIGMA2])
        return {
            "kind": kind,
            "dimension": dimension,
            "structure_seed": structure_seed,
            "keys": keys,
            "truth": truth,
            "lower": lower,
            "upper": upper,
            "a": ROSENBROCK_A,
            "b": ROSENBROCK_B,
            "mu": ROSENBROCK_MU,
            "n1": ROSENBROCK_N1,
            "n2": dimension - 1,
            "sigma1": SIGMA1,
            "sigma2": SIGMA2,
        }
    raise ValueError(f"unknown kind {kind!r}")


class RefusedRealisation(ValueError):
    """A data realisation outside :data:`MAX_DEVIATION_SIGMA` (module docstring)."""


def simulate_data(name: str, data_seed: int, struct: dict | None = None) -> np.ndarray:
    """The observation vector ``y`` — pure and reproducible from the two seeds."""
    struct = structure(name) if struct is None else struct
    dimension = struct["dimension"]
    rng = np.random.default_rng([struct["structure_seed"], int(data_seed)])
    z = rng.standard_normal(dimension)
    if np.max(np.abs(z)) > MAX_DEVIATION_SIGMA:
        raise RefusedRealisation(
            f"{name} data_seed {data_seed}: a whitened deviation {np.max(np.abs(z)):.2f} "
            f"exceeds {MAX_DEVIATION_SIGMA} sigma (protocol §2); choose another data_seed"
        )
    if struct["kind"] == "gaussian":
        return struct["truth"] + struct["eigenvectors"] @ (np.sqrt(struct["eigenvalues"]) * z)
    y = np.empty(dimension)
    y[0] = struct["mu"] + struct["sigma1"] * z[0]
    y[1:] = struct["sigma2"] * z[1:]
    return y


def dataset_dir(name: str, data_seed: int | None = None, root: _Path | None = None) -> _Path:
    root = root if root is not None else _REPO_ROOT
    base = root / "dataset" / name
    if data_seed is None or int(data_seed) == DEFAULT_DATA_SEED:
        return base
    return base / f"data_seed{int(data_seed)}"


def _floats(values) -> list[float]:
    return [float(v) for v in np.asarray(values).ravel()]


def model_json(struct: dict) -> dict:
    """The structure as committed (``model.json``)."""
    out = {
        "family": FAMILY,
        "kind": struct["kind"],
        "dimension": struct["dimension"],
        "structure_seed": struct["structure_seed"],
    }
    if struct["kind"] == "gaussian":
        out["eigenvalues"] = _floats(struct["eigenvalues"])
        out["eigenvectors"] = [_floats(row) for row in struct["eigenvectors"]]
        out["condition_number"] = float(
            np.max(struct["eigenvalues"]) / np.min(struct["eigenvalues"])
        )
        out["marginal_sigma"] = _floats(struct["marginal_sigma"])
        out["note"] = "Sigma = eigenvectors @ diag(eigenvalues) @ eigenvectors.T (columns)"
    else:
        for field in ("a", "b", "mu", "n1", "n2", "sigma1", "sigma2"):
            out[field] = struct[field]
        out["note"] = (
            "hybrid Rosenbrock (Pagani et al. 2022), n1 = 2: x1 ~ N(y1, sigma1^2), "
            "x_j | x1 ~ N(x1^2 + y_j, sigma2^2), j = 2..D"
        )
    return out


def simulate(name: str, data_seed: int | None = None, output_root: _Path | None = None) -> _Path:
    """Write one dataset realisation; returns its directory."""
    seed = DEFAULT_DATA_SEED if data_seed is None else int(data_seed)
    struct = structure(name)
    y = simulate_data(name, seed, struct)
    path = dataset_dir(name, seed, output_root)
    path.mkdir(parents=True, exist_ok=True)

    def dump(filename, payload):
        (path / filename).write_text(json.dumps(payload, indent=1) + "\n")

    keys = struct["keys"]
    dump("data.json", _floats(y))
    dump("model.json", model_json(struct))
    dump(
        "priors.json",
        {
            "kind": "uniform",
            "lower": dict(zip(keys, _floats(struct["lower"]))),
            "upper": dict(zip(keys, _floats(struct["upper"]))),
        },
    )
    dump(
        "truth.json",
        {
            "dataset": name,
            "family": FAMILY,
            "kind": struct["kind"],
            "dimension": struct["dimension"],
            "data_seed": seed,
            "structure_seed": struct["structure_seed"],
            "generator": GENERATOR,
            "max_deviation_sigma": MAX_DEVIATION_SIGMA,
            "parameters": dict(zip(keys, _floats(struct["truth"]))),
        },
    )
    print(f"  wrote {path}")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", choices=sorted(DATASETS), default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--output-root", type=_Path, default=None)
    args = parser.parse_args()
    names = [args.name] if args.name else list(DATASETS)
    for dataset_name in names:
        simulate(dataset_name, data_seed=args.data_seed, output_root=args.output_root)

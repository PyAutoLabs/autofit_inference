"""Exact posterior quantities of the ``correlated_nd`` family — numpy only, no autofit.

Every number the ``correlated_nd@1`` verdicts rest on (protocol
``wiki/project/protocol_correlated_nd.md`` §2, §6) is computed here from a committed
dataset directory, by construction rather than by a reference search:

- **ln Z.** The likelihood is a normalised density in ``theta`` and the prior is uniform
  on the box of volume ``V``, so ``Z = mass / V`` with ``mass`` the posterior mass inside
  the box. Gaussians: ``mass = 1 - eps`` with ``eps`` bounded by a union bound
  (``truncation_bound``; below 1e-25 on every committed realisation) and
  ``ln Z = -ln V``. Banana: ``mass`` is integrated exactly (1-D Gauss–Legendre over
  ``x1``; every ``x_j`` factor is a closed-form normal CDF) and ``ln Z = ln mass - ln V``.
- **Marginals** in the convention of ``searches/_posterior.marginal`` (weighted quantiles
  0.025 / 0.16 / 0.5 / 0.84 / 0.975, ``sigma = (q84 - q16) / 2``, plus the mean). Gaussian:
  ``y_i + sqrt(Sigma_ii) Phi^-1(q)`` (the box truncation moves them by less than its bound).
  Banana: ``x1`` is a truncated normal (closed form); each ``x_j`` CDF is a 1-D integral,
  inverted by bisection to 1e-10 of ``sigma2``.
- **MAP / max ln L.** The mode is ``y`` (Gaussian) or ``(y1, y1^2 + y_j)`` (banana), inside
  the box by construction; there ``ln L`` is the normalisation alone. PyAutoFit's uniform
  prior contributes 0 inside its box, so the MAP log posterior equals it.
- **Whitening** (protocol §4(b)3): the exact map to ``N(0, I)`` —
  ``u = diag(lambda)^-1/2 Q^T (theta - y)`` (Gaussian) or
  ``u1 = (x1 - y1) / sigma1``, ``u_j = (x_j - x1^2 - y_j) / sigma2`` (banana, unit
  Jacobian).
- **Exact draws** for the §6 self-test and the control rows: direct sampling, with a draw
  outside the box rejected.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

#: The quantiles ``searches/_posterior.marginal`` reports, in its order.
QUANTILES = (0.025, 0.16, 0.5, 0.84, 0.975)
_STD = NormalDist()
#: ``(Phi^-1(0.84) - Phi^-1(0.16)) / 2``: the protocol's sigma of a unit normal (0.9945).
UNIT_SIGMA = (_STD.inv_cdf(0.84) - _STD.inv_cdf(0.16)) / 2.0

#: Composite Gauss–Legendre over x1 for the banana integrals: panels × nodes per panel,
#: over ``y1 -/+ X1_SPAN_SIGMA sigma1`` clipped to the box (outside it the x1 factor is
#: below 1e-40 of its peak).
X1_PANELS = 4000
X1_NODES = 8
X1_SPAN_SIGMA = 14.0


@dataclass(frozen=True)
class Dataset:
    """One committed realisation."""

    name: str
    kind: str
    dimension: int
    data_seed: int
    keys: tuple[str, ...]
    y: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    truth: np.ndarray
    structure_seed: int = 0
    eigenvalues: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    sigma1: float | None = None
    sigma2: float | None = None

    @property
    def log_volume(self) -> float:
        return float(np.sum(np.log(self.upper - self.lower)))


def load(dataset_path: Path) -> Dataset:
    path = Path(dataset_path)
    data = json.loads((path / "data.json").read_text())
    model = json.loads((path / "model.json").read_text())
    priors = json.loads((path / "priors.json").read_text())
    truth = json.loads((path / "truth.json").read_text())
    keys = tuple(truth["parameters"])
    common = {
        "name": truth["dataset"],
        "kind": model["kind"],
        "dimension": int(model["dimension"]),
        "data_seed": int(truth["data_seed"]),
        "keys": keys,
        "y": np.asarray(data, dtype=float),
        "lower": np.asarray([priors["lower"][k] for k in keys], dtype=float),
        "upper": np.asarray([priors["upper"][k] for k in keys], dtype=float),
        "truth": np.asarray([truth["parameters"][k] for k in keys], dtype=float),
        "structure_seed": int(truth["structure_seed"]),
    }
    if model["kind"] == "gaussian":
        return Dataset(
            **common,
            eigenvalues=np.asarray(model["eigenvalues"], dtype=float),
            eigenvectors=np.asarray(model["eigenvectors"], dtype=float),
        )
    return Dataset(**common, sigma1=float(model["sigma1"]), sigma2=float(model["sigma2"]))


# ---------------------------------------------------------------------------
# Likelihood (the analytic form every backend is checked against, §6)
# ---------------------------------------------------------------------------


def max_log_likelihood(ds: Dataset) -> float:
    """``ln L`` at the mode: the normalisation of the density."""
    if ds.kind == "gaussian":
        return float(-0.5 * np.sum(np.log(2.0 * math.pi * ds.eigenvalues)))
    return float(
        -0.5 * math.log(2.0 * math.pi * ds.sigma1**2)
        - 0.5 * (ds.dimension - 1) * math.log(2.0 * math.pi * ds.sigma2**2)
    )


def whiten(ds: Dataset, matrix) -> np.ndarray:
    """The exact map of samples (rows) to ``N(0, I)`` coordinates."""
    theta = np.atleast_2d(np.asarray(matrix, dtype=float))
    if ds.kind == "gaussian":
        return ((theta - ds.y) @ ds.eigenvectors) / np.sqrt(ds.eigenvalues)
    u = np.empty_like(theta)
    x1 = theta[:, 0]
    u[:, 0] = (x1 - ds.y[0]) / ds.sigma1
    u[:, 1:] = (theta[:, 1:] - (x1**2)[:, None] - ds.y[1:]) / ds.sigma2
    return u


def log_likelihood(ds: Dataset, matrix) -> np.ndarray:
    """``ln L`` of every row (numpy, analytic)."""
    u = whiten(ds, matrix)
    return max_log_likelihood(ds) - 0.5 * np.sum(u * u, axis=1)


def mode(ds: Dataset) -> np.ndarray:
    if ds.kind == "gaussian":
        return ds.y.copy()
    out = ds.y.copy()
    out[1:] = ds.y[0] ** 2 + ds.y[1:]
    return out


# ---------------------------------------------------------------------------
# Banana integrals (exact up to quadrature error ~1e-14)
# ---------------------------------------------------------------------------


def _ndtr(x):
    from scipy.special import ndtr

    return ndtr(x)


def _x1_nodes(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre nodes and weights for the x1 integral over the box, with the
    x1 normal density folded into the weights."""
    lo = max(ds.lower[0], ds.y[0] - X1_SPAN_SIGMA * ds.sigma1)
    hi = min(ds.upper[0], ds.y[0] + X1_SPAN_SIGMA * ds.sigma1)
    t, w = np.polynomial.legendre.leggauss(X1_NODES)
    edges = np.linspace(lo, hi, X1_PANELS + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    nodes = (mid[:, None] + half[:, None] * t[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    density = np.exp(-0.5 * ((nodes - ds.y[0]) / ds.sigma1) ** 2) / (
        ds.sigma1 * math.sqrt(2.0 * math.pi)
    )
    return nodes, weights * density


def _xj_box_mass(ds: Dataset, nodes: np.ndarray) -> np.ndarray:
    """``P(x_j in box | x1)`` for every node (rows) and every j >= 2 (columns)."""
    centre = nodes[:, None] ** 2 + ds.y[None, 1:]
    hi = _ndtr((ds.upper[None, 1:] - centre) / ds.sigma2)
    lo = _ndtr((ds.lower[None, 1:] - centre) / ds.sigma2)
    return hi - lo


def banana_mass(ds: Dataset) -> float:
    nodes, weights = _x1_nodes(ds)
    return float(np.sum(weights * np.prod(_xj_box_mass(ds, nodes), axis=1)))


def _banana_x1_marginal(ds: Dataset) -> dict:
    """x1 under the box: the x_j box factors (each 1 - O(1e-9)) are kept in the CDF."""
    nodes, weights = _x1_nodes(ds)
    w = weights * np.prod(_xj_box_mass(ds, nodes), axis=1)
    # Mid-node cumulative weight (each node holds half its own weight), so the quantile
    # carries no half-node bias.
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    q = np.interp(QUANTILES, cdf, nodes)
    mean = float(np.sum(w * nodes) / np.sum(w))
    return _marginal_dict(q, mean)


def _banana_xj_marginal(ds: Dataset, j: int, nodes, weights, box) -> dict:
    """``F_j(t) = E_x1[ P(x_j <= t | x1) * prod_{k != j} P(x_k in box | x1) ] / mass``."""
    others = np.prod(np.delete(box, j - 1, axis=1), axis=1)
    w = weights * others
    centre = nodes**2 + ds.y[j]
    lo_cdf = _ndtr((ds.lower[j] - centre) / ds.sigma2)
    total = float(np.sum(w * (box[:, j - 1])))

    def cdf(t: float) -> float:
        return float(np.sum(w * (_ndtr((t - centre) / ds.sigma2) - lo_cdf))) / total

    out = []
    for q in QUANTILES:
        a, b = ds.lower[j], ds.upper[j]
        for _ in range(200):
            m = 0.5 * (a + b)
            if cdf(m) < q:
                a = m
            else:
                b = m
            if b - a < 1e-10 * ds.sigma2:
                break
        out.append(0.5 * (a + b))
    # E[x_j | x1] = x1^2 + y_j (the truncation moves it by < 1e-9 sigma2).
    mean = float(np.sum(w * box[:, j - 1] * centre) / total)
    return _marginal_dict(np.asarray(out), mean)


def _marginal_dict(q, mean: float) -> dict:
    return {
        "median": float(q[2]),
        "sigma": float(q[3] - q[1]) / 2.0,
        "lower_1": float(q[1]),
        "upper_1": float(q[3]),
        "lower_2": float(q[0]),
        "upper_2": float(q[4]),
        "mean": float(mean),
    }


# ---------------------------------------------------------------------------
# The exact posterior summary
# ---------------------------------------------------------------------------


def gaussian_truncation_bound(ds: Dataset) -> float:
    """Union bound on the posterior mass outside the box: ``sum_i P(|theta_i - y_i| >
    distance to the nearer edge)`` with the exact marginal ``N(y_i, Sigma_ii)``."""
    sigma = np.sqrt((ds.eigenvectors**2) @ ds.eigenvalues)
    below = (ds.y - ds.lower) / sigma
    above = (ds.upper - ds.y) / sigma
    return float(sum(_upper_tail(b) + _upper_tail(a) for b, a in zip(below, above)))


def _upper_tail(x: float) -> float:
    """``P(Z > x)`` for a unit normal, via ``erfc`` (no cancellation in the far tail)."""
    return 0.5 * math.erfc(float(x) / math.sqrt(2.0))


def log_evidence(ds: Dataset) -> tuple[float, dict]:
    """``(ln Z, how)``."""
    if ds.kind == "gaussian":
        bound = gaussian_truncation_bound(ds)
        return -ds.log_volume, {
            "method": "closed form: ln Z = -ln V (normalised Gaussian likelihood, uniform box)",
            "truncation_bound": bound,
            "log_evidence_error_bound": -math.log1p(-bound) if bound < 1 else float("inf"),
        }
    mass = banana_mass(ds)
    return math.log(mass) - ds.log_volume, {
        "method": "ln Z = ln(box mass) - ln V; box mass by Gauss-Legendre over x1 "
        f"({X1_PANELS} panels x {X1_NODES} nodes) with closed-form normal CDFs in x_j",
        "box_mass": mass,
        "truncation_bound": 1.0 - mass,
    }


def marginals(ds: Dataset) -> dict:
    """``{key: marginal}`` in the ``_posterior.marginal`` convention."""
    if ds.kind == "gaussian":
        sigma = np.sqrt((ds.eigenvectors**2) @ ds.eigenvalues)
        z = np.asarray([_STD.inv_cdf(q) for q in QUANTILES])
        return {
            key: _marginal_dict(ds.y[i] + sigma[i] * z, ds.y[i]) for i, key in enumerate(ds.keys)
        }
    nodes, weights = _x1_nodes(ds)
    box = _xj_box_mass(ds, nodes)
    out = {ds.keys[0]: _banana_x1_marginal(ds)}
    for j in range(1, ds.dimension):
        out[ds.keys[j]] = _banana_xj_marginal(ds, j, nodes, weights, box)
    return out


def information(ds: Dataset, log_z: float) -> float:
    """Kullback–Leibler information ``H = E_post[ln L] - ln Z`` (nats): the prior-to-
    posterior compression a nested sampler must traverse (§7 live-point rule)."""
    return float(max_log_likelihood(ds) - 0.5 * ds.dimension - log_z)


def sample(ds: Dataset, n: int, seed: int = 0) -> np.ndarray:
    """``n`` exact iid posterior draws (rejection of the rare out-of-box draw)."""
    rng = np.random.default_rng([ds.structure_seed, ds.data_seed, int(seed)])
    out = []
    have = 0
    while have < n:
        z = rng.standard_normal((n, ds.dimension))
        if ds.kind == "gaussian":
            theta = ds.y + (z * np.sqrt(ds.eigenvalues)) @ ds.eigenvectors.T
        else:
            theta = np.empty_like(z)
            theta[:, 0] = ds.y[0] + ds.sigma1 * z[:, 0]
            theta[:, 1:] = theta[:, [0]] ** 2 + ds.y[1:] + ds.sigma2 * z[:, 1:]
        keep = np.all((theta >= ds.lower) & (theta <= ds.upper), axis=1)
        out.append(theta[keep])
        have += int(keep.sum())
    return np.concatenate(out)[:n]


def whitened_stats(ds: Dataset, matrix, weights) -> dict:
    """Protocol §4(b)3 on one weighted sample set: every whitened coordinate's median (in
    units of the unit-normal sigma) and sigma ratio. Imported lazily by the runner."""
    from searches import _posterior as post

    u = whiten(ds, matrix)
    stats = post.posterior_stats(u, weights, [f"u{i:03d}" for i in range(ds.dimension)])
    medians = {k: abs(s["median"]) / UNIT_SIGMA for k, s in stats.items()}
    ratios = {k: s["sigma"] / UNIT_SIGMA for k, s in stats.items()}
    worst_median = max(medians, key=medians.get)
    lo, hi = min(ratios, key=ratios.get), max(ratios, key=ratios.get)
    return {
        "basis": "exact whitening map (protocol correlated_nd §4(b)3)",
        "n_coordinates": ds.dimension,
        "worst_median": medians[worst_median],
        "worst_median_key": worst_median,
        "sigma_ratio_min": ratios[lo],
        "sigma_ratio_min_key": lo,
        "sigma_ratio_max": ratios[hi],
        "sigma_ratio_max_key": hi,
    }

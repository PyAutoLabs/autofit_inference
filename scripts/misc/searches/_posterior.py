"""Posterior statistics shared by the search runner, the reference runs and the tests.

Pure numpy (no autofit import), so every number a verdict rests on can be unit-tested
from arrays alone. The definitions are the ones ``wiki/project/protocol_gaussian_x3.md``
pre-registers:

- **Marginals** — weighted quantiles; ``sigma = (q84 - q16) / 2``.
- **Relabelled statistics** (convention D15(c)) — every sample's Gaussian components are
  sorted by centre before the marginals are taken. Under the ordered-centre assertions
  this is the identity; it is recorded for every row so an exchangeable or
  assertion-violating search is still comparable.
- **Label permutations** — the number of distinct centre orderings carrying at least
  :data:`MODE_MIN_WEIGHT` of the posterior weight in the *raw* samples.
- **Modes found** — distinct clusters of the relabelled centre triple: greedy
  clustering over **every** sample with its weight, heaviest sample first, a sample
  joining the first cluster whose seed is within :data:`MODE_RADIUS_PX` pixels on every
  centre, counting clusters that carry at least :data:`MODE_MIN_WEIGHT` of the weight.
  No sample is discarded, so every cluster's weight is its true posterior mass.
- **ESS** — Kish's ``(sum w)^2 / sum w^2`` over the normalised weights.
"""

from __future__ import annotations

import numpy as np

#: A label ordering or a mode counts only above this posterior weight.
MODE_MIN_WEIGHT = 0.01

#: Two relabelled centre triples belong to one mode when every centre differs by at most
#: this many pixels. Placeholder, calibrated in the pilot then frozen (protocol §Calibration).
MODE_RADIUS_PX = 5.0

COMPONENTS = ("g0", "g1", "g2")


def normalised_weights(weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    total = w.sum()
    if total <= 0:
        raise ValueError("posterior weights sum to zero")
    return w / total


def weighted_quantile(values, quantiles, weights) -> np.ndarray:
    """Weighted quantiles by linear interpolation of the cumulative weight at sample
    midpoints (the convention ``af.marginalize`` follows)."""
    values = np.asarray(values, dtype=float)
    w = normalised_weights(weights)
    order = np.argsort(values)
    v, w = values[order], w[order]
    cdf = np.cumsum(w) - 0.5 * w
    return np.interp(np.asarray(quantiles, dtype=float), cdf, v)


def marginal(values, weights) -> dict:
    q = weighted_quantile(values, [0.025, 0.16, 0.5, 0.84, 0.975], weights)
    w = normalised_weights(weights)
    mean = float(np.sum(w * np.asarray(values, dtype=float)))
    return {
        "median": float(q[2]),
        "sigma": float(q[3] - q[1]) / 2.0,
        "lower_1": float(q[1]),
        "upper_1": float(q[3]),
        "lower_2": float(q[0]),
        "upper_2": float(q[4]),
        "mean": mean,
    }


def posterior_stats(matrix, weights, keys) -> dict:
    """``{key: marginal}`` for every column of ``matrix`` (samples x parameters)."""
    matrix = np.asarray(matrix, dtype=float)
    return {key: marginal(matrix[:, i], weights) for i, key in enumerate(keys)}


def _centre_columns(keys) -> list[int]:
    return [list(keys).index(f"{name}.centre") for name in COMPONENTS]


def relabel_matrix(matrix, keys) -> np.ndarray:
    """Sort every row's (centre, normalization, sigma) triples by centre."""
    matrix = np.array(matrix, dtype=float, copy=True)
    keys = list(keys)
    blocks = [
        [keys.index(f"{name}.{p}") for p in ("centre", "normalization", "sigma")]
        for name in COMPONENTS
    ]
    centres = matrix[:, [block[0] for block in blocks]]
    order = np.argsort(centres, axis=1, kind="stable")
    out = matrix.copy()
    for target, block in enumerate(blocks):
        for p, column in enumerate(block):
            source_columns = np.array([blocks[k][p] for k in range(len(blocks))])
            out[:, column] = matrix[np.arange(matrix.shape[0]), source_columns[order[:, target]]]
    return out


def label_permutations(matrix, weights, keys, min_weight: float = MODE_MIN_WEIGHT) -> int:
    """Distinct raw centre orderings carrying at least ``min_weight`` of the weight."""
    w = normalised_weights(weights)
    centres = np.asarray(matrix, dtype=float)[:, _centre_columns(keys)]
    orderings = [tuple(row) for row in np.argsort(centres, axis=1, kind="stable")]
    totals: dict[tuple, float] = {}
    for ordering, weight in zip(orderings, w):
        totals[ordering] = totals.get(ordering, 0.0) + weight
    return sum(1 for total in totals.values() if total >= min_weight)


def mode_clusters(
    matrix,
    weights,
    keys,
    radius: float = MODE_RADIUS_PX,
    min_weight: float = MODE_MIN_WEIGHT,
) -> list[dict]:
    """Clusters of the relabelled centre triple, heaviest first: ``[{centres, weight}]``.

    Every sample is clustered with its weight preserved (no truncation, no resampling).
    The greedy rule is evaluated in vectorised form: the heaviest unassigned sample
    seeds a cluster and every unassigned sample within ``radius`` of it on every centre
    joins it. This is the same assignment as visiting samples heaviest first and joining
    the first earlier seed in range, because a later seed never takes a sample an earlier
    seed could claim. Clustering stops once the unassigned weight is below
    ``min_weight``: no later cluster can reach the threshold, and later clusters never
    change the weight of earlier ones, so the kept clusters are exact.
    """
    w = normalised_weights(weights)
    relabelled = relabel_matrix(matrix, keys)[:, _centre_columns(keys)]
    order = np.argsort(-w, kind="stable")
    points, w = relabelled[order], w[order]
    unassigned = np.ones(len(w), dtype=bool)
    kept = []
    while True:
        remaining = np.flatnonzero(unassigned)
        if remaining.size == 0 or w[remaining].sum() < min_weight:
            break
        seed = points[remaining[0]]
        members = remaining[np.all(np.abs(points[remaining] - seed) <= radius, axis=1)]
        unassigned[members] = False
        total = float(w[members].sum())
        if total >= min_weight:
            kept.append({"centres": [float(c) for c in seed], "weight": total})
    return sorted(kept, key=lambda cluster: -cluster["weight"])


def kish_ess(weights) -> float:
    w = normalised_weights(weights)
    return float(1.0 / np.sum(w * w))


def equal_weight_draws(matrix, weights, n: int, seed: int = 0) -> np.ndarray:
    """``n`` rows resampled with probability proportional to the weight (seeded)."""
    rng = np.random.default_rng(seed)
    w = normalised_weights(weights)
    index = rng.choice(len(w), size=n, replace=True, p=w)
    return np.asarray(matrix, dtype=float)[index]


def summarise_samples(
    matrix,
    weights,
    keys,
    log_likelihood=None,
    log_posterior=None,
) -> dict:
    """Every protocol statistic for one set of weighted samples."""
    matrix = np.asarray(matrix, dtype=float)
    keys = list(keys)
    out = {
        "n_samples": int(matrix.shape[0]),
        "ess_kish": kish_ess(weights),
        "posterior": posterior_stats(matrix, weights, keys),
        "posterior_relabelled": posterior_stats(relabel_matrix(matrix, keys), weights, keys),
        "label_permutations_found": label_permutations(matrix, weights, keys),
        "modes": mode_clusters(matrix, weights, keys),
    }
    out["modes_found"] = len(out["modes"])
    if log_likelihood is not None:
        log_likelihood = np.asarray(log_likelihood, dtype=float)
        best = int(np.nanargmax(log_likelihood))
        out["max_log_likelihood"] = float(log_likelihood[best])
        out["max_log_likelihood_vector"] = [float(v) for v in matrix[best]]
    if log_posterior is not None:
        log_posterior = np.asarray(log_posterior, dtype=float)
        best = int(np.nanargmax(log_posterior))
        out["max_log_posterior"] = float(log_posterior[best])
        out["max_log_posterior_vector"] = [float(v) for v in matrix[best]]
    return out

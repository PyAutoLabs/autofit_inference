"""Protocol statistics: weighted marginals, relabelling, label permutations, modes, ESS."""

import numpy as np
from searches import _posterior as post

KEYS = [
    "g0.centre",
    "g0.normalization",
    "g0.sigma",
    "g1.centre",
    "g1.normalization",
    "g1.sigma",
    "g2.centre",
    "g2.normalization",
    "g2.sigma",
    "background.level",
]
ORDERED = [25.0, 30.0, 3.0, 45.0, 50.0, 6.0, 60.0, 40.0, 10.0, 0.02]


def permuted(vector, order):
    blocks = [vector[0:3], vector[3:6], vector[6:9]]
    out = []
    for index in order:
        out.extend(blocks[index])
    return out + [vector[9]]


def test_relabel_sorts_components_by_centre():
    rows = np.array([permuted(ORDERED, (2, 0, 1)), ORDERED, permuted(ORDERED, (1, 2, 0))])
    relabelled = post.relabel_matrix(rows, KEYS)
    for row in relabelled:
        np.testing.assert_allclose(row, ORDERED)


def test_label_permutations_counts_raw_orderings():
    rows = np.array([ORDERED, permuted(ORDERED, (1, 0, 2)), permuted(ORDERED, (2, 1, 0))])
    assert post.label_permutations(rows, [1, 1, 1], KEYS) == 3
    assert post.label_permutations(rows, [1, 1e-6, 1e-6], KEYS) == 1


def test_modes_found_merges_permutations_and_splits_distinct_solutions():
    other = list(ORDERED)
    other[6] = 80.0  # a genuinely different g2 centre
    rows = np.array([ORDERED, permuted(ORDERED, (2, 1, 0)), other])
    modes = post.mode_clusters(rows, [1, 1, 1], KEYS)
    assert len(modes) == 2
    assert abs(modes[0]["weight"] - 2 / 3) < 1e-12


def test_weighted_quantiles_and_marginal():
    values = np.arange(1001, dtype=float)
    stats = post.marginal(values, np.ones_like(values))
    assert abs(stats["median"] - 500.0) < 1.0
    assert abs(stats["sigma"] - 0.34 * 1000) < 5.0


def test_kish_ess():
    assert post.kish_ess([1, 1, 1, 1]) == 4.0
    assert abs(post.kish_ess([1, 0, 0, 0]) - 1.0) < 1e-12


def test_summarise_never_interchanges_logl_and_logp():
    rows = np.array([ORDERED, ORDERED])
    rows[1, 0] = 25.5
    out = post.summarise_samples(
        rows, [1, 1], KEYS, log_likelihood=[1.0, 2.0], log_posterior=[5.0, 3.0]
    )
    assert out["max_log_likelihood"] == 2.0 and out["max_log_likelihood_vector"][0] == 25.5
    assert out["max_log_posterior"] == 5.0 and out["max_log_posterior_vector"][0] == 25.0

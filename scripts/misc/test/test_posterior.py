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


def test_modes_cluster_every_sample_and_keep_a_minority_mode():
    """A 90 % / 10 % two-mode posterior whose minor mode has the lighter per-sample
    weights: clustering only the heaviest samples would drop it (review finding 1)."""
    rng = np.random.default_rng(0)
    other = list(ORDERED)
    other[6] = 80.0
    rows = np.array([ORDERED] * 5000 + [other] * 5000, dtype=float)
    rows[:, [0, 3, 6]] += rng.normal(0.0, 0.5, size=(10000, 3))
    weights = np.r_[np.full(5000, 0.9 / 5000), np.full(5000, 0.1 / 5000)]
    modes = post.mode_clusters(rows, weights, KEYS)
    assert len(modes) == 2
    assert abs(modes[0]["weight"] - 0.9) < 1e-9 and abs(modes[1]["weight"] - 0.1) < 1e-9
    assert abs(modes[1]["centres"][2] - 80.0) < 2.0
    assert abs(sum(m["weight"] for m in modes) - 1.0) < 1e-9


def test_chain_diagnostics_flag_a_stuck_chain():
    import numpy as np
    from searches._posterior import chain_diagnostics

    keys = [f"g{i}.{p}" for i in range(3) for p in ("centre", "normalization", "sigma")]
    keys.append("background.level")
    rng = np.random.default_rng(0)
    base = np.array([20, 1, 1, 50, 1, 1, 80, 1, 1, 0.0])
    mixed = base + rng.normal(size=(2000, 8, 10))
    good = chain_diagnostics(mixed, keys)
    assert good["rhat_max"] < 1.01 and good["ess_bulk_min"] > 4000
    stuck = mixed.copy()
    stuck[:, :4, 9] += 3.0
    bad = chain_diagnostics(stuck, keys)
    assert bad["rhat_max"] > 1.2 and bad["ess_bulk_min"] < 100

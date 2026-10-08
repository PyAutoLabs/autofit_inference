"""The committed datasets are reproducible from their seed and match the protocol."""

import json
from pathlib import Path as _Path

import numpy as np
import pytest

ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "ruff.toml").exists())
from simulators import gaussian_x3 as sim


@pytest.mark.parametrize("name", sorted(sim.DATASETS))
def test_committed_dataset_reproduces_from_seed(name):
    components, seed = sim.DATASETS[name]
    data, noise_map, _ = sim.simulate_arrays(components, seed)
    directory = ROOT / "dataset" / name
    committed = np.asarray(json.loads((directory / "data.json").read_text()))
    # rtol 1e-12, not bit equality: the seeded noise is exact, but exp() in the Gaussian
    # profile can differ by an ULP across NumPy builds/platforms (noise sigma is 0.04).
    np.testing.assert_allclose(committed, data, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        np.asarray(json.loads((directory / "noise_map.json").read_text())),
        noise_map,
        rtol=1e-12,
        atol=0,
    )
    truth = json.loads((directory / "truth.json").read_text())
    assert truth["data_seed"] == seed
    assert truth["parameters"] == sim.truth_parameters(components)


def test_same_seed_same_data_and_new_seed_new_data():
    components, seed = sim.DATASETS["gaussian_x3_blend"]
    a, _, _ = sim.simulate_arrays(components, seed)
    b, _, _ = sim.simulate_arrays(components, seed)
    c, _, _ = sim.simulate_arrays(components, seed + 1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_blend_numbers_are_the_preregistered_ones():
    components, seed = sim.DATASETS["gaussian_x3_blend"]
    assert components == [(25.0, 30.0, 3.0), (45.0, 50.0, 6.0), (60.0, 40.0, 10.0)]
    assert seed == 1
    assert sim.PIXELS == 100 and sim.NOISE_SIGMA == 0.04 and sim.BACKGROUND == 0.02


def test_separated_control_is_disjoint():
    components, _ = sim.DATASETS["gaussian_x3_separated"]
    for (c0, _, s0), (c1, _, s1) in zip(components, components[1:]):
        assert c1 - c0 >= 3 * (s0 + s1)
    centre, _, sigma = components[-1]
    assert centre + 3 * sigma < sim.PIXELS


def test_non_default_seed_writes_a_separate_realisation(tmp_path):
    path = sim.simulate("gaussian_x3_blend", data_seed=7, output_root=tmp_path)
    assert path == tmp_path / "dataset" / "gaussian_x3_blend" / "data_seed7"
    assert json.loads((path / "truth.json").read_text())["data_seed"] == 7

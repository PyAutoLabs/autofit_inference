"""
Simulator: the ``gaussian_x3`` datasets
=======================================

Simulates the two committed ``gaussian_x3`` datasets and writes ``data.json``,
``noise_map.json``, ``model.json`` (the generating instance, one entry per component)
and ``truth.json`` (the flat truth, the seed and the generator) into
``dataset/<name>/``:

- ``gaussian_x3_blend`` — centres 25 / 45 / 60, sigma 3 / 6 / 10, normalizations
  30 / 50 / 40, background 0.02. Components g1 and g2 overlap, which creates real
  correlation and a degenerate-ridge risk without being pathological.
- ``gaussian_x3_separated`` — the control: the same normalizations and background, with
  centres 20 / 50 / 80 and sigma 3 / 4 / 5, so neighbouring components are separated by
  at least ``3 (sigma_i + sigma_j)`` pixels and never overlap. A low success rate on the
  blend can then be attributed to the problem or to the search.

100 pixels (``x = 0..99``, the ``af.ex`` convention) and a constant noise sigma of 0.04
(``af.ex.util``'s ``SIGNAL_TO_NOISE_RATIO = 25``), so every ``af.ex`` tool can read the
datasets. The noise is drawn from an explicit ``np.random.default_rng(data_seed)`` —
never the unseeded ``af.ex.util`` helper — and the ``data_seed`` is recorded in
``truth.json``. ``--data-seed`` writes a further realisation (wave 2 uses five) under
``dataset/<name>/data_seed<n>/`` and leaves the committed default untouched.

The pattern is ``autofit_visualization/scripts/misc/simulators/gaussian.py``'s: a
``DATASETS`` table, a fixed seed per dataset and committed JSON.

Usage
-----

    python scripts/misc/simulators/gaussian_x3.py                 # both, default seeds
    python scripts/misc/simulators/gaussian_x3.py --name gaussian_x3_blend --data-seed 7
"""

from __future__ import annotations

import json
from pathlib import Path as _Path

import numpy as np


def _repo_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autofit_inference root (ruff.toml) not found")


_REPO_ROOT = _repo_root()

PIXELS = 100
NOISE_SIGMA = 0.04
BACKGROUND = 0.02

#: name -> (components [(centre, normalization, sigma)] in centre order, default data_seed)
DATASETS = {
    "gaussian_x3_blend": (
        [(25.0, 30.0, 3.0), (45.0, 50.0, 6.0), (60.0, 40.0, 10.0)],
        1,
    ),
    "gaussian_x3_separated": (
        [(20.0, 30.0, 3.0), (50.0, 50.0, 4.0), (80.0, 40.0, 5.0)],
        2,
    ),
}


def gaussian_profile(xvalues: np.ndarray, centre: float, normalization: float, sigma: float):
    """``af.ex.Gaussian.model_data_from`` written out, so the simulator needs no autofit."""
    return (
        normalization
        / (sigma * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * np.square((xvalues - centre) / sigma))
    )


def truth_parameters(components, background: float = BACKGROUND) -> dict[str, float]:
    out = {}
    for index, (centre, normalization, sigma) in enumerate(components):
        out[f"g{index}.centre"] = float(centre)
        out[f"g{index}.normalization"] = float(normalization)
        out[f"g{index}.sigma"] = float(sigma)
    out["background.level"] = float(background)
    return out


def simulate_arrays(components, data_seed: int, background: float = BACKGROUND):
    """``(data, noise_map, model_data)`` — pure and reproducible from ``data_seed``."""
    xvalues = np.arange(PIXELS, dtype=float)
    model_data = np.full(PIXELS, background, dtype=float)
    for centre, normalization, sigma in components:
        model_data = model_data + gaussian_profile(xvalues, centre, normalization, sigma)
    rng = np.random.default_rng(data_seed)
    data = model_data + rng.normal(0.0, NOISE_SIGMA, PIXELS)
    noise_map = np.full(PIXELS, NOISE_SIGMA, dtype=float)
    return data, noise_map, model_data


def _model_json(components, background: float) -> dict:
    """The generating instance, in the ``af.ex`` ``model.json`` shape, one entry per
    component (keys match the fitted model's names)."""
    out = {}
    for index, (centre, normalization, sigma) in enumerate(components):
        out[f"g{index}"] = {
            "type": "instance",
            "class_path": "autofit.example.model.Gaussian",
            "arguments": {"centre": centre, "normalization": normalization, "sigma": sigma},
        }
    out["background"] = {
        "type": "instance",
        "class_path": "models.gaussian_x3.Background",
        "arguments": {"level": background},
    }
    return out


def dataset_dir(name: str, data_seed: int | None = None, root: _Path | None = None) -> _Path:
    root = root if root is not None else _REPO_ROOT
    base = root / "dataset" / name
    default_seed = DATASETS[name][1]
    if data_seed is None or data_seed == default_seed:
        return base
    return base / f"data_seed{data_seed}"


def simulate(name: str, data_seed: int | None = None, output_root: _Path | None = None) -> _Path:
    """Write one dataset; returns its directory."""
    components, default_seed = DATASETS[name]
    seed = default_seed if data_seed is None else int(data_seed)
    data, noise_map, model_data = simulate_arrays(components, seed)
    path = dataset_dir(name, seed, output_root)
    path.mkdir(parents=True, exist_ok=True)

    def dump(filename, payload):
        (path / filename).write_text(json.dumps(payload, indent=4) + "\n")

    dump("data.json", [float(v) for v in data])
    dump("noise_map.json", [float(v) for v in noise_map])
    dump("model.json", _model_json(components, BACKGROUND))
    dump(
        "truth.json",
        {
            "dataset": name,
            "data_seed": seed,
            "generator": "numpy.random.default_rng(data_seed).normal(0, noise_sigma, pixels)",
            "pixels": PIXELS,
            "noise_sigma": NOISE_SIGMA,
            "parameters": truth_parameters(components),
            "max_model_value": float(np.max(model_data)),
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

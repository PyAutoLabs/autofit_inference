"""BlackJAXNUTS on ``gaussian_x3_blend`` — one config, one seed, one row.

    python scripts/gaussian_x3_blend/searches/blackjax_nuts_warm/gaussian_x3.py --config-name local_jax_cpu_fp64 --seed 0

Task: posterior. Legs: JAX only (PyAutoFit refuses a numpy Analysis). Settings: ``warmup_200_samples_500`` (default). The row lands under
``results/searches/gaussian_x3_blend/``; judged against ``results/reference/gaussian_x3_blend/<backend>/`` by
protocol ``gaussian_x3@1`` (``wiki/project/protocol_gaussian_x3.md``).

warm start from a short Nautilus (n_live=100, same seed): its best point starts the chain and its covariance seeds the mass matrix; the provider's wall is recorded and counts toward wall per right answer (protocol §8).
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(_ROOT / "scripts" / "misc"))

from searches._runner import run_search  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(
        run_search(
            sampler="blackjax_nuts_warm",
            dataset_class="gaussian_x3_blend",
            model_type="gaussian_x3",
        )
    )

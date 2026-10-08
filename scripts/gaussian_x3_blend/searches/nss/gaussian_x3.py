"""NSS on ``gaussian_x3_blend`` — one config, one seed, one row.

    python scripts/gaussian_x3_blend/searches/nss/gaussian_x3.py --config-name local_jax_cpu_fp64 --seed 0

Task: evidence. Legs: JAX only (PyAutoFit refuses a numpy Analysis). Settings: ``n_live_200`` (default). The row lands under
``results/searches/gaussian_x3_blend/``; judged against ``results/reference/gaussian_x3_blend/<backend>/`` by
protocol ``gaussian_x3@1`` (``wiki/project/protocol_gaussian_x3.md``).

NSS on the blend is deferred until phase A3b (NSS onto Fitness).
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(_ROOT / "scripts" / "misc"))

from searches._runner import run_search  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(
        run_search(sampler="nss", dataset_class="gaussian_x3_blend", model_type="gaussian_x3")
    )

"""MultiStartAdam on ``corr_gaussian_d100`` (family ``correlated_nd``) — one config, one seed, one row.

    python scripts/corr_gaussian_d100/searches/multi_start_adam/correlated_nd.py --config-name local_jax_cpu_fp64 --seed 0

Settings: ``starts_48_steps_300`` (default). The row lands under ``results/searches/corr_gaussian_d100/``; judged against the exact
reference ``results/reference/corr_gaussian_d100/<backend>/`` by protocol ``correlated_nd@1``
(``wiki/project/protocol_correlated_nd.md``). Deferred cells (protocol §7) are refused.
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(_ROOT / "scripts" / "misc"))

from searches._runner import run_search  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(
        run_search(
            sampler="multi_start_adam",
            dataset_class="corr_gaussian_d100",
            model_type="correlated_nd",
        )
    )

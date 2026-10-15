"""Zeus on ``corr_gaussian_d20`` (family ``correlated_nd``) — one config, one seed, one row.

    python scripts/corr_gaussian_d20/searches/zeus/correlated_nd.py --config-name local_numpy_fp64 --seed 0

Settings: ``walkers_50_steps_2000`` (default). The row lands under ``results/searches/corr_gaussian_d20/``; judged against the exact
reference ``results/reference/corr_gaussian_d20/<backend>/`` by protocol ``correlated_nd@1``
(``wiki/project/protocol_correlated_nd.md``). Deferred cells (protocol §7) are refused.
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(_ROOT / "scripts" / "misc"))

from searches._runner import run_search  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(
        run_search(sampler="zeus", dataset_class="corr_gaussian_d20", model_type="correlated_nd")
    )

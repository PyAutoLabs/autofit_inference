"""BlackJAXNUTS on ``rosenbrock_d20`` (family ``correlated_nd``) — one config, one seed, one row.

    python scripts/rosenbrock_d20/searches/blackjax_nuts/correlated_nd.py --config-name local_jax_cpu_fp64 --seed 0

Settings: ``warmup_200_samples_500`` (default). The row lands under ``results/searches/rosenbrock_d20/``; judged against the exact
reference ``results/reference/rosenbrock_d20/<backend>/`` by protocol ``correlated_nd@1``
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
            sampler="blackjax_nuts", dataset_class="rosenbrock_d20", model_type="correlated_nd"
        )
    )

"""Nautilus on ``gaussian_x3_separated`` — one config, one seed, one row.

    python scripts/gaussian_x3_separated/searches/nautilus/gaussian_x3.py --config-name local_numpy_fp64 --seed 0
    python scripts/gaussian_x3_separated/searches/nautilus/gaussian_x3.py --settings n_live_400 --seed 3

Settings: ``n_live_100`` (default), ``n_live_200``, ``n_live_400``. The row lands under
``results/searches/gaussian_x3_separated/``; judged against ``results/reference/gaussian_x3_separated/<backend>/`` by
protocol ``gaussian_x3@1`` (``wiki/project/protocol_gaussian_x3.md``).
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(_ROOT / "scripts" / "misc"))

from searches._runner import run_search  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(
        run_search(
            sampler="nautilus", dataset_class="gaussian_x3_separated", model_type="gaussian_x3"
        )
    )

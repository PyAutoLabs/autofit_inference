"""Write the censored row of a ``correlated_nd`` pilot run that the cap killed without one.

Protocol ``wiki/project/protocol_correlated_nd.md`` §7: a run stopped at the pilot's
operational cap is a failed attempt censored at the cap. The runner writes that row itself
on SIGTERM, but Python handles a signal only between bytecodes, so a run inside one long
native call (a jitted XLA loop, a large numpy step) can reach ``timeout``'s SIGKILL with no
row — and would then be ``missing`` for ever. The pilot submit calls this after every run
that exited non-zero: when the process was stopped by ``timeout`` (exit 124, or 137 after
the kill grace) **and** no row exists, it writes a minimal row that
``searches/_correlated_nd.admit`` accepts, with ``status`` ``failed: Terminated (no row …)``
and ``process_elapsed_s`` equal to the process limit, so accounting counts a censored
failure. Any other exit, or an existing row, writes nothing. Standard library only.

    python scripts/misc/tooling/cnd_cap_stub.py --dataset D --sampler S --settings X \
        --config C --data-seed 1 --seed 0 --limit 14520 --exit-code 124
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(ROOT), str(ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from searches import _correlated_nd as cnd  # noqa: E402
from searches._runner import SAMPLERS, config_segment, results_path, run_id  # noqa: E402

#: ``timeout``'s exit statuses: 124 after its SIGTERM, 137 when it had to SIGKILL.
TIMEOUT_EXITS = (124, 137)


def stub_row(args) -> dict:
    spec = SAMPLERS[args.sampler]
    backend = "jax_cpu" if "_jax_cpu_" in args.config else "numpy"
    config_id = config_segment(args.config, cnd.COMPILE_CACHE if backend == "jax_cpu" else None)
    rid = run_id(
        cnd.MODEL_TYPE,
        args.dataset,
        args.data_seed,
        args.sampler,
        args.settings,
        config_id,
        args.seed,
    )
    return {
        "schema_version": 1,
        "protocol_id": cnd.PROTOCOL_ID,
        "wave": cnd.WAVE,
        "pilot": True,
        "run_id": rid,
        "model": cnd.MODEL_TYPE,
        "sampler": args.sampler,
        "sampler_class": spec.cls,
        "task": spec.task,
        "family": spec.family,
        "settings_name": args.settings,
        "settings": spec.settings[args.settings],
        "config_name": args.config,
        "config_id": config_id,
        "backend": backend,
        "dataset": args.dataset,
        "dataset_class": args.dataset,
        "data_seed": args.data_seed,
        "seed": args.seed,
        "assertion_mechanism": "none",
        # The submit's pin guards passed before this run started (they exit otherwise).
        "pyautofit_commit": cnd.PYAUTOFIT_PIN,
        "status": f"failed: Terminated (no row: killed by timeout, exit {args.exit_code})",
        "completed": False,
        "resumed": False,
        "total_wall_s": None,
        "process_elapsed_s": float(args.limit),
        "stub": "written by scripts/misc/tooling/cnd_cap_stub.py (protocol correlated_nd §7)",
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("--dataset", "--sampler", "--settings", "--config"):
        parser.add_argument(name, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--limit", type=float, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.exit_code not in TIMEOUT_EXITS:
        return 0
    row = stub_row(args)
    path = results_path(
        args.root,
        args.dataset,
        args.data_seed,
        args.sampler,
        args.settings,
        row["config_id"],
        args.seed,
    )
    if path.exists():
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2) + "\n")
    print(f"  censored stub row: {path.relative_to(args.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

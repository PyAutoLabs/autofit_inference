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

**Out of memory (B5c-2a).** SLURM's cgroup OOM killer also ends a run with SIGKILL, which
``timeout`` reports as exit 137, so the pilot wrote this cap stub for its four
``OUT_OF_MEMORY`` tasks (263/273/304/383) although none reached the cap. With
``--elapsed-s`` (the process's own wall) a 137 exit below the limit writes nothing: it is
not a cap kill. ``--reconcile-oom <sacct.tsv>`` rewrites the stub of every task ``sacct``
records as ``OUT_OF_MEMORY`` into an ``oom`` row (``searches/_correlated_nd.OOM_STATUS``)
carrying the measured MaxRSS, the elapsed wall and the request; it replaces only a stub
(never a runner-written row) and maps a task to its run by the manifest
(``hpc/batch_cpu/correlated_nd/pilot.tsv``, task N = line N + 2).

    python scripts/misc/tooling/cnd_cap_stub.py --dataset D --sampler S --settings X \
        --config C --data-seed 1 --seed 0 --limit 14520 --exit-code 124
    python scripts/misc/tooling/cnd_cap_stub.py \
        --reconcile-oom results/accounting/correlated_nd_pilot_sacct.tsv
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

#: ``timeout``'s exit statuses: 124 after its SIGTERM, 137 when it had to SIGKILL (or
#: when anything else — SLURM's OOM killer — sent the run a SIGKILL).
TIMEOUT_EXITS = (124, 137)
MANIFEST = ROOT / "hpc" / "batch_cpu" / "correlated_nd" / "pilot.tsv"


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


def read_sacct(path: Path) -> list[dict]:
    """The committed sacct extract (``#`` comment lines, then a header row)."""
    lines = [line for line in Path(path).read_text().splitlines() if not line.startswith("#")]
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line]


def oom_row(stub: dict, record: dict) -> dict:
    """The ``oom`` row of one task: its stub's identity with the sacct facts."""
    elapsed = float(record["elapsed_s"])
    max_rss_kb = int(record["max_rss_kb"]) if record.get("max_rss_kb") else None
    row = dict(stub)
    row.update(
        {
            "status": (
                f"{cnd.OOM_STATUS} (no row: SLURM killed the run at its --mem "
                f"{record['req_mem']} request after {elapsed:.0f} s)"
            ),
            "completed": False,
            "total_wall_s": None,
            "process_elapsed_s": elapsed,
            "oom": {
                "slurm_state": record["state"],
                "slurm_job": record["job"],
                "req_mem": record["req_mem"],
                "max_rss_kb": max_rss_kb,
                "max_rss_gb": round(max_rss_kb / 1024**2, 2) if max_rss_kb else None,
                "max_rss_note": "sacct samples RSS periodically: the peak at the kill was "
                "at least the request",
                "elapsed_s": elapsed,
                "node": record.get("node"),
            },
            "stub": "written by scripts/misc/tooling/cnd_cap_stub.py --reconcile-oom from "
            "sacct (B5c-2a); replaces the cap stub the exit-137 path wrote",
        }
    )
    return row


def reconcile_oom(sacct: Path, root: Path = ROOT, manifest: Path = MANIFEST) -> list[Path]:
    """Rewrite the stub of every ``OUT_OF_MEMORY`` task as an ``oom`` row; returns the
    paths written. A runner-written row (no ``stub`` key) is never replaced."""
    tasks = Path(manifest).read_text().splitlines()[1:]
    written = []
    for record in read_sacct(sacct):
        if record["state"] != "OUT_OF_MEMORY":
            continue
        dataset, sampler, settings, config, data_seed, _, seed = tasks[int(record["task"])].split(
            "\t"
        )
        backend = "jax_cpu" if "_jax_cpu_" in config else "numpy"
        config_id = config_segment(config, cnd.COMPILE_CACHE if backend == "jax_cpu" else None)
        path = results_path(root, dataset, int(data_seed), sampler, settings, config_id, int(seed))
        existing = json.loads(path.read_text()) if path.exists() else None
        if existing is not None and not existing.get("stub"):
            raise SystemExit(f"{path}: a runner-written row exists for OOM task {record['task']}")
        base = existing or stub_row(
            argparse.Namespace(
                dataset=dataset,
                sampler=sampler,
                settings=settings,
                config=config,
                data_seed=int(data_seed),
                seed=int(seed),
                limit=0.0,
                exit_code=137,
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(oom_row(base, record), indent=2) + "\n")
        written.append(path)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reconcile-oom", type=Path, metavar="SACCT_TSV")
    for name in ("--dataset", "--sampler", "--settings", "--config"):
        parser.add_argument(name)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--elapsed-s", type=float, help="the process's own wall, if known")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.reconcile_oom:
        for path in reconcile_oom(args.reconcile_oom, args.root):
            print(f"  oom row: {path.relative_to(args.root)}")
        return 0
    missing = [
        name
        for name in ("dataset", "sampler", "settings", "config", "data_seed", "seed", "limit")
        if getattr(args, name) is None
    ] + (["exit_code"] if args.exit_code is None else [])
    if missing:
        parser.error("missing " + ", ".join(f"--{m.replace('_', '-')}" for m in missing))
    if args.exit_code not in TIMEOUT_EXITS:
        return 0
    if args.elapsed_s is not None and args.elapsed_s < args.limit:
        return 0  # a SIGKILL below the cap (SLURM's OOM killer): sacct reconciles it
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

"""Shared CLI, environment and provenance helpers for every autofit_inference script.

Generalised from ``autolens_inference/_inference_cli.py``: the generic pieces
(``default_cores``, ``device_info_dict``, the config-name regex that refuses a lying
name, the dataset auto-simulate hook) are kept; the lens-specific ones (inversion,
mesh, SLaM stages, instruments) are dropped.

The config-name grammar
-----------------------

``{where}_{backend}_{precision}`` with

- ``where`` — ``local`` or ``ral``: the hardware the run actually used. A ``ral`` name
  outside a SLURM job, or a ``local`` name inside one, is refused (the
  ``hpc_a100_numba_cpu`` CPU legs of autolens_inference ran on ``ral`` nodes under a name
  that said A100 — Cortex run 350682 — and nothing downstream could tell).
- ``backend`` — ``numpy`` (the callback path: ``use_jax=False``) or ``jax_cpu``
  (``use_jax=True``, ``JAX_PLATFORMS=cpu``). There is no GPU leg at birth: a 100-pixel
  likelihood is dispatch-bound (protocol §Hardware).
- ``precision`` — ``fp64`` or ``fp32``. ``fp32`` exists only on ``jax_cpu``
  (``JAX_ENABLE_X64=False``); ``numpy_fp32`` is refused.

``--backend`` / ``--precision`` default from the config name; passing one that
disagrees with the name is refused.

Every leaf imports this after the standard root-finding preamble::

    import sys
    from pathlib import Path

    _ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
    sys.path.insert(0, str(_ROOT))
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

WHERES = ("local", "ral")
BACKENDS = ("numpy", "jax_cpu")
PRECISIONS = ("fp64", "fp32")

#: ``(where, backend, precision)`` — one group per field.
CONFIG_NAME_RE = re.compile(r"^(local|ral)_(numpy|jax_cpu)_(fp64|fp32)$")

#: The datasets a leaf may name (``dataset/<name>/``).
DATASETS = ("gaussian_x3_blend", "gaussian_x3_separated")

#: ``--compile-cache``: ``cold`` disables JAX's persistent compilation cache so
#: ``compile_s`` is a full trace + lower + compile; ``warm`` leaves it on. The two are
#: separate runs, never one run's first and second call (protocol §Timing).
COMPILE_CACHES = ("cold", "warm")


def default_cores() -> int:
    """``SLURM_CPUS_PER_TASK`` inside a job, else the machine's CPU count."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            return int(slurm)
        except ValueError:
            pass
    return os.cpu_count() or 1


def in_slurm_job() -> bool:
    return bool(os.environ.get("SLURM_JOB_ID"))


@dataclass(frozen=True)
class InferenceCLI:
    config_name: str
    where: str
    backend: str
    precision: str
    seed: int
    data_seed: int | None
    dataset: str
    output_dir: Path | None
    cores: int
    compile_cache: str
    settings: str | None


def parse_config_name(config_name: str) -> tuple[str, str, str]:
    match = CONFIG_NAME_RE.match(config_name or "")
    if match is None:
        raise ValueError(
            f"config name {config_name!r} does not match "
            "{local,ral}_{numpy,jax_cpu}_{fp64,fp32}"
        )
    return match.group(1), match.group(2), match.group(3)


def validate_cli(
    config_name: str,
    backend: str | None,
    precision: str | None,
    *,
    in_slurm: bool | None = None,
) -> tuple[str, str, str]:
    """Return ``(where, backend, precision)`` or raise ``ValueError`` naming the lie."""
    where, name_backend, name_precision = parse_config_name(config_name)
    if backend is not None and backend != name_backend:
        raise ValueError(f"--backend {backend} disagrees with config name {config_name}")
    if precision is not None and precision != name_precision:
        raise ValueError(f"--precision {precision} disagrees with config name {config_name}")
    if name_backend == "numpy" and name_precision == "fp32":
        raise ValueError("numpy_fp32 is not a leg: the numpy callback path is fp64 only")
    slurm = in_slurm_job() if in_slurm is None else in_slurm
    if where == "ral" and not slurm:
        raise ValueError(f"{config_name} names RAL but this is not a SLURM job")
    if where == "local" and slurm:
        raise ValueError(f"{config_name} names a local run inside SLURM job")
    return where, name_backend, name_precision


def build_parser(default_config_name: str = "local_numpy_fp64") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="autofit_inference run flags")
    parser.add_argument("--config-name", default=default_config_name)
    parser.add_argument("--backend", choices=BACKENDS, default=None)
    parser.add_argument("--precision", choices=PRECISIONS, default=None)
    parser.add_argument("--seed", type=int, default=0, help="search seed")
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="data realisation (default: the committed dataset's seed)",
    )
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--compile-cache", choices=COMPILE_CACHES, default="warm")
    parser.add_argument(
        "--settings",
        default=None,
        help="a named settings variant of the sampler (e.g. n_live_200); default = the leaf's",
    )
    return parser


def parse_inference_cli(
    argv=None,
    default_config_name: str = "local_numpy_fp64",
    default_dataset: str = "gaussian_x3_blend",
) -> InferenceCLI:
    """Parse and validate; exits with status 2 on a refused combination."""
    args = build_parser(default_config_name).parse_args(argv)
    try:
        where, backend, precision = validate_cli(args.config_name, args.backend, args.precision)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    return InferenceCLI(
        config_name=args.config_name,
        where=where,
        backend=backend,
        precision=precision,
        seed=int(args.seed),
        data_seed=args.data_seed,
        dataset=args.dataset or default_dataset,
        output_dir=args.output_dir,
        cores=args.cores if args.cores is not None else default_cores(),
        compile_cache=args.compile_cache,
        settings=args.settings,
    )


def set_backend_env(backend: str, precision: str, cores: int, compile_cache: str = "warm") -> dict:
    """Export the environment the backend needs **before** autofit is imported.

    ``JAX_ENABLE_X64`` is always exported explicitly (an fp64 config without it silently
    runs fp32); ``NPROC`` sizes XLA's CPU thread pool (unset, a job takes the node);
    ``PYAUTO_DISABLE_JAX=1`` is the process-wide belt to the numpy leg's
    ``use_jax=False`` braces.
    """
    env = {"JAX_ENABLE_X64": "False" if precision == "fp32" else "True"}
    if backend == "numpy":
        env["PYAUTO_DISABLE_JAX"] = "1"
    elif backend == "jax_cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env["NPROC"] = str(cores)
        if compile_cache == "cold":
            env["JAX_COMPILATION_CACHE_DIR"] = ""
    else:  # pragma: no cover — argparse constrains the vocabulary
        raise ValueError(f"unknown backend {backend!r}")
    os.environ.update(env)
    return env


def device_info_dict(backend: str) -> dict:
    """Host/device provenance; JAX is imported only on a JAX leg."""
    info = {
        "xla_flags": os.environ.get("XLA_FLAGS") or None,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS") or None,
        "nproc": os.environ.get("NPROC") or None,
        "cpu_count": os.cpu_count(),
    }
    if backend.startswith("jax"):
        try:
            import jax

            info["backend"] = jax.default_backend()
            info["device"] = str(jax.devices()[0])
            info["jax_version"] = jax.__version__
            info["x64"] = bool(jax.config.read("jax_enable_x64"))
        except Exception as exc:  # pragma: no cover
            info["backend"] = f"unavailable ({type(exc).__name__})"
    else:
        info["backend"] = "numpy"
    return info


def library_revisions(libraries=("autonerves", "autofit")) -> dict:
    """``{package: full git SHA}`` of each library's checkout, ``None`` when not a checkout."""
    import importlib

    revisions: dict[str, str | None] = {}
    for name in libraries:
        try:
            module = importlib.import_module(name)
            directory = Path(module.__file__).resolve().parent
            revisions[name] = (
                subprocess.check_output(
                    ["git", "-C", str(directory), "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                .decode()
                .strip()
            )
        except Exception:
            revisions[name] = None
    return revisions


def dataset_path(dataset: str, data_seed: int | None = None, root: Path = ROOT) -> Path:
    """``dataset/<name>/`` (or ``…/data_seed<n>/`` for a non-default realisation)."""
    sys.path.insert(0, str(root / "scripts" / "misc"))
    from simulators.gaussian_x3 import dataset_dir

    return dataset_dir(dataset, data_seed, root)


def auto_simulate_if_missing(dataset: str, data_seed: int | None = None, root: Path = ROOT) -> Path:
    """Simulate the dataset when its ``data.json`` is absent; return its directory."""
    path = dataset_path(dataset, data_seed, root)
    if (path / "data.json").exists():
        return path
    command = [
        sys.executable,
        str(root / "scripts" / "misc" / "simulators" / "gaussian_x3.py"),
        "--name",
        dataset,
        "--output-root",
        str(root),
    ]
    if data_seed is not None:
        command += ["--data-seed", str(data_seed)]
    print(f"  [auto-simulate] {path} missing; running {' '.join(command[1:])}")
    subprocess.run(command, check=True)
    return path

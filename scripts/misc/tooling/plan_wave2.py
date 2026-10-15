"""Plan scored wave 2 on RAL: the task tables and the submit scripts under ``hpc/batch_cpu/``.

Generated from :mod:`searches._wave2` (the expected-run manifest), the calibration record's
frozen timeouts (``results/calibration/gaussian_x3_wave1.json``) and the pilot-derived rates
in ``wall/rates.py``. Nothing here is hand-edited: change the planner, run ``--write``,
commit; ``--check`` (lint) fails when a committed file differs from what it generates.

What it writes
--------------

- ``hpc/batch_cpu/wave2/*.tsv`` — one line per array task (header first; the submit reads
  line ``SLURM_ARRAY_TASK_ID + 2``).
- ``submit_wave2_reference_<dataset>`` — the per-realisation reference runs (protocol §6:
  3 Nautilus ``n_live=2000`` + 3 DynestyStatic ``nlive=1000`` per backend) for the four
  realisations of :data:`_wave2.DATA_SEEDS` that are not the dataset's committed default,
  plus, on the blend, the JAX Nautilus seed 1 the B3 wrap-up deferred (protocol §10) and
  the NSS ``ln 3!`` constant-likelihood measurement (§3) that NSS's evidence verdict needs.
- ``submit_wave2_reference_build_<dataset>`` — per (backend, new realisation): the MAP
  reference, then ``build_reference.py``. Submitted ``--dependency=afterany`` on the
  reference array. (The JAX blend default realisation is rebuilt on the laptop in B5b: its
  seeds 0 and 2 are pilot-era runs whose ``samples.csv`` is not on RAL.)
- ``submit_wave2_scored_<dataset>_<backend>`` — every planned cell × realisation, seeds
  blocked so a task carries ~2 h of median pilot wall; each run under its frozen ``@2``
  timeout plus ``_wave2.RUN_OVERHEAD_S`` (``timeout -s TERM``; the runner writes the failure
  row). Submitted ``--dependency=afterany`` on the reference builds, so every realisation's
  reference exists when its rows are judged at write time.
- ``submit_wave2_probe`` — the probe-first cells' 5-run probes (no pilot wall:
  protocol §11 A2), under :data:`_wave2.PROBE_CAP_S` per run.
- ``submit_wave2_remainder`` (B5b, once ``results/calibration/gaussian_x3_wave2_probe.json``
  exists) — the rest of every probe-first cell: every realisation × seed except the probe's
  own runs, blocked on the probe median, each run under its probe-derived ``@2`` timeout
  plus the run overhead.

The scored arrays and the probe are generated from the menu as B5a submitted it (the
probe-first cells stay out of the scored arrays and in the probe even once their timeouts
are written), so the committed as-run files never move.

    python scripts/misc/tooling/plan_wave2.py            # print the plan
    python scripts/misc/tooling/plan_wave2.py --write    # write tables + submits
    python scripts/misc/tooling/plan_wave2.py --check    # lint gate
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(ROOT / "scripts" / "misc"), str(ROOT / "scripts" / "misc" / "tooling")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_catalogue as catalogue  # noqa: E402
from searches import _protocol as protocol  # noqa: E402
from searches import _wave2 as wave2  # noqa: E402
from searches._runner import SAMPLERS  # noqa: E402
from wall import rates  # noqa: E402

BATCH = ROOT / "hpc" / "batch_cpu"
TABLES = BATCH / "wave2"
RAL_ROOT = "/mnt/ral/jnightin/autofit_inference"

#: Seconds added to a task's ``--time`` beyond the sum of its runs' limits.
TASK_MARGIN_S = 600.0

#: Laptop-measured walls sizing RAL jobs: the device differs, so every row asks for 3x
#: (the gate's floor for a rate is 1.5x).
HEADROOM = 3.0

#: Per-task resources. The pilot ran one search per process at ``--cores 1`` (numpy
#: single-threaded, ``NPROC=1``); two CPUs leave XLA's runtime thread room.
CPUS_PER_TASK = 2
MEM = "6gb"

#: Concurrency (``%N``) per array. Four scored arrays at %60 = 240 tasks = 480 CPUs, about
#: a fifth of ``ral`` (2188 CPUs), on the five 236-CPU nodes only (:data:`EXCLUDE`).
SCORED_THROTTLE = 60
REFERENCE_THROTTLE = 50
BUILD_THROTTLE = 8
PROBE_THROTTLE = 20

#: The four 252-CPU ``ral`` nodes are the only ones a 252-CPU whole-node job fits on (other
#: campaigns run those); these 2-CPU tasks stay off them so they never fragment them.
EXCLUDE = "euclid-ral-compute-[1-4]"

MAIL = "james.w.nightingale@durham.ac.uk"

#: Reference walls measured in B2/B3 (laptop), per (dataset, backend): the slowest run.
REFERENCE_SETTINGS = {"nautilus": (0, 1, 2), "dynesty_static": (0, 1, 2)}
MAP_WALL = ("results/reference/gaussian_x3_blend/jax_cpu/map_reference.json", "wall_s")
#: ``build_reference.py`` on the blend numpy reference, laptop, 2026-10-14 (3.6 s).
BUILD_WALL_S = 3.6


# ---------------------------------------------------------------------------
# Rates (wall/rates.py) — pilot walls per dataset × cell × backend
# ---------------------------------------------------------------------------


def ran_to_the_end(row: dict) -> bool:
    """An attempt whose wall measures the cell: it completed, or failed on its own (a
    non-finite answer, a trace error) — never one stopped from outside (``Terminated``:
    the pilot's wall cap), whose wall is a censored lower bound."""
    return "Terminated" not in str(row.get("status"))


def pilot_walls(root: Path = ROOT) -> dict[tuple[str, str, str, str], list[float]]:
    """Pilot attempts' walls (warm-start provider included), per (dataset, sampler,
    settings, backend), over every attempt that ran to its own end. (The calibration's
    timeouts pool the completed ones over both datasets; a rate also counts a cell whose
    runs all fail on their own, e.g. BFGS numpy and cold NUTS on the blend.)"""
    walls: dict[tuple, list[float]] = defaultdict(list)
    for _, row in catalogue.scan_rows(root):
        cost = protocol.attempt_wall_s(row)
        if ran_to_the_end(row) and cost is not None:
            key = (row["dataset"], row["sampler"], row["settings_name"], row["backend"])
            walls[key].append(cost)
    return walls


def device(backend: str) -> str:
    return f"laptop_{backend}"


def derived_rates(root: Path = ROOT) -> dict[tuple, float]:
    """``wall/rates.py`` ``STEP_RATE`` as the pilot rows give it: per (dataset, cell,
    backend), the median wall of one search run (a "step" = one seed) of the cell's
    slowest settings on that dataset."""
    medians: dict[tuple, float] = {}
    for (dataset, sampler, _settings, backend), values in pilot_walls(root).items():
        key = (
            dataset,
            f"searches/{sampler}",
            "gaussian_x3",
            device(backend),
            "fp64",
            1,
            None,
        )
        medians[key] = max(medians.get(key, 0.0), statistics.median(values))
    return medians


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def slurm_time(seconds: float) -> str:
    total = int(math.ceil(seconds / 60.0)) * 60
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    return f"{days}-{hours:02d}:{minutes:02d}:00" if days else f"{hours:02d}:{minutes:02d}:00"


def table(header: tuple[str, ...], lines: list[str]) -> str:
    return "\t".join(header) + "\n" + "".join(line + "\n" for line in lines)


def header(
    title: str,
    why: list[str],
    basis: list[str],
    sbatch: dict,
    basis_source: str = "walls measured on the B2/B3 laptop",
) -> str:
    out = [
        "#!/bin/bash -l",
        "#",
        "# GENERATED by scripts/misc/tooling/plan_wave2.py -- do not edit by hand. Change the",
        "# planner, run it with --write and commit; lint runs it with --check.",
        "#",
        f"# {title}",
        "#",
    ]
    out += [f"# {line}" if line else "#" for line in why]
    out += [
        "#",
        "# CPU jobs run on --partition=ral ONLY: never gpu, ral,gpu or gpu,ral (the human's hard",
        "# rule of 2026-09-30; PyAutoMind/policy/hpc_ral.md). The --exclude keeps these small",
        "# tasks off the four 252-CPU nodes, the only ones a whole-node job of another campaign",
        "# fits on. The PyAuto libraries come from the PRIVATE pin under",
        f"# {RAL_ROOT}/{wave2.PIN_DIR}/, PYTHONPATH-prefixed below; the",
        "# shared /mnt/ral/jnightin/PyAuto mirror provides only the venv and is never pulled.",
        "#",
        f"# WALL-BASIS: -- one row per cell this submit runs; {basis_source}",
    ]
    out += [f"#   {line}" for line in basis]
    for key, value in sbatch.items():
        # Long options take ``=``; short ones (-J, -o, -e) a space.
        out.append(f"#SBATCH {key}={value}" if key.startswith("--") else f"#SBATCH {key} {value}")
    return "\n".join(out) + "\n"


def environment() -> str:
    return f"""
export AP_ROOT={RAL_ROOT}
source $AP_ROOT/activate.sh

# The private pin: PyAutoFit {wave2.PYAUTOFIT_PIN}, PyAutoNerves
# {wave2.PYAUTONERVES_PIN}. Prefixed, so it wins over the shared checkouts that
# activate.sh puts on PYTHONPATH; a wave-2 run refuses any other PyAutoFit commit.
# $AF_PIN/site holds the third-party packages the pin needs that the shared venv lacks
# ({", ".join(wave2.PIN_SITE_PACKAGES)}), unpacked from their wheels: the shared venv is
# never modified.
export AF_PIN=$AP_ROOT/{wave2.PIN_DIR}
export PYTHONPATH="$AF_PIN/PyAutoNerves:$AF_PIN/PyAutoFit:$AF_PIN/site${{PYTHONPATH:+:$PYTHONPATH}}"

export JAX_PLATFORMS=cpu
# Every submit exports this explicitly: without it an fp64 config silently runs fp32.
export JAX_ENABLE_X64=True
# One search per process at --cores 1, as in the pilot.
export NPROC=1

echo "=========================================="
date
cd $AP_ROOT
# Refuse to run anything unless both pins are at their commits and are what imports.
test "$(git -C $AF_PIN/PyAutoFit rev-parse HEAD)" = "{wave2.PYAUTOFIT_PIN}"
test "$(git -C $AF_PIN/PyAutoNerves rev-parse HEAD)" = "{wave2.PYAUTONERVES_PIN}"
python3 -c "import autofit, autonerves, os; pin = os.environ['AF_PIN']; assert autofit.__file__.startswith(pin) and autonerves.__file__.startswith(pin), (autofit.__file__, autonerves.__file__); print('autofit', autofit.__version__, autofit.__file__); print('autonerves', autonerves.__file__)"
"""


def read_line(tsv: str, names: str) -> str:
    return f"""
TASKS=hpc/batch_cpu/wave2/{tsv}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$TASKS")
if [ -z "$LINE" ]; then echo "no task line $SLURM_ARRAY_TASK_ID in $TASKS" >&2; exit 1; fi
read -r {names} <<< "$LINE"
echo "task $SLURM_ARRAY_TASK_ID: $LINE"
"""


FOOTER = """
echo "Finished."
date
"""


def sbatch(job: str, n_tasks: int, throttle: int, time_s: float, cpus: int = CPUS_PER_TASK):
    return {
        "-J": job,
        "--partition": "ral",
        "--exclude": EXCLUDE,
        "--ntasks": 1,
        "--cpus-per-task": cpus,
        "--mem": MEM,
        "--time": slurm_time(time_s),
        "--array": f"0-{n_tasks - 1}%{min(throttle, n_tasks)}",
        "-o": "output/output.%A_%a.out",
        "-e": "error/error.%A_%a.err",
        "--mail-type": "END,FAIL",
        "--mail-user": MAIL,
    }


def search_call(dataset: str, sampler: str) -> str:
    return (
        f'timeout -s TERM -k {int(wave2.KILL_GRACE_S)} "$LIMIT" python3 scripts/{dataset}/searches/{sampler}/gaussian_x3.py '
        '--wave 2 --config-name "$CONFIG" --settings "$SETTINGS" --data-seed "$DATA_SEED" '
        '--seed "$SEED" --cores 1 || echo "run '
        f"{dataset}/{sampler}/$SETTINGS/data_seed$DATA_SEED/seed$SEED exited $? (its row, if "
        'any, records the failure)"'
    )


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def scored(dataset: str, backend: str, step_rate: dict) -> tuple[str, str, int]:
    tasks = wave2.scored_tasks(dataset, backend)
    lines = [t.line() for t in tasks]
    samplers = list(dict.fromkeys(t.cell.sampler for t in tasks))
    basis, walls = [], []
    for sampler in samplers:
        cell_tasks = [t for t in tasks if t.cell.sampler == sampler]
        steps = max(len(t.seeds) for t in cell_tasks)
        key = (dataset, f"searches/{sampler}", "gaussian_x3", device(backend), "fp64", 1, None)
        rate = step_rate[key]
        walls.append(HEADROOM * rate * steps)
        basis += [
            f"cell: {dataset}/searches/{sampler}/gaussian_x3  device: {device(backend)}  "
            f"precision: fp64",
            f"lanes: 1  steps: {steps}  rate: {rate:.4f}  source: rates  headroom: {HEADROOM:g}",
        ]
    time_s = max([max(t.budget_s for t in tasks) + TASK_MARGIN_S, *walls])
    name = f"scored_{dataset}_{backend}"
    why = [
        f"Scored wave 2 (gaussian_x3@2), {dataset}, {backend}: {len(tasks)} array tasks.",
        "Every planned cell x 5 data realisations x 50 seeds, seeds blocked to ~2 h of median",
        "pilot wall per task.",
        "Each run is limited to its frozen @2 timeout plus the run overhead; a run that hits",
        "it writes a failure row (SIGTERM -> Terminated) and the task moves to the next seed.",
        "Submit with --dependency=afterany on the reference builds (hpc/README.md).",
    ]
    text = header(
        why[0],
        why[1:],
        basis,
        sbatch(
            f"w2_{dataset.removeprefix('gaussian_x3_')}_{backend}",
            len(tasks),
            SCORED_THROTTLE,
            time_s,
        ),
    )
    text += environment()
    text += read_line(f"{name}.tsv", "SAMPLER SETTINGS CONFIG DATA_SEED LIMIT SEEDS")
    text += 'for SEED in ${SEEDS//,/ }; do\n    case "$SAMPLER" in\n'
    for sampler in samplers:
        text += f"        {sampler})\n            {search_call(dataset, sampler)}\n            ;;\n"
    text += '        *) echo "unknown sampler $SAMPLER" >&2; exit 1 ;;\n    esac\ndone\n'
    text += FOOTER
    tsv = table(("sampler", "settings", "config", "data_seed", "limit_s", "seeds"), lines)
    return text, tsv, len(tasks)


def probe() -> tuple[str, str, int]:
    tasks = wave2.probe_tasks()
    lines = [f"{t.cell.dataset}\t{t.line()}" for t in tasks]
    cells = list(dict.fromkeys((t.cell.dataset, t.cell.sampler) for t in tasks))
    basis = []
    for dataset, sampler in cells:
        basis += [
            f"cell: {dataset}/searches/{sampler}/gaussian_x3  device: ral_cpu  precision: fp64",
            "lanes: 1  steps: 1  source: unmeasured  probe-first: yes",
        ]
    time_s = max(t.budget_s for t in tasks) + TASK_MARGIN_S
    why = [
        f"Wave-2 probes: {len(tasks)} array tasks, one run each, for the probe-first cells.",
        "These cells have no pilot wall (Zeus on JAX was stopped at the pilot's wall cap; NSS",
        "runs its new 50-step entry), so they have no frozen timeout yet. Each runs 5 seeds on",
        "data realisation 1 under a per-run cap; the cell's timeout is then max(600 s, 10 x",
        "the probe median) (protocol 11 A2) and its remaining runs are submitted in B5b.",
    ]
    text = header(why[0], why[1:], basis, sbatch("w2_probe", len(tasks), PROBE_THROTTLE, time_s))
    text += environment()
    text += read_line("probe.tsv", "DATASET SAMPLER SETTINGS CONFIG DATA_SEED LIMIT SEEDS")
    text += 'for SEED in ${SEEDS//,/ }; do\n    case "$DATASET/$SAMPLER" in\n'
    for dataset, sampler in cells:
        text += f"        {dataset}/{sampler})\n            {search_call(dataset, sampler)}\n            ;;\n"
    text += '        *) echo "unknown cell $DATASET/$SAMPLER" >&2; exit 1 ;;\n    esac\ndone\n'
    text += FOOTER
    tsv = table(
        ("dataset", "sampler", "settings", "config", "data_seed", "limit_s", "seeds"), lines
    )
    return text, tsv, len(tasks)


def remainder() -> tuple[str, str, int] | None:
    tasks = wave2.remainder_tasks()
    if not tasks:
        return None
    lines = [f"{t.cell.dataset}\t{t.line()}" for t in tasks]
    cells = list(dict.fromkeys((t.cell.dataset, t.cell.sampler) for t in tasks))
    probe_ref = wave2.PROBE_CALIBRATION.relative_to(ROOT).as_posix()
    basis = []
    for dataset, sampler in cells:
        cell_tasks = [t for t in tasks if (t.cell.dataset, t.cell.sampler) == (dataset, sampler)]
        steps = max(len(t.seeds) for t in cell_tasks)
        median = cell_tasks[0].cell.median_pilot_wall_s
        basis += [
            f"cell: {dataset}/searches/{sampler}/gaussian_x3  device: ral_cpu  precision: fp64",
            f"lanes: 1  steps: {steps}  source: measured-wall  wall: {median * steps:.0f}  "
            f"ref: {probe_ref}  headroom: {HEADROOM:g}",
        ]
    runs = sum(len(t.seeds) for t in tasks)
    time_s = max(t.budget_s for t in tasks) + TASK_MARGIN_S
    why = [
        f"Wave-2 probe-first remainder: {len(tasks)} array tasks, {runs} runs.",
        "The probe-first cells (Zeus on JAX, NSS's 50-step entry; both datasets) after their",
        "5-run probe (job 408246): every realisation x seed except the probe's own runs. Each",
        "run is limited to its @2 timeout, max(600 s, 10 x the probe median) (protocol 11 A2,",
        f"{probe_ref}), plus the run overhead; a run that hits it writes a failure row.",
        "Submitted in B5b; pulled and judged in B5b-2.",
    ]
    text = header(
        why[0],
        why[1:],
        basis,
        sbatch("w2_remainder", len(tasks), SCORED_THROTTLE, time_s),
        basis_source="walls are the RAL probe medians x seeds per task",
    )
    text += environment()
    text += read_line("remainder.tsv", "DATASET SAMPLER SETTINGS CONFIG DATA_SEED LIMIT SEEDS")
    text += 'for SEED in ${SEEDS//,/ }; do\n    case "$DATASET/$SAMPLER" in\n'
    for dataset, sampler in cells:
        text += f"        {dataset}/{sampler})\n            {search_call(dataset, sampler)}\n            ;;\n"
    text += '        *) echo "unknown cell $DATASET/$SAMPLER" >&2; exit 1 ;;\n    esac\ndone\n'
    text += FOOTER
    tsv = table(
        ("dataset", "sampler", "settings", "config", "data_seed", "limit_s", "seeds"), lines
    )
    return text, tsv, len(tasks)


def new_realisations(dataset: str) -> list[int]:
    return [d for d in wave2.DATA_SEEDS if d != protocol.DEFAULT_DATA_SEED[dataset]]


def reference_walls(dataset: str) -> tuple[float, str]:
    slowest, ref = 0.0, ""
    for path in sorted((ROOT / "results" / "reference" / dataset).glob("*/*_seed*.json")):
        wall = (protocol.load_json(path) or {}).get("total_wall_s") or 0.0
        if wall > slowest:
            slowest, ref = wall, path.relative_to(ROOT).as_posix()
    return slowest, ref


def reference(dataset: str) -> tuple[str, str, int]:
    lines = []
    for backend, config in wave2.CONFIG_BY_BACKEND.items():
        for data_seed in new_realisations(dataset):
            for sampler, seeds in REFERENCE_SETTINGS.items():
                for seed in seeds:
                    lines.append(f"run\t{sampler}\t{seed}\t{data_seed}\t{config}")
    extra = []
    if dataset == "gaussian_x3_blend":
        default = protocol.DEFAULT_DATA_SEED[dataset]
        jax = wave2.CONFIG_BY_BACKEND["jax_cpu"]
        lines.append(f"run\tnautilus\t1\t{default}\t{jax}")
        lines.append(f"constant\tnss\t0\t{default}\t{jax}")
        extra = [
            "Also the JAX blend Nautilus reference seed 1 on the default realisation, deferred at",
            "the B3 wrap-up (protocol 10), and the NSS ln 3! constant-likelihood measurement",
            "(protocol 3) that NSS's blend evidence verdict needs.",
        ]
    wall, ref = reference_walls(dataset)
    basis = [
        "cell: misc/reference/run_reference  device: laptop  precision: fp64",
        f"source: measured-wall  wall: {wall:.0f}  ref: {ref}  headroom: {HEADROOM:g}",
    ]
    if dataset == "gaussian_x3_blend":
        basis += [
            "cell: misc/reference/run_constant_likelihood  device: ral_cpu  precision: fp64",
            "lanes: 1  steps: 1  source: unmeasured  probe-first: yes",
        ]
    time_s = HEADROOM * wall + TASK_MARGIN_S
    why = [
        f"Wave-2 references, {dataset}: {len(lines)} array tasks.",
        f"Protocol 6 per backend and per new data realisation {new_realisations(dataset)}:",
        "Nautilus n_live=2000 and DynestyStatic nlive=1000, seeds 0-2 each.",
        *extra,
    ]
    text = header(
        why[0],
        why[1:],
        basis,
        sbatch(
            f"w2_ref_{dataset.removeprefix('gaussian_x3_')}", len(lines), REFERENCE_THROTTLE, time_s
        ),
    )
    text += environment()
    text += read_line(f"reference_{dataset}.tsv", "KIND SAMPLER SEED DATA_SEED CONFIG")
    text += f"""case "$KIND" in
    run)
        python3 scripts/misc/reference/run_reference.py --dataset {dataset} --sampler "$SAMPLER" --seed "$SEED" --data-seed "$DATA_SEED" --config-name "$CONFIG"
        ;;
"""
    if dataset == "gaussian_x3_blend":
        text += """    constant)
        python3 scripts/misc/reference/run_constant_likelihood.py --config-name "$CONFIG" --samplers "$SAMPLER"
        ;;
"""
    text += '    *) echo "unknown kind $KIND" >&2; exit 1 ;;\nesac\n'
    text += FOOTER
    return text, table(("kind", "sampler", "seed", "data_seed", "config"), lines), len(lines)


def build(dataset: str) -> tuple[str, str, int]:
    lines = [
        f"{backend}\t{config}\t{data_seed}"
        for backend, config in wave2.CONFIG_BY_BACKEND.items()
        for data_seed in new_realisations(dataset)
    ]
    map_wall = (protocol.load_json(ROOT / MAP_WALL[0]) or {}).get(MAP_WALL[1]) or 0.0
    basis = [
        "cell: misc/reference/run_map_reference  device: laptop  precision: fp64",
        f"source: measured-wall  wall: {map_wall:.0f}  ref: {MAP_WALL[0]}  headroom: {HEADROOM:g}",
        "cell: misc/reference/build_reference  device: laptop  precision: fp64",
        f"source: measured-wall  wall: {BUILD_WALL_S:g}  ref: plan_wave2.BUILD_WALL_S  "
        f"headroom: {HEADROOM:g}",
    ]
    time_s = max(3600.0, HEADROOM * (map_wall + BUILD_WALL_S) + TASK_MARGIN_S)
    why = [
        f"Wave-2 reference builds, {dataset}: {len(lines)} array tasks.",
        "One per (backend, new realisation): the MAP reference (protocol 6), then",
        "build_reference.py, which pools the six runs and writes reference.json. Submit with",
        "--dependency=afterany on the reference array; a reference whose runs disagree is",
        "recorded as such (protocol 6).",
    ]
    text = header(
        why[0],
        why[1:],
        basis,
        sbatch(
            f"w2_build_{dataset.removeprefix('gaussian_x3_')}", len(lines), BUILD_THROTTLE, time_s
        ),
    )
    text += environment()
    text += read_line(f"reference_build_{dataset}.tsv", "BACKEND CONFIG DATA_SEED")
    text += f"""python3 scripts/misc/reference/run_map_reference.py --dataset {dataset} --data-seed "$DATA_SEED" --config-name "$CONFIG"
python3 scripts/misc/reference/build_reference.py --dataset {dataset} --backend "$BACKEND" --data-seed "$DATA_SEED"
"""
    text += FOOTER
    return text, table(("backend", "config", "data_seed"), lines), len(lines)


def plan() -> dict[Path, str]:
    """``{path: content}`` for every generated file."""
    step_rate = rates.STEP_RATE
    files: dict[Path, str] = {}
    for dataset in wave2.DATASETS:
        text, tsv, _ = reference(dataset)
        files[BATCH / f"submit_wave2_reference_{dataset}"] = text
        files[TABLES / f"reference_{dataset}.tsv"] = tsv
        text, tsv, _ = build(dataset)
        files[BATCH / f"submit_wave2_reference_build_{dataset}"] = text
        files[TABLES / f"reference_build_{dataset}.tsv"] = tsv
        for backend in wave2.CONFIG_BY_BACKEND:
            text, tsv, _ = scored(dataset, backend, step_rate)
            files[BATCH / f"submit_wave2_scored_{dataset}_{backend}"] = text
            files[TABLES / f"scored_{dataset}_{backend}.tsv"] = tsv
    text, tsv, _ = probe()
    files[BATCH / "submit_wave2_probe"] = text
    files[TABLES / "probe.tsv"] = tsv
    rest = remainder()
    if rest is not None:
        text, tsv, _ = rest
        files[BATCH / "submit_wave2_remainder"] = text
        files[TABLES / "remainder.tsv"] = tsv
    return files


def counts(files: dict[Path, str]) -> dict[str, int]:
    """Array tasks and runs per table."""
    out = {}
    for path, content in files.items():
        if path.suffix != ".tsv":
            continue
        lines = content.splitlines()[1:]
        header_fields = content.splitlines()[0].split("\t")
        runs = len(lines)
        if "seeds" in header_fields:
            col = header_fields.index("seeds")
            runs = sum(len(line.split("\t")[col].split(",")) for line in lines)
        out[path.name] = {"tasks": len(lines), "runs": runs}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if rates.STEP_RATE != derived_rates():
        print("wall/rates.py STEP_RATE differs from the pilot walls; update it", file=sys.stderr)
        return 1
    files = plan()
    if args.check:
        stale = [p for p, c in files.items() if not p.exists() or p.read_text() != c]
        known = {*BATCH.glob("submit_wave2_*"), *TABLES.glob("*.tsv")}
        extra = sorted(known - set(files))
        for path in stale + extra:
            print(f"stale: {path.relative_to(ROOT)}", file=sys.stderr)
        if stale or extra:
            print("run plan_wave2.py --write", file=sys.stderr)
            return 1
        print(f"wave-2 plan is current ({len(files)} files)")
        return 0
    if args.write:
        TABLES.mkdir(parents=True, exist_ok=True)
        for path, content in files.items():
            path.write_text(content)
            if path.name.startswith("submit_"):
                path.chmod(0o755)
        print(f"wrote {len(files)} files")
    for name, c in counts(files).items():
        print(f"  {name:48s} tasks {c['tasks']:4d}  runs {c['runs']:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

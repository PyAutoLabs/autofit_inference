# autofit_inference — Agent Instructions

This repo is the single home for **non-linear search and inference benchmarking for
PyAutoFit**: which searches — gradient-free or gradient-based — find the right answer
fastest, and how reliably, on toy likelihoods where the truth is known. The first benchmark
is the `gaussian_x3_blend` + `gaussian_x3_separated` pair: three 1D Gaussians, once blended
and once well separated. It is the sibling of
[`autofit_profiling`](https://github.com/PyAutoLabs/autofit_profiling), which owns PyAutoFit
**timing**, and the PyAutoFit counterpart of
[`autolens_inference`](https://github.com/PyAutoLabs/autolens_inference).

It is a collection of standalone scripts, **not** an installable package — there is no
`pyproject.toml`. These are the canonical, agent-agnostic instructions. `README.md` is the
human-facing overview; `CORTEX.md` says where the rulings of record live.

**Status:** skeleton born 2026-10-07 (PyAutoMind#492, phase B1 of the
`search-extensibility` epic); phase B2 (2026-10-08) added the harness, the two
`gaussian_x3` datasets, the pre-registered protocol and the reference posteriors. There
is no wave-1 row yet: those are B3.

## Where this repo is going

The work arrives in later phases of the `search-extensibility` epic, whose ledger is
`PyAutoMind/draft/research/autofit/search_extensibility_epic.md`
([on GitHub](https://github.com/PyAutoLabs/PyAutoMind/blob/main/draft/research/autofit/search_extensibility_epic.md)):

- **B2** (done) — the benchmark harness, the toy-likelihood datasets, the protocol and the
  reference posteriors.
- **B3** — the first runs (the wave-1 pilot). From B3 this repo reports to
  [PyAutoInsight](https://github.com/PyAutoLabs/PyAutoInsight) as instance `fit`
  (`inference-summary@1`), and its
  [PyAutoCortex](https://github.com/PyAutoLabs/PyAutoCortex) row (`projects.yaml` key
  `autofit_inference`, currently `planned`) flips to `active`.

Until then, do not commit rows under `results/searches/` outside the phase that owns them.

## The protocol comes first

[`wiki/project/protocol_gaussian_x3.md`](wiki/project/protocol_gaussian_x3.md)
(`gaussian_x3@1`) pre-registers what "the right answer" means: point/MAP searches against
the MAP reference, posterior searches on relabelled marginals plus a predictive check and
mode coverage, evidence searches on `ln Z` within one backend under the measured `ln 3!`
offset, and convergence reported separately. Its first commit predates every result row,
and it changes only through a new version committed before the first row it judges.
The thresholds live once, in `scripts/misc/searches/_protocol.py`; PLACEHOLDER values are
calibrated in the wave-1 pilot and then frozen. Wave 1 ranks nothing.

## Import model and leaves

`ruff.toml` is the root sentinel: every script walks up to it and puts the root and
`scripts/misc/` on `sys.path`. A leaf is
`scripts/<dataset_class>/searches/<sampler>/<model_type>.py` and calls
`run_search(sampler=..., dataset_class=..., model_type=...)` with **literal** keyword
arguments — the Brain samplers faculty reads the cell from that call by AST
(`test_runner.py` pins it). The backend environment is exported before autofit is imported.

## Outputs are KEPT

`config/general.yaml` sets `hpc_mode: false`, `remove_files: false` and
`samples_to_csv: true`: the run tree under `output/` (gitignored) is the evidence a row is
rebuilt from, and `scripts/misc/reference/build_reference.py` pools the reference runs'
`samples.csv` from it. Only the small result rows under `results/` are committed.

## Who owns what

- **This repo** executes runs and holds their small committed result rows.
- **PyAutoCortex** records the science: runs, observations and human conclusions, under the
  `autofit_inference` row. Link those records; do not keep competing conclusions here.
- **PyAutoInsight** holds inference campaign intent and aggregates the `inference-summary`
  export once it exists.
- **PyAutoMind** holds bounded implementation tasks, PR lifecycle and claims.

## Repository Structure

```
ruff.toml                  lint config AND the root sentinel (leaf scripts walk up to it)
activate.sh                RAL shared venv + PYTHONPATH (PyAutoNerves, PyAutoFit only)
_autofit_inference_cli.py  flags, the {local,ral}_{numpy,jax_cpu}_{fp64,fp32} grammar, env
config/general.yaml        PyAutoConf overrides (outputs kept)
dataset/<name>/            committed data/noise_map/model/truth JSON (seeded)
scripts/<dataset>/searches/<sampler>/gaussian_x3.py   the leaves
scripts/misc/models/       the gaussian_x3 model, priors and Analysis
scripts/misc/searches/     _runner.py (run_search + registry), _posterior.py, _protocol.py,
                           _metrics.py (MLTracker)
scripts/misc/reference/    reference-posterior, MAP and constant-likelihood runs + combiner
scripts/misc/simulators/   the seeded dataset simulator
scripts/misc/tooling/      build_readme.py, export_inference_summary.py
scripts/misc/wall/         the WALL-BASIS submit gate (rates table empty)
scripts/misc/test/         unit tests + the exporter fixture
results/reference/         reference posteriors per dataset and backend
results/searches/          result rows (from B3)
dashboard/summary.json     inference-summary@1, generated
hpc/                       hpc/sync (laptop-side RAL driver) + the batch_cpu template
wiki/project/              the ledger (state.md) and the protocol
CORTEX.md                  where the rulings of record live
```

## RAL

The RAL copy of this repo is a **git clone** at `/mnt/ral/jnightin/autofit_inference`.
Code goes over with `git pull` on the login node; results come back with `hpc/sync pull`
(`cp hpc/sync.conf.example hpc/sync.conf` first; `sync.conf` is gitignored). Runs here are
CPU runs on `--partition=ral` — there is no `cpu` partition on RAL, and CPU arrays never go
on `gpu`. The PyAuto* libraries resolve from the shared checkouts on `PYTHONPATH`
(`source activate.sh`) and are updated with `HPCPullPyAuto`, never pip-installed. Full
detail in [`hpc/README.md`](hpc/README.md).

## Testing

The PR gate is `.github/workflows/lint.yml` on Python 3.12: `ruff check .`,
`ruff format --check .`, `pytest scripts/misc/test` (the exporter fixture is validated
against a fresh PyAutoInsight checkout), `build_readme.py --check`,
`export_inference_summary.py --check`, `check_submits.py --check` and a `lychee` link
check. It installs no PyAuto library. `.github/workflows/witness.yml` does: it runs one
seed of Nautilus (`n_live=100`, numpy) on `gaussian_x3_blend` from PyAutoFit `main` and
fails unless the exporter judges the row `accepted` against the committed reference.

After a change under `results/` or to the exporter, commit it, then regenerate
`dashboard/summary.json` and commit that separately: the summary stamps the latest
commit touching its producer paths, so it cannot sit in the same commit.

## Related Repos

- `../autofit_profiling` — PyAutoFit timing (search.fit breakdowns, EP/graphical baselines).
- `../PyAutoFit` — the library being exercised (plus `../PyAutoNerves` on `PYTHONPATH`).
- `../autolens_inference` — the PyAutoLens sibling this repo is modelled on.
- `../PyAutoCortex`, `../PyAutoInsight` — the science ledger and the inference dashboard.

## Task Workflows

Keep `ruff check .` and `ruff format --check .` clean, do not commit machine-specific
absolute paths, and flag any change that affects PyAutoFit itself in your PR.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->

<!-- repos_sync:deliverable:begin -->
## Sessions end at their deliverable

A session ends when it reports its deliverable — never arm anything that
outlives the turn to wait for CI, a review or a merge: no `send_later`, no
`subscribe_pr_activity`, no `CronCreate`, no `ScheduleWakeup`, no `/loop`, no
`RemoteTrigger` create/update/run. Judge once, report, stop; the human re-runs
`/prm` (or the batch review) when it is green. Measured: five batch members
armed hourly check-ins on 2026-08-31, and a mobile `/prm` re-armed a 60-minute
`send_later` hourly all night on 2026-09-03 with no task active, draining usage.
<!-- repos_sync:deliverable:end -->

<!-- repos_sync:filing:begin -->
## Where to file

Questions, help with code or an analysis, ideas, bug reports and results from a
user or collaborator — or an agent acting for one — go to
<https://github.com/orgs/PyAutoLabs/discussions> in the matching category
(Help & Questions, Ideas & Proposals, Bugs & Errors, Show and tell;
Announcements is maintainers-only), never to this repo's Issues. An agent never
runs `gh issue create` for such a report: it drafts the title, category and
body and hands them to the human (sessions cannot create Discussions). Only the
development flow — Mind prompt → `/start_dev` → `/create_issue` → one issue per
task → PR — opens issues here. Why: `PyAutoMind/policy/community_surface.md`.
<!-- repos_sync:filing:end -->

<!-- repos_sync:standards:begin -->
## Shared standards

Before changing a shared interface, consult the applicable
[organism standard](https://github.com/PyAutoLabs/PyAutoBrain/blob/main/docs/standards.md)
on demand, identify affected consumers, and validate their adoption. Change
generated guidance at its canonical source and regenerate.
<!-- repos_sync:standards:end -->

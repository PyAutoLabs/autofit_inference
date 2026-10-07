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

**Status:** skeleton born 2026-10-07 (PyAutoMind#492), phase B1 of the
`search-extensibility` epic. There is no harness, no dataset and no result yet.

## Where this repo is going

The work arrives in later phases of the `search-extensibility` epic, whose ledger is
`PyAutoMind/draft/research/autofit/search_extensibility_epic.md`
([on GitHub](https://github.com/PyAutoLabs/PyAutoMind/blob/main/draft/research/autofit/search_extensibility_epic.md)):

- **B2** — the benchmark harness and the toy-likelihood datasets.
- **B3** — the first runs (the wave-1 pilot). From B3 this repo reports to
  [PyAutoInsight](https://github.com/PyAutoLabs/PyAutoInsight) as instance `fit`
  (`inference-summary@1`), and its
  [PyAutoCortex](https://github.com/PyAutoLabs/PyAutoCortex) row (`projects.yaml` key
  `autofit_inference`, currently `planned`) flips to `active`.

Until then, do not add scripts, results or exporters outside the phase that owns them.

## Who owns what

- **This repo** executes runs and holds their small committed result rows.
- **PyAutoCortex** records the science: runs, observations and human conclusions, under the
  `autofit_inference` row. Link those records; do not keep competing conclusions here.
- **PyAutoInsight** holds inference campaign intent and aggregates the `inference-summary`
  export once it exists.
- **PyAutoMind** holds bounded implementation tasks, PR lifecycle and claims.

## Repository Structure

```
ruff.toml        lint config AND the root sentinel (leaf scripts walk up to it)
activate.sh      RAL shared venv + PYTHONPATH (PyAutoNerves, PyAutoFit only)
hpc/             hpc/sync (laptop-side RAL driver) + the batch_cpu submit template
wiki/project/    the project ledger (state.md) + its journal-entry template
CORTEX.md        where the rulings of record live
```

`scripts/`, `results/` and `output/` arrive with the harness in B2.

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
`ruff format --check .` and a `lychee` link check over `README.md` and `AGENTS.md`. It
installs no PyAuto library and runs no script.

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

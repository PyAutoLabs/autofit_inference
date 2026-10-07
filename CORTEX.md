# Where the rulings of record live

The **rulings of record** for this repository — what is decided, and therefore what a run
is allowed to assume — live in
[PyAutoCortex](https://github.com/PyAutoLabs/PyAutoCortex), in `projects.yaml`, row
`autofit_inference`. That row is the machine-readable body map for this repo: the remote,
the RAL root, the sync CLI and its verbs, the ledger, the assistant and the witness every
run is judged by.

**Status: `planned`.** The row was added 2026-10-07 at phase B1 of the
`search-extensibility` epic (PyAutoMind#492), before any run exists. It stays `planned`
until phase B3's first runs (the wave-1 pilot) flip it to `active` with `cortex.py new`,
which also creates the Cortex-side ledger `projects/autofit_inference.md`.

From B3, inference campaign intent and evidence aggregate in
[PyAutoInsight](https://github.com/PyAutoLabs/PyAutoInsight) as instance `fit`
(`inference-summary@1`). Existing project drivers execute runs; PyAutoMind retains
bounded implementation issues, PR lifecycle and claims.

## The project ledger

**`wiki/project/state.md`** is this project's ledger (`ledger: wiki/project/state.md` in the
Cortex row): commentary and journal, newest last, one entry per run or finding, copied from
`wiki/project/_template.md`. Where the ledger and a Cortex ruling disagree, the ruling
counts.

- **<https://pyautolabs.github.io/PyAutoCortex/>** — the current scientific board.
- **<https://pyautolabs.github.io/PyAutoInsight/>** — inference campaigns and evidence.

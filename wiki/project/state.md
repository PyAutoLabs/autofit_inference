# autofit_inference — project state

The Cortex ledger for this repository (`PyAutoCortex/projects.yaml`, row
`autofit_inference`, `ledger: wiki/project/state.md`). Commentary, not the register:
where the ledger and a Cortex ruling disagree, the ruling counts.

## Science goal

Which PyAutoFit non-linear searches — gradient-free and gradient-based — find the right
answer fastest, and how reliably, from a cold start and across seeds, on toy likelihoods
where the truth is known. The first benchmark is `gaussian_x3_blend` +
`gaussian_x3_separated`.

## Now: harness, datasets, protocol and the numpy reference (B2); wave 1 is B3

B2 (2026-10-08) added the harness (`run_search`, the `{local,ral}_{numpy,jax_cpu}_{fp64,fp32}`
grammar, MLTracker, the exporter with producer-asserted verdicts), the seeded
`gaussian_x3_blend` / `gaussian_x3_separated` datasets, the pre-registered protocol
[`protocol_gaussian_x3.md`](protocol_gaussian_x3.md) (`gaussian_x3@1`, committed before
any row) and the reference posteriors. Reference status lives in the protocol's §10:
the numpy blend reference is complete (three Nautilus `n_live=2000` runs agreeing to
0.019 nat and 0.015 σ; MAP ln P 186.668); the JAX Nautilus legs and both separated-control
references are pending, and B3 must land them before judging rows that need them. The
CI witness (`witness.yml`) runs 1 seed of Nautilus `n_live=100` and requires `accepted`.

Born 2026-10-07 as phase B1 of the `search-extensibility` epic (PyAutoMind#492; ledger
`PyAutoMind/draft/research/autofit/search_extensibility_epic.md`). The repo has a lint-green
skeleton, is registered in the Mind, Heart and Cortex (`status: planned`), and has a clone
on RAL. B2 brings the harness and datasets; B3's first runs flip the Cortex row to
`active` and start the `fit` instance in PyAutoInsight (`inference-summary@1`).

## Runs

No wave-1 rows yet. Reference runs (not wave-1 rows) are under `results/reference/`.

## Journal

Copy `_template.md` to the bottom of this section, newest last.

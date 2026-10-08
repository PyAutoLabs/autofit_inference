# autofit_inference — project state

The Cortex ledger for this repository (`PyAutoCortex/projects.yaml`, row
`autofit_inference`, `ledger: wiki/project/state.md`). Commentary, not the register:
where the ledger and a Cortex ruling disagree, the ruling counts.

## Science goal

Which PyAutoFit non-linear searches — gradient-free and gradient-based — find the right
answer fastest, and how reliably, from a cold start and across seeds, on toy likelihoods
where the truth is known. The first benchmark is `gaussian_x3_blend` +
`gaussian_x3_separated`.

## Now: wave-1 pilot stopped at 223 of 520 runs (B3); the rest is wave 2 on RAL

B3 (2026-10-08) ran the wave-1 pilot of `gaussian_x3@1` on one shared laptop and was
stopped by the wrap-up ruling with **223 of 520** expected runs on disk
(`results/searches/`, every row flagged `pilot`). The pilot **ranks nothing**; it
calibrates the protocol. Every run without a row is `deferred` with its reason in
`catalogue/search_catalogue.json` and `dashboard/summary.json`: wave 2 on RAL
`--partition=ral`, NSS to phase A3b. The pilot ran on PyAutoFit main `0dbf258c4f5e`,
**before phase A2** (PyAutoFit#1679, gradients under JAX): the BFGS/LBFGS non-finite
results are expected to change once A2 merges and are not a judgement on those searches.
References: numpy blend and both separated references are complete; the JAX blend
reference is pending at 2 of 3 Nautilus runs (seed 1 deferred), so JAX blend rows are
`not_assessed`. The calibration record (`results/calibration/gaussian_x3_wave1.json`)
keeps the σ-ratio band and ppc tolerance at `@1` (17 calibration rows < 20); the `@2`
freeze is deferred and still precedes any wave-2 row (protocol §7).

Born 2026-10-07 as phase B1 of the `search-extensibility` epic (PyAutoMind#492; ledger
`PyAutoMind/draft/research/autofit/search_extensibility_epic.md`). The repo has a lint-green
skeleton, is registered in the Mind, Heart and Cortex (`status: planned`), and has a clone
on RAL. B2 brought the harness and datasets; B3's pilot flips the Cortex row to
`active` and starts the `fit` instance in PyAutoInsight (`inference-summary@1`).

## Runs

Wave-1 pilot rows (223, `pilot: true`) are under `results/searches/`; reference runs are
under `results/reference/`. Per-cell counts: README "Search rows" and the catalogue.

## Journal

Copy `_template.md` to the bottom of this section, newest last.

### 2026-10-08 — wave-1 pilot, stopped at 223 of 520 runs

**What ran.** Local CPU (one shared WSL laptop, no RAL job ids), `local_numpy_fp64` and
`local_jax_cpu_fp64` (warm cache), both datasets, search seeds 0–9 where reached, on
PyAutoFit main `0dbf258c4f5e` (pre-A2). 223 rows: point/MAP 160 (every planned seed),
evidence 40, posterior 23; one BlackJAXNUTS row is a failed trace
(`TracerBoolConversionError` on the asserted blend). Also the JAX blend Nautilus
`n_live=2000` reference seeds 0 and 2. Stopped by the B3 wrap-up ruling; 297 runs
deferred (wave 2 on RAL `--partition=ral`; NSS to A3b).

**What we learned.** Pilot only — nothing is ranked. Accepted / attempted under `@1`:
separated evidence 15/20, separated point/MAP 10/80, separated posterior 3/11, blend
evidence 7/20 (10 not assessed: JAX blend reference pending), blend point/MAP 0/80 (60
not assessed), blend posterior 0/12 (8 not assessed). BFGS/LBFGS non-finite results
precede A2 and are expected to change. Calibration: 17 nested calibration rows (< 20),
so the σ-ratio band and ppc tolerance stay at `@1`.

**Next.** Run the deferred cells (and the JAX blend reference seed 1) on RAL
`--partition=ral` on a PyAutoFit revision after A2, then freeze `gaussian_x3@2`.

# autofit_inference — project state

The Cortex ledger for this repository (`PyAutoCortex/projects.yaml`, row
`autofit_inference`, `ledger: wiki/project/state.md`). Commentary, not the register:
where the ledger and a Cortex ruling disagree, the ruling counts.

## Science goal

Which PyAutoFit non-linear searches — gradient-free and gradient-based — find the right
answer fastest, and how reliably, from a cold start and across seeds, on toy likelihoods
where the truth is known. The first benchmark is `gaussian_x3_blend` +
`gaussian_x3_separated`.

## Now: `gaussian_x3@2` frozen; scored wave 2 on RAL (B5a)

**Freeze (B5a, 2026-10-14).** `gaussian_x3@2` (protocol §11 A2, `_protocol.FROZEN_V2`) was
committed in **`72d6faaf1717`** (`72d6faaf17171af1573de0f1d2431dbf4444a947`) on branch `feature/search-ext-b5a-wave2`
(autofit_inference#8) **before any scored run existed**. It freezes the calibration
record's six values (σ-ratio band [0.5, 2.0] and ppc tolerance 10 kept at `@1` on 17 rows;
mode-coverage radius 5.5 px; ESS/R-hat bars kept) and the timeout/censoring rules; the 223
pilot rows stay `@1` with unchanged verdicts. The phase gates are met: A2 (PyAutoFit#1679)
and A4 (PyAutoFit#1685–#1687, merged as `df6d218f1`).

**Scored wave 2.** 50 seeds × realisations 1–5 × every non-deferred leg × both datasets
(13 000 expected runs: 11 750 scored now, 20 probe runs for the 4 probe-first cells whose
other 980 runs follow the probe, 250 deferred — blend warm NUTS), on RAL `--partition=ral`
against a private PyAutoFit pinned at `df6d218f10cd` (PyAutoNerves `7f3055b7f3d1`) under
`/mnt/ral/jnightin/autofit_inference/.pyauto_pin/`, never the shared mirror. References
for the four new realisations per dataset run first. Submission and job ids: `## Runs`.
Pull, catalogue and `@2` verdicts are the next task (B5b).

**Carried to B5b (review of the B5a branch).** (1) The JAX blend reference on realisation 1
can only be built on the laptop (its seeds 0 and 2 are pilot-era runs whose `samples.csv`
is not on RAL): pull Nautilus seed 1, rebuild, and decide explicitly whether the pilot's
`@1` JAX blend rows (today `not_assessed`, reference pending) are re-judged against it —
that is a reassessment (review 06 item 8), not a silent change. Until then the scored
JAX blend realisation-1 rows are written `not_assessed`. (2) The `@2` timeouts are 10× the
laptop pilot medians. The B5a dry run on RAL (job 408236, `euclid-ral-compute-10-1`,
2 CPUs, from a throwaway copy so none of its rows is a wave row) imported the pins and
measured RAL **4–7× faster** than the laptop pilot: DynestyStatic reference leg 122 s
(laptop 518–543 s), DynestyStatic `nlive_200` numpy 28 s (pilot median 173 s),
MultiStartAdam 14 s (102 s), NSS 50-step 26 s. The timeouts are therefore lenient on
RAL, never tighter than the protocol intends — recorded as a limitation of every
censoring decision. It also found `zeus` missing from the shared venv: the pin carries
it in `.pyauto_pin/site` (the shared venv is not modified). (3) The pilot tooling (catalogue,
calibration, `inference-summary@1`) reads wave-1 rows only, so pulling wave-2 rows never
moves the pilot outputs.

## Before: wave-1 pilot stopped at 223 of 520 runs (B3); the rest is wave 2 on RAL

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
keeps the σ-ratio band and ppc tolerance at `@1` (17 calibration rows < 20).

**Freeze gate (protocol §7, 2026-10-14).** `gaussian_x3@2` is frozen from that existing
calibration **before any scored run**; the pilot is not restarted, and its 297 deferred
runs become scored wave-2 runs, never calibration input. The freeze proceeds on **17
calibration rows against the 20-row target**. That is acceptable because the
pre-declared < 20 rule already keeps the σ-ratio band and ppc tolerance at `@1`, so
`@2` loosens nothing on thin evidence; and the 3 missing rows cannot come from the pilot
output on disk (every completed nested row with a complete reference is already counted,
JAX blend nested rows wait on the pending JAX blend reference, and the only row-less run
directories are two interrupted non-nested runs). Scored wave 2 still waits for a
PyAutoFit revision after phases A2 **and** A4.

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

**Next.** Freeze `gaussian_x3@2` from the existing 17-row calibration (protocol §7)
before any scored run; do not restart the pilot. Then, on a PyAutoFit revision frozen
after phases A2 and A4, run scored wave 2 (which absorbs the deferred cells) and the JAX
blend reference seed 1 on RAL `--partition=ral` (phase B5).

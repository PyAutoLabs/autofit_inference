# autofit_inference — project state

The Cortex ledger for this repository (`PyAutoCortex/projects.yaml`, row
`autofit_inference`, `ledger: wiki/project/state.md`). Commentary, not the register:
where the ledger and a Cortex ruling disagree, the ruling counts.

## Science goal

Which PyAutoFit non-linear searches — gradient-free and gradient-based — find the right
answer fastest, and how reliably, from a cold start and across seeds, on toy likelihoods
where the truth is known. The first benchmark is `gaussian_x3_blend` +
`gaussian_x3_separated`.

## Now: scored wave 2 judged under `gaussian_x3@2` (B5b); probe-first remainder on RAL

**Judged (B5b, 2026-10-15).** All 11,770 pulled wave-2 rows (11,750 scored arrays + the
20-run probe) are admitted by `_wave2.admit`. Every row is at the pin `df6d218f10cd` under
`gaussian_x3@2`, and none is refused. Every expected run is accounted for
(`results/accounting/gaussian_x3_wave2.json`, `judge_wave2.py`):

- **scored arrays:** 11,750 expected; 10,580 rows, 1,170 failed attempts, 0 missing, 0
  operational kills;
- **probe:** 20 of 20 rows;
- **probe-first remainder:** 980 missing, submitted as job 409930;
- **deferred:** 250 (blend warm NUTS).

Every one of the 20 references, for 2 datasets × 2 backends × 5 realisations, is
`complete`. Each consists of the Nautilus runs only: on every one, DynestyStatic's ln Z
disagrees with Nautilus by 1.3–7.3 nat (§6, recorded as a limitation). The JAX blend
realisation-1 reference was rebuilt on the laptop from Nautilus seeds 0 and 2 (pilot) and
seed 1 (RAL).

The `gaussian_x3@2` section of `catalogue/search_catalogue.json` carries each leg's
success rate (Wilson 95 %) and expected wall per right answer (bootstrap 95 %). It orders
the closed legs with at least one right answer within one task and one dataset. The
README's "Scored wave 2" table renders it, and the journal entry below gives the headline.

**What stays open.**

- **Probe-first remainder.** The probe-first cells (Zeus JAX, NSS 50-step) got their `@2`
  timeouts from the probe (`results/calibration/gaussian_x3_wave2_probe.json`): NSS 600 s
  (the floor; median 26.6 s) and Zeus 1168 s (median 116.8 s). Their remaining 980 runs
  are RAL job 409930, to be pulled and judged in B5b-2. Until then those legs are `open`
  and not ordered.
- **Wave-2 summary records.** `dashboard/summary.json` withholds the wave-2 records on
  size (~84 MB for 11,770 records; question on autofit_inference#10). It carries the
  pilot records, the `protocols` list and the wave-2 accounting.
- **Pilot rows.** The 223 pilot rows are not re-judged (B5b ruling 1). The 78 JAX blend
  pilot rows stay `not_assessed` with "reference completed after the pilot (B5b)".

## Before: `gaussian_x3@2` frozen; scored wave 2 on RAL (B5a)

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

**Scored wave 2, submitted 2026-10-14 17:19 BST** on RAL `--partition=ral` (excluding the
four 252-CPU nodes), 2 CPUs and 6 GB per task, from branch `feature/search-ext-b5a-wave2`
at `f0b3eef`, against the private pin PyAutoFit `df6d218f10cd` / PyAutoNerves
`7f3055b7f3d1` (+ `zeus` 2.5.4 in `.pyauto_pin/site`). `squeue` at submission: other
campaigns held `ral` with three 252-CPU `euclid` Pilot__Xlarge jobs (408225, 408226,
408230) and one 236-CPU `mlam` job (408241); 1196 of 2188 `ral` CPUs idle; no pending GPU
jobs. Nothing is pulled or scored in B5a.

| job | array (throttle) | what | depends on |
|---|---|---|---|
| 408244 | 0-49 (%50) | blend references: realisations 2–5 × both backends × Nautilus/DynestyStatic seeds 0–2, + JAX Nautilus seed 1 on realisation 1, + NSS `ln 3!` | — |
| 408245 | 0-47 (%48) | separated references: realisations 1, 3, 4, 5 × both backends × 6 runs | — |
| 408246 | 0-19 (%20) | probe: Zeus JAX and NSS 50-step, both datasets, realisation 1, seeds 0–4 | — |
| 408346 | 0-7 (%8) | blend reference builds (MAP + `build_reference.py`) | afterany 408244 |
| 408347 | 0-7 (%8) | separated reference builds | afterany 408245 |
| 408348 | 0-379 (%60) | scored blend numpy, 2250 runs | afterany 408346, 408347 |
| 408365 | 0-344 (%60) | scored blend jax_cpu, 3500 runs | afterany 408346, 408347 |
| 408366 | 0-379 (%60) | scored separated numpy, 2250 runs | afterany 408346, 408347 |
| 408367 | 0-389 (%60) | scored separated jax_cpu, 3750 runs | afterany 408346, 408347 |

Dry run (not part of the wave): 408236. Peak footprint 240 scored tasks = 480 CPUs.

**All nine finished and were pulled in B5b (2026-10-15).** sacct: all 1,629 array tasks
`COMPLETED`, exit 0. The table gives each job's wall from its first start to its last end:

| job | wall | pulled |
|---|---|---|
| 408244 | 1:04 | rows in `results/reference/gaussian_x3_blend/` |
| 408245 | 0:26 | rows in `results/reference/gaussian_x3_separated/` |
| 408246 | 0:03 | 20 probe rows |
| 408346 | 0:01 | reference builds |
| 408347 | 0:01 | reference builds |
| 408348 | 1:53 | 2250 rows |
| 408365 | 1:46 | 3500 rows |
| 408366 | 1:51 | 2250 rows |
| 408367 | 1:57 | 3750 rows |

The pull was `results/` only, as a parallel per-directory tar on the login node: rsync
ran at ~0.16 files/s over the RAL NFS. `output/` was not pulled, because of laptop disk;
only the one run directory the JAX blend reference needs came back.

**Probe-first remainder, submitted 2026-10-15 (B5b):**

| job | array (throttle) | what |
|---|---|---|
| 409930 | 0-19 (%20) | `submit_wave2_remainder`: Zeus JAX + NSS 50-step, both datasets, the 980 runs after the probe, each under its probe-derived `@2` timeout; `--partition=ral`, same `--exclude` and pin guards as B5a |

The submit and its table were generated by `plan_wave2.py` and copied by `scp` into the
RAL clone, which stays at `f0b3eef`. Its runner, protocol values and pin are unchanged by
B5b; the sha256 of both files matches the branch.

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

### 2026-10-15 — scored wave 2 judged under `gaussian_x3@2` (B5b)

**What ran.** RAL `--partition=ral`, jobs 408244–408367 (B5a), against the private
PyAutoFit pin `df6d218f10cd`. 11,770 rows came back: 50 seeds × 5 realisations per leg,
both datasets, plus the 20-run probe. Every row is admitted and every expected run is
accounted for (`results/accounting/gaussian_x3_wave2.json`). Failed attempts are part of
the success rates below:

- 920 non-finite results on the blend: BFGS 250 + 250 and LBFGS 210 + 210, numpy + JAX;
- 250 `TracerBoolConversionError` (cold NUTS on the asserted blend).

The 980 probe-first remainder runs are job 409930.

**What we learned.** These are the `@2` verdicts for the closed legs. They are compared
only within one task and one dataset (`catalogue/search_catalogue.json`, `gaussian_x3@2`);
"wall" is the expected wall per right answer on RAL.

- **Evidence.**
  - Nautilus is accepted on 250/250 at every `n_live` on both datasets and both backends,
    for 250–520 s per right answer.
  - DynestyStatic is accepted on 36/250 (blend) and 103/250 (separated), and
    DynestyDynamic on 59/250 and 145/250. Their failures are mostly §4(c): ln Z misses
    the Nautilus-only reference by more than 1 nat, matching the DynestyStatic–Nautilus
    disagreement inside every reference.
  - Because DynestyStatic is cheap, its wall per right answer (62–190 s) is the lowest
    despite the low success rate. The ordering follows §8 strictly, so read the success
    rate beside it.
- **Point/MAP.**
  - MultiStartProdigy is the only search that succeeds: 152/250 on the blend (18 s per
    right answer) and 250/250 on the separated control (9 s).
  - BFGS, LBFGS, MultiStartAdam, MultiStartADABelief and MultiStartLion are 0/250 on both
    datasets. On the blend, BFGS and LBFGS are mostly non-finite; elsewhere they stop at
    local optima far below the reference MAP.
  - LBFGS returning +inf on blend numpy at the pin is recorded as an outcome (carried
    decision c), not fixed.
- **Posterior.**
  - Warm BlackJAXNUTS on the separated control is 250/250 (237 s).
  - Zeus numpy on the separated control is 198/250 (216 s).
  - Emcee, cold NUTS, SMC, and Zeus numpy on the blend are 0/250. Emcee does not mix in
    2000 steps (its medians sit many σ from the reference), and cold NUTS fails at trace time on the blend.
- **numpy vs JAX.** The two backends gave identical acceptance counts for every Dynesty
  and Nautilus leg (same seeds, data and algorithm). Only the walls differ.

Two limitations bear on these numbers:

1. **Lenient timeouts.** The pilot-derived timeouts are lenient on RAL (10× laptop
   medians; RAL is 4–7× faster). No scored attempt hit its timeout.
2. **Pilot not re-judged.** The pilot is not re-judged: its 78 JAX blend rows stay
   `not_assessed` ("reference completed after the pilot (B5b)").

**Next.**

1. **B5b-2.** Pull job 409930 and judge the Zeus JAX and NSS cells under their probe
   timeouts, closing the wave.
2. **Summary form.** The human chooses the form for the wave-2 summary records
   (autofit_inference#10).


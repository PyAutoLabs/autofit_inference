# Protocol `gaussian_x3@1` — pre-registered success criteria

**Pre-registered 2026-10-08, before any wave-1 row exists** (search-extensibility epic,
phase B2; decisions D15 and D16 of `PyAutoMind/draft/research/autofit/search_extensibility_epic_report.md`
§8). Git proves the order: this file's first commit predates every file under
`results/searches/`. The exporter stamps every verdict it emits with
`protocol_id: gaussian_x3@1`.

**Versions.** `gaussian_x3@1` judges the 223 wave-1 pilot rows (their verdicts are
unchanged). `gaussian_x3@2` — this document with amendment **A2** (§11) applied — was
frozen on 2026-10-14 before any scored run and judges every scored wave-2 row; the
exporter stamps each verdict with the protocol of its row's wave
(`_protocol.protocol_for`), and every `@2` verdict reason starts with `gaussian_x3@2:`.

**Change rule.** Nothing in §1–§9 changes after a wave-1 row is committed except
through a new protocol version (`gaussian_x3@2`, …) whose diff and reason are recorded
in §11, committed before the first row judged by it. Values marked **PLACEHOLDER** are
calibrated in the pilot (wave 1) and then frozen (D16, §7). §10 (reference status) is
bookkeeping, not criteria, and is updated as references land.

## 1. The question

Which PyAutoFit searches find the right answer on `gaussian_x3`, how reliably across
seeds, and at what wall-clock cost per right answer. A search answers one of three
**tasks**, and is judged only within its task:

| Task | Searches (wave 1) | Judged by |
|---|---|---|
| point/MAP | LBFGS, BFGS, MultiStartAdam, MultiStartProdigy, MultiStartADABelief, MultiStartLion | §4(a) |
| posterior | Emcee, Zeus (numpy and JAX legs), BlackJAXNUTS (cold, and warm from a short Nautilus), SMC | §4(b) |
| evidence | Nautilus (`n_live` 100/200/400), DynestyStatic, DynestyDynamic, NSS | §4(b) and §4(c) |

Tasks are never ranked against each other, and the catalogue groups by task, not by
algorithm family. Every registered search appears with a status: `measured`,
`unsupported`, `deferred` or `failed`. `Drawer` is a sanity floor and is not
benchmarked. **NSS on the blend is `deferred` until phase A3b** (NSS onto `Fitness`).

## 2. Model, priors and likelihood

Defined once, in `scripts/misc/models/gaussian_x3.py`.

- Three `af.ex.Gaussian` (`g0`, `g1`, `g2`: `centre`, `normalization`, `sigma`) plus a
  thin `Background(level)` summed in the Analysis: **10 free parameters**.
- Priors, shared by the three Gaussians: centre `U(0, 100)`, normalization
  `LogUniform(1e-2, 1e2)`, sigma `U(0.5, 30)`; background level `U(-1, 1)`.
- Ordered-centre assertions `g0.centre < g1.centre < g2.centre` (§3).
- **The `gaussian_x3_separated` control (amendment A1, §11)** replaces the shared centre
  prior with disjoint ones, `g0 U(0, 35)`, `g1 U(35, 65)`, `g2 U(65, 100)` (truth
  centres 20 / 50 / 80, each 15 px inside its range), and has **no** assertions; every
  other prior is as above. The model is chosen per dataset in `build_model(dataset=…)`.
- Likelihood: Gaussian noise with the normalisation kept,
  `ln L = -0.5 Σ [((d − m)/n)² + ln(2π n²)]`, on `x = 0..99`.
- Point/MAP searches optimise the **log posterior** (PyAutoFit's MLE searches add the
  log prior; a LogUniform prior contributes `−ln x`). Every row records both
  `max_log_likelihood` and `max_log_posterior`, and they are never interchanged.

## 3. Label convention (D15)

The three Gaussians are exchangeable: every good fit has `3! = 6` label-permuted
copies. The convention is chosen per dataset (amendment A1, §11): `gaussian_x3_blend`
uses (a); the `gaussian_x3_separated` control uses D15 option **(b)**, disjoint centre
priors with no assertions (`assertion_mechanism: disjoint_priors`), which excludes no
prior volume, so its `ln 3!` offset is `0` by construction and is not measured. Both
record (c).

- **(a) The model fitted is the user-facing one** (the blend), with the ordered-centre
  assertions.
  Their enforcement differs by backend: on numpy a violated assertion raises inside
  `instance_from_vector` and `Fitness` returns the resample sentinel; on JAX the
  assertions are a traced boolean applied with `xp.where` to the figure of merit. The
  two mechanisms, and different samplers' initialisers, can count the excluded prior
  volume differently, so **log evidences are compared only within one backend, one
  assertion mechanism and one sampler-offset convention** (below).
- **(c) Every row also records relabelled statistics**: each sample's components are
  sorted by centre before marginals are taken (`posterior_relabelled`, the basis of
  every §4 comparison), plus `label_permutations_found` (raw orderings carrying ≥ 1 %
  of the weight) and `modes_found` (clusters of the relabelled centre triple; a sample
  joins a cluster when every centre is within **5 px — PLACEHOLDER** of its seed;
  clusters below 1 % weight are dropped; `scripts/misc/searches/_posterior.py`).
- **The `ln 3! ≈ 1.792` nat convention.** The ordered region is exactly 1/6 of the
  exchangeable prior volume. A sampler that counts the excluded volume reports
  `ln Z_ordered = ln Z_normalised − ln 3!`; one whose initial live points are drawn
  only from the allowed region reports `ln Z_normalised` directly. The offset each
  (sampler, backend) applies is **measured, not assumed**, on a near-constant
  likelihood (`ln L = −10⁻³ level²`, exact exchangeable evidence
  `ln(√(π/10⁻³) erf(√10⁻³)/2) ≈ −3.3 × 10⁻⁴`): one run with the assertions and one
  without, per sampler and backend. The validation passes when
  `ln Z_assert − ln Z_exch` is within 0.1 nat of either `−ln 3!` or `0`; the measured
  offset (`0` or `−ln 3!`) is recorded in `results/reference/constant_likelihood.json`,
  and every evidence comparison uses `ln Z_normalised = ln Z − offset`. A result in
  neither window fails the validation and blocks §4(c) for that sampler and backend.

## 4. Success criteria (per run)

All comparisons are against the **reference** of §5 for the **same dataset and the
same backend**, using relabelled statistics. A missing reference makes the verdict
`not_assessed` with that reason.

**(a) point/MAP** — `max_log_posterior ≥ ref_MAP_log_posterior − 1` nat (the MLTracker
tolerance). Point/MAP runs are judged on (a) alone. Posterior and evidence rows record
(a) as a diagnostic, not a criterion.

**(b) posterior accuracy** (feeds Insight `scientific.acceptance`), all of:

1. every parameter: `|median − ref_median| / ref_σ ≤ 1`;
2. every parameter: `σ_run / σ_ref ∈ [0.5, 2.0]` — **PLACEHOLDER band**;
3. posterior-predictive residual: `ppc_χ² − ref_ppc_χ² ≤ 10` — **PLACEHOLDER**, where
   `ppc_χ²` is the χ² of the pixel-wise median of 500 equal-weight posterior draws'
   model curves;
4. mode coverage: every reference mode carrying ≥ 5 % of the reference weight has a run
   mode whose centres lie within the §3 mode radius.

**(c) evidence** — `|ln Z_normalised − ref_ln Z_normalised| ≤ 1` nat, same backend and
same assertion mechanism, under the §3 offset convention.

**Verdict.** point/MAP: `accepted` iff (a). posterior: `accepted` iff (b). evidence:
`accepted` iff (b) and (c). Otherwise `rejected`, and `reason` lists every failed
criterion with its number. A run that did not complete is `rejected` with reason
`not completed: <status>`.

`truth_delta_sigma` against the generating truth is recorded too. It checks the data
and the model, not the search: across data seeds it should look roughly N(0, 1).

## 5. Convergence diagnostics (separate from acceptance)

Feeds Insight `scientific.convergence` (`converged` / `not_converged` /
`not_assessed`). Convergence never implies acceptance, nor the reverse.

- **Nested** (Nautilus, Dynesty, NSS): `converged` iff the run completed, its own
  termination criterion was met (`dlogz` for Dynesty, `f_live` / `n_eff` for
  Nautilus) and the Kish ESS of the weighted samples is ≥ **500 — PLACEHOLDER**.
- **Chains** (Emcee, Zeus, NUTS, SMC): `converged` iff split-R-hat ≤ **1.01** and bulk
  ESS ≥ **400** on every parameter — **PLACEHOLDERS**. Where a search exposes no
  chains, `not_assessed` with that reason.
- **point/MAP**: `not_assessed` (an optimiser's exit flag is not a convergence
  diagnostic; (a) is the test).

## 6. Reference posteriors and the MAP reference

Computed **per dataset and per backend** (numpy and JAX), with the dataset's §3 label
convention (the blend's assertions; the control's disjoint priors, A1):

- 3 long `Nautilus` runs (`n_live=2000`, seeds 0, 1, 2) and 3 long `DynestyStatic`
  runs (`nlive=1000`; PyAutoFit forwards no seed to Dynesty, so its seeds label the run
  only — the search-level seed is phase A4).
- **Agreement required**: `ln Z_normalised` spread (max − min over all six) ≤ 0.2 nat,
  and every relabelled median within 0.1 `ref_σ` of the pooled median. If they do not
  agree, §10 records why, and the reference is the pooled set of the runs that do, with
  the disagreement listed as a limitation of every verdict that uses it.
- The reference values are the pooled (equal-weight-per-run) relabelled marginals,
  `ref_ln Z_normalised` = the mean over runs, and `ref_ppc_χ²` = the mean over runs.
- **MAP reference**: a long MultiStart (`MultiStartAdam`, many starts) followed by an
  LBFGS polish started from the reference's best (max log posterior) sample; the higher
  of the two polished log posteriors is `ref_MAP_log_posterior`.
- Committed as JSON under `results/reference/<dataset>/<backend>/`.

## 7. Calibration: pilot, then freeze (D16)

- Every PLACEHOLDER (σ-ratio band, ppc tolerance, mode radius, ESS and R-hat bars), the
  per-search timeout and the censoring rule are calibrated on the **reference runs'
  seed scatter and the wave-1 pilot**, never on fewer seeds than the pilot has.
- The calibrated values are frozen as `gaussian_x3@2`, committed before the first
  wave-2 row. Wave-1 verdicts stay stamped `@1`.
- Until then a timeout is censored as a failure at **10× the median wall of
  Nautilus `n_live=100` on the same config — PLACEHOLDER**.
- **Status at the B3 wrap-up (2026-10-08).** The pilot was stopped by the wrap-up
  ruling at 223 of its 520 expected runs (the rest deferred to wave 2 on RAL
  `--partition=ral`; NSS to phase A3b). `results/calibration/gaussian_x3_wave1.json`
  records the evidence the pre-declared rules give on those rows: 17 calibration rows
  (< 20), so the σ-ratio band and the ppc tolerance stay at their `@1` values. The
  pilot ran on PyAutoFit main `0dbf258c4f5e`, before phase A2 (PyAutoFit#1679,
  gradients under JAX).
- **Freeze gate (B3c, 2026-10-14) — the one rule.** `gaussian_x3@2` is frozen from the
  calibration that already exists (`results/calibration/gaussian_x3_wave1.json`, 17
  calibration rows) and committed **before any scored run**. The pilot is **not**
  restarted and its 297 deferred runs are **not** calibration input: they move to the
  scored wave 2 and are judged under `@2`. Scored wave 2 still waits for a PyAutoFit
  revision frozen after phases **A2 and A4** (§9). Any calibration extension a human
  authorizes later is a separately identified run set, committed under its own record
  and excluded from scored results; it never changes `@2` after the first scored row.
- **Frozen (B5a, 2026-10-14).** `gaussian_x3@2` is §11 amendment A2, with the values in
  `_protocol.FROZEN_V2` (pinned to the record's `frozen` by a test): σ-ratio band
  [0.5, 2.0] and ppc tolerance 10 (kept at `@1`: 17 rows < 20), §4(b)4 mode radius
  **5.5 px** (raised from 5 to clear 10 × the largest reference centre σ, 5.32 px; ≤ half
  the smallest truth separation, 7.5 px), nested Kish ESS 500, chain R-hat 1.01 and bulk
  ESS 400 (kept: standard), and the timeout and censoring rules below. Freeze commit:
  `72d6faaf1717` (72d6faaf17171af1573de0f1d2431dbf4444a947) (autofit_inference), committed before any scored wave-2 run.
- **Why 17 rows suffice for the freeze.** The < 20 rule is pre-declared and
  conservative: below it the σ-ratio band and ppc tolerance keep their `@1` values, so
  `@2` widens nothing from thin evidence. The 3 missing rows cannot be recovered from
  the pilot output on disk: every completed nested row with a complete reference is
  already among the 17; the JAX blend nested rows need the pending JAX blend reference
  (seed 1 not run); the only run directories without a row are two interrupted
  non-nested runs (separated SMC and warm NUTS, JAX, seed 1). Reaching 20 would need
  new compute, which the B3 wrap-up ruling ended.

## 8. Headline numbers

Per (search × settings × config), never per best seed:

- **success rate** = accepted / attempted, with a **Wilson 95 % interval**; every
  attempt counts, including failures, crashes and timeouts;
- **expected wall per right answer** = Σ `total_wall_s` over all attempts (failed and
  timed-out attempts included, plus any warm-start provider's cost, e.g. the short
  Nautilus a warm NUTS starts from) / number of successes, with a **bootstrap 95 %
  interval** over seeds (10 000 resamples, seed 0). With **zero successes** it is
  reported as unbounded, alongside the lower bound `mean_wall / wilson_upper`.
- Supporting columns: evals-to-target and time-to-target against the **shared target**
  `ref_max_log_likelihood − 1` nat (MLTracker); ESS and ESS/s; `compile_s`; the
  admission bar (`per_call_s`, `likelihood_share`).
- **Timing labels.** On the numpy callback path MLTracker times every evaluation:
  `time_to_target_basis: observed`. Under JAX the likelihood runs inside `jit`/`vmap`
  and the time is interpolated from the eval index: `estimated`. `per_call_s` and
  `likelihood_share` are always `estimated`. `compile_s` comes from separate cold
  (persistent cache disabled) and warm runs (`--compile-cache`), never one run's first
  and second call.

> **Presentation note — post-hoc, not pre-registered (adopted 2026-10-15, B5b-2).**
> This note was written **after seeing the wave-2 data** (DynestyStatic, accepted on 14 %
> of its blend evidence attempts, led that task on expected wall per right answer). It is
> a convention for how the catalogue *presents* the headline numbers above, **not** a
> criterion of this protocol: it changes no verdict, no threshold and no headline number,
> it belongs to no protocol version, and the pre-registered rules of §4–§8 are untouched.
> The `gaussian_x3@2` section of `catalogue/search_catalogue.json` orders legs within one
> task and one dataset (never across) by expected wall per right answer (ascending), then
> success rate, and positions **only** closed, fully-judged legs whose success rate is
> **≥ 50 % over ≥ 20 attempts**. Every other closed leg, zero-success legs included, is
> listed below the ranked block as **unreliable** with its numbers. Every leg shows its
> success rate (Wilson 95 %) and its attempt count. The values live once, in
> `scripts/misc/tooling/build_catalogue.py` (`RANK_FLOOR`).

## 9. Waves

- **Wave 1 is an exploratory pilot; it ranks nothing.** 10 search seeds, local CPU,
  `local_numpy_fp64` and `local_jax_cpu_fp64` where the search is JAX-native, on
  `gaussian_x3_blend` **and** the `gaussian_x3_separated` control. It may run on
  PyAutoFit `main`; every row records the PyAutoFit commit and is flagged `pilot`.
- **Wave 2 is scored.** 50 search seeds × 5 data realisations, on a PyAutoFit revision
  frozen after phases A2 and A4 merge. RAL arrays on **`--partition=ral` only — never
  `gpu`, `ral,gpu` or `gpu,ral`**. This is the human's hard rule of 2026-09-30: CPU
  arrays on `ral,gpu` occupied all 124 CPUs on euclid-ral-gpu-1/-2 and left all 8 A100s
  idle but unschedulable for hours. `hpc/batch_cpu/template` carries the same rule.
- **No GPU leg.** The 100-pixel likelihood is dispatch-bound; an A100 says nothing about
  a search here. There is no `batch_gpu` template.
- **Wave 2's identity (A2).** A scored row belongs to the wave only when its key
  `(dataset, sampler, settings_name, config_id, data_seed, protocol_id, pyautofit_sha)`
  names a menu cell, a realisation 1–5, `gaussian_x3@2` and the pinned PyAutoFit revision
  (`scripts/misc/searches/_wave2.py`); any other row is refused and listed, never counted.
  Every expected run is `row`, `failed`, `missing` or `deferred` (§7 censoring applies to
  `failed` and, once the wave closes, to `missing`).

## 10. Reference status (bookkeeping)

| Dataset | Backend | Nautilus ×3 | DynestyStatic ×3 | MAP | Agreement |
|---|---|---|---|---|---|
| gaussian_x3_blend | numpy | complete | complete | complete | Nautilus: ln Z spread 0.019 nat, medians 0.015 σ. All six: medians 0.062 σ, ln Z spread 2.33 nat (DynestyStatic) → reference = the Nautilus runs |
| gaussian_x3_blend | jax_cpu | pending (2 of 3: seeds 0 and 2 in B3; seed 1 deferred to wave 2 at the B3 wrap-up ruling) | complete | complete | pending |
| gaussian_x3_separated | numpy | complete (B3) | complete (B3) | complete (ln P 175.706) | Nautilus: ln Z spread 0.006 nat, medians 0.010 σ. All six: medians 0.071 σ, ln Z spread 0.49 nat (DynestyStatic) → reference = the Nautilus runs (ln Z 122.931, offset 0: disjoint priors) |
| gaussian_x3_separated | jax_cpu | complete (B3) | complete (B3) | complete (ln P 175.706) | Nautilus: identical to numpy (same seeds, same samples to 1e-12). All six: medians 0.062 σ, ln Z spread 1.32 nat (DynestyStatic) → reference = the Nautilus runs |

The table covers each dataset's committed realisation (blend `data_seed` 1, separated
`data_seed` 2). **Wave-2 realisations (B5a, 2026-10-14):** the other four of 1–5 on each
dataset (blend 2–5, separated 1, 3, 4, 5) are committed under `dataset/<name>/data_seed<d>/`;
their references (3 Nautilus + 3 DynestyStatic + MAP per backend, under
`results/reference/<dataset>/<backend>/data_seed<d>/`) and the JAX blend Nautilus seed 1
are **scheduled** on RAL at the wave-2 pin, job ids in `wiki/project/state.md`. The new
references are made at a later PyAutoFit revision than the B2/B3 ones; the JAX blend
reference will pool pilot-era seeds 0 and 2 with the pinned seed 1 (a limitation it
records). NSS's `ln 3!` offset on the blend (§3) is measured in the same submission.

Constant-likelihood `ln 3!` validation (`results/reference/constant_likelihood.json`):
Nautilus counts the excluded volume (offset `−ln 3!`; measured −1.825 on numpy and JAX at
`n_live=2000`; a first `n_live=500` attempt measured −1.972, outside the window, and is
kept under `attempts`); DynestyStatic draws its initial live points from the allowed
region only (offset `0`, measured 0.000 on both backends).

Rebuilt 2026-10-08 after the B2 adversary review: the mode clustering now covers every
sample with its weight (it had kept only the 4000 heaviest), so the blend/numpy reference
mode weight is 1.000 (was 0.103, the retained mass) and every per-run `modes` weight is
1.000 (was 0.26–0.74); the centres, `modes_found` (1), the pooled medians and σ, ln Z
(137.061; normalised 138.853) and `ppc_chi2` (64.71) are unchanged. The reference now
records its identity `data_seed: 1`, `assertion_mechanism: raise_resample`.

Observation recorded for the pilot, not a criterion: DynestyStatic at PyAutoFit's
default `rwalk` (`walks=5`, `nlive=1000`) reproduces the posterior (medians within
0.06 σ) but its ln Z scatters by 2.3 nat (numpy) and 5.6 nat (JAX) across seeds.

## 11. Amendments

Amendments made before any row they could affect exists. Each names the sections it
changes; the change rule (top) still governs anything after a wave-1 row.

- **A1 — 2026-10-08, the separated control is D15 option (b).** Changes §2 (the
  control's priors), §3 (per-dataset convention) and §6 (the reference's convention).
  *Why:* the epic report's D15 decision (§8.2) makes `gaussian_x3_separated` the
  disjoint-prior control, option (b), so a low blend success rate can be attributed to
  the problem or to the search; the first B2 implementation fitted the control with the
  blend's shared `U(0, 100)` centre priors and ordering assertions, which keeps the
  assertion-volume and prior-geometry effects the control exists to remove (review
  finding 10). *What:* disjoint centre priors `U(0, 35)`, `U(35, 65)`, `U(65, 100)`, no
  assertions, `assertion_mechanism: disjoint_priors`, evidence offset `0` by
  construction. *Timing:* no `gaussian_x3_separated` reference run, reference or row
  existed when it was made (§10), so nothing is rerun and no verdict changes.

- **A2 — 2026-10-14, `gaussian_x3@2`: the freeze for scored wave 2** (B5a, PyAutoMind
  `search_ext_b5a_freeze_and_scored_wave2.md`). Committed before any scored wave-2 run
  (freeze commit `72d6faaf1717` (72d6faaf17171af1573de0f1d2431dbf4444a947)); judges only rows of wave 2; wave-1 verdicts stay
  `@1`. Changes:
  - *§4(b)2, §4(b)3, §5 — the PLACEHOLDERs are frozen* from
    `results/calibration/gaussian_x3_wave1.json`: σ-ratio band [0.5, 2.0], ppc tolerance
    10, Kish ESS 500, R-hat 1.01, bulk ESS 400 — all unchanged in value (17 calibration
    rows < 20 keep the band and tolerance; the ESS/R-hat bars are standard and are not
    loosened because no pilot chain row met the bulk-ESS bar at pilot settings).
  - *§4(b)4 — the mode-coverage radius is 5.5 px* (was 5): the pre-declared bounds are
    ≥ 10 × the largest reference centre σ (5.32 px) and ≤ half the smallest truth centre
    separation (7.5 px). The §3 clustering radius that writes `modes` onto rows and
    references stays 5 px, so rows of every wave and every reference (including the
    pilot-era ones) are clustered by one rule; only the coverage test widens.
  - *§7 — timeouts and censoring.* Per (search × settings × backend):
    `max(600 s, 10 × the median pilot total wall)` (the record's `timeouts`; a run's
    process limit adds 120 s for imports and the admission bar, the censoring test uses
    `total_wall_s`). A cell with no pilot wall (Zeus on JAX, every pilot attempt stopped at
    the wall cap; NSS's new entry) is **probe-first**: five runs on realisation 1, seeds
    0–4, under a 12 h cap; its timeout is the same rule over the probe's median, set before
    the rest of the cell is submitted. Censoring as the record states: a run that hit the
    timeout, crashed, wrote a non-finite answer or left no row is an attempted failure
    with `total_wall_s` set to the timeout when it ran past it; never dropped.
  - *§1, §9 — what wave 2 runs.* NSS is no longer deferred (phase A3b merged) and runs a
    wave-2 settings entry `n_live_200_mcmc_steps_50` (50 slice-MCMC steps ≈ 5 × ndim: at the
    default 5 the A3b witness lost the mode on seed 1 and ln Z by ~27 nat on seed 2; at 50
    seeds 0–2 recovered every centre inside the reference spread; the pilot's `n_live_200`
    entry stays as the pilot's deferred cell). Every search receives the A4 search-level
    `seed` (PyAutoFit#1686), so the 50 seeds are 50 distinct attempts for Dynesty, Emcee,
    Zeus and the optimisers too. 50 seeds × realisations 1–5 × every non-deferred leg ×
    both datasets on `ral_{numpy,jax_cpu}_fp64`, against a private PyAutoFit checkout pinned
    at `df6d218f10cd` (the A4 merge, newest `main` at submission; no release contains it) —
    never the shared RAL mirror. BlackJAXNUTS warm-started on the blend stays **deferred**:
    the cold leg still fails at trace time at the pin (TracerBoolConversionError on the
    asserted blend), so each warm attempt would spend its Nautilus provider and fail; the
    cold leg runs and records the failure.
  *Why:* §7's freeze gate (B3c) — freeze from the calibration that exists, before any
  scored run, never restarting the pilot. *Timing:* no wave-2 row existed when it was
  committed; the arrays were submitted afterwards (job ids in `state.md`).

- **Note N1 — 2026-10-15, the catalogue's reliability floor (B5b-2; not an amendment).**
  The §8 presentation note was added after the scored wave-2 rows existed, so under the
  change rule it cannot be a criterion, and it is not one: it changes no verdict, no
  threshold and no headline number, and it creates no protocol version. *Why:* the
  catalogue's own sort key (expected wall per right answer) let a leg that is right on
  14 % of attempts lead its task; the human-delegated ruling of 2026-10-15
  (autofit_inference#12, ruling 2) keeps that key but ranks only legs with a success rate
  ≥ 50 % over ≥ 20 attempts and lists the rest as unreliable, with every number shown.

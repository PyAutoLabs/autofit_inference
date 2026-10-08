# Protocol `gaussian_x3@1` — pre-registered success criteria

**Pre-registered 2026-10-08, before any wave-1 row exists** (search-extensibility epic,
phase B2; decisions D15 and D16 of `PyAutoMind/draft/research/autofit/search_extensibility_epic_report.md`
§8). Git proves the order: this file's first commit predates every file under
`results/searches/`. The exporter stamps every verdict it emits with
`protocol_id: gaussian_x3@1`.

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
- Likelihood: Gaussian noise with the normalisation kept,
  `ln L = -0.5 Σ [((d − m)/n)² + ln(2π n²)]`, on `x = 0..99`.
- Point/MAP searches optimise the **log posterior** (PyAutoFit's MLE searches add the
  log prior; a LogUniform prior contributes `−ln x`). Every row records both
  `max_log_likelihood` and `max_log_posterior`, and they are never interchanged.

## 3. Label convention (D15)

The three Gaussians are exchangeable: every good fit has `3! = 6` label-permuted
copies.

- **(a) The model fitted is the user-facing one**, with the ordered-centre assertions.
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

Computed **per dataset and per backend** (numpy and JAX), with the §3 assertions:

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

## 10. Reference status (bookkeeping)

| Dataset | Backend | Nautilus ×3 | DynestyStatic ×3 | MAP | Agreement |
|---|---|---|---|---|---|
| gaussian_x3_blend | numpy | complete | complete | complete | Nautilus: ln Z spread 0.019 nat, medians 0.015 σ. All six: medians 0.062 σ, ln Z spread 2.33 nat (DynestyStatic) → reference = the Nautilus runs |
| gaussian_x3_blend | jax_cpu | pending (stopped: beyond the B2 budget) | complete | complete | pending |
| gaussian_x3_separated | numpy | pending (B3, before its wave-1 rows) | pending | pending | pending |
| gaussian_x3_separated | jax_cpu | pending (B3, before its wave-1 rows) | pending | pending | pending |

Constant-likelihood `ln 3!` validation (`results/reference/constant_likelihood.json`):
Nautilus counts the excluded volume (offset `−ln 3!`; measured −1.825 on numpy and JAX at
`n_live=2000`; a first `n_live=500` attempt measured −1.972, outside the window, and is
kept under `attempts`); DynestyStatic draws its initial live points from the allowed
region only (offset `0`, measured 0.000 on both backends).

Observation recorded for the pilot, not a criterion: DynestyStatic at PyAutoFit's
default `rwalk` (`walks=5`, `nlive=1000`) reproduces the posterior (medians within
0.06 σ) but its ln Z scatters by 2.3 nat (numpy) and 5.6 nat (JAX) across seeds.

## 11. Amendments

None.

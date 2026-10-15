# Protocol `correlated_nd@1` — pre-registered success criteria

**Pre-registered 2026-10-15, before any `correlated_nd` row exists** (search-extensibility
epic, phase B5c; autofit_inference#14). Git proves the order: this file's first commit
predates every file under `results/searches/corr_gaussian_*/` and
`results/searches/rosenbrock_d20/`. The runner stamps every row of this family with
`protocol_id: correlated_nd@1`, and every verdict reason starts with `correlated_nd@1:`.
The freeze commit is recorded in §7 by a later commit that cites it.

**Why a second family.** `gaussian_x3` (protocol `gaussian_x3@1/@2`) tests label switching
and a blend at 10 parameters; its posterior is close to Gaussian and low-dimensional. A
ranking from one family says nothing about the geometries where searches really differ.
`correlated_nd` stresses a different axis: scaling with dimension and adaptation to an
ill-conditioned, correlated or curved posterior (mass-matrix adaptation for gradient
chains, bound construction for nested samplers). Its posterior and evidence are **exact by
construction**, so no reference search is run.

**Change rule** (as `gaussian_x3`). Nothing in §1–§9 changes after a `correlated_nd` row is
committed except through a new protocol version (`correlated_nd@2`, …) whose diff and
reason are recorded in §11, committed before the first row judged by it. Values marked
**PLACEHOLDER** are calibrated on the pilot (§7) and then frozen as `correlated_nd@2`
before any scored row. §10 is bookkeeping, not criteria.

**Families are never compared.** This protocol judges only `correlated_nd` rows; the
catalogue groups them by task and by dataset in their own section, separately from
`gaussian_x3`, and **ranks nothing across families** (§9).

## 1. The question

Which PyAutoFit searches find the right answer on high-dimensional, ill-conditioned
targets, how reliably across seeds, and at what wall-clock cost per right answer. A
search answers one of three **tasks** and is judged only within its task and dataset:

| Task | Searches (the gaussian_x3 wave-2 menu, unchanged) | Judged by |
|---|---|---|
| point/MAP | LBFGS, BFGS, MultiStartAdam, MultiStartProdigy, MultiStartADABelief, MultiStartLion | §4(a) |
| posterior | Emcee, Zeus (numpy and JAX legs), BlackJAXNUTS (cold, and warm from a short Nautilus), SMC | §4(b) |
| evidence | Nautilus (`n_live` 100/200/400), DynestyStatic (`nlive` 200), DynestyDynamic (default), NSS (`n_live_200_mcmc_steps_50`) | §4(b) and §4(c) |

The menu — searches, settings names and kwargs, backends (`numpy` where PyAutoFit
supports it, `jax_cpu` where the search is JAX-native) — is the `gaussian_x3` wave-2 menu
verbatim (`searches/_runner.SAMPLERS`, `searches/_wave2.SETTINGS`). It is **not retuned**
for this family: how a fixed configuration copes with dimension is part of what is
measured. A cell that cannot answer its task by construction at a dimension is
**deferred** with its reason (§7), never silently dropped. Every registered search
appears with a status: `measured`, `deferred` or `failed`.

## 2. Datasets, model, priors and likelihood

Defined once: `scripts/misc/simulators/correlated_nd.py` (the generator),
`scripts/misc/models/correlated_nd.py` (model and Analysis) and
`scripts/misc/models/correlated_nd_exact.py` (every exact quantity).

**Four datasets** (`dataset/<name>/`; one structure each, fixed by a `structure_seed`):

| Dataset | Kind | D | structure_seed | ln Z (realisation 1) | H (nat) | max ln L |
|---|---|---|---|---|---|---|
| `corr_gaussian_d20` | correlated Gaussian | 20 | 2020 | −93.0195 | 53.13 | −29.892 |
| `corr_gaussian_d50` | correlated Gaussian | 50 | 2050 | −233.0188 | 133.29 | −74.729 |
| `corr_gaussian_d100` | correlated Gaussian | 100 | 2100 | −467.0880 | 267.63 | −149.458 |
| `rosenbrock_d20` | hybrid Rosenbrock banana | 20 | 3020 | −140.9460 | 133.29 | 2.344 |

(`H = E_post[ln L] − ln Z` is the prior-to-posterior information; ln Z and H are exact,
§6.)

- **Correlated Gaussian.** `L(θ) = N(y; θ, Σ)` with `Σ = Q diag(λ) Qᵀ`: `Q` a seeded
  Haar-random orthogonal matrix (QR of a standard-normal matrix, signs fixed by
  `diag(R) > 0`), `λ` log-spaced from 0.1 to 100, so the **condition number is 10³
  exactly**. Truth `θ*` seeded in `U(−10, 10)` per axis; data
  `y = θ* + Q (√λ ⊙ z)`, `z ~ N(0, I)`. Under the prior box the posterior is `N(y, Σ)`.
- **Hybrid Rosenbrock banana** (Pagani, Wiegand & Nadarajah 2022, *Scand. J. Stat.* 49,
  657) with `n1 = 2`, `n2 = 19`, `a = 1/20`, `b = 100/20`, `μ = 1`:
  `x1 | y ~ N(y1, σ1²)`, `x_j | x1, y ~ N(x1² + y_j, σ2²)` for `j = 2…20`, with
  `σ1 = √10`, `σ2 = √0.1`. The map `(x1, x_j) → (x1, x_j − x1²)` has unit Jacobian, so the
  density is normalised exactly and sampled directly. Truth `x1* = 1`, `x_j* = 1`; data
  `y1 = 1 + σ1 z0`, `y_j = σ2 z_j`.
- **Priors.** One `UniformPrior` per parameter (`x000 … x{D−1}`, an `af.Collection` in key
  order) on a box fixed by the structure, never by the data (`priors.json`): Gaussians,
  `θ*_i − m⁻_i σ_i … θ*_i + m⁺_i σ_i` with `σ_i = √Σ_ii` and margins `m±` seeded in
  `U(12, 16)`; banana, `x1 ∈ 1 ∓ U(10, 12) σ1` and `x_j ∈ [−c⁻ σ2, max(x1)² + c⁺ σ2]` with
  `c±` seeded in `U(12, 16)`. No assertions and no label symmetry: the label convention of
  `gaussian_x3` §3 does not arise, `assertion_mechanism` is `none` and the evidence offset
  is 0.
- **Likelihood** — a normalised density in θ, so `ln Z = ln(box mass) − ln V` exactly
  (`V` the box volume): Gaussian
  `ln L = −½ |diag(λ)^−½ Qᵀ (θ − y)|² − ½ Σ ln(2πλ)`; banana
  `ln L = −½ [((x1 − y1)/σ1)² + Σ_j ((x_j − x1² − y_j)/σ2)²] − ½ ln(2πσ1²) − ½ (D−1) ln(2πσ2²)`.
  One `xp` expression serves the numpy and JAX backends.
- **Realisations.** A `data_seed` draws `z` from `default_rng([structure_seed, data_seed])`;
  the default realisation is `data_seed` 1. A realisation with any `|z| > 5` is refused by
  the generator, which bounds the box truncation (§6).
- Point/MAP searches optimise the log posterior; inside the box PyAutoFit's uniform prior
  contributes 0, so log posterior = log likelihood there. Every row records both
  `max_log_likelihood` and `max_log_posterior`.

## 3. Label convention

None: the parameters are not exchangeable. `posterior_relabelled` on rows and references
is the posterior itself; `modes`, `modes_found` and `label_permutations_found` are `null`.

## 4. Success criteria (per run)

All comparisons are against the **exact reference** of §6 for the **same dataset, the
same backend and the same realisation** (identity `dataset, backend, data_seed,
assertion_mechanism`, `_protocol.REFERENCE_IDENTITY`). A missing or failed reference makes
the verdict `not_assessed` with that reason.

**(a) point/MAP** — `max_log_posterior ≥ ref_MAP_log_posterior − 1` nat, the exact MAP
being the mode (`y`, or `(y1, y1² + y_j)` for the banana). Point/MAP runs are judged on (a)
alone; other rows record it as a diagnostic.

**(b) posterior accuracy**, all of:

1. every parameter: `|median − ref_median| / ref_σ ≤ 1`;
2. every parameter: `σ_run / σ_ref ∈ [0.5, 2.0]` — **PLACEHOLDER band**;
3. **whitened coordinates** (this family's replacement for `gaussian_x3`'s
   posterior-predictive and mode criteria): map every sample through the exact whitening
   map, `u = diag(λ)^−½ Qᵀ (θ − y)` (Gaussian) or `u1 = (x1 − y1)/σ1`,
   `u_j = (x_j − x1² − y_j)/σ2` (banana), under which the true posterior is `N(0, I)`.
   Every coordinate's `|median|` is ≤ 1 unit sigma and its sigma ratio lies in
   **[0.5, 2.0] — PLACEHOLDER band**. Sigma is `(q84 − q16)/2` throughout, so a unit
   normal's is `Φ⁻¹(0.84) = 0.9945` and ratios are taken against it. This tests the
   correlation structure and the curvature, which per-parameter marginals cannot: a
   search that matches every marginal of a correlated Gaussian but samples an
   axis-aligned ellipsoid fails it along the stiff eigen-directions.

**(c) evidence** — `|ln Z − ref ln Z| ≤ 1` nat (no offset: no excluded prior volume).

**Verdict.** point/MAP: `accepted` iff (a). posterior: `accepted` iff (b). evidence:
`accepted` iff (b) and (c). Otherwise `rejected`, with every failed criterion listed by
number. A run that did not complete is `rejected` with `not completed: <status>`.
`truth_delta_sigma` against the generating truth is recorded (across realisations it
should look roughly N(0, 1)); it checks the data, not the search.

## 5. Convergence diagnostics (separate from acceptance)

As `gaussian_x3` §5, with its values as **PLACEHOLDERS** for this family: nested —
`converged` iff the run completed, its own termination criterion was met and the Kish ESS
≥ **500**; chains — split R-hat ≤ **1.01** and bulk ESS ≥ **400** on every parameter;
point/MAP — `not_assessed`. Convergence never implies acceptance, nor the reverse.

## 6. The exact reference and its agreement test

Written per dataset × backend × realisation by
`scripts/misc/reference/exact_correlated_nd.py` to
`results/reference/<dataset>/<backend>/reference.json` (`data_seed<d>/` for a non-default
realisation): `status`, the identity, `log_evidence` (= `log_evidence_normalised`),
`information_nats`, `max_log_likelihood`, `map.log_posterior`, the exact marginals
(`posterior`, `posterior_relabelled`) in the `_posterior.marginal` convention, and how
each was computed.

- **ln Z.** Gaussian: `ln Z = −ln V`, the box truncation bounded by a union bound over the
  exact marginals (`truncation_bound`, below 1e-25 on every committed realisation, so ln Z
  is exact to that). Banana: the box mass is integrated (Gauss–Legendre over `x1`,
  4000 × 8 nodes; every `x_j` factor a closed-form normal CDF) and `ln Z = ln mass − ln V`.
- **Marginals.** Gaussian: `y_i + σ_i Φ⁻¹(q)`. Banana: `x1` from the integrated `x1`
  density, each `x_j` by inverting its integrated CDF (bisection to 1e-10 σ2).

**Agreement required** (the §6 test of `gaussian_x3`, against an exact target instead of
between reference runs); a reference is `complete` only when every item passes, else
`failed` with its reasons and nothing is judged against it:

1. *Backend identity.* The backend's own figure of merit — PyAutoFit `Fitness.call` over the
   family's Analysis, the path every search calls (`jax.jit` on JAX) — equals the analytic
   `ln L` at the exact mode and 63 exact posterior draws to **1e-8 nat**.
2. *Exact draws.* 200 000 iid exact posterior draws, summarised by the runner's own
   statistics code: every median within **0.02** reference σ, every σ ratio and every
   whitened σ ratio within **1 ± 0.02**, every whitened median within **0.02**.
3. *Truncation.* The fraction of 200 000 unconstrained exact draws inside the box agrees
   with the box mass used for ln Z (within 5 binomial σ, or exactly 1).
4. *Control verdict.* A row built from those draws, with the exact ln Z, is `accepted` on
   the posterior and the evidence task by `correlated_nd@1`.

## 7. Calibration, deferrals and the pilot

**Deferral rules** — a cell is deferred at a dimension, with the reason recorded on every
expected run (`searches/_correlated_nd.deferral`), when it cannot answer its task by
construction:

1. *Live-point floor (nested).* Nested sampling's ln Z scatter is ≈ `√(H / n_live)`
   (Skilling 2006). A leg with `n_live < H` has an expected scatter above the 1-nat
   tolerance of §4(c). `DynestyDynamic`'s default `nlive_init` is 500.
2. *Walker floor (ensemble chains).* Emcee and Zeus refuse fewer than `2D` walkers, so the
   menu's 50 walkers are an invalid configuration at `D ≥ 26`.
3. *Warm-start provider.* Warm BlackJAXNUTS starts from Nautilus `n_live_100`; where that
   provider is deferred, the warm leg is deferred with it.

Applied to the menu (26 cells per dataset):

| Dataset | Deferred cells (both backends where the search has them) | Planned cells |
|---|---|---|
| `corr_gaussian_d20` | none | 26 |
| `corr_gaussian_d50` | Nautilus `n_live_100` (rule 1: 100 < 133.3); Emcee, Zeus (rule 2); warm NUTS (rule 3) | 19 |
| `corr_gaussian_d100` | Nautilus `n_live_100` and `n_live_200`, DynestyStatic `nlive_200`, NSS (rule 1: < 267.6); Emcee, Zeus (rule 2); warm NUTS (rule 3) | 14 |
| `rosenbrock_d20` | Nautilus `n_live_100` (rule 1: 100 < 133.3); warm NUTS (rule 3) | 23 |

**The pilot** (wave 1, exploratory, ranks nothing; flagged `pilot: true`): **10 search
seeds (0–9) × realisation 1 × every planned cell × the four datasets = 820 runs**
(1040 expected, 220 deferred), on RAL `--partition=ral` against the private PyAutoFit pin
`df6d218f10cd` (PyAutoNerves `7f3055b7f3d1`) — the pin of `gaussian_x3` wave 2 — under
`.pyauto_pin/` in the RAL clone, never the shared mirror. Every search receives the A4
search-level `seed`. One run per array task, each under a **4 h operational cap** on
`search.fit` (`timeout -s TERM`; the runner writes a `Terminated` failure row): a run
stopped at the cap is a failed attempt whose wall is a censored lower bound. The cap is not
a protocol timeout.

**Calibration, then freeze.** Every PLACEHOLDER (the two σ-ratio bands, the ESS and R-hat
bars), the per-cell timeout and the censoring rule are calibrated on the pilot and frozen as
`correlated_nd@2`, committed before the first scored row, by the `gaussian_x3` rules where
they apply: timeouts `max(600 s, 10 × the median pilot total wall)` per (search × settings ×
backend × dataset); a band may move only on ≥ 20 completed pilot rows of its task (the
`gaussian_x3` < 20 rule), and never so far that 200 000 exact draws (§6 item 2) would fail
it. A cell whose every pilot attempt hit the cap is deferred from the scored wave with
that reason. Pilot verdicts stay `@1`.

**Freeze commit:** recorded here by the commit that follows it (B5c, 2026-10-15).

## 8. Headline numbers

As `gaussian_x3` §8, per (search × settings × config × dataset), never per best seed:
success rate with a Wilson 95 % interval over every attempt; expected wall per right
answer (Σ `total_wall_s` of all attempts, warm-start provider included, / successes) with a
bootstrap 95 % interval over seeds, unbounded at zero successes (with its lower bound);
the supporting columns and timing labels of that section. **Pre-registered here**
(it was a post-hoc presentation note for `gaussian_x3@2`): the catalogue orders legs
within one task and one dataset only, by expected wall per right answer then success
rate, and ranks only closed legs with success ≥ 50 % over ≥ 20 attempts; every other
closed leg is listed below as unreliable with its numbers.

## 9. Waves

- **Wave 1 is the pilot (§7); it ranks nothing.**
- **The scored wave** (B5c-2 onwards): 50 search seeds × 3 realisations (data seeds 1–3)
  × every non-deferred cell, under `correlated_nd@2`, on RAL `--partition=ral` only —
  never `gpu`, `ral,gpu` or `gpu,ral` (the human's hard rule of 2026-09-30) — at a PyAutoFit
  pin recorded with the freeze. Each realisation is judged only against its own exact
  reference.
- **Identity.** A row belongs to a wave only when its key `(dataset, sampler,
  settings_name, config_id, data_seed, protocol_id, pyautofit_sha)` names a cell of the
  wave, an admitted realisation, the wave's protocol and the pinned PyAutoFit, and its
  `model` is `correlated_nd` (`searches/_correlated_nd.admit`); any other row is refused
  and listed. Every expected run is `row`, `failed`, `missing` or `deferred`.
- **No GPU leg** (as `gaussian_x3`): the likelihood is a D × D matrix product at most.
- **No cross-family ranking.** The catalogue's `correlated_nd` section groups by task and
  dataset, separately from `gaussian_x3`. A statement that a search is "better" holds
  within one task and one dataset of one family.

## 10. Reference status (bookkeeping)

| Dataset | Backend | Realisation 1 |
|---|---|---|
| `corr_gaussian_d20` | numpy, jax_cpu | pending (written with the generator, B5c) |
| `corr_gaussian_d50` | numpy, jax_cpu | pending |
| `corr_gaussian_d100` | numpy, jax_cpu | pending |
| `rosenbrock_d20` | numpy, jax_cpu | pending |

## 11. Amendments

None.

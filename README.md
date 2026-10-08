autofit_inference
=================

Non-linear search and inference benchmarking for
[PyAutoFit](https://github.com/PyAutoLabs/PyAutoFit) — the proving ground for *which
search finds the right answer fastest*, on toy likelihoods where the truth is known.

It is the sibling of [autofit_profiling](https://github.com/PyAutoLabs/autofit_profiling)
(PyAutoFit timing) and the PyAutoFit counterpart of
[autolens_inference](https://github.com/PyAutoLabs/autolens_inference).

**Status: wave-1 pilot, partial (phase B3 of the
[search-extensibility epic](https://github.com/PyAutoLabs/PyAutoMind/blob/main/draft/research/autofit/search_extensibility_epic.md)).**
The pilot ranks nothing: its rows calibrate the protocol's PLACEHOLDER thresholds, they
do not order the searches. It was stopped by the B3 wrap-up ruling (2026-10-08) with the
rows counted below on disk; every run without a row is deferred to wave 2 on RAL
(`--partition=ral`), and NSS to phase A3b. It ran on PyAutoFit main **before** phase A2
(gradients under JAX, PyAutoFit#1679), so the BFGS/LBFGS non-finite results are expected
to change once A2 merges and are not a judgement on those searches. The repo reports to
[PyAutoInsight](https://github.com/PyAutoLabs/PyAutoInsight) through `inference-summary@1`.

Agents and contributors: start with [`AGENTS.md`](AGENTS.md). Rulings of record live in
[PyAutoCortex](https://github.com/PyAutoLabs/PyAutoCortex) — see [`CORTEX.md`](CORTEX.md).

## The benchmark: `gaussian_x3`

Three 1D `af.ex.Gaussian` plus a constant `Background(level)`: 10 free parameters, with
ordered-centre assertions `g0.centre < g1.centre < g2.centre`, on two committed datasets
(`dataset/`, reproducible from `scripts/misc/simulators/gaussian_x3.py`):

- `gaussian_x3_blend` — centres 25 / 45 / 60, σ 3 / 6 / 10; g1 and g2 overlap.
- `gaussian_x3_separated` — the disjoint control (centres 20 / 50 / 80, σ 3 / 4 / 5),
  fitted with disjoint centre priors and no assertions (D15 option (b); protocol §11 A1).

Every run is judged by the pre-registered protocol
[`wiki/project/protocol_gaussian_x3.md`](wiki/project/protocol_gaussian_x3.md)
(`gaussian_x3@1`), committed before any result row: point/MAP searches against a MAP
reference, posterior searches on relabelled marginals, a predictive check and mode
coverage, evidence searches on `ln Z` within one backend under the `ln 3!` convention.
Wave 1 is a pilot that ranks nothing; its PLACEHOLDER thresholds are calibrated and then
frozen before the scored wave 2.

## Running

```bash
source activate.sh
python scripts/gaussian_x3_blend/searches/nautilus/gaussian_x3.py \
    --config-name local_numpy_fp64 --settings n_live_100 --seed 0
python scripts/misc/tooling/export_inference_summary.py   # dashboard/summary.json
python scripts/misc/tooling/build_readme.py               # the tables below
```

Config names are `{local,ral}_{numpy,jax_cpu}_{fp64,fp32}`; there is no GPU leg. RAL runs
use `--partition=ral` only (`hpc/README.md`).

## Searches by task

Every registered search, grouped by the task it answers. Tasks are never ranked against
each other.

<!-- BEGIN auto-table:catalogue -->
**point/MAP** — judged by protocol §4(a): max_log_posterior >= ref MAP - 1 nat.

| Search class | Status | Accepted / attempted (blend: numpy, jax) | Accepted / attempted (separated: numpy, jax) | Reason |
|---|---|---|---|---|
| `Drawer` | unsupported | —; — | —; — | not benchmarked: Drawer is a sanity floor that answers no task (protocol §1) |
| `BFGS` | measured | 0/10; 0/10 | 0/10; 0/10 | 22 usable of 40 attempted wave-1 runs; pilot ran on PyAutoFit main pre-A2; the non-finite results are expected to change once A2 (PyAutoFit#1679, gradients under JAX) merges, so these rows are not a judgement on the search |
| `LBFGS` | measured | 0/10; 0/10 | 0/10; 0/10 | 23 usable of 40 attempted wave-1 runs; pilot ran on PyAutoFit main pre-A2; the non-finite results are expected to change once A2 (PyAutoFit#1679, gradients under JAX) merges, so these rows are not a judgement on the search |
| `MultiStartAdam` | measured | —; 0/10 | —; 0/10 | 20 usable of 20 attempted wave-1 runs |
| `MultiStartADABelief` | measured | —; 0/10 | —; 0/10 | 20 usable of 20 attempted wave-1 runs |
| `MultiStartLion` | measured | —; 0/10 | —; 0/10 | 20 usable of 20 attempted wave-1 runs |
| `MultiStartProdigy` | measured | —; 0/10 | —; 10/10 | 20 usable of 20 attempted wave-1 runs |

**posterior** — judged by protocol §4(b): relabelled medians, sigma-ratio band, ppc chi2, mode coverage.

| Search class | Status | Accepted / attempted (blend: numpy, jax) | Accepted / attempted (separated: numpy, jax) | Reason |
|---|---|---|---|---|
| `Emcee` | measured | 0/2; 0/2 | 0/2; 0/2 | 7 usable of 8 attempted wave-1 runs |
| `Zeus` | measured | 0/2; 0/1 | 2/2; 0/1 | 3 usable of 6 attempted wave-1 runs |
| `BlackJAXNUTS` | measured | —; 0/2 (`blackjax_nuts` `warmup_200_samples_500`) · 0/1 (`blackjax_nuts_warm` `warmup_200_samples_500`) | —; 0/2 (`blackjax_nuts` `warmup_200_samples_500`) · 1/1 (`blackjax_nuts_warm` `warmup_200_samples_500`) | 3 usable of 6 attempted wave-1 runs |
| `SMC` | measured | —; 0/2 | —; 0/1 | 3 usable of 3 attempted wave-1 runs |

**evidence** — judged by protocol §4(b) and §4(c): posterior accuracy and ln Z within 1 nat.

| Search class | Status | Accepted / attempted (blend: numpy, jax) | Accepted / attempted (separated: numpy, jax) | Reason |
|---|---|---|---|---|
| `DynestyStatic` | measured | 0/2; 0/2 | 1/2; 0/2 | 8 usable of 8 attempted wave-1 runs |
| `DynestyDynamic` | measured | 2/2; 0/2 | 1/2; 1/2 | 8 usable of 8 attempted wave-1 runs |
| `Nautilus` | measured | 2/2 (`nautilus` `n_live_100`) · 1/2 (`nautilus` `n_live_200`) · 2/2 (`nautilus` `n_live_400`); 0/2 (`nautilus` `n_live_100`) · 0/2 (`nautilus` `n_live_200`) · 0/2 (`nautilus` `n_live_400`) | 2/2 (`nautilus` `n_live_100`) · 2/2 (`nautilus` `n_live_200`) · 2/2 (`nautilus` `n_live_400`); 2/2 (`nautilus` `n_live_100`) · 2/2 (`nautilus` `n_live_200`) · 2/2 (`nautilus` `n_live_400`) | 23 usable of 24 attempted wave-1 runs |
| `NSS` | deferred | —; 0/0 | —; 0/0 | NSS is deferred until phase A3b (NSS onto Fitness): the B3 rule defers it on the blend, and the harness registers it deferred as a whole, so the separated control is deferred with it |

Wave-1 coverage: 223 rows of 520 expected runs; PyAutoFit `0dbf258c4f5e`. Accepted counts are judged by protocol `gaussian_x3@1`; an attempt that could not be judged (no complete reference) is not counted as accepted.

**Pilot stopped.** Wave 1 was stopped by the B3 wrap-up ruling (2026-10-08) before its expected-run manifest was complete: every run without a row is deferred to wave 2 on RAL --partition=ral (NSS to phase A3b). The pilot ran on PyAutoFit main before phase A2 (gradients under JAX, PyAutoFit#1679) merged.
<!-- END auto-table:catalogue -->

## Reference posteriors

Per dataset and backend: three long Nautilus (`n_live=2000`) and three long
DynestyStatic (`nlive=1000`) runs, plus the MAP reference (protocol §6). Committed under
`results/reference/`.

<!-- BEGIN auto-table:references -->
| Dataset | Backend | Status | Included runs | ln Z (normalised) | ln Z spread | Worst median offset (σ) | MAP ln P |
|---|---|---|---|---|---|---|---|
| `gaussian_x3_blend` | `jax_cpu` | pending | 0 | — | — | — | — |
| `gaussian_x3_blend` | `numpy` | complete | 3 | 138.853 | 0.019 | 0.015 | 186.668 |
| `gaussian_x3_separated` | `jax_cpu` | complete | 3 | 122.931 | 0.006 | 0.010 | 175.706 |
| `gaussian_x3_separated` | `numpy` | complete | 3 | 122.931 | 0.006 | 0.010 | 175.706 |

ln 3! convention (constant likelihood): `dynesty_dynamic`/`jax_cpu`: offset -1.792 (validated, measured -1.813); `dynesty_dynamic`/`numpy`: offset -1.792 (validated, measured -1.848); `dynesty_static`/`jax_cpu`: offset 0.000 (validated, measured 0.000); `dynesty_static`/`numpy`: offset 0.000 (validated, measured 0.000); `nautilus`/`jax_cpu`: offset -1.792 (validated, measured -1.825); `nautilus`/`numpy`: offset -1.792 (validated, measured -1.825).
<!-- END auto-table:references -->

## Search rows

<!-- BEGIN auto-table:searches -->
**point/MAP**

| Dataset | Search | Settings | Config | Attempted | Accepted | Rejected | Not assessed | Converged | Success rate (Wilson 95 %) | Median wall (s) | Wall per right answer (s) | Deferred seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `gaussian_x3_blend` | `bfgs` | `default` | `local_numpy_fp64` | 10 | 0 | 10 | 0 | 0 | 0% [0%, 28%] | 9.0 | unbounded (≥ 32) | — |
| `gaussian_x3_blend` | `bfgs` | `default` | `local_jax_cpu_fp64_cache_warm` | 10 | 0 | 0 | 10 | 0 | — | 15.1 | — | — |
| `gaussian_x3_separated` | `bfgs` | `default` | `local_numpy_fp64` | 10 | 0 | 10 | 0 | 0 | 0% [0%, 28%] | 32.2 | unbounded (≥ 116) | — |
| `gaussian_x3_separated` | `bfgs` | `default` | `local_jax_cpu_fp64_cache_warm` | 10 | 0 | 10 | 0 | 0 | 0% [0%, 28%] | 35.8 | unbounded (≥ 130) | — |
| `gaussian_x3_blend` | `lbfgs` | `default` | `local_numpy_fp64` | 10 | 0 | 10 | 0 | 0 | 0% [0%, 28%] | 7.4 | unbounded (≥ 55) | — |
| `gaussian_x3_blend` | `lbfgs` | `default` | `local_jax_cpu_fp64_cache_warm` | 10 | 0 | 0 | 10 | 0 | — | 13.1 | — | — |
| `gaussian_x3_separated` | `lbfgs` | `default` | `local_numpy_fp64` | 10 | 0 | 10 | 0 | 0 | 0% [0%, 28%] | 34.9 | unbounded (≥ 129) | — |
| `gaussian_x3_separated` | `lbfgs` | `default` | `local_jax_cpu_fp64_cache_warm` | 10 | 0 | 10 | 0 | 0 | 0% [0%, 28%] | 38.6 | unbounded (≥ 140) | — |
| `gaussian_x3_blend` | `multi_start_adam` | `starts_48_steps_300` | `local_jax_cpu_fp64_cache_warm` | 10 | 0 | 0 | 10 | 0 | — | 102.3 | — | — |
| `gaussian_x3_separated` | `multi_start_adam` | `starts_48_steps_300` | `local_jax_cpu_fp64_cache_warm` | 10 | 0 | 10 | 0 | 0 | 0% [0%, 28%] | 69.0 | unbounded (≥ 246) | — |
| `gaussian_x3_blend` | `multi_start_adabelief` | `starts_48_steps_300` | `local_jax_cpu_fp64_cache_warm` | 10 | 0 | 0 | 10 | 0 | — | 86.8 | — | — |
| `gaussian_x3_separated` | `multi_start_adabelief` | `starts_48_steps_300` | `local_jax_cpu_fp64_cache_warm` | 10 | 0 | 10 | 0 | 0 | 0% [0%, 28%] | 71.0 | unbounded (≥ 257) | — |
| `gaussian_x3_blend` | `multi_start_lion` | `starts_48_steps_300` | `local_jax_cpu_fp64_cache_warm` | 10 | 0 | 0 | 10 | 0 | — | 56.2 | — | — |
| `gaussian_x3_separated` | `multi_start_lion` | `starts_48_steps_300` | `local_jax_cpu_fp64_cache_warm` | 10 | 0 | 10 | 0 | 0 | 0% [0%, 28%] | 54.3 | unbounded (≥ 184) | — |
| `gaussian_x3_blend` | `multi_start_prodigy` | `starts_48_steps_300` | `local_jax_cpu_fp64_cache_warm` | 10 | 0 | 0 | 10 | 0 | — | 75.5 | — | — |
| `gaussian_x3_separated` | `multi_start_prodigy` | `starts_48_steps_300` | `local_jax_cpu_fp64_cache_warm` | 10 | 10 | 0 | 0 | 0 | 100% [72%, 100%] | 63.7 | 66 | — |

**posterior**

| Dataset | Search | Settings | Config | Attempted | Accepted | Rejected | Not assessed | Converged | Success rate (Wilson 95 %) | Median wall (s) | Wall per right answer (s) | Deferred seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `gaussian_x3_blend` | `emcee` | `walkers_50_steps_2000` | `local_numpy_fp64` | 2 | 0 | 2 | 0 | 0 | 0% [0%, 66%] | 234.4 | unbounded (≥ 356) | 8 |
| `gaussian_x3_blend` | `emcee` | `walkers_50_steps_2000` | `local_jax_cpu_fp64_cache_warm` | 2 | 0 | 0 | 2 | 0 | — | 1499.8 | — | 8 |
| `gaussian_x3_separated` | `emcee` | `walkers_50_steps_2000` | `local_numpy_fp64` | 2 | 0 | 2 | 0 | 0 | 0% [0%, 66%] | 198.7 | unbounded (≥ 302) | 8 |
| `gaussian_x3_separated` | `emcee` | `walkers_50_steps_2000` | `local_jax_cpu_fp64_cache_warm` | 2 | 0 | 2 | 0 | 0 | 0% [0%, 66%] | 1757.8 | unbounded (≥ 2673) | 8 |
| `gaussian_x3_blend` | `zeus` | `walkers_50_steps_2000` | `local_numpy_fp64` | 2 | 0 | 2 | 0 | 0 | 0% [0%, 66%] | 1283.2 | unbounded (≥ 1951) | 8 |
| `gaussian_x3_blend` | `zeus` | `walkers_50_steps_2000` | `local_jax_cpu_fp64_cache_warm` | 1 | 0 | 0 | 1 | 0 | — | 1792.3 | — | 9 |
| `gaussian_x3_separated` | `zeus` | `walkers_50_steps_2000` | `local_numpy_fp64` | 2 | 2 | 0 | 0 | 0 | 100% [34%, 100%] | 1587.0 | 1587 | 8 |
| `gaussian_x3_separated` | `zeus` | `walkers_50_steps_2000` | `local_jax_cpu_fp64_cache_warm` | 1 | 0 | 1 | 0 | 0 | 0% [0%, 79%] | 3595.5 | unbounded (≥ 4531) | 9 |
| `gaussian_x3_blend` | `blackjax_nuts` | `warmup_200_samples_500` | `local_jax_cpu_fp64_cache_warm` | 2 | 0 | 0 | 2 | 0 | — | 25.9 | — | 8 |
| `gaussian_x3_blend` | `blackjax_nuts_warm` | `warmup_200_samples_500` | `local_jax_cpu_fp64_cache_warm` | 1 | 0 | 0 | 1 | 0 | — | 1233.7 | — | 9 |
| `gaussian_x3_separated` | `blackjax_nuts` | `warmup_200_samples_500` | `local_jax_cpu_fp64_cache_warm` | 2 | 0 | 2 | 0 | 0 | 0% [0%, 66%] | 166.0 | unbounded (≥ 252) | 8 |
| `gaussian_x3_separated` | `blackjax_nuts_warm` | `warmup_200_samples_500` | `local_jax_cpu_fp64_cache_warm` | 1 | 1 | 0 | 0 | 0 | 100% [21%, 100%] | 1150.2 | 1150 | 9 |
| `gaussian_x3_blend` | `smc` | `default` | `local_jax_cpu_fp64_cache_warm` | 2 | 0 | 0 | 2 | 0 | — | 379.1 | — | 8 |
| `gaussian_x3_separated` | `smc` | `default` | `local_jax_cpu_fp64_cache_warm` | 1 | 0 | 1 | 0 | 0 | 0% [0%, 79%] | 322.4 | unbounded (≥ 406) | 9 |

**evidence**

| Dataset | Search | Settings | Config | Attempted | Accepted | Rejected | Not assessed | Converged | Success rate (Wilson 95 %) | Median wall (s) | Wall per right answer (s) | Deferred seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `gaussian_x3_blend` | `dynesty_static` | `nlive_200` | `local_numpy_fp64` | 2 | 0 | 2 | 0 | 0 | 0% [0%, 66%] | 172.9 | unbounded (≥ 263) | 8 |
| `gaussian_x3_blend` | `dynesty_static` | `nlive_200` | `local_jax_cpu_fp64_cache_warm` | 2 | 0 | 0 | 2 | 0 | — | 166.8 | — | 8 |
| `gaussian_x3_separated` | `dynesty_static` | `nlive_200` | `local_numpy_fp64` | 2 | 1 | 1 | 0 | 0 | 50% [9%, 91%] | 228.8 | 458 | 8 |
| `gaussian_x3_separated` | `dynesty_static` | `nlive_200` | `local_jax_cpu_fp64_cache_warm` | 2 | 0 | 2 | 0 | 0 | 0% [0%, 66%] | 205.3 | unbounded (≥ 312) | 8 |
| `gaussian_x3_blend` | `dynesty_dynamic` | `default` | `local_numpy_fp64` | 2 | 2 | 0 | 0 | 0 | 100% [34%, 100%] | 557.4 | 557 | 8 |
| `gaussian_x3_blend` | `dynesty_dynamic` | `default` | `local_jax_cpu_fp64_cache_warm` | 2 | 0 | 0 | 2 | 0 | — | 841.7 | — | 8 |
| `gaussian_x3_separated` | `dynesty_dynamic` | `default` | `local_numpy_fp64` | 2 | 1 | 1 | 0 | 0 | 50% [9%, 91%] | 850.2 | 1700 | 8 |
| `gaussian_x3_separated` | `dynesty_dynamic` | `default` | `local_jax_cpu_fp64_cache_warm` | 2 | 1 | 1 | 0 | 0 | 50% [9%, 91%] | 822.4 | 1645 | 8 |
| `gaussian_x3_blend` | `nautilus` | `n_live_100` | `local_numpy_fp64` | 2 | 2 | 0 | 0 | 2 | 100% [34%, 100%] | 1062.7 | 1063 | 8 |
| `gaussian_x3_blend` | `nautilus` | `n_live_100` | `local_jax_cpu_fp64_cache_warm` | 2 | 0 | 0 | 2 | 2 | — | 1041.4 | — | 8 |
| `gaussian_x3_blend` | `nautilus` | `n_live_200` | `local_numpy_fp64` | 2 | 1 | 1 | 0 | 1 | 50% [9%, 91%] | 1455.4 | 2911 | 8 |
| `gaussian_x3_blend` | `nautilus` | `n_live_200` | `local_jax_cpu_fp64_cache_warm` | 2 | 0 | 0 | 2 | 2 | — | 1021.4 | — | 8 |
| `gaussian_x3_blend` | `nautilus` | `n_live_400` | `local_numpy_fp64` | 2 | 2 | 0 | 0 | 2 | 100% [34%, 100%] | 2493.8 | 2494 | 8 |
| `gaussian_x3_blend` | `nautilus` | `n_live_400` | `local_jax_cpu_fp64_cache_warm` | 2 | 0 | 0 | 2 | 2 | — | 1593.3 | — | 8 |
| `gaussian_x3_separated` | `nautilus` | `n_live_100` | `local_numpy_fp64` | 2 | 2 | 0 | 0 | 2 | 100% [34%, 100%] | 1173.4 | 1173 | 8 |
| `gaussian_x3_separated` | `nautilus` | `n_live_100` | `local_jax_cpu_fp64_cache_warm` | 2 | 2 | 0 | 0 | 2 | 100% [34%, 100%] | 1261.4 | 1261 | 8 |
| `gaussian_x3_separated` | `nautilus` | `n_live_200` | `local_numpy_fp64` | 2 | 2 | 0 | 0 | 2 | 100% [34%, 100%] | 1443.4 | 1443 | 8 |
| `gaussian_x3_separated` | `nautilus` | `n_live_200` | `local_jax_cpu_fp64_cache_warm` | 2 | 2 | 0 | 0 | 2 | 100% [34%, 100%] | 1452.8 | 1453 | 8 |
| `gaussian_x3_separated` | `nautilus` | `n_live_400` | `local_numpy_fp64` | 2 | 2 | 0 | 0 | 2 | 100% [34%, 100%] | 1962.4 | 1962 | 8 |
| `gaussian_x3_separated` | `nautilus` | `n_live_400` | `local_jax_cpu_fp64_cache_warm` | 2 | 2 | 0 | 0 | 2 | 100% [34%, 100%] | 1857.1 | 1857 | 8 |

<!-- END auto-table:searches -->

autofit_inference
=================

Non-linear search and inference benchmarking for
[PyAutoFit](https://github.com/PyAutoLabs/PyAutoFit) — the proving ground for *which
search finds the right answer fastest*, on toy likelihoods where the truth is known.

It is the sibling of [autofit_profiling](https://github.com/PyAutoLabs/autofit_profiling)
(PyAutoFit timing) and the PyAutoFit counterpart of
[autolens_inference](https://github.com/PyAutoLabs/autolens_inference).

**Status: scored wave 2 complete and judged under `gaussian_x3@2` (phases B5b and B5b-2
of the
[search-extensibility epic](https://github.com/PyAutoLabs/PyAutoMind/blob/main/draft/research/autofit/search_extensibility_epic.md)).**

- **Scored wave 2.** 12,750 rows ran on RAL `--partition=ral` against PyAutoFit pinned at
  `df6d218f10cd`: 50 seeds × 5 data realisations per leg, both datasets, every leg
  closed (250 blend warm-NUTS runs deferred). They are judged under the frozen `@2`
  thresholds and ranked within one task and one dataset among the legs that clear a
  reliability floor (see "Scored wave 2" below).
- **A second family, `correlated_nd` (B5c).** High-D correlated Gaussians and a
  Rosenbrock banana with exact references, under its own pre-registered protocol
  `correlated_nd@1`; its 930-run pilot runs on RAL and ranks nothing (see "The second
  family" below).
- **The wave-1 pilot** (223 rows, below) ranks nothing. It calibrated the protocol and
  ran on PyAutoFit main before phase A2.
- **Summary.** The repo reports to
  [PyAutoInsight](https://github.com/PyAutoLabs/PyAutoInsight) through
  `inference-summary@3`: `dashboard/summary.json` is an index of per-protocol, per-leg
  record files under `dashboard/records/`, so no file approaches GitHub's size limits and
  a refresh re-copies only the shards that changed.

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

## The second family: `correlated_nd` (B5c)

High-dimensional, ill-conditioned targets with **exact** posteriors and evidences, so no
reference search is run: correlated Gaussians at D = 20, 50 and 100 with condition number
10³ (a seeded Haar-random rotation of log-spaced eigenvalues), and a D = 20 hybrid
Rosenbrock banana (Pagani et al. 2022). Datasets come from
`scripts/misc/simulators/correlated_nd.py`; the exact references
(`scripts/misc/reference/exact_correlated_nd.py`) pass a per-backend agreement test before
any row is judged. Rows are judged by their own pre-registered protocol,
[`wiki/project/protocol_correlated_nd.md`](wiki/project/protocol_correlated_nd.md)
(`correlated_nd@1`), and catalogued by task and dataset **apart from `gaussian_x3`;
nothing is ranked across families.** The pilot (930 runs: 10 seeds × every non-deferred
cell of the gaussian_x3 wave-2 menu × the four datasets) runs on RAL `--partition=ral`.

<!-- BEGIN auto-table:correlated_nd -->
| Dataset | D | Exact ln Z | H (nat) | Exact references | Planned cells | Deferred cells | Pilot rows (failed) |
|---|---|---|---|---|---|---|---|
| `corr_gaussian_d20` | 20 | -93.020 | 53.1 | `jax_cpu` complete, `numpy` complete | 26 | 0 | 0 (0) |
| `corr_gaussian_d50` | 50 | -233.019 | 133.3 | `jax_cpu` complete, `numpy` complete | 19 | 7 | 0 (0) |
| `corr_gaussian_d100` | 100 | -467.088 | 267.6 | `jax_cpu` complete, `numpy` complete | 14 | 12 | 0 (0) |
| `rosenbrock_d20` | 20 | -140.946 | 133.3 | `jax_cpu` complete, `numpy` complete | 23 | 3 | 0 (0) |

Pilot: 1040 expected runs — 0 rows, 0 failed, 820 not yet pulled, 220 deferred with reasons (`catalogue/search_catalogue.json`, section `correlated_nd@1`). Wave 1 (pilot): ranks nothing (protocol correlated_nd §9). Cross-family ranking: none: this family is catalogued apart from gaussian_x3 and no leg is ever ranked or compared across families (protocol correlated_nd §9).
<!-- END auto-table:correlated_nd -->

## Running

```bash
source activate.sh
python scripts/gaussian_x3_blend/searches/nautilus/gaussian_x3.py \
    --config-name local_numpy_fp64 --settings n_live_100 --seed 0
python scripts/misc/tooling/export_inference_summary.py   # dashboard/summary.json + records/
python scripts/misc/tooling/build_readme.py               # the tables below
```

Config names are `{local,ral}_{numpy,jax_cpu}_{fp64,fp32}`; there is no GPU leg. RAL runs
use `--partition=ral` only (`hpc/README.md`).

## Scored wave 2 (`gaussian_x3@2`)

The scored wave (protocol §8, §9): 50 search seeds × 5 data realisations per leg, judged
under the frozen `@2` thresholds against each realisation's own reference. Legs are
ordered within one task and one dataset only, never across; the full record, with every
leg's counts, censoring and bootstrap intervals, is the `gaussian_x3@2` section of
`catalogue/search_catalogue.json`, and the attempt accounting is
`results/accounting/gaussian_x3_wave2.json`.

<!-- BEGIN auto-table:wave2 -->
`gaussian_x3@2` at PyAutoFit `df6d218f10cd`: 12750 admitted rows of 12750 pulled (0 refused), 13000 expected runs. Protocol §7: an attempt that hit the timeout, crashed or wrote a non-finite answer is a failed attempt; in a closed leg a missing run is a failed attempt censored at the timeout; an operational kill (Terminated below the timeout) is not an attempt (rerun it).

**Order (#):** by expected wall per right answer, then success rate, among closed legs that clear the reliability floor (success rate ≥ 50% over ≥ 20 attempts); `=n` marks a leg whose bootstrap interval overlaps the one above it. Closed legs below the floor (zero-success legs included) follow, marked **unreliable** with their numbers and no position; open or unjudged legs come last. The floor is a post-hoc presentation convention adopted 2026-10-15 after seeing the wave-2 data (protocol §8 note), not a pre-registered criterion: it changes no verdict.

**point/MAP** — `gaussian_x3_blend`

| # | Search | Settings | Config | Attempted | Accepted | Success rate (Wilson 95 %) | Wall per right answer (s, bootstrap 95 %) | Median attempt wall (s) | Timeout (s) | State |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `multi_start_prodigy` | `starts_48_steps_300` | `ral_jax_cpu_fp64_cache_warm` | 250 | 152 | 61% [55%, 67%] | 18 [17, 20] | 11.4 | 714 | closed |
| unreliable | `bfgs` | `default` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 171) | 2.8 | 600 | closed |
| unreliable | `bfgs` | `default` | `ral_numpy_fp64` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 175) | 2.6 | 600 | closed |
| unreliable | `lbfgs` | `default` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 236) | 3.0 | 600 | closed |
| unreliable | `lbfgs` | `default` | `ral_numpy_fp64` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 214) | 2.6 | 600 | closed |
| unreliable | `multi_start_adabelief` | `starts_48_steps_300` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 794) | 11.4 | 761 | closed |
| unreliable | `multi_start_adam` | `starts_48_steps_300` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 833) | 12.3 | 812 | closed |
| unreliable | `multi_start_lion` | `starts_48_steps_300` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 535) | 8.3 | 600 | closed |

**point/MAP** — `gaussian_x3_separated`

| # | Search | Settings | Config | Attempted | Accepted | Success rate (Wilson 95 %) | Wall per right answer (s, bootstrap 95 %) | Median attempt wall (s) | Timeout (s) | State |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `multi_start_prodigy` | `starts_48_steps_300` | `ral_jax_cpu_fp64_cache_warm` | 250 | 250 | 100% [98%, 100%] | 9 [9, 9] | 8.8 | 714 | closed |
| unreliable | `bfgs` | `default` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 322) | 4.6 | 600 | closed |
| unreliable | `bfgs` | `default` | `ral_numpy_fp64` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 437) | 6.7 | 600 | closed |
| unreliable | `lbfgs` | `default` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 352) | 4.9 | 600 | closed |
| unreliable | `lbfgs` | `default` | `ral_numpy_fp64` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 462) | 7.1 | 600 | closed |
| unreliable | `multi_start_adabelief` | `starts_48_steps_300` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 703) | 10.6 | 761 | closed |
| unreliable | `multi_start_adam` | `starts_48_steps_300` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 697) | 10.8 | 812 | closed |
| unreliable | `multi_start_lion` | `starts_48_steps_300` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 389) | 5.9 | 600 | closed |

**posterior** — `gaussian_x3_blend`

| # | Search | Settings | Config | Attempted | Accepted | Success rate (Wilson 95 %) | Wall per right answer (s, bootstrap 95 %) | Median attempt wall (s) | Timeout (s) | State |
|---|---|---|---|---|---|---|---|---|---|---|
| unreliable | `blackjax_nuts` | `warmup_200_samples_500` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 418) | 6.2 | 1660 | closed |
| unreliable | `emcee` | `walkers_50_steps_2000` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 4896) | 79.3 | 14000 | closed |
| unreliable | `emcee` | `walkers_50_steps_2000` | `ral_numpy_fp64` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 4063) | 65.3 | 1987 | closed |
| unreliable | `smc` | `default` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 1921) | 30.0 | 3224 | closed |
| unreliable | `zeus` | `walkers_50_steps_2000` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 6933) | 104.2 | 1168 | closed |
| unreliable | `zeus` | `walkers_50_steps_2000` | `ral_numpy_fp64` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 11233) | 173.1 | 15775 | closed |
| — | `blackjax_nuts_warm` | `warmup_200_samples_500` | `ral_jax_cpu_fp64_cache_warm` | 0 | 0 | — | — | — | 11502 | deferred |

**posterior** — `gaussian_x3_separated`

| # | Search | Settings | Config | Attempted | Accepted | Success rate (Wilson 95 %) | Wall per right answer (s, bootstrap 95 %) | Median attempt wall (s) | Timeout (s) | State |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `zeus` | `walkers_50_steps_2000` | `ral_jax_cpu_fp64_cache_warm` | 250 | 190 | 76% [70%, 81%] | 136 [126, 148] | 108.7 | 1168 | closed |
| 2 | `zeus` | `walkers_50_steps_2000` | `ral_numpy_fp64` | 250 | 198 | 79% [74%, 84%] | 216 [201, 233] | 175.9 | 15775 | closed |
| =3 | `blackjax_nuts_warm` | `warmup_200_samples_500` | `ral_jax_cpu_fp64_cache_warm` | 250 | 250 | 100% [98%, 100%] | 237 [233, 241] | 225.3 | 11502 | closed |
| unreliable | `blackjax_nuts` | `warmup_200_samples_500` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 1490) | 17.4 | 1660 | closed |
| unreliable | `emcee` | `walkers_50_steps_2000` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 5137) | 83.7 | 14000 | closed |
| unreliable | `emcee` | `walkers_50_steps_2000` | `ral_numpy_fp64` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 4085) | 58.3 | 1987 | closed |
| unreliable | `smc` | `default` | `ral_jax_cpu_fp64_cache_warm` | 250 | 0 | 0% [0%, 2%] | unbounded (≥ 1310) | 18.5 | 3224 | closed |

**evidence** — `gaussian_x3_blend`

| # | Search | Settings | Config | Attempted | Accepted | Success rate (Wilson 95 %) | Wall per right answer (s, bootstrap 95 %) | Median attempt wall (s) | Timeout (s) | State |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nss` | `n_live_200_mcmc_steps_50` | `ral_jax_cpu_fp64_cache_warm` | 250 | 158 | 63% [57%, 69%] | 33 [30, 36] | 20.5 | 600 | closed |
| 2 | `nautilus` | `n_live_100` | `ral_numpy_fp64` | 250 | 250 | 100% [98%, 100%] | 247 [242, 251] | 247.1 | 10627 | closed |
| =3 | `nautilus` | `n_live_100` | `ral_jax_cpu_fp64_cache_warm` | 250 | 250 | 100% [98%, 100%] | 253 [248, 259] | 252.9 | 10414 | closed |
| 4 | `nautilus` | `n_live_200` | `ral_jax_cpu_fp64_cache_warm` | 250 | 250 | 100% [98%, 100%] | 299 [297, 302] | 297.7 | 11613 | closed |
| =5 | `nautilus` | `n_live_200` | `ral_numpy_fp64` | 250 | 250 | 100% [98%, 100%] | 306 [301, 312] | 300.1 | 11130 | closed |
| 6 | `nautilus` | `n_live_400` | `ral_jax_cpu_fp64_cache_warm` | 250 | 250 | 100% [98%, 100%] | 513 [508, 518] | 504.3 | 15933 | closed |
| =7 | `nautilus` | `n_live_400` | `ral_numpy_fp64` | 250 | 250 | 100% [98%, 100%] | 522 [514, 531] | 507.9 | 19654 | closed |
| unreliable | `dynesty_dynamic` | `default` | `ral_jax_cpu_fp64_cache_warm` | 250 | 59 | 24% [19%, 29%] | 378 [307, 484] | 87.4 | 8224 | closed |
| unreliable | `dynesty_dynamic` | `default` | `ral_numpy_fp64` | 250 | 59 | 24% [19%, 29%] | 346 [281, 444] | 80.2 | 7558 | closed |
| unreliable | `dynesty_static` | `nlive_200` | `ral_jax_cpu_fp64_cache_warm` | 250 | 36 | 14% [11%, 19%] | 190 [144, 266] | 27.0 | 1946 | closed |
| unreliable | `dynesty_static` | `nlive_200` | `ral_numpy_fp64` | 250 | 36 | 14% [11%, 19%] | 170 [128, 237] | 24.2 | 2048 | closed |

**evidence** — `gaussian_x3_separated`

| # | Search | Settings | Config | Attempted | Accepted | Success rate (Wilson 95 %) | Wall per right answer (s, bootstrap 95 %) | Median attempt wall (s) | Timeout (s) | State |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nss` | `n_live_200_mcmc_steps_50` | `ral_jax_cpu_fp64_cache_warm` | 250 | 210 | 84% [79%, 88%] | 24 [23, 26] | 20.2 | 600 | closed |
| 2 | `dynesty_dynamic` | `default` | `ral_numpy_fp64` | 250 | 145 | 58% [52%, 64%] | 143 [130, 160] | 83.2 | 7558 | closed |
| =3 | `dynesty_dynamic` | `default` | `ral_jax_cpu_fp64_cache_warm` | 250 | 145 | 58% [52%, 64%] | 156 [141, 175] | 90.8 | 8224 | closed |
| 4 | `nautilus` | `n_live_200` | `ral_jax_cpu_fp64_cache_warm` | 250 | 250 | 100% [98%, 100%] | 284 [276, 292] | 293.7 | 11613 | closed |
| =5 | `nautilus` | `n_live_200` | `ral_numpy_fp64` | 250 | 250 | 100% [98%, 100%] | 293 [285, 301] | 298.9 | 11130 | closed |
| 6 | `nautilus` | `n_live_100` | `ral_jax_cpu_fp64_cache_warm` | 250 | 250 | 100% [98%, 100%] | 339 [332, 347] | 324.2 | 10414 | closed |
| =7 | `nautilus` | `n_live_100` | `ral_numpy_fp64` | 250 | 250 | 100% [98%, 100%] | 344 [336, 352] | 332.1 | 10627 | closed |
| 8 | `nautilus` | `n_live_400` | `ral_numpy_fp64` | 250 | 250 | 100% [98%, 100%] | 393 [387, 399] | 395.3 | 19654 | closed |
| 9 | `nautilus` | `n_live_400` | `ral_jax_cpu_fp64_cache_warm` | 250 | 250 | 100% [98%, 100%] | 414 [407, 422] | 420.6 | 15933 | closed |
| unreliable | `dynesty_static` | `nlive_200` | `ral_jax_cpu_fp64_cache_warm` | 250 | 103 | 41% [35%, 47%] | 69 [60, 81] | 28.3 | 1946 | closed |
| unreliable | `dynesty_static` | `nlive_200` | `ral_numpy_fp64` | 250 | 103 | 41% [35%, 47%] | 62 [54, 72] | 25.2 | 2048 | closed |

- Pilot-derived timeouts are 10 x laptop medians and RAL ran 4-7 x faster than the laptop (B5a dry run): censoring is lenient on RAL for those cells
- Probe-first cells (Zeus on JAX, NSS 50-step) take 10 x their RAL probe median or the 600 s floor (results/calibration/gaussian_x3_wave2_probe.json); their remainder ran as RAL job 409930 (pulled in B5b-2)
<!-- END auto-table:wave2 -->

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
| Dataset | Backend | Realisation | Status | Included runs | ln Z (normalised) | ln Z spread | Worst median offset (σ) | MAP ln P |
|---|---|---|---|---|---|---|---|---|
| `gaussian_x3_blend` | `jax_cpu` | 1 | complete | 3 | 138.855 | 0.012 | 0.016 | 186.668 |
| `gaussian_x3_blend` | `jax_cpu` | 2 | complete | 3 | 129.504 | 0.002 | 0.012 | 176.854 |
| `gaussian_x3_blend` | `jax_cpu` | 3 | complete | 3 | 118.115 | 0.002 | 0.017 | 165.734 |
| `gaussian_x3_blend` | `jax_cpu` | 4 | complete | 3 | 122.156 | 0.004 | 0.018 | 170.125 |
| `gaussian_x3_blend` | `jax_cpu` | 5 | complete | 3 | 137.686 | 0.016 | 0.015 | 185.496 |
| `gaussian_x3_blend` | `numpy` | 1 | complete | 3 | 138.853 | 0.019 | 0.015 | 186.668 |
| `gaussian_x3_blend` | `numpy` | 2 | complete | 3 | 129.504 | 0.002 | 0.012 | 176.854 |
| `gaussian_x3_blend` | `numpy` | 3 | complete | 3 | 118.115 | 0.002 | 0.017 | 165.734 |
| `gaussian_x3_blend` | `numpy` | 4 | complete | 3 | 122.156 | 0.004 | 0.018 | 170.125 |
| `gaussian_x3_blend` | `numpy` | 5 | complete | 3 | 137.686 | 0.016 | 0.015 | 185.496 |
| `gaussian_x3_separated` | `jax_cpu` | 1 | complete | 3 | 132.082 | 0.019 | 0.013 | 184.871 |
| `gaussian_x3_separated` | `jax_cpu` | 2 | complete | 3 | 122.931 | 0.006 | 0.010 | 175.706 |
| `gaussian_x3_separated` | `jax_cpu` | 3 | complete | 3 | 112.731 | 0.017 | 0.015 | 165.550 |
| `gaussian_x3_separated` | `jax_cpu` | 4 | complete | 3 | 115.743 | 0.003 | 0.012 | 168.500 |
| `gaussian_x3_separated` | `jax_cpu` | 5 | complete | 3 | 130.263 | 0.019 | 0.010 | 182.991 |
| `gaussian_x3_separated` | `numpy` | 1 | complete | 3 | 132.082 | 0.019 | 0.013 | 184.871 |
| `gaussian_x3_separated` | `numpy` | 2 | complete | 3 | 122.931 | 0.006 | 0.010 | 175.706 |
| `gaussian_x3_separated` | `numpy` | 3 | complete | 3 | 112.731 | 0.017 | 0.015 | 165.550 |
| `gaussian_x3_separated` | `numpy` | 4 | complete | 3 | 115.743 | 0.003 | 0.012 | 168.500 |
| `gaussian_x3_separated` | `numpy` | 5 | complete | 3 | 130.263 | 0.019 | 0.010 | 182.991 |

ln 3! convention (constant likelihood): `dynesty_dynamic`/`jax_cpu`: offset -1.792 (validated, measured -1.813); `dynesty_dynamic`/`numpy`: offset -1.792 (validated, measured -1.848); `dynesty_static`/`jax_cpu`: offset 0.000 (validated, measured 0.000); `dynesty_static`/`numpy`: offset 0.000 (validated, measured 0.000); `nautilus`/`jax_cpu`: offset -1.792 (validated, measured -1.825); `nautilus`/`numpy`: offset -1.792 (validated, measured -1.825); `nss`/`jax_cpu`: offset -1.792 (validated, measured -1.728).
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

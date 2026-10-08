autofit_inference
=================

Non-linear search and inference benchmarking for
[PyAutoFit](https://github.com/PyAutoLabs/PyAutoFit) — the proving ground for *which
search finds the right answer fastest*, on toy likelihoods where the truth is known.

It is the sibling of [autofit_profiling](https://github.com/PyAutoLabs/autofit_profiling)
(PyAutoFit timing) and the PyAutoFit counterpart of
[autolens_inference](https://github.com/PyAutoLabs/autolens_inference).

**Status: harness, datasets, protocol and reference posteriors (phase B2 of the
[search-extensibility epic](https://github.com/PyAutoLabs/PyAutoMind/blob/main/draft/research/autofit/search_extensibility_epic.md)).
No wave-1 row yet** — the pilot runs in B3, when the repo starts reporting to
[PyAutoInsight](https://github.com/PyAutoLabs/PyAutoInsight).

Agents and contributors: start with [`AGENTS.md`](AGENTS.md). Rulings of record live in
[PyAutoCortex](https://github.com/PyAutoLabs/PyAutoCortex) — see [`CORTEX.md`](CORTEX.md).

## The benchmark: `gaussian_x3`

Three 1D `af.ex.Gaussian` plus a constant `Background(level)`: 10 free parameters, with
ordered-centre assertions `g0.centre < g1.centre < g2.centre`, on two committed datasets
(`dataset/`, reproducible from `scripts/misc/simulators/gaussian_x3.py`):

- `gaussian_x3_blend` — centres 25 / 45 / 60, σ 3 / 6 / 10; g1 and g2 overlap.
- `gaussian_x3_separated` — the disjoint control (centres 20 / 50 / 80, σ 3 / 4 / 5).

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
| Task | Search | PyAutoFit class | Status | Settings | Note |
|---|---|---|---|---|---|
| point/MAP | `lbfgs` | `LBFGS` | registered | `default` | — |
| point/MAP | `bfgs` | `BFGS` | registered | `default` | — |
| point/MAP | `multi_start_adam` | `MultiStartAdam` | registered | `starts_48_steps_300` | — |
| point/MAP | `multi_start_prodigy` | `MultiStartProdigy` | registered | `starts_48_steps_300` | — |
| point/MAP | `multi_start_adabelief` | `MultiStartADABelief` | registered | `starts_48_steps_300` | — |
| point/MAP | `multi_start_lion` | `MultiStartLion` | registered | `starts_48_steps_300` | — |
| posterior | `emcee` | `Emcee` | registered | `walkers_50_steps_2000` | — |
| posterior | `zeus` | `Zeus` | registered | `walkers_50_steps_2000` | — |
| posterior | `blackjax_nuts` | `BlackJAXNUTS` | registered | `warmup_200_samples_500` | — |
| posterior | `smc` | `SMC` | registered | `default` | — |
| evidence | `nautilus` | `Nautilus` | registered | `n_live_100`, `n_live_200`, `n_live_400` | — |
| evidence | `dynesty_static` | `DynestyStatic` | registered | `nlive_200` | — |
| evidence | `dynesty_dynamic` | `DynestyDynamic` | registered | `default` | — |
| evidence | `nss` | `NSS` | deferred | `n_live_200` | NSS on the blend is deferred until phase A3b (NSS onto Fitness) |
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

ln 3! convention (constant likelihood): `dynesty_static`/`jax_cpu`: offset 0.000 (validated, measured 0.000); `dynesty_static`/`numpy`: offset 0.000 (validated, measured 0.000); `nautilus`/`jax_cpu`: offset -1.792 (validated, measured -1.825); `nautilus`/`numpy`: offset -1.792 (validated, measured -1.825).
<!-- END auto-table:references -->

## Search rows

<!-- BEGIN auto-table:searches -->
_No search rows yet — wave 1 (the pilot) runs in phase B3._
<!-- END auto-table:searches -->

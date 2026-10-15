"""Per-cell measured step rates for HPC submit `--time` estimates.

Copied from ``autolens_inference/scripts/misc/wall/rates.py`` and shipped empty, as
autolens_inference shipped its own: this repo inherits no rate from any other project.
Populated only from rates measured in this repo, on the cell the row names; nothing is
interpolated, extrapolated or carried across cells.

**The B3 pilot rows (B5a, 2026-10-14).** A "step" here is one search run (one seed), and
the rate is the median wall of the pilot's attempts of that cell on that dataset and
backend — every attempt that ran to its own end (completed, or failed on its own: a
non-finite answer, a trace error), never one the pilot's wall cap ``Terminated`` (its wall
is a censored lower bound) — with a warm start's provider wall included (protocol §8). A
cell with several settings carries its slowest settings' median (a submit is sized by its
slowest run). ``scripts/misc/tooling/plan_wave2.py`` re-derives the table from the
committed rows and refuses to plan when it differs.

Keys are ``(dataset, cell, instrument, device, precision, n_lanes, batch_size)``; the
``instrument`` slot holds the cell's leaf (the ``<model_type>``). Add a row with an
inline comment naming the RAL job and date, plus a ``PROVENANCE`` entry saying what it
must NOT be used for. `wall/check_submits.py` enforces the rule on every submit.
"""

from __future__ import annotations

# Key: (dataset, cell, instrument, device, precision, n_lanes, batch_size)
# Value: seconds per step, measured on that exact configuration, in this repo.
STEP_RATE: dict[tuple[str, str, str, str, str, int, int | None], float] = {
    # default: median of 10 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/bfgs",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 15.102919349999866,
    # default: median of 10 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/bfgs",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 9.021306599999662,
    # warmup_200_samples_500: median of 2 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/blackjax_nuts",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 25.940788050002084,
    # warmup_200_samples_500: median of 1 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/blackjax_nuts_warm",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 1233.7245691000026,
    # default: median of 2 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/dynesty_dynamic",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 841.7413687000007,
    # default: median of 2 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/dynesty_dynamic",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 557.4172283500011,
    # nlive_200: median of 2 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/dynesty_static",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 166.79425195000113,
    # nlive_200: median of 2 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/dynesty_static",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 172.92721325000093,
    # walkers_50_steps_2000: median of 1 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/emcee",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 1203.6711538000018,
    # walkers_50_steps_2000: median of 2 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/emcee",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 234.3571006500024,
    # default: median of 10 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/lbfgs",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 13.09298740000122,
    # default: median of 10 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/lbfgs",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 7.37818574999983,
    # starts_48_steps_300: median of 10 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/multi_start_adabelief",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 86.79080475000046,
    # starts_48_steps_300: median of 10 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/multi_start_adam",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 102.29083899999932,
    # starts_48_steps_300: median of 10 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/multi_start_lion",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 56.20928135000031,
    # starts_48_steps_300: median of 10 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/multi_start_prodigy",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 75.50850340000034,
    # n_live_400: median of 2 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/nautilus",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 1593.2795949500032,
    # n_live_400: median of 2 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/nautilus",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 2493.76463265,
    # default: median of 2 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/smc",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 379.1261949500022,
    # walkers_50_steps_2000: median of 1 pilot attempt(s)
    (
        "gaussian_x3_blend",
        "searches/zeus",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 769.202040199998,
    # default: median of 10 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/bfgs",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 35.79175750000104,
    # default: median of 10 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/bfgs",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 32.232422099999894,
    # warmup_200_samples_500: median of 2 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/blackjax_nuts",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 166.01127349999842,
    # warmup_200_samples_500: median of 1 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/blackjax_nuts_warm",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 1150.158048399997,
    # default: median of 2 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/dynesty_dynamic",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 822.3972178499989,
    # default: median of 2 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/dynesty_dynamic",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 850.1735010499997,
    # nlive_200: median of 2 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/dynesty_static",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 205.33680074999938,
    # nlive_200: median of 2 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/dynesty_static",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 228.75237444999766,
    # walkers_50_steps_2000: median of 2 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/emcee",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 1757.8389142999968,
    # walkers_50_steps_2000: median of 2 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/emcee",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 198.72722005000105,
    # default: median of 10 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/lbfgs",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 38.60408059999827,
    # default: median of 10 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/lbfgs",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 34.926994400000694,
    # starts_48_steps_300: median of 10 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/multi_start_adabelief",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 70.9817132499993,
    # starts_48_steps_300: median of 10 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/multi_start_adam",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 68.99266859999989,
    # starts_48_steps_300: median of 10 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/multi_start_lion",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 54.30726159999904,
    # starts_48_steps_300: median of 10 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/multi_start_prodigy",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 63.70203609999953,
    # n_live_400: median of 2 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/nautilus",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 1857.120405200003,
    # n_live_400: median of 2 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/nautilus",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 1962.4101040999994,
    # default: median of 1 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/smc",
        "gaussian_x3",
        "laptop_jax_cpu",
        "fp64",
        1,
        None,
    ): 322.3606273000041,
    # walkers_50_steps_2000: median of 2 pilot attempt(s)
    (
        "gaussian_x3_separated",
        "searches/zeus",
        "gaussian_x3",
        "laptop_numpy",
        "fp64",
        1,
        None,
    ): 1587.019891850001,
}

#: One entry per key prefix in ``STEP_RATE``: what ran, on which job, and the
#: configuration the row is and is not valid for.
PROVENANCE: dict[str, str] = {
    "laptop_numpy": (
        "B3 wave-1 pilot rows (results/searches/, pilot: true), local_numpy_fp64 at --cores 1 "
        "on one shared 8-CPU WSL laptop (DESKTOP-H143S82), PyAutoFit 0dbf258c4f5e (pre-A2), "
        "2026-10-08; no SLURM job. Valid for sizing a run of this cell at these settings. NOT "
        "valid as a RAL measurement: a RAL submit citing it states the device and asks for "
        "3x headroom (the rate floor is 1.5x), and NOT for other settings or a later "
        "library revision's behaviour beyond sizing"
    ),
    "laptop_jax_cpu": (
        "B3 wave-1 pilot rows, local_jax_cpu_fp64 (warm compile cache, NPROC 1 or 8), same "
        "laptop, revision and date. Same validity as laptop_numpy; Zeus on JAX has no row "
        "(every pilot attempt was Terminated at the wall cap) and must probe first"
    ),
}


class UnmeasuredCellError(KeyError):
    """No measured step rate exists for the requested configuration.

    Raised instead of returning a nearby cell's rate. Falling back across cells
    is the bug — see this module's docstring.
    """


def step_rate_for(
    dataset: str,
    cell: str,
    instrument: str,
    device: str,
    precision: str,
    n_lanes: int,
    batch_size: int | None = None,
) -> float:
    """Seconds per step for exactly this configuration.

    There is **no** nearest-neighbour fallback: an unmeasured configuration
    raises `UnmeasuredCellError` rather than silently answering with a rate
    measured on a different cell, lane count or batching.
    """
    key = (dataset, cell, instrument, device, precision, n_lanes, batch_size)
    try:
        return STEP_RATE[key]
    except KeyError:
        raise UnmeasuredCellError(
            f"no measured step rate for {key!r}. Do not substitute another cell's rate — "
            f"run one short arm on this cell and add the row to wall/rates.py, or declare "
            f"`source: unmeasured` with `probe-first: yes` in the submit's WALL-BASIS block."
        ) from None


def wall_estimate(rate_s_per_step: float, n_steps: int, compile_s: float = 0.0) -> float:
    """Estimated wall seconds for `n_steps` at `rate_s_per_step`, plus compile."""
    return rate_s_per_step * n_steps + compile_s

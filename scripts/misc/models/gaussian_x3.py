"""The ``gaussian_x3`` benchmark model: three 1D Gaussians plus a constant background.

This module is the single definition of the model, the likelihood and the priors that
every leaf, reference run and test in this repo fits. The numbers are pre-registered in
``wiki/project/protocol_gaussian_x3.md``; change them there first, never here alone.

The model (10 free parameters)
------------------------------

- ``g0``, ``g1``, ``g2`` — three ``af.ex.Gaussian`` (``centre``, ``normalization``,
  ``sigma``) with broad, shared priors: centre ``U(0, 100)``, normalization
  ``LogUniform(1e-2, 1e2)``, sigma ``U(0.5, 30)``.
- ``background`` — a thin :class:`Background` (``level``), prior ``U(-1, 1)``.

The components are exchangeable, so every good fit has ``3! = 6`` label-permuted copies.
The user-facing model breaks the symmetry with ordered-centre assertions
``g0.centre < g1.centre < g2.centre`` (protocol convention D15(a)). How PyAutoFit enforces
them depends on the backend: on numpy a violated assertion raises inside
``instance_from_vector`` and ``Fitness`` returns the resample sentinel; on JAX the
assertions are a traced boolean applied with ``xp.where`` to the figure of merit
(``autofit/non_linear/fitness.py``, ``Fitness.call``). The two mechanisms see different
effective prior volumes, so log evidences are compared only within one backend.

The likelihood
--------------

Gaussian noise with the normalisation term kept, so the log evidence is an absolute,
comparable number::

    ln L = -0.5 * sum( ((d - m) / n)^2 + ln(2 pi n^2) )

where ``m = g0(x) + g1(x) + g2(x) + level`` on ``x = 0..99``.
"""

from __future__ import annotations

import json
from pathlib import Path

import autofit as af
import numpy as np

#: Model identity recorded in every row.
MODEL_ID = "gaussian_x3"
MODEL_DESCRIPTION = "3 af.ex.Gaussian + Background(level), 10 free, ordered-centre assertions"

#: Component names, in centre order (the assertion order).
COMPONENTS = ("g0", "g1", "g2")

#: Pre-registered priors (protocol §Model).
CENTRE_PRIOR = (0.0, 100.0)
NORMALIZATION_PRIOR = (1.0e-2, 1.0e2)
SIGMA_PRIOR = (0.5, 30.0)
BACKGROUND_PRIOR = (-1.0, 1.0)

PRIORS_RECORD = {
    "centre": f"U({CENTRE_PRIOR[0]:g}, {CENTRE_PRIOR[1]:g})",
    "normalization": f"LogUniform({NORMALIZATION_PRIOR[0]:g}, {NORMALIZATION_PRIOR[1]:g})",
    "sigma": f"U({SIGMA_PRIOR[0]:g}, {SIGMA_PRIOR[1]:g})",
    "background.level": f"U({BACKGROUND_PRIOR[0]:g}, {BACKGROUND_PRIOR[1]:g})",
    "assertions": "g0.centre < g1.centre < g2.centre",
}

#: The parameter keys of a row, in the model's own vector order (checked by
#: :func:`parameter_keys` against ``model.paths`` at run time).
PARAMETER_KEYS = (
    "g0.centre",
    "g0.normalization",
    "g0.sigma",
    "g1.centre",
    "g1.normalization",
    "g1.sigma",
    "g2.centre",
    "g2.normalization",
    "g2.sigma",
    "background.level",
)


class Background:
    """A constant level added to every pixel — the tenth parameter."""

    def __init__(self, level: float = 0.0):
        self.level = level

    def model_data_from(self, xvalues, xp=np):
        return xp.zeros_like(xvalues, dtype=float) + self.level

    def _tree_flatten(self):
        return (self.level,), None

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children)


def model_data_from(instance, xvalues, xp=np):
    """The summed model curve for an instance (or any object with ``g0..g2`` and
    ``background`` attributes)."""
    total = instance.background.model_data_from(xvalues=xvalues, xp=xp)
    for name in COMPONENTS:
        total = total + getattr(instance, name).model_data_from(xvalues=xvalues, xp=xp)
    return total


def build_model(assertions: bool = True):
    """The 10-parameter ``af.Collection``. ``assertions=False`` gives the exchangeable
    model (used only by the constant-likelihood convention check)."""
    components = {}
    for name in COMPONENTS:
        gaussian = af.Model(af.ex.Gaussian)
        gaussian.centre = af.UniformPrior(lower_limit=CENTRE_PRIOR[0], upper_limit=CENTRE_PRIOR[1])
        gaussian.normalization = af.LogUniformPrior(
            lower_limit=NORMALIZATION_PRIOR[0], upper_limit=NORMALIZATION_PRIOR[1]
        )
        gaussian.sigma = af.UniformPrior(lower_limit=SIGMA_PRIOR[0], upper_limit=SIGMA_PRIOR[1])
        components[name] = gaussian
    background = af.Model(Background)
    background.level = af.UniformPrior(
        lower_limit=BACKGROUND_PRIOR[0], upper_limit=BACKGROUND_PRIOR[1]
    )
    model = af.Collection(**components, background=background)
    if assertions:
        model.add_assertion(model.g0.centre < model.g1.centre)
        model.add_assertion(model.g1.centre < model.g2.centre)
    return model


def parameter_keys(model) -> list[str]:
    """Dotted keys (``g0.centre`` …) in the model's vector order."""
    return [".".join(path) for path in model.paths]


class AnalysisGaussianX3(af.Analysis):
    """Gaussian log likelihood of the summed model curve, noise normalisation kept.

    Module-level (not built in a factory) so DynestyStatic's checkpoint can pickle it.
    Import this module only after the backend environment is set
    (``_autofit_inference_cli.set_backend_env``).
    """

    def __init__(self, data, noise_map, use_jax: bool = False):
        super().__init__(use_jax=use_jax)
        self.data = np.asarray(data, dtype=float)
        self.noise_map = np.asarray(noise_map, dtype=float)
        self.xvalues = np.arange(self.data.shape[0], dtype=float)
        self.log_noise_normalization = float(np.sum(np.log(2.0 * np.pi * self.noise_map**2)))

    def log_likelihood_function(self, instance):
        xp = self._xp
        model_data = model_data_from(instance, xp.asarray(self.xvalues), xp=xp)
        chi_squared = xp.sum(((xp.asarray(self.data) - model_data) / self.noise_map) ** 2)
        return -0.5 * (chi_squared + self.log_noise_normalization)


class TrackedAnalysisGaussianX3(AnalysisGaussianX3):
    """``AnalysisGaussianX3`` that hands every evaluated log likelihood to an MLTracker.

    Only the numpy callback path can do this (under JAX the likelihood runs inside
    ``jit``/``vmap``, so ``tracker`` is ``None`` there and times are estimated, protocol
    §8). Assertion-violating points never reach the likelihood, so the tracker counts
    only evaluations that passed the assertions. Module-level so Dynesty can pickle it.
    """

    def __init__(self, data, noise_map, use_jax: bool = False, tracker=None):
        super().__init__(data=data, noise_map=noise_map, use_jax=use_jax)
        self.tracker = tracker

    def log_likelihood_function(self, instance):
        value = super().log_likelihood_function(instance)
        if self.tracker is not None:
            self.tracker.record(float(value))
        return value


#: The constant-likelihood check's tie-breaking slope (nats): nested samplers stall on
#: an exactly flat likelihood, so ``ln L = -TIE_BREAK * level^2`` with ``level`` in
#: ``[-1, 1]``. Its exact evidence on the exchangeable prior is
#: ``ln(sqrt(pi / TIE_BREAK) * erf(sqrt(TIE_BREAK)) / 2)``, about ``-TIE_BREAK / 3``.
TIE_BREAK = 1.0e-3


def constant_log_evidence_exact(tie_break: float = TIE_BREAK) -> float:
    """``ln Z`` of the constant-likelihood analysis on the exchangeable model."""
    import math

    return math.log(math.sqrt(math.pi / tie_break) * math.erf(math.sqrt(tie_break)) / 2.0)


class AnalysisConstant(af.Analysis):
    """Near-constant log likelihood (``-TIE_BREAK * level^2``): its log evidence is the
    log of the prior volume the sampler counts, which is how the protocol validates the
    ``ln 3!`` label convention."""

    def log_likelihood_function(self, instance):
        level = instance.background.level
        return -TIE_BREAK * level * level


def load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(data, noise_map, truth)`` from a committed dataset directory."""
    dataset_path = Path(dataset_path)
    data = np.asarray(json.loads((dataset_path / "data.json").read_text()), dtype=float)
    noise_map = np.asarray(json.loads((dataset_path / "noise_map.json").read_text()), dtype=float)
    truth = json.loads((dataset_path / "truth.json").read_text())
    return data, noise_map, truth


def truth_dict(truth_record: dict) -> dict[str, float]:
    """Flatten ``truth.json`` into ``{"g0.centre": …, "background.level": …}``."""
    return {key: float(value) for key, value in truth_record["parameters"].items()}


def relabel_vector(vector, keys=PARAMETER_KEYS):
    """Sort one parameter vector's Gaussian components by centre (convention D15(c)).

    The ordered-centre assertions already make this the identity for a sample that
    satisfies them; it matters for exchangeable models and for searches that report
    samples outside the assertion region.
    """
    values = dict(zip(keys, vector))
    triples = sorted(
        (
            (values[f"{name}.centre"], values[f"{name}.normalization"], values[f"{name}.sigma"])
            for name in COMPONENTS
        ),
        key=lambda triple: triple[0],
    )
    out = {}
    for name, (centre, normalization, sigma) in zip(COMPONENTS, triples):
        out[f"{name}.centre"] = centre
        out[f"{name}.normalization"] = normalization
        out[f"{name}.sigma"] = sigma
    out["background.level"] = values["background.level"]
    return [out[key] for key in keys]

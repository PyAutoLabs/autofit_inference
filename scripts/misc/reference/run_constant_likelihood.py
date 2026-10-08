"""Validate the ``ln 3!`` label convention on a near-constant likelihood (protocol §3).

With ``ln L = −10⁻³ level²`` the evidence is (to 3 × 10⁻⁴ nat) the fraction of the prior
volume the sampler counts. On the exchangeable model (no assertions) that is the whole
prior, ``ln Z_exch ≈ 0``. With the ordered-centre assertions the allowed region is exactly
1/6 of the prior: a sampler that counts the excluded volume reports ``−ln 3!``; one whose
initial live points come only from the allowed region reports ``0``. This script measures
which, per sampler and backend, and writes ``results/reference/constant_likelihood.json``:

    {"offsets": {"<sampler>": {"<backend>": {"offset", "validated", ...}}}}

``offset`` is ``0`` or ``−ln 3!``; ``validated`` is true when the measured difference sits
within 0.1 nat of one of them. Each (sampler, backend) runs once with and once without
the assertions; results from earlier invocations are kept (one invocation per backend).

    python scripts/misc/reference/run_constant_likelihood.py --config-name local_numpy_fp64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _autofit_inference_cli import (  # noqa: E402
    device_info_dict,
    library_revisions,
    set_backend_env,
    validate_cli,
)

SETTINGS = {
    "nautilus": {"n_live": 500, "seed": 0},
    "dynesty_static": {"nlive": 500},
    # B3: the pilot runs DynestyDynamic as an evidence search, so its offset is measured too.
    "dynesty_dynamic": {"nlive_init": 500},
}
CLASSES = {
    "nautilus": "Nautilus",
    "dynesty_static": "DynestyStatic",
    "dynesty_dynamic": "DynestyDynamic",
}
#: The live-point keyword ``--live`` overrides, per sampler.
LIVE_KWARG = {"nautilus": "n_live", "dynesty_static": "nlive", "dynesty_dynamic": "nlive_init"}
WINDOW_NAT = 0.1


def classify(difference: float) -> tuple[float | None, bool]:
    """``(offset, validated)`` for a measured ``ln Z_assert − ln Z_exch``."""
    for offset in (0.0, -math.log(6.0)):
        if abs(difference - offset) <= WINDOW_NAT:
            return offset, True
    return None, False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config-name", default="local_numpy_fp64")
    parser.add_argument("--samplers", nargs="+", default=sorted(SETTINGS))
    parser.add_argument(
        "--live",
        type=int,
        default=None,
        help="override n_live / nlive (every attempt is kept under `attempts`)",
    )
    args = parser.parse_args(argv)
    where, backend, precision = validate_cli(args.config_name, None, None)
    set_backend_env(backend, precision, cores=1)

    import autofit as af
    from models import gaussian_x3 as gx3

    output_root = (_ROOT / "output").resolve()
    af.conf.instance.push(new_path=_ROOT / "config", output_path=output_root)
    path = _ROOT / "results" / "reference" / "constant_likelihood.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    record.update(
        {
            "schema_version": 1,
            "kind": "constant_likelihood",
            "protocol_id": "gaussian_x3@1",
            "log_likelihood": f"-{gx3.TIE_BREAK:g} * level^2",
            "exact_log_evidence_exchangeable": gx3.constant_log_evidence_exact(),
            "ln_3_factorial": math.log(6.0),
            "window_nat": WINDOW_NAT,
        }
    )
    offsets = record.setdefault("offsets", {})
    analysis = gx3.AnalysisConstant(use_jax=backend == "jax_cpu")
    for sampler in args.samplers:
        settings = dict(SETTINGS[sampler])
        if args.live is not None:
            settings[LIVE_KWARG[sampler]] = args.live
        tag = "_".join(f"{k}{v}" for k, v in sorted(settings.items()))
        values = {}
        for label, assertions in (("assert", True), ("exch", False)):
            model = gx3.build_model(assertions=assertions)
            search = getattr(af, CLASSES[sampler])(
                path_prefix=Path("reference") / "constant_likelihood" / args.config_name,
                name=f"{sampler}_{label}_{tag}",
                **settings,
            )
            start = time.perf_counter()
            result = search.fit(model=model, analysis=analysis)
            values[label] = {
                "log_evidence": float(result.samples.log_evidence),
                "wall_s": time.perf_counter() - start,
            }
        difference = values["assert"]["log_evidence"] - values["exch"]["log_evidence"]
        offset, validated = classify(difference)
        previous = offsets.get(sampler, {}).get(backend)
        attempts = (previous or {}).pop("attempts", []) if previous else []
        if previous:
            attempts.append(previous)
        offsets.setdefault(sampler, {})[backend] = {
            "attempts": attempts,
            "offset": offset,
            "validated": validated,
            "difference": difference,
            "runs": values,
            "settings": settings,
            "config_name": args.config_name,
            "where": where,
            "assertion_mechanism": "xp_where_penalty" if backend == "jax_cpu" else "raise_resample",
            "version": af.__version__,
            "library_revisions": library_revisions(),
            "device": device_info_dict(backend),
            "host": os.uname().nodename,
        }
        print(
            f"  {sampler}/{backend}: ln Z assert {values['assert']['log_evidence']:.3f}, "
            f"exch {values['exch']['log_evidence']:.3f}, diff {difference:.3f} -> "
            f"offset {offset} validated {validated}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

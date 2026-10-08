"""Combine the long reference runs into ``results/reference/<dataset>/<backend>/reference.json``.

Protocol ``gaussian_x3@1`` §6. Reads the per-run rows ``run_reference.py`` wrote, the
``ln 3!`` offsets ``run_constant_likelihood.py`` measured and the MAP reference
``run_map_reference.py`` wrote, and records:

- per-run ``ln Z``, the normalised ``ln Z − offset`` and the agreement statistics
  (``log_evidence_spread`` over runs; ``median_agreement_sigma`` = the worst
  ``|run median − pooled median| / pooled σ``);
- the pooled marginals: every included run's weighted samples (``samples.csv`` from its
  kept output directory), each run's weights normalised to sum to one, so each run counts
  equally;
- ``status: complete`` when at least the three Nautilus runs exist; ``included_runs``:
  all runs when they agree (§6: 0.2 nat, 0.1 σ), otherwise the agreeing family, with the
  disagreement written into ``limitations``.

Run locally after the reference runs (it needs their ``output/`` directories)::

    python scripts/misc/reference/build_reference.py --dataset gaussian_x3_blend --backend numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402
from searches import _posterior as post  # noqa: E402
from searches import _protocol as protocol  # noqa: E402

LOGZ_AGREEMENT_NAT = 0.2
MEDIAN_AGREEMENT_SIGMA = 0.1
SAMPLERS = ("nautilus", "dynesty_static")


def read_samples_csv(path: Path, keys) -> tuple[np.ndarray, np.ndarray]:
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = [h.strip() for h in next(reader)]
        rows = [[float(v) for v in row] for row in reader if row]
    table = np.asarray(rows, dtype=float)
    columns = [header.index(k) for k in keys]
    return table[:, columns], table[:, header.index("weight")]


def run_samples(run: dict, keys) -> tuple[np.ndarray, np.ndarray] | None:
    output = _ROOT / "output" / run["output_path"] / "files" / "samples.csv"
    if not output.exists():
        return None
    return read_samples_csv(output, keys)


def agreement(runs: list[dict], pooled: dict, offsets) -> dict:
    normalised = [
        protocol.normalised_log_evidence(
            r["log_evidence"], protocol.evidence_offset(offsets, r["sampler"], r["backend"])
        )
        for r in runs
    ]
    raw = [r["log_evidence"] for r in runs]
    values = [v for v in normalised if v is not None]
    worst, worst_key = 0.0, None
    for r in runs:
        for key, stats in pooled.items():
            delta = abs(r["posterior_relabelled"][key]["median"] - stats["median"]) / stats["sigma"]
            if delta > worst:
                worst, worst_key = delta, f"{r['sampler']}_seed{r['seed']}:{key}"
    spread = (max(values) - min(values)) if len(values) == len(runs) and values else None
    return {
        "log_evidence_raw": raw,
        "log_evidence_normalised": normalised,
        "log_evidence_spread": spread,
        "log_evidence_raw_spread": max(raw) - min(raw),
        "median_agreement_sigma": worst,
        "median_agreement_worst": worst_key,
        "logz_ok": spread is not None and spread <= LOGZ_AGREEMENT_NAT,
        "median_ok": worst <= MEDIAN_AGREEMENT_SIGMA,
    }


def pool(runs: list[dict], keys) -> tuple[dict, list[dict], str]:
    blocks, weights = [], []
    for r in runs:
        loaded = run_samples(r, keys)
        if loaded is None:
            break
        matrix, w = loaded
        blocks.append(matrix)
        weights.append(post.normalised_weights(w))
    if len(blocks) == len(runs):
        matrix = np.vstack(blocks)
        w = np.concatenate(weights) / len(runs)
        return (
            post.posterior_stats(post.relabel_matrix(matrix, keys), w, keys),
            post.mode_clusters(matrix, w, keys),
            "pooled samples.csv (equal weight per run)",
        )
    stats = {
        key: {
            field: float(np.mean([r["posterior_relabelled"][key][field] for r in runs]))
            for field in runs[0]["posterior_relabelled"][key]
        }
        for key in keys
    }
    return stats, runs[0]["modes"], "mean of per-run marginals (output/ not available)"


def build(dataset: str, backend: str) -> dict:
    directory = _ROOT / "results" / "reference" / dataset / backend
    runs = []
    for sampler in SAMPLERS:
        for path in sorted(directory.glob(f"{sampler}_seed*.json")):
            runs.append(json.loads(path.read_text()))
    offsets = protocol.load_json(protocol.offsets_file(_ROOT))
    map_ref = protocol.load_json(directory / "map_reference.json")
    by_sampler = {s: [r for r in runs if r["sampler"] == s] for s in SAMPLERS}
    out = {
        "schema_version": 1,
        "kind": "reference",
        "protocol_id": protocol.PROTOCOL_ID,
        "dataset": dataset,
        "backend": backend,
        "runs": [f"{r['sampler']}_seed{r['seed']}" for r in runs],
        "limitations": [],
    }
    if len(by_sampler["nautilus"]) < 3:
        out["status"] = "pending"
        out["limitations"].append("fewer than three Nautilus reference runs")
        return out
    keys = list(runs[0]["posterior_relabelled"])
    out["assertion_mechanism"] = runs[0]["assertion_mechanism"]

    all_stats, _, _ = pool(runs, keys)
    all_agree = agreement(runs, all_stats, offsets)
    included = runs
    if not (all_agree["logz_ok"] and all_agree["median_ok"]):
        families = {}
        for sampler, members in by_sampler.items():
            if len(members) >= 3:
                stats, _, _ = pool(members, keys)
                families[sampler] = agreement(members, stats, offsets)
        agreeing = [s for s, a in families.items() if a["median_ok"] and a["logz_ok"]]
        chosen = "nautilus" if "nautilus" in agreeing or not agreeing else agreeing[0]
        included = by_sampler[chosen]
        out["family_agreement"] = families
        out["limitations"].append(
            f"the six runs disagree (ln Z spread {all_agree['log_evidence_spread']}, "
            f"worst median {all_agree['median_agreement_sigma']:.2f} sigma at "
            f"{all_agree['median_agreement_worst']}); the reference is the {chosen} runs only"
        )
    stats, modes, pooling = pool(included, keys)
    included_agree = agreement(included, stats, offsets)
    normalised = [v for v in included_agree["log_evidence_normalised"] if v is not None]
    out.update(
        {
            "status": "complete",
            "included_runs": [f"{r['sampler']}_seed{r['seed']}" for r in included],
            "pooling": pooling,
            "agreement_all": all_agree,
            "agreement_included": included_agree,
            "posterior_relabelled": stats,
            "modes": modes,
            "log_evidence": float(np.mean([r["log_evidence"] for r in included])),
            "log_evidence_normalised": float(np.mean(normalised)) if normalised else None,
            "ppc_chi2": float(np.mean([r["ppc_chi2"] for r in included])),
            "max_log_likelihood": float(max(r["max_log_likelihood"] for r in runs)),
            "max_log_posterior_sampled": float(max(r["max_log_posterior"] for r in runs)),
            "best_sample_vector": max(runs, key=lambda r: r["max_log_posterior"])[
                "max_log_posterior_vector"
            ],
            "map": (map_ref or {}).get("map"),
            "truths": runs[0]["truths"],
            "pyautofit_commits": sorted(
                {(r.get("library_revisions") or {}).get("autofit") or "unknown" for r in runs}
            ),
        }
    )
    if normalised == []:
        out["limitations"].append("no validated ln 3! offset: log_evidence_normalised is null")
    if out["map"] is None:
        out["limitations"].append("MAP reference pending: point/MAP rows are not assessable")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="gaussian_x3_blend")
    parser.add_argument("--backend", choices=("numpy", "jax_cpu"), default="numpy")
    args = parser.parse_args(argv)
    out = build(args.dataset, args.backend)
    path = _ROOT / "results" / "reference" / args.dataset / args.backend / "reference.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path} (status {out['status']})")
    for line in out["limitations"]:
        print(f"  limitation: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate paired BAFU benchmark runs and write a compact summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(direct: dict[str, Any], iterative: dict[str, Any]) -> dict[str, Any]:
    direct_records = direct["records"]
    iterative_records = iterative["records"]
    if len(direct_records) != len(iterative_records):
        raise ValueError("Run lengths differ")

    for left, right in zip(direct_records, iterative_records):
        if left["iteration"] != right["iteration"]:
            raise ValueError("Iteration indices differ")
        if left["technosphere_fingerprint"] != right["technosphere_fingerprint"]:
            raise ValueError(f"Technosphere mismatch at {left['iteration']}")
        if left["biosphere_fingerprint"] != right["biosphere_fingerprint"]:
            raise ValueError(f"Biosphere mismatch at {left['iteration']}")

    direct_scores = np.array([record["score"] for record in direct_records])
    iterative_scores = np.array([record["score"] for record in iterative_records])
    absolute = np.abs(iterative_scores - direct_scores)
    relative = absolute / np.maximum(np.abs(direct_scores), np.finfo(float).tiny)
    direct_times = [record["seconds"] for record in direct_records]
    iterative_times = [record["seconds"] for record in iterative_records]
    gmres_iterations = [record["gmres_iterations"] for record in iterative_records]
    iterative_residuals = [record["relative_residual"] for record in iterative_records]

    return {
        "samples": len(direct_records),
        "paired_matrix_fingerprints_match": True,
        "functional_unit": direct["activity"],
        "method": direct["method"],
        "method_unit": direct["method_unit"],
        "technosphere_shape": direct["technosphere_shape"],
        "technosphere_nnz": direct["technosphere_nnz"],
        "direct": {
            "backend": direct["direct_backend"],
            "calculation_seconds": direct["calculation_seconds"],
            "median_iteration_seconds_excluding_first": median(direct_times[1:]),
            "incremental_peak_rss_bytes": direct["incremental_peak_rss_bytes"],
            "peak_rss_bytes": direct["peak_rss_bytes"],
        },
        "jacobi_gmres": {
            "calculation_seconds": iterative["calculation_seconds"],
            "median_iteration_seconds_excluding_first": median(iterative_times[1:]),
            "incremental_peak_rss_bytes": iterative["incremental_peak_rss_bytes"],
            "peak_rss_bytes": iterative["peak_rss_bytes"],
            "use_guess": iterative["use_guess"],
            "rtol": iterative["rtol"],
            "median_gmres_iterations": median(gmres_iterations),
            "maximum_gmres_iterations": max(gmres_iterations),
            "maximum_relative_residual": max(iterative_residuals),
            "stochastic_matrix_rebind_fix": iterative["stochastic_matrix_rebind_fix"],
        },
        "score_agreement": {
            "maximum_absolute_difference": float(absolute.max()),
            "median_absolute_difference": float(np.median(absolute)),
            "maximum_relative_difference": float(relative.max()),
            "median_relative_difference": float(np.median(relative)),
            "direct_mean": float(direct_scores.mean()),
            "direct_standard_deviation": float(direct_scores.std(ddof=1)),
            "iterative_mean": float(iterative_scores.mean()),
            "iterative_standard_deviation": float(iterative_scores.std(ddof=1)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("direct", type=Path)
    parser.add_argument("iterative", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(load(args.direct), load(args.iterative))
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

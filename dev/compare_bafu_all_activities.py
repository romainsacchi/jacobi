"""Compare fixed-matrix all-activity BAFU score runs."""

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
    left = direct["records"]
    right = iterative["records"]
    if len(left) != len(right):
        raise ValueError("Activity counts differ")
    if [record["activity_id"] for record in left] != [
        record["activity_id"] for record in right
    ]:
        raise ValueError("Activity order differs")

    direct_scores = np.array([record["score"] for record in left])
    iterative_scores = np.array([record["score"] for record in right])
    absolute = np.abs(iterative_scores - direct_scores)
    relative = absolute / np.maximum(np.abs(direct_scores), np.finfo(float).tiny)
    return {
        "activity_count": len(left),
        "same_activity_order": True,
        "direct": {
            "calculation_seconds": direct["calculation_seconds"],
            "seconds_per_activity": direct["seconds_per_activity"],
            "incremental_peak_rss_bytes": direct["incremental_peak_rss_bytes"],
            "factorized_once": direct["factorized_once"],
            "characterization_matrix_built_once": direct[
                "characterization_matrix_built_once"
            ],
        },
        "jacobi_gmres": {
            "calculation_seconds": iterative["calculation_seconds"],
            "seconds_per_activity": iterative["seconds_per_activity"],
            "incremental_peak_rss_bytes": iterative["incremental_peak_rss_bytes"],
            "rtol": iterative["rtol"],
            "use_guess": iterative["use_guess"],
            "median_gmres_iterations": iterative["median_gmres_iterations"],
            "maximum_relative_residual": iterative["maximum_relative_residual"],
            "characterization_matrix_built_once": iterative[
                "characterization_matrix_built_once"
            ],
        },
        "score_agreement": {
            "maximum_absolute_difference": float(absolute.max()),
            "median_absolute_difference": float(np.median(absolute)),
            "maximum_relative_difference": float(relative.max()),
            "median_relative_difference": float(np.median(relative)),
            "direct_mean": float(direct_scores.mean()),
            "iterative_mean": float(iterative_scores.mean()),
            "direct_median": float(median(direct_scores)),
            "iterative_median": float(median(iterative_scores)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("direct", type=Path)
    parser.add_argument("iterative", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(load(args.direct), load(args.iterative))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

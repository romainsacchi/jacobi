"""Score every activity in the BAFU database with one fixed LCA system."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.benchmark_bafu import (
    MemoryMonitor,
    install_gmres_instrumentation,
    relative_residual,
    stochastic_jacobi_class,
    version,
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import bw2calc as bc
    import bw2data as bd

    if args.project not in {project.name for project in bd.projects}:
        raise ValueError(f"Missing Brightway project: {args.project}")
    bd.projects.set_current(args.project)
    if args.database not in bd.databases:
        raise ValueError(f"Missing database {args.database!r}")

    method = tuple(args.method)
    if method not in bd.methods:
        raise ValueError(f"Missing LCIA method: {method!r}")

    activities = sorted(bd.Database(args.database), key=lambda node: node.id)
    if args.limit is not None:
        activities = activities[: args.limit]
    if not activities:
        raise ValueError("No activities found")

    instrument = {"iterations": None, "info": None}
    if args.solver == "jacobi-gmres":
        instrument = install_gmres_instrumentation()
        calculator = stochastic_jacobi_class(bc.JacobiGMRESLCA)
    else:
        calculator = bc.LCA

    first_demand = {activities[0].id: 1.0}
    kwargs: dict[str, Any] = {
        "demand": first_demand,
        "method": method,
        "use_distributions": False,
    }
    if args.solver == "jacobi-gmres":
        kwargs.update(
            rtol=args.rtol,
            atol=0.0,
            restart=args.restart,
            maxiter=args.maxiter,
            use_guess=args.use_guess,
        )

    object_started = time.perf_counter()
    lca = calculator(**kwargs)
    object_seconds = time.perf_counter() - object_started
    load_started = time.perf_counter()
    lca.load_lci_data()
    # Build the characterization matrix once; every later lcia(demand=...) reuses it.
    lca.load_lcia_data()
    matrix_load_seconds = time.perf_counter() - load_started

    records: list[dict[str, Any]] = []
    with MemoryMonitor() as memory:
        calculation_started = time.perf_counter()
        for index, activity in enumerate(activities):
            started = time.perf_counter()
            demand = {activity.id: 1.0}
            if index == 0:
                # Direct LCA factorizes once; subsequent lci(demand=...) calls reuse lca.solver.
                lca.lci(demand=demand, factorize=args.solver == "direct")
                lca.lcia()
            else:
                lca.lcia(demand=demand)
            records.append(
                {
                    "index": index,
                    "activity_id": activity.id,
                    "code": activity.get("code"),
                    "name": activity.get("name"),
                    "reference_product": activity.get("reference product"),
                    "location": activity.get("location"),
                    "unit": activity.get("unit"),
                    "seconds": time.perf_counter() - started,
                    "score": float(lca.score),
                    "relative_residual": relative_residual(lca),
                    "gmres_iterations": instrument["iterations"],
                    "gmres_info": instrument["info"],
                }
            )
        calculation_seconds = time.perf_counter() - calculation_started

    return {
        "kind": "bafu-all-activities",
        "solver": args.solver,
        "direct_backend": "scikit-umfpack" if args.solver == "direct" else None,
        "project": args.project,
        "database": args.database,
        "activity_count": len(records),
        "method": list(method),
        "method_unit": bd.Method(method).metadata.get("unit"),
        "stochastic": False,
        "rtol": args.rtol if args.solver == "jacobi-gmres" else None,
        "use_guess": args.use_guess if args.solver == "jacobi-gmres" else None,
        "factorized_once": args.solver == "direct",
        "characterization_matrix_built_once": True,
        "object_seconds": object_seconds,
        "matrix_load_seconds": matrix_load_seconds,
        "calculation_seconds": calculation_seconds,
        "seconds_per_activity": calculation_seconds / len(records),
        "baseline_rss_bytes": memory.baseline_bytes,
        "peak_rss_bytes": memory.peak_bytes,
        "incremental_peak_rss_bytes": memory.incremental_peak_bytes,
        "maximum_relative_residual": max(
            record["relative_residual"] for record in records
        ),
        "median_gmres_iterations": (
            float(np.median([record["gmres_iterations"] for record in records]))
            if args.solver == "jacobi-gmres"
            else None
        ),
        "records": records,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "bw2calc": version("bw2calc"),
            "bw2data": version("bw2data"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "scikit_umfpack": version("scikit-umfpack"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("direct", "jacobi-gmres"), required=True)
    parser.add_argument("--project", default="brightcon-2026")
    parser.add_argument("--database", default="bafu")
    parser.add_argument(
        "--method",
        nargs=3,
        default=("IPCC 2021", "climate change", "global warming potential (GWP100)"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--restart", type=int, default=50)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument(
        "--use-guess", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "solver": result["solver"],
                "activity_count": result["activity_count"],
                "calculation_seconds": result["calculation_seconds"],
                "seconds_per_activity": result["seconds_per_activity"],
                "maximum_relative_residual": result["maximum_relative_residual"],
            }
        )
    )


if __name__ == "__main__":
    main()

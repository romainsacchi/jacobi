"""Orchestrate isolated synthetic solver benchmarks with optional time guards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_worker(
    python: str,
    worker: Path,
    solver: str,
    size: int,
    topology: str,
    degree: int,
    density: float,
    diagonal_span: float,
    rtol: float,
    matrix_family: str,
    blocks: int,
    rhs_count: int,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    command = [
        python,
        str(worker),
        "--solver",
        solver,
        "--size",
        str(size),
        "--topology",
        topology,
        "--degree",
        str(degree),
        "--density",
        str(density),
        "--diagonal-span",
        str(diagonal_span),
        "--rtol",
        str(rtol),
        "--matrix-family",
        matrix_family,
        "--blocks",
        str(blocks),
        "--rhs-count",
        str(rhs_count),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "kind": "synthetic",
            "solver": solver,
            "size": size,
            "topology": topology,
            "degree": degree if topology == "constant-degree" else None,
            "target_density": density if topology == "fixed-density" else None,
            "diagonal_span_orders": diagonal_span,
            "rtol": rtol,
            "matrix_family": matrix_family,
            "rhs_count": rhs_count,
            "timed_out": True,
            "status": "TIMEOUT",
            "worker_wall_seconds": time.perf_counter() - started,
        }
    except subprocess.CalledProcessError as error:
        return {
            "kind": "synthetic",
            "solver": solver,
            "size": size,
            "topology": topology,
            "degree": degree if topology == "constant-degree" else None,
            "target_density": density if topology == "fixed-density" else None,
            "diagonal_span_orders": diagonal_span,
            "rtol": rtol,
            "matrix_family": matrix_family,
            "rhs_count": rhs_count,
            "timed_out": False,
            "status": "FAILED",
            "worker_wall_seconds": time.perf_counter() - started,
            "error": error.stderr[-2000:],
        }

    result = json.loads(completed.stdout.strip().splitlines()[-1])
    result["timed_out"] = False
    result["status"] = "COMPLETED"
    result["worker_wall_seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--worker", type=Path, default=Path("dev/benchmark_synthetic.py")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2500, 5000])
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["numpy-dense", "superlu", "umfpack", "pardiso", "gmres", "jacobi-gmres"],
    )
    parser.add_argument(
        "--topology",
        choices=("constant-degree", "fixed-density", "banded"),
        default="constant-degree",
    )
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--density", type=float, default=0.001)
    parser.add_argument("--degrees", type=int, nargs="+")
    parser.add_argument("--densities", type=float, nargs="+")
    parser.add_argument(
        "--matrix-family", choices=("lca-random", "io-block", "banded"), default="lca-random"
    )
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--rhs-count", type=int, default=1)
    parser.add_argument("--rhs-counts", type=int, nargs="+")
    parser.add_argument("--diagonal-span", type=float, default=4.0)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--dense-max", type=int, default=2500)
    parser.add_argument("--run-timeout", type=float)
    parser.add_argument("--total-budget", type=float)
    parser.add_argument(
        "--max-estimated-construction-mib",
        type=float,
        help="Skip a matrix before construction when estimated working memory exceeds this cap",
    )
    parser.add_argument(
        "--construction-memory-multiplier",
        type=float,
        default=3.0,
        help="Multiplier applied to estimated CSC storage for construction intermediates",
    )
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    suite_started = time.perf_counter()
    stop = False
    degrees = args.degrees or [args.degree]
    densities = args.densities or [args.density]
    rhs_counts = args.rhs_counts or [args.rhs_count]
    cases = [
        (size, degree, density, rhs_count)
        for size in args.sizes
        for degree in (degrees if args.topology == "constant-degree" else [args.degree])
        for density in (densities if args.topology == "fixed-density" else [args.density])
        for rhs_count in rhs_counts
    ]
    for size, degree, density, rhs_count in cases:
        estimated_nnz = (
            size * (degree + 1)
            if args.topology == "constant-degree"
            else size * (2 * degree + 1)
            if args.topology == "banded"
            else int(size * size * density) + size
        )
        estimated_storage_bytes = estimated_nnz * 12 + (size + 1) * 4
        estimated_construction_mib = (
            estimated_storage_bytes * args.construction_memory_multiplier / 2**20
        )
        guarded = (
            args.max_estimated_construction_mib is not None
            and estimated_construction_mib > args.max_estimated_construction_mib
        )
        for solver in args.solvers:
            if solver == "numpy-dense" and size > args.dense_max:
                continue
            if guarded:
                result = {
                    "kind": "synthetic",
                    "solver": solver,
                    "size": size,
                    "topology": args.topology,
                    "degree": degree if args.topology in {"constant-degree", "banded"} else None,
                    "target_density": density if args.topology == "fixed-density" else None,
                    "matrix_family": "banded" if args.topology == "banded" else args.matrix_family,
                    "rhs_count": rhs_count,
                    "estimated_nnz": estimated_nnz,
                    "estimated_construction_mib": estimated_construction_mib,
                    "status": "SKIPPED",
                    "skip_reason": "MEMORY GUARD",
                    "timed_out": False,
                }
                results.append(result)
                print(json.dumps(result), flush=True)
                continue
            if (
                args.total_budget is not None
                and time.perf_counter() - suite_started >= args.total_budget
            ):
                stop = True
                break
            result = run_worker(
                args.python,
                args.worker,
                solver,
                size,
                args.topology,
                degree,
                density,
                args.diagonal_span,
                args.rtol,
                args.matrix_family,
                args.blocks,
                rhs_count,
                args.run_timeout,
            )
            result["estimated_nnz"] = estimated_nnz
            result["estimated_construction_mib"] = estimated_construction_mib
            results.append(result)
            print(
                json.dumps(
                    {
                        key: result.get(key)
                        for key in (
                            "solver",
                            "size",
                            "degree",
                            "target_density",
                            "matrix_family",
                            "rhs_count",
                            "solve_seconds",
                            "factorization_seconds",
                            "rhs_solve_seconds",
                            "lu_fill_ratio",
                            "incremental_peak_rss_bytes",
                            "iterations",
                            "relative_residual",
                            "timed_out",
                            "status",
                            "error",
                        )
                    }
                ),
                flush=True,
            )
        if stop:
            break

    payload = {
        "suite_wall_seconds": time.perf_counter() - suite_started,
        "stopped_by_total_budget": stop,
        "run_timeout_seconds": args.run_timeout,
        "total_budget_seconds": args.total_budget,
        "max_estimated_construction_mib": args.max_estimated_construction_mib,
        "construction_memory_multiplier": args.construction_memory_multiplier,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

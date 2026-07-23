"""Orchestrate isolated synthetic solver benchmarks with time guards."""

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
    timeout_seconds: float,
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
            "timed_out": True,
            "worker_wall_seconds": time.perf_counter() - started,
        }
    except subprocess.CalledProcessError as error:
        return {
            "kind": "synthetic",
            "solver": solver,
            "size": size,
            "topology": topology,
            "timed_out": False,
            "worker_wall_seconds": time.perf_counter() - started,
            "error": error.stderr[-2000:],
        }

    result = json.loads(completed.stdout.strip().splitlines()[-1])
    result["timed_out"] = False
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
        default=["numpy-dense", "superlu", "umfpack", "gmres", "jacobi-gmres"],
    )
    parser.add_argument(
        "--topology",
        choices=("constant-degree", "fixed-density"),
        default="constant-degree",
    )
    parser.add_argument("--dense-max", type=int, default=2500)
    parser.add_argument("--run-timeout", type=float, default=20.0)
    parser.add_argument("--total-budget", type=float, default=60.0)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    suite_started = time.perf_counter()
    stop = False
    for size in args.sizes:
        for solver in args.solvers:
            if solver == "numpy-dense" and size > args.dense_max:
                continue
            if time.perf_counter() - suite_started >= args.total_budget:
                stop = True
                break
            result = run_worker(
                args.python,
                args.worker,
                solver,
                size,
                args.topology,
                args.run_timeout,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        key: result.get(key)
                        for key in (
                            "solver",
                            "size",
                            "solve_seconds",
                            "incremental_peak_rss_bytes",
                            "iterations",
                            "relative_residual",
                            "timed_out",
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
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

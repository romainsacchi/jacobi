"""Benchmark direct or Jacobi-GMRES Brightway calculations on BAFU.

Run each solver in a separate process. The script records only aggregate
calculation metadata, matrix fingerprints, and LCIA scores.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil


class MemoryMonitor:
    def __init__(self, interval_seconds: float = 0.002):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.baseline_bytes = self.process.memory_info().rss
        self.peak_bytes = self.baseline_bytes
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
            time.sleep(self.interval_seconds)

    def __enter__(self) -> "MemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    @property
    def incremental_peak_bytes(self) -> int:
        return max(0, self.peak_bytes - self.baseline_bytes)


def sparse_fingerprint(matrix: Any) -> str:
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    digest = hashlib.blake2b(digest_size=12)
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def relative_residual(lca: Any) -> float:
    numerator = np.linalg.norm(
        lca.technosphere_matrix @ lca.supply_array - lca.demand_array
    )
    denominator = np.linalg.norm(lca.demand_array)
    return float(numerator / denominator)


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def find_activity(database: Any, code: str) -> Any:
    matches = [node for node in database if node.get("code") == code]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one activity with code {code!r}; found {len(matches)}"
        )
    activity = matches[0]
    expected = (
        "Electricity, low voltage, at grid",
        "Electricity, low voltage, at grid",
        "CH",
        "kilowatt hour",
    )
    observed = (
        activity.get("name"),
        activity.get("reference product"),
        activity.get("location"),
        activity.get("unit"),
    )
    if observed != expected:
        raise ValueError(f"Unexpected functional unit metadata: {observed!r}")
    return activity


def install_gmres_instrumentation() -> dict[str, Any]:
    import bw2calc.jacobi_gmres_lca as module

    state: dict[str, Any] = {"iterations": None, "info": None}
    original = module.gmres

    def monitored_gmres(*args: Any, **kwargs: Any) -> tuple[np.ndarray, int]:
        residuals: list[float] = []
        kwargs["callback"] = residuals.append
        kwargs["callback_type"] = "pr_norm"
        solution, info = original(*args, **kwargs)
        state["iterations"] = len(residuals)
        state["info"] = int(info)
        return solution, info

    module.gmres = monitored_gmres
    return state


def stochastic_jacobi_class(base: Any) -> Any:
    class StochasticJacobiGMRESLCA(base):
        """Keep the active matrix attached to the iterated matrix manager."""

        def after_matrix_iteration(self) -> None:
            self.technosphere_matrix = self.technosphere_mm.matrix
            self._matrix_prepared = False
            self._cached_preconditioner = None

    return StochasticJacobiGMRESLCA


def run(args: argparse.Namespace) -> dict[str, Any]:
    import bw2calc as bc
    import bw2data as bd

    if args.project not in {project.name for project in bd.projects}:
        raise ValueError(f"Missing Brightway project: {args.project}")
    bd.projects.set_current(args.project)
    if args.database not in bd.databases:
        raise ValueError(
            f"Missing database {args.database!r}; available: {sorted(bd.databases)}"
        )

    method = tuple(args.method)
    if method not in bd.methods:
        raise ValueError(f"Missing LCIA method: {method!r}")
    activity = find_activity(bd.Database(args.database), args.activity_code)

    instrument = {"iterations": None, "info": None}
    if args.solver == "jacobi-gmres":
        instrument = install_gmres_instrumentation()
        calculator = (
            stochastic_jacobi_class(bc.JacobiGMRESLCA)
            if args.stochastic_fix
            else bc.JacobiGMRESLCA
        )
    else:
        calculator = bc.LCA

    kwargs: dict[str, Any] = {
        "demand": {activity.id: 1.0},
        "method": method,
        "use_distributions": args.stochastic,
        "seed_override": args.seed,
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

    matrix_load_started = time.perf_counter()
    lca.load_lci_data()
    lca.load_lcia_data()
    matrix_load_seconds = time.perf_counter() - matrix_load_started

    records: list[dict[str, Any]] = []
    with MemoryMonitor() as memory:
        calculation_started = time.perf_counter()
        iteration_started = time.perf_counter()
        lca.lci()
        lca.lcia()
        records.append(
            {
                "iteration": 0,
                "seconds": time.perf_counter() - iteration_started,
                "score": float(lca.score),
                "relative_residual": relative_residual(lca),
                "gmres_iterations": instrument["iterations"],
                "gmres_info": instrument["info"],
                "technosphere_fingerprint": sparse_fingerprint(lca.technosphere_matrix),
                "biosphere_fingerprint": sparse_fingerprint(lca.biosphere_matrix),
            }
        )

        for index in range(1, args.iterations):
            iteration_started = time.perf_counter()
            next(lca)
            records.append(
                {
                    "iteration": index,
                    "seconds": time.perf_counter() - iteration_started,
                    "score": float(lca.score),
                    "relative_residual": relative_residual(lca),
                    "gmres_iterations": instrument["iterations"],
                    "gmres_info": instrument["info"],
                    "technosphere_fingerprint": sparse_fingerprint(
                        lca.technosphere_matrix
                    ),
                    "biosphere_fingerprint": sparse_fingerprint(lca.biosphere_matrix),
                }
            )
        calculation_seconds = time.perf_counter() - calculation_started

    matrix = lca.technosphere_matrix
    biosphere = lca.biosphere_matrix
    return {
        "kind": "bafu",
        "solver": args.solver,
        "direct_backend": "scikit-umfpack" if args.solver == "direct" else None,
        "project": args.project,
        "database": args.database,
        "activity": {
            "key": list(activity.key),
            "id": activity.id,
            "name": activity.get("name"),
            "reference_product": activity.get("reference product"),
            "location": activity.get("location"),
            "unit": activity.get("unit"),
            "amount": 1.0,
        },
        "method": list(method),
        "method_unit": bd.Method(method).metadata.get("unit"),
        "stochastic": args.stochastic,
        "stochastic_matrix_rebind_fix": (
            args.stochastic_fix if args.solver == "jacobi-gmres" else None
        ),
        "seed": args.seed,
        "iterations_requested": args.iterations,
        "use_guess": args.use_guess if args.solver == "jacobi-gmres" else None,
        "rtol": args.rtol if args.solver == "jacobi-gmres" else None,
        "restart": args.restart if args.solver == "jacobi-gmres" else None,
        "maxiter": args.maxiter if args.solver == "jacobi-gmres" else None,
        "technosphere_shape": list(matrix.shape),
        "technosphere_nnz": int(matrix.nnz),
        "technosphere_density": float(matrix.nnz / np.prod(matrix.shape)),
        "biosphere_shape": list(biosphere.shape),
        "biosphere_nnz": int(biosphere.nnz),
        "object_seconds": object_seconds,
        "matrix_load_seconds": matrix_load_seconds,
        "calculation_seconds": calculation_seconds,
        "baseline_rss_bytes": memory.baseline_bytes,
        "peak_rss_bytes": memory.peak_bytes,
        "incremental_peak_rss_bytes": memory.incremental_peak_bytes,
        "records": records,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "physical_memory_bytes": psutil.virtual_memory().total,
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
    parser.add_argument("--activity-code", default="bafu-219622")
    parser.add_argument(
        "--method",
        nargs=3,
        default=(
            "IPCC 2021",
            "climate change",
            "global warming potential (GWP100)",
        ),
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--stochastic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--stochastic-fix", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-guess", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--restart", type=int, default=50)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run(args)
    rendered = json.dumps(result, sort_keys=True)
    if args.quiet:
        print(
            json.dumps(
                {
                    "solver": result["solver"],
                    "iterations": result["iterations_requested"],
                    "calculation_seconds": result["calculation_seconds"],
                    "incremental_peak_rss_bytes": result["incremental_peak_rss_bytes"],
                    "maximum_relative_residual": max(
                        record["relative_residual"] for record in result["records"]
                    ),
                }
            )
        )
    else:
        print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

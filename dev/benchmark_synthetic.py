"""Run one isolated synthetic linear-system benchmark and emit JSON.

Each invocation handles one matrix and one solver so peak RSS measurements are
not contaminated by allocations retained by an earlier solver.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import psutil
import scipy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, spsolve


@dataclass
class MemoryMonitor:
    interval_seconds: float = 0.002

    def __post_init__(self) -> None:
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


def constant_degree_matrix(
    n: int, degree: int, seed: int, diagonal_span: float
) -> sparse.csc_matrix:
    if degree < 1 or degree >= n:
        raise ValueError("degree must be between 1 and n - 1")
    rng = np.random.default_rng(seed)
    columns = np.repeat(np.arange(n, dtype=np.int64), degree)
    rows = rng.integers(0, n, size=n * degree, dtype=np.int64)
    diagonal_hits = rows == columns
    rows[diagonal_hits] = (rows[diagonal_hits] + 1) % n

    # Keep each column's absolute off-diagonal sum comfortably below one.
    coefficients = rng.uniform(0.01, 0.04, size=n * degree)
    off_diagonal = sparse.coo_matrix(
        (-coefficients, (rows, columns)), shape=(n, n)
    ).tocsc()
    matrix = sparse.eye(n, format="csc") + off_diagonal
    row_scales = 10 ** rng.uniform(-diagonal_span / 2, diagonal_span / 2, size=n)
    matrix = sparse.diags(row_scales, format="csc") @ matrix
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def fixed_density_matrix(
    n: int, density: float, seed: int, diagonal_span: float
) -> sparse.csc_matrix:
    if not 0 < density < 1:
        raise ValueError("density must be between zero and one")
    rng = np.random.default_rng(seed)
    off_diagonal = sparse.random(
        n,
        n,
        density=density,
        format="csc",
        random_state=rng,
        data_rvs=lambda size: rng.uniform(0.01, 0.04, size),
    )
    off_diagonal.setdiag(0.0)
    off_diagonal.eliminate_zeros()

    # Normalize columns so the increasing density does not make A singular.
    column_sums = np.asarray(off_diagonal.sum(axis=0)).ravel()
    scale = np.ones(n)
    nonzero = column_sums > 0
    scale[nonzero] = np.minimum(1.0, 0.4 / column_sums[nonzero])
    off_diagonal = off_diagonal @ sparse.diags(scale, format="csc")
    matrix = sparse.eye(n, format="csc") - off_diagonal
    row_scales = 10 ** rng.uniform(-diagonal_span / 2, diagonal_span / 2, size=n)
    matrix = sparse.diags(row_scales, format="csc") @ matrix
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def matrix_fingerprint(matrix: sparse.csc_matrix) -> str:
    digest = hashlib.blake2b(digest_size=12)
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def solve(
    matrix: sparse.csc_matrix,
    demand: np.ndarray,
    solver: str,
    rtol: float,
    restart: int,
    maxiter: int,
) -> tuple[np.ndarray, int | None, int]:
    if solver == "numpy-dense":
        return np.linalg.solve(matrix.toarray(), demand), None, 0
    if solver == "superlu":
        return spsolve(matrix, demand, use_umfpack=False), None, 0
    if solver == "umfpack":
        import scikits.umfpack  # noqa: F401

        return spsolve(matrix, demand, use_umfpack=True), None, 0

    residual_history: list[float] = []
    preconditioner = None
    if solver == "jacobi-gmres":
        diagonal = matrix.diagonal()
        if np.any(diagonal == 0):
            raise ValueError("Jacobi requires a non-zero matrix diagonal")
        inverse_diagonal = 1.0 / diagonal
        preconditioner = LinearOperator(
            matrix.shape,
            matvec=lambda vector: inverse_diagonal * vector,
            dtype=matrix.dtype,
        )
    elif solver != "gmres":
        raise ValueError(f"Unknown solver: {solver}")

    solution, info = gmres(
        matrix,
        demand,
        M=preconditioner,
        rtol=rtol,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
        callback=residual_history.append,
        callback_type="pr_norm",
    )
    return solution, int(info), len(residual_history)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    if args.topology == "constant-degree":
        matrix = constant_degree_matrix(
            args.size, args.degree, args.seed, args.diagonal_span
        )
    else:
        matrix = fixed_density_matrix(
            args.size, args.density, args.seed, args.diagonal_span
        )
    generation_seconds = time.perf_counter() - started

    demand = np.zeros(args.size)
    demand[0] = 1.0
    matrix_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes

    with MemoryMonitor() as memory:
        started = time.perf_counter()
        solution, info, iterations = solve(
            matrix,
            demand,
            args.solver,
            args.rtol,
            args.restart,
            args.maxiter,
        )
        solve_seconds = time.perf_counter() - started

    relative_residual = float(
        np.linalg.norm(matrix @ solution - demand) / np.linalg.norm(demand)
    )
    converged = info in (None, 0) and relative_residual <= max(args.rtol * 10, 1e-12)

    return {
        "kind": "synthetic",
        "topology": args.topology,
        "solver": args.solver,
        "size": args.size,
        "shape": [args.size, args.size],
        "degree": args.degree if args.topology == "constant-degree" else None,
        "target_density": args.density if args.topology == "fixed-density" else None,
        "diagonal_span_orders": args.diagonal_span,
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (args.size * args.size)),
        "matrix_storage_bytes": int(matrix_bytes),
        "matrix_fingerprint": matrix_fingerprint(matrix),
        "seed": args.seed,
        "rtol": args.rtol,
        "restart": args.restart,
        "maxiter": args.maxiter,
        "generation_seconds": generation_seconds,
        "solve_seconds": solve_seconds,
        "baseline_rss_bytes": memory.baseline_bytes,
        "peak_rss_bytes": memory.peak_bytes,
        "incremental_peak_rss_bytes": memory.incremental_peak_bytes,
        "iterations": iterations,
        "solver_info": info,
        "relative_residual": relative_residual,
        "converged": converged,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_umfpack": package_version("scikit-umfpack"),
            "physical_memory_bytes": psutil.virtual_memory().total,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=("numpy-dense", "superlu", "umfpack", "gmres", "jacobi-gmres"),
        required=True,
    )
    parser.add_argument(
        "--topology",
        choices=("constant-degree", "fixed-density"),
        default="constant-degree",
    )
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--density", type=float, default=0.001)
    parser.add_argument("--diagonal-span", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--restart", type=int, default=50)
    parser.add_argument("--maxiter", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()

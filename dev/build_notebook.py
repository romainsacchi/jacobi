"""Build the Brightcon presentation notebook with deterministic cell IDs."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "jupyter-notebook" / "brightcon-2026-jacobi-gmres.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


cells = [
    markdown(r"""
# From 8 minutes to 4 seconds
## Solving large systems with Jacobi + GMRES

**Romain Sacchi (PSI) · Brightcon 2026 · 25 September 2026 · 8 minutes**  
Open Tools and Development · Aalborg University & online

> Direct sparse factorisation is an excellent default—until fill-in makes memory and runtime scale much faster than the matrix itself. An iterative solve makes that trade-off explicit and controllable.
"""),
    markdown(r"""
## Run of show

1. Start with bare synthetic technosphere matrices and solve $Ax=b$ five ways.
2. Scale the same matrix family while measuring runtime, peak RSS, iterations, and residuals.
3. Reuse one fixed synthetic matrix across many right-hand sides.
4. Rebuild paired synthetic matrices across repeated samples.

Synthetic workers run in isolated subprocesses with explicit timeouts and a pre-construction memory
guard. Unsafe cells are labeled `SKIPPED`; failed or timed-out cells are preserved as diagnostics,
and only an unavailable live suite falls back to committed calibration results.
"""),
    code(r"""
from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy
from IPython.display import Markdown, display
from scipy.sparse.linalg import LinearOperator, gmres, spsolve


def find_repo_root(start: Path = Path.cwd()) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "dev" / "benchmark_synthetic.py").exists():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the repository")


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIVE = os.environ.get("BRIGHTCON_LIVE", "1") != "0"
BW_PROJECT = os.environ.get("BRIGHTCON_BW_PROJECT", "brightcon-2026")
DATABASE = "bafu"
ACTIVITY_CODE = "bafu-219622"
METHOD = ("IPCC 2021", "climate change", "global warming potential (GWP100)")
SEED = 2026
RTOL = 1e-4
try:
    import pypardiso  # noqa: F401
    PARDISO_AVAILABLE = True
except ImportError:
    PARDISO_AVAILABLE = False
DIRECT_SOLVERS = ["umfpack"] + (["pardiso"] if PARDISO_AVAILABLE else [])

{
    "live workers": LIVE,
    "worker Python": sys.executable,
    "Brightway project": BW_PROJECT,
    "NumPy": np.__version__,
    "SciPy": scipy.__version__,
    "scikit-umfpack": importlib.metadata.version("scikit-umfpack"),
}
"""),
    markdown(r"""
# 1 · Strip LCA down to $Ax=b$

- $A$: technosphere matrix—positive production diagonal, negative inputs.
- $b$: demand vector—one unit of the functional unit.
- $x$: supply array—how much each activity must produce.

The first matrix is deliberately ill-scaled across four orders of magnitude. This makes the purpose of diagonal preconditioning visible without any biosphere or characterization matrices.
"""),
    code(r"""
from dev.benchmark_synthetic import constant_degree_matrix

A_tiny = constant_degree_matrix(n=12, degree=2, seed=SEED, diagonal_span=4.0)
b_tiny = np.zeros(12)
b_tiny[0] = 1.0

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].spy(A_tiny, markersize=7, color="#315B7D")
axes[0].set_title(f"Sparse structure: {A_tiny.nnz} nonzeros")
axes[0].set_xlabel("activity")
axes[0].set_ylabel("product")
axes[1].bar(np.arange(12), np.abs(A_tiny.diagonal()), color="#12A594")
axes[1].set_yscale("log")
axes[1].set_title("Production diagonal")
axes[1].set_xlabel("activity")
axes[1].set_ylabel("absolute value, log scale")
fig.tight_layout()
plt.show()
"""),
    markdown(r"""
## Five approaches, one solution

- Dense LAPACK through `numpy.linalg.solve`—useful only while $A$ is small.
- Sparse SuperLU, UMFPACK, and optionally Pardiso—robust direct factorisations.
- GMRES—approximate Krylov solve with an explicit residual tolerance.
- Jacobi + GMRES—left-precondition with $D^{-1}$, the reciprocal diagonal.

The entire Jacobi preconditioner is the `LinearOperator` below; no dense inverse is created.
"""),
    code(r"""
def timed(function):
    started = perf_counter()
    value = function()
    return value, perf_counter() - started


def krylov(matrix, demand, *, jacobi: bool):
    history = []
    preconditioner = None
    if jacobi:
        inverse_diagonal = 1.0 / matrix.diagonal()
        preconditioner = LinearOperator(
            matrix.shape,
            matvec=lambda vector: inverse_diagonal * vector,
            dtype=matrix.dtype,
        )
    solution, info = gmres(
        matrix,
        demand,
        M=preconditioner,
        rtol=RTOL,
        atol=0.0,
        restart=50,
        maxiter=300,
        callback=history.append,
        callback_type="pr_norm",
    )
    return solution, info, len(history)


tiny_runs = []
tiny_solver_runs = [
    ("NumPy dense", lambda: (np.linalg.solve(A_tiny.toarray(), b_tiny), None, 0)),
    ("SuperLU", lambda: (spsolve(A_tiny, b_tiny, use_umfpack=False), None, 0)),
    ("UMFPACK", lambda: (spsolve(A_tiny, b_tiny, use_umfpack=True), None, 0)),
    ("GMRES", lambda: krylov(A_tiny, b_tiny, jacobi=False)),
    ("Jacobi + GMRES", lambda: krylov(A_tiny, b_tiny, jacobi=True)),
]
if PARDISO_AVAILABLE:
    from pypardiso import spsolve as pardiso_spsolve
    tiny_solver_runs.insert(3, ("Pardiso", lambda: (pardiso_spsolve(A_tiny, b_tiny), None, 0)))
for label, function in tiny_solver_runs:
    (solution, info, iterations), seconds = timed(function)
    residual = np.linalg.norm(A_tiny @ solution - b_tiny) / np.linalg.norm(b_tiny)
    tiny_runs.append(
        {"solver": label, "seconds": seconds, "iterations": iterations, "residual": residual, "info": info}
    )

pd.DataFrame(tiny_runs).style.format({"seconds": "{:.2e}", "residual": "{:.2e}"})
"""),
    markdown(r"""
### Audience prompt

What happens to the benefit of Jacobi if every production diagonal is already one? The answer cell repeats the comparison on a unit-scaled matrix.
"""),
    code(r"""
A_unit = constant_degree_matrix(n=500, degree=8, seed=SEED, diagonal_span=0.0)
b_unit = np.zeros(500)
b_unit[0] = 1.0
_, _, plain_iterations = krylov(A_unit, b_unit, jacobi=False)
_, _, jacobi_iterations = krylov(A_unit, b_unit, jacobi=True)
{"plain GMRES iterations": plain_iterations, "Jacobi + GMRES iterations": jacobi_iterations}
"""),
    markdown(r"""
# 2 · Scale the matrix, isolate every solver

Each solver runs in a fresh subprocess. This prevents one factorisation from contaminating the next solver's peak memory. A 2 ms RSS sampler captures allocations made by NumPy, SciPy, UMFPACK, and BLAS—not only Python objects. Incremental RSS is measured above the post-matrix-construction baseline.

Main series: approximately eight inputs per activity. Dense solving stops at 2,500 activities; the remaining isolated workers run to completion without a benchmark timeout.
"""),
    code(r"""
def run_checked(command: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def live_or_stored_synthetic():
    live_path = ROOT / "results" / "live_synthetic.json"
    fallback = ROOT / "results" / "synthetic_calibration.json"
    if LIVE:
        command = [
            sys.executable,
            "dev/run_synthetic_suite.py",
            "--python", sys.executable,
            "--output", str(live_path),
            "--sizes", "500", "1000", "2500", "5000", "10000",
            "--rtol", str(RTOL),
        ]
        try:
            run_checked(command)
            return json.loads(live_path.read_text()), "LIVE"
        except Exception as error:
            return json.loads(fallback.read_text()), f"STORED FALLBACK ({type(error).__name__})"
    return json.loads(fallback.read_text()), "STORED RESULTS"


synthetic_payload, synthetic_source = live_or_stored_synthetic()
display(Markdown(f"**Result source: {synthetic_source}**"))
synthetic = pd.json_normalize(synthetic_payload["results"])
synthetic["memory_mib"] = synthetic["incremental_peak_rss_bytes"] / 2**20
synthetic[["solver", "size", "solve_seconds", "memory_mib", "iterations", "relative_residual", "timed_out"]]
"""),
    code(r"""
solver_order = ["numpy-dense", "superlu", *DIRECT_SOLVERS, "gmres", "jacobi-gmres"]
colors = {
    "numpy-dense": "#7F8C8D",
    "superlu": "#C44E52",
    "umfpack": "#DD8452",
    "pardiso": "#937860",
    "gmres": "#4C72B0",
    "jacobi-gmres": "#12A594",
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
valid = synthetic[synthetic["solve_seconds"].notna()]
for solver in solver_order:
    subset = valid[valid.solver == solver].sort_values("size")
    if subset.empty:
        continue
    axes[0].plot(subset["size"], subset["solve_seconds"], "o-", label=solver, color=colors[solver])
    axes[1].plot(subset["size"], subset["memory_mib"], "o-", label=solver, color=colors[solver])

axes[0].set_yscale("log")
axes[0].set_ylabel("solve time [s, log]")
axes[1].set_yscale("log")
axes[1].set_ylabel("incremental peak RSS [MiB, log]")
for axis in axes:
    axis.set_xlabel("matrix rows / columns")
    axis.grid(alpha=0.25, which="both")
axes[0].set_title("Factorisation time separates")
axes[1].set_title("Fill-in dominates memory")
axes[0].legend(frameon=False, ncol=2, fontsize=8)
fig.tight_layout()
plt.show()
"""),
    markdown(r"""
### Connectivity versus size

The first scaling curve keeps roughly eight inputs per activity. This additional sweep holds the
number of inputs per activity at 5 or 25 while changing matrix size. It separates the effect of
database size from the effect of denser connectivity.
"""),
    code(r"""
def live_or_stored_connectivity():
    live_path = ROOT / "results" / "live_connectivity.json"
    fallback = ROOT / "results" / "synthetic_connectivity_calibration.json"
    if LIVE:
        command = [
            sys.executable, "dev/run_synthetic_suite.py", "--python", sys.executable,
            "--output", str(live_path), "--sizes", "1000", "5000", "10000",
            "--solvers", *DIRECT_SOLVERS, "jacobi-gmres", "--topology", "constant-degree",
            "--degrees", "5", "25", "--rtol", str(RTOL), "--run-timeout", "60",
        ]
        try:
            run_checked(command)
            return json.loads(live_path.read_text()), "LIVE"
        except Exception as error:
            return json.loads(fallback.read_text()), f"STORED FALLBACK ({type(error).__name__})"
    return json.loads(fallback.read_text()), "STORED RESULTS"


connectivity_payload, connectivity_source = live_or_stored_connectivity()
display(Markdown(f"**Result source: {connectivity_source}**"))
connectivity = pd.json_normalize(connectivity_payload["results"])
connectivity["memory_mib"] = connectivity["incremental_peak_rss_bytes"] / 2**20
connectivity["speedup_vs_umfpack"] = np.nan
for (size, degree), subset in connectivity.groupby(["size", "degree"]):
    direct = subset.loc[subset.solver == "umfpack", "solve_seconds"]
    if len(direct) == 1:
        connectivity.loc[subset.index, "speedup_vs_umfpack"] = direct.iloc[0] / subset["solve_seconds"]
display(connectivity[["solver", "size", "degree", "solve_seconds", "memory_mib", "speedup_vs_umfpack", "status"]])
"""),
    markdown(r"""
### Large synthetic systems

The same fixed-connectivity construction is also attempted at 50,000, 100,000, and 300,000 rows.
Dense and SuperLU are omitted at these sizes; UMFPACK, GMRES, and Jacobi + GMRES run in isolated
workers with explicit status, timeout, and total-budget fields.
"""),
    code(r"""
def live_or_stored_large_scaling():
    live_path = ROOT / "results" / "live_synthetic_large.json"
    fallback = ROOT / "results" / "synthetic_large_calibration.json"
    if LIVE:
        available_mib = psutil.virtual_memory().available / 2**20
        construction_guard_mib = int(min(8192, max(512, available_mib * 0.25)))
        command = [
            sys.executable, "dev/run_synthetic_suite.py", "--python", sys.executable,
            "--output", str(live_path), "--sizes", "50000", "100000", "300000",
            "--solvers", *DIRECT_SOLVERS, "gmres", "jacobi-gmres",
            "--topology", "constant-degree", "--degree", "8", "--rtol", str(RTOL),
            "--run-timeout", "120", "--total-budget", "900",
            "--max-estimated-construction-mib", str(construction_guard_mib),
        ]
        try:
            run_checked(command, timeout=960)
            return json.loads(live_path.read_text()), "LIVE"
        except Exception as error:
            return json.loads(fallback.read_text()), f"STORED FALLBACK ({type(error).__name__})"
    return json.loads(fallback.read_text()), "STORED RESULTS"


large_payload, large_source = live_or_stored_large_scaling()
display(Markdown(f"**Result source: {large_source}**"))
large_results = pd.json_normalize(large_payload["results"])
large_results["memory_mib"] = large_results["incremental_peak_rss_bytes"] / 2**20
display(large_results[["solver", "size", "degree", "nnz", "status", "worker_wall_seconds", "solve_seconds", "memory_mib", "iterations", "relative_residual"]])
"""),
    markdown(r"""
## Guarded size–density stress test

With constant density, nonzeros grow with $n^2$, not $n$. The guarded grid below adds 0.3%, 1%,
3%, 5%, 10%, and 15% density to the original 0.1% case. High-density cases are intentionally
limited to smaller matrices: the pre-construction guard labels unsafe cases `SKIPPED` instead of
allocating them. The `io-block` family is also included to show that density alone does not capture
input-output structure.
"""),
    code(r"""
def live_or_stored_density_grid():
    live_path = ROOT / "results" / "live_density_grid.json"
    fallback = ROOT / "results" / "synthetic_density_grid_calibration.json"
    if LIVE:
        available_mib = psutil.virtual_memory().available / 2**20
        construction_guard_mib = int(min(8192, max(512, available_mib * 0.25)))
        command = [
            sys.executable, "dev/run_synthetic_suite.py", "--python", sys.executable,
            "--output", str(live_path), "--sizes", "1000", "2500", "5000", "10000", "20000",
            "--solvers", *DIRECT_SOLVERS, "gmres", "jacobi-gmres",
            "--topology", "fixed-density", "--densities",
            "0.001", "0.003", "0.01", "0.03", "0.05", "0.1", "0.15",
            "--matrix-family", "lca-random", "--rtol", str(RTOL),
            "--run-timeout", "90", "--total-budget", "900", "--max-estimated-construction-mib", str(construction_guard_mib),
            "--construction-memory-multiplier", "3",
        ]
        try:
            run_checked(command, timeout=600)
            return json.loads(live_path.read_text()), "LIVE"
        except Exception as error:
            return json.loads(fallback.read_text()), f"STORED FALLBACK ({type(error).__name__})"
    return json.loads(fallback.read_text()), "STORED RESULTS"


density_payload, density_source = live_or_stored_density_grid()
display(Markdown(f"**Result source: {density_source}**"))
density_results = pd.json_normalize(density_payload["results"])
density_results["incremental_peak_MiB"] = density_results["incremental_peak_rss_bytes"] / 2**20
density_results["runtime_status"] = density_results.get("status", "COMPLETED")
display(density_results[["solver", "size", "target_density", "nnz", "runtime_status", "solve_seconds", "incremental_peak_MiB", "iterations", "relative_residual"]])

completed = density_results[density_results.runtime_status == "COMPLETED"].copy()
completed["log10_speedup_umfpack_over_jacobi"] = np.nan
for (size, density), subset in completed.groupby(["size", "target_density"]):
    direct = subset.loc[subset.solver == "umfpack", "solve_seconds"]
    iterative = subset.loc[subset.solver == "jacobi-gmres", "solve_seconds"]
    if len(direct) == 1 and len(iterative) == 1:
        value = np.log10(direct.iloc[0] / iterative.iloc[0])
        completed.loc[subset.index, "log10_speedup_umfpack_over_jacobi"] = value

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for solver, subset in completed.groupby("solver"):
    axes[0].scatter(subset["density"], subset["solve_seconds"], label=solver, color=colors[solver], alpha=0.8)
    axes[0].plot(subset["density"], subset["solve_seconds"], color=colors[solver], alpha=0.35)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("realized density [log]")
axes[0].set_ylabel("solve time [s, log]")
axes[0].set_title("Runtime versus density")
axes[0].grid(alpha=0.25, which="both")

pivot = completed[completed.solver == "jacobi-gmres"].pivot_table(index="size", columns="target_density", values="log10_speedup_umfpack_over_jacobi")
image = axes[1].imshow(pivot, aspect="auto", cmap="PiYG", vmin=-2, vmax=2)
axes[1].set_xticks(range(len(pivot.columns)), [f"{value:.1%}" for value in pivot.columns], rotation=45, ha="right")
axes[1].set_yticks(range(len(pivot.index)), pivot.index)
axes[1].set_xlabel("target density")
axes[1].set_ylabel("matrix size")
axes[1].set_title("log₁₀(UMFPACK / Jacobi time)")
fig.colorbar(image, ax=axes[1], label="positive = Jacobi faster")
fig.tight_layout()
plt.show()
"""),
    markdown(r"""
### Matrix structure sensitivity

At the same nominal densities, block-structured input-output matrices can behave differently from
uniform random cross-links. This small companion sweep keeps the high-density cases bounded while
showing that density is a proxy for structure, not a complete explanation.
"""),
    code(r"""
def live_or_stored_io_block():
    live_path = ROOT / "results" / "live_density_ioblock.json"
    fallback = ROOT / "results" / "synthetic_density_ioblock_calibration.json"
    if LIVE:
        available_mib = psutil.virtual_memory().available / 2**20
        construction_guard_mib = int(min(8192, max(512, available_mib * 0.25)))
        command = [
            sys.executable, "dev/run_synthetic_suite.py", "--python", sys.executable,
            "--output", str(live_path), "--sizes", "1000", "2500", "5000", "10000",
            "--solvers", *DIRECT_SOLVERS, "jacobi-gmres", "--topology", "fixed-density",
            "--densities", "0.01", "0.05", "0.15", "--matrix-family", "io-block",
            "--blocks", "8", "--rtol", str(RTOL), "--run-timeout", "90", "--total-budget", "600",
            "--max-estimated-construction-mib", str(construction_guard_mib),
        ]
        try:
            run_checked(command, timeout=300)
            return json.loads(live_path.read_text()), "LIVE"
        except Exception as error:
            return json.loads(fallback.read_text()), f"STORED FALLBACK ({type(error).__name__})"
    return json.loads(fallback.read_text()), "STORED RESULTS"


ioblock_payload, ioblock_source = live_or_stored_io_block()
display(Markdown(f"**Result source: {ioblock_source}**"))
ioblock = pd.json_normalize(ioblock_payload["results"])
ioblock["memory_mib"] = ioblock["incremental_peak_rss_bytes"] / 2**20
display(ioblock[["solver", "size", "target_density", "nnz", "solve_seconds", "memory_mib", "iterations", "status"]])
"""),
    markdown(r"""
# 3 · The same switch inside Brightway

```python
from bw2calc import LCA, JacobiGMRESLCA

direct = LCA(demand, method=method)
iterative = JacobiGMRESLCA(
    demand,
    method=method,
    rtol=1e-4,
    use_guess=True,
)
```

The presentation project is prepared before the clock starts. The notebook validates the exact project, database, functional unit, and method; it does not import Excel live.
"""),
    code(r"""
BW_READY = False
BW_ERROR = None
try:
    import bw2calc as bc
    import bw2data as bd

    if BW_PROJECT not in {project.name for project in bd.projects}:
        raise ValueError(f"Missing project: {BW_PROJECT}")
    bd.projects.set_current(BW_PROJECT)
    if DATABASE not in bd.databases:
        raise ValueError(f"Missing database: {DATABASE}")
    activity = bd.get_node(database=DATABASE, code=ACTIVITY_CODE)
    observed = (
        activity.get("name"), activity.get("reference product"),
        activity.get("location"), activity.get("unit"),
    )
    expected = (
        "Electricity, low voltage, at grid", "Electricity, low voltage, at grid",
        "CH", "kilowatt hour",
    )
    if observed != expected:
        raise ValueError(f"Unexpected functional unit: {observed}")
    if METHOD not in bd.methods:
        raise ValueError(f"Missing method: {METHOD}")
    BW_READY = True
except Exception as error:
    BW_ERROR = f"{type(error).__name__}: {error}"

{
    "ready for live BAFU workers": BW_READY and LIVE,
    "project": BW_PROJECT,
    "database": DATABASE,
    "activity": ACTIVITY_CODE,
    "method": METHOD,
    "preflight error": BW_ERROR,
}
"""),
    markdown(r"""
> **Scope and local workaround—recheck before Brightcon:** in our tests this affects stochastic `JacobiGMRESLCA` only. Its CSC preparation detaches the active technosphere matrix from the stochastic matrix manager, so later Monte Carlo iterations would reuse the first matrix. Direct `LCA` does not take this route, and deterministic Jacobi does not advance the matrix. The fix is **not** in the installed `bw2calc 2.5.0`; this repository adds an `after_matrix_iteration` rebind in the benchmark worker. Remove the local guard once the behavior is corrected upstream.
"""),
    code(r"""
def run_bafu_worker(solver: str, iterations: int, stochastic: bool, output: Path):
    command = [
        sys.executable,
        "dev/benchmark_bafu.py",
        "--solver", solver,
        "--project", BW_PROJECT,
        "--iterations", str(iterations),
        "--output", str(output),
        "--quiet",
    ]
    if solver == "jacobi-gmres":
        command.extend(["--rtol", str(RTOL), "--use-guess"])
    if not stochastic:
        command.append("--no-stochastic")
    run_checked(command, timeout=45)
    return json.loads(output.read_text())


def deterministic_runs():
    fallback_direct = ROOT / "results" / "bafu_direct_deterministic.json"
    fallback_iterative = ROOT / "results" / "bafu_jacobi_deterministic.json"
    if LIVE and BW_READY:
        try:
            direct = run_bafu_worker("direct", 1, False, ROOT / "results" / "live_bafu_direct_deterministic.json")
            iterative = run_bafu_worker("jacobi-gmres", 1, False, ROOT / "results" / "live_bafu_jacobi_deterministic.json")
            return direct, iterative, "LIVE"
        except Exception as error:
            source = f"STORED FALLBACK ({type(error).__name__})"
    else:
        source = "STORED RESULTS"
    return json.loads(fallback_direct.read_text()), json.loads(fallback_iterative.read_text()), source


det_direct, det_iterative, deterministic_source = deterministic_runs()
display(Markdown(f"**Result source: {deterministic_source}**"))
pd.DataFrame([
    {
        "solver": "UMFPACK",
        "seconds": det_direct["calculation_seconds"],
        "incremental peak MiB": det_direct["incremental_peak_rss_bytes"] / 2**20,
        "score": det_direct["records"][0]["score"],
        "relative residual": det_direct["records"][0]["relative_residual"],
    },
    {
        "solver": "Jacobi + GMRES",
        "seconds": det_iterative["calculation_seconds"],
        "incremental peak MiB": det_iterative["incremental_peak_rss_bytes"] / 2**20,
        "score": det_iterative["records"][0]["score"],
        "relative residual": det_iterative["records"][0]["relative_residual"],
    },
]).style.format({"seconds": "{:.3f}", "incremental peak MiB": "{:.1f}", "score": "{:.8f}", "relative residual": "{:.2e}"})
"""),
    markdown(r"""
# 4 · 500 paired BAFU Monte Carlo samples at `rtol=1e-4`

- Functional unit: **1 kWh Swiss low-voltage electricity at grid**.
- Impact: **IPCC 2021 GWP100**.
- Both solvers receive the same seed and identical sampled technosphere and biosphere matrices.
- Each pair is accepted only if both matrix fingerprints match.
- Jacobi+GMRES uses `rtol=1e-4` and `use_guess=True`, reusing the preceding supply array as its initial guess.
"""),
    code(r"""
from dev.compare_bafu_runs import summarize


def monte_carlo_runs():
    fallback_direct = ROOT / "results" / "bafu_direct_500.json"
    fallback_iterative = ROOT / "results" / "bafu_jacobi_500.json"
    if LIVE and BW_READY:
        try:
            direct = run_bafu_worker("direct", 500, True, ROOT / "results" / "live_bafu_direct_500.json")
            iterative = run_bafu_worker("jacobi-gmres", 500, True, ROOT / "results" / "live_bafu_jacobi_500.json")
            return direct, iterative, "LIVE"
        except Exception as error:
            source = f"STORED FALLBACK ({type(error).__name__})"
    else:
        source = "STORED RESULTS"
    return json.loads(fallback_direct.read_text()), json.loads(fallback_iterative.read_text()), source


mc_direct, mc_iterative, mc_source = monte_carlo_runs()
mc_summary = summarize(mc_direct, mc_iterative)  # Raises if any paired fingerprint differs.
display(Markdown(f"**Result source: {mc_source} · paired fingerprints match: YES**"))

pd.DataFrame([
    {
        "solver": "UMFPACK",
        "500 samples [s]": mc_summary["direct"]["calculation_seconds"],
        "median iteration [ms]": mc_summary["direct"]["median_iteration_seconds_excluding_first"] * 1000,
        "incremental peak [MiB]": mc_summary["direct"]["incremental_peak_rss_bytes"] / 2**20,
        "median GMRES iterations": np.nan,
        "warm start": "—",
    },
    {
        "solver": "Jacobi + GMRES",
        "500 samples [s]": mc_summary["jacobi_gmres"]["calculation_seconds"],
        "median iteration [ms]": mc_summary["jacobi_gmres"]["median_iteration_seconds_excluding_first"] * 1000,
        "incremental peak [MiB]": mc_summary["jacobi_gmres"]["incremental_peak_rss_bytes"] / 2**20,
        "median GMRES iterations": mc_summary["jacobi_gmres"]["median_gmres_iterations"],
        "warm start": "yes",
    },
]).style.format({"500 samples [s]": "{:.2f}", "median iteration [ms]": "{:.1f}", "incremental peak [MiB]": "{:.1f}", "median GMRES iterations": "{:.0f}"})
"""),
    code(r"""
direct_scores = np.array([record["score"] for record in mc_direct["records"]])
iterative_scores = np.array([record["score"] for record in mc_iterative["records"]])

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
labels = ["UMFPACK", "Jacobi + GMRES"]
axes[0, 0].bar(labels, [mc_summary["direct"]["calculation_seconds"], mc_summary["jacobi_gmres"]["calculation_seconds"]], color=["#DD8452", "#12A594"])
axes[0, 0].set_ylabel("500 samples [s]")
axes[0, 0].set_title("Audited tolerance changes the winner")

axes[0, 1].bar(labels, [mc_summary["direct"]["incremental_peak_rss_bytes"] / 2**20, mc_summary["jacobi_gmres"]["incremental_peak_rss_bytes"] / 2**20], color=["#DD8452", "#12A594"])
axes[0, 1].set_ylabel("incremental peak RSS [MiB]")
axes[0, 1].set_title("Peak memory is similar at this scale")

limits = [min(direct_scores.min(), iterative_scores.min()), max(direct_scores.max(), iterative_scores.max())]
axes[1, 0].scatter(direct_scores, iterative_scores, s=18, alpha=0.7, color="#315B7D")
axes[1, 0].plot(limits, limits, "--", color="black", linewidth=1)
axes[1, 0].set_xlabel("UMFPACK score")
axes[1, 0].set_ylabel("Jacobi + GMRES score")
axes[1, 0].set_title(f"max relative difference: {mc_summary['score_agreement']['maximum_relative_difference']:.1e}")

for values, label, color in ((direct_scores, "UMFPACK", "#DD8452"), (iterative_scores, "Jacobi + GMRES", "#12A594")):
    ordered = np.sort(values)
    axes[1, 1].step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post", label=label, color=color)
axes[1, 1].set_xlabel("GWP100 [kg CO$_2$-eq / kWh]")
axes[1, 1].set_ylabel("empirical cumulative probability")
axes[1, 1].set_title("Score distributions overlap")
axes[1, 1].legend(frameon=False)

for axis in axes.ravel():
    axis.grid(alpha=0.2)
fig.tight_layout()
plt.show()
"""),
    markdown(r"""
## Dataset-wide fixed-matrix check: first 100 BAFU activities

This is separate from Monte Carlo: the technosphere, biosphere, and characterization matrices are built once, then each of the first 100 BAFU activities is scored as a one-unit demand. UMFPACK factorizes once on the first demand and reuses that factorization; Jacobi+GMRES uses `rtol=1e-4` and `use_guess=True`.
"""),
    code(r"""
from dev.compare_bafu_all_activities import summarize as summarize_all_activities


def all_activity_runs():
    fallback_direct = ROOT / "results" / "bafu_all_direct_100.json"
    fallback_iterative = ROOT / "results" / "bafu_all_jacobi_100.json"
    if LIVE and BW_READY:
        try:
            direct_command = [
                sys.executable, "dev/benchmark_bafu_all_activities.py",
                "--solver", "direct", "--project", BW_PROJECT, "--limit", "100",
                "--output", str(ROOT / "results" / "live_bafu_all_direct_100.json"),
            ]
            iterative_command = [
                sys.executable, "dev/benchmark_bafu_all_activities.py",
                "--solver", "jacobi-gmres", "--project", BW_PROJECT, "--limit", "100",
                "--rtol", str(RTOL), "--use-guess",
                "--output", str(ROOT / "results" / "live_bafu_all_jacobi_100.json"),
            ]
            run_checked(direct_command, timeout=240)
            run_checked(iterative_command, timeout=240)
            return (
                json.loads((ROOT / "results" / "live_bafu_all_direct_100.json").read_text()),
                json.loads((ROOT / "results" / "live_bafu_all_jacobi_100.json").read_text()),
                "LIVE",
            )
        except Exception as error:
            source = f"STORED FALLBACK ({type(error).__name__})"
    else:
        source = "STORED RESULTS"
    return json.loads(fallback_direct.read_text()), json.loads(fallback_iterative.read_text()), source


all_direct, all_iterative, all_source = all_activity_runs()
all_summary = summarize_all_activities(all_direct, all_iterative)
display(Markdown(f"**Result source: {all_source} · same activity order: YES**"))
pd.DataFrame([
    {
        "solver": "UMFPACK",
        "100 dataset scores [s]": all_summary["direct"]["calculation_seconds"],
        "per dataset [ms]": all_summary["direct"]["seconds_per_activity"] * 1000,
        "incremental peak [MiB]": all_summary["direct"]["incremental_peak_rss_bytes"] / 2**20,
        "factorized once": "yes",
    },
    {
        "solver": "Jacobi + GMRES",
        "100 dataset scores [s]": all_summary["jacobi_gmres"]["calculation_seconds"],
        "per dataset [ms]": all_summary["jacobi_gmres"]["seconds_per_activity"] * 1000,
        "incremental peak [MiB]": all_summary["jacobi_gmres"]["incremental_peak_rss_bytes"] / 2**20,
        "factorized once": "no",
    },
]).style.format({"100 dataset scores [s]": "{:.2f}", "per dataset [ms]": "{:.2f}", "incremental peak [MiB]": "{:.1f}"})

direct_activity_scores = np.array([record["score"] for record in all_direct["records"]])
iterative_activity_scores = np.array([record["score"] for record in all_iterative["records"]])
relative_activity_difference = np.abs(iterative_activity_scores - direct_activity_scores) / np.maximum(np.abs(direct_activity_scores), np.finfo(float).tiny)
fig, axis = plt.subplots(figsize=(8.5, 3.5))
axis.plot(np.arange(1, len(relative_activity_difference) + 1), relative_activity_difference, ".", color="#315B7D")
axis.axhline(1e-4, linestyle="--", color="#DD8452", linewidth=1, label="0.01% reference")
axis.set_yscale("log")
axis.set_xlabel("BAFU activity rank by database ID")
axis.set_ylabel("relative score difference [log]")
axis.set_title(f"Dataset-wide score agreement; maximum = {relative_activity_difference.max():.1e}")
axis.grid(alpha=0.2, which="both")
axis.legend(frameon=False)
fig.tight_layout()
plt.show()
"""),
    markdown(r"""
# Takeaways

1. **The crossover is structural.** As random cross-links and density generate fill-in, direct runtime and RAM rise sharply.
2. **A fixed matrix favors reuse.** Direct factorization amortizes across many right-hand sides.
3. **Changing matrices favor factorization-free iteration.** Jacobi+GMRES avoids paying a new LU factorization each sample.
4. **Approximation must be audited.** Report `rtol`, GMRES status, the measured $Ax-b$ residual, and solution agreement.
5. **Large systems change the practical limit.** Jacobi+GMRES can remain feasible after direct factorization runs out of memory.

> Use direct solving by default; switch when measured factorisation cost—not fashion—justifies it.
"""),
    markdown(r"""
## Presenter preflight—not part of the timed talk

- Run once on the conference VM with `BRIGHTCON_LIVE=1`.
- Confirm the direct backend is UMFPACK or explicitly relabel it if Pardiso is used.
- Keep the committed results as a fallback, but never present them as live output.
"""),
]


# Remove the Brightway/BAFU presentation cells from the main notebook. The BAFU workers remain
# available as optional supporting material, but the talk itself focuses on synthetic technosphere
# systems and solver behavior.
start = next((i for i, cell in enumerate(cells) if cell.cell_type == "markdown" and cell.source.startswith("# 3 ·")), None)
end = next((i for i, cell in enumerate(cells) if cell.cell_type == "markdown" and cell.source.startswith("# Takeaways")), None)
if start is not None and end is not None:
    cells[start:end] = [
        markdown("""
# 3 · Fixed matrix, many right-hand sides

One fixed technosphere matrix can serve many demands. UMFPACK factorizes once and reuses its factors;
Jacobi + GMRES solves each right-hand side iteratively.
"""),
        code("""
rhs_output = ROOT / "results" / "live_synthetic_rhs_sweep.json"
rhs_command = [sys.executable, "dev/run_synthetic_suite.py", "--python", sys.executable,
    "--output", str(rhs_output), "--sizes", "5000", "--solvers", *DIRECT_SOLVERS, "jacobi-gmres",
    "--topology", "constant-degree", "--degree", "8", "--rhs-counts", "1", "10", "100", "1000",
    "--rtol", str(RTOL), "--run-timeout", "120"]
run_checked(rhs_command, timeout=600)
rhs = pd.json_normalize(json.loads(rhs_output.read_text())["results"])
display(rhs[["solver", "rhs_count", "factorization_seconds", "rhs_solve_seconds", "solve_seconds", "relative_residual"]])
fig, axis = plt.subplots(figsize=(8.5, 4))
for solver, subset in rhs.groupby("solver"):
    axis.plot(subset["rhs_count"], subset["solve_seconds"], "o-", label=solver, color=colors[solver])
axis.set_xscale("log"); axis.set_yscale("log")
axis.set_xlabel("right-hand sides on one fixed matrix"); axis.set_ylabel("total runtime [s, log]")
axis.set_title("Factorization amortization creates the crossover"); axis.grid(alpha=0.25, which="both")
axis.legend(frameon=False); fig.tight_layout(); plt.show()
"""),
        markdown("""
# 4 · Changing matrix, repeated solves

This synthetic Monte Carlo analogue rebuilds the technosphere matrix for every sample. Both solvers
receive the same paired matrices and demand; residuals and convergence are recorded.
"""),
        code("""
from dev.benchmark_synthetic import constant_degree_matrix, solve
demand = np.zeros(5000); demand[0] = 1.0
records = []
for solver in (*DIRECT_SOLVERS, "jacobi-gmres"):
    started = perf_counter(); residuals = []; iterations = []
    for sample in range(20):
        matrix = constant_degree_matrix(5000, 8, SEED + sample, 4.0)
        solve_started = perf_counter()
        solution, info, count = solve(matrix, demand, solver, RTOL, 50, 300)
        elapsed = perf_counter() - solve_started
        residual = float(np.linalg.norm(matrix @ solution - demand) / np.linalg.norm(demand))
        residuals.append(residual); iterations.append(count)
        records.append({"solver": solver, "sample": sample, "solve_seconds": elapsed, "iterations": count, "relative_residual": residual, "info": info})
    records.append({"solver": solver, "sample": "TOTAL", "solve_seconds": perf_counter() - started, "iterations": int(np.median(iterations)), "relative_residual": max(residuals), "info": None})
changing = pd.DataFrame(records)
display(changing[changing["sample"] == "TOTAL"])
fig, axis = plt.subplots(figsize=(8.5, 4))
for solver, subset in changing[changing["sample"] != "TOTAL"].groupby("solver"):
    axis.plot(subset["sample"], subset["solve_seconds"], "o-", label=solver, color=colors[solver])
axis.set_xlabel("sample; matrix rebuilt each time"); axis.set_ylabel("solve runtime [s]")
axis.set_title("Repeated factorization versus factorization-free iteration"); axis.grid(alpha=0.25)
axis.legend(frameon=False); fig.tight_layout(); plt.show()
"""),
        markdown("""
# 5 · JacobiGMRESLCA in bw2calc

In Brightway, `bw2calc.JacobiGMRESLCA` changes the technosphere solve while leaving biosphere and
characterization processing unchanged. Main arguments are `demand`, `method`, `rtol`, `use_guess`,
`restart`, and `maxiter`.

```python
from bw2calc import JacobiGMRESLCA
lca = JacobiGMRESLCA({activity.id: 1.0}, method, rtol=1e-4, use_guess=True)
lca.lci()
lca.lcia()
print(lca.score)
```

Check convergence, residual, and agreement before interpreting an iterative score.
"""),
        markdown("""
# When to use Jacobi + GMRES

Use it when the technosphere is very large and sparse, the matrix changes repeatedly, or direct
factorization approaches the machine memory limit and a controlled tolerance is acceptable.

Prefer direct UMFPACK or Pardiso when the matrix is small or moderate, many demands reuse one fixed
matrix, exact/direct solves are required, or factorization fits comfortably in memory and can be
amortized. Decide from measured factorization cost, memory, convergence, and agreement—not runtime
alone.
"""),
    ]

for index, cell in enumerate(cells, start=1):
    cell["id"] = f"brightcon-{index:02d}"

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")

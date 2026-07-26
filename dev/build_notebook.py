"""Build the Brightcon presentation notebook with deterministic cell IDs."""

from __future__ import annotations

from pathlib import Path
import ast

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "jupyter-notebook" / "brightcon-2026-jacobi-gmres.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def hidden_code(source: str):
    cell = code(source)
    cell.metadata["jupyter"] = {"source_hidden": True}
    cell.metadata["tags"] = ["hide-input"]
    return cell


embedded_worker_code = Path(__file__).with_name("benchmark_synthetic.py").read_text()
# The notebook needs the worker's reusable functions, not its command-line entry point.
_tree = ast.parse(embedded_worker_code)
_wanted = {"constant_degree_matrix", "banded_matrix", "solve"}
_parts = [
    {
        "constant_degree_matrix": """# Build a matrix with a fixed number of links per activity.
""",
        "banded_matrix": """# Build a locally connected matrix with low direct-solver fill-in.
""",
        "solve": """# Solve with a direct method or with GMRES, optionally scaled by the diagonal.
""",
    }.get(node.name, "") + ast.get_source_segment(embedded_worker_code, node)
    for node in _tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _wanted
]
embedded_benchmark_code = """# Notebook helpers
# ----------------
# These two functions create a sparse technosphere-like matrix and solve Ax = b.
# The benchmark cells below use them directly; the larger worker code is kept private.

import time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, spsolve
""" + "\n\n".join(_parts) + "\n"
embedded_suite_code = Path(__file__).with_name("run_synthetic_suite.py").read_text()
embedded_suite_code = embedded_suite_code.replace('Path("dev/benchmark_synthetic.py")', 'Path("benchmark_synthetic.py")')

cells = [
    hidden_code(embedded_benchmark_code + "\n" + """
import tempfile
import json
import subprocess
import sys

_embedded_dir = Path(tempfile.mkdtemp(prefix="jacobi_notebook_"))
_embedded_worker = _embedded_dir / "benchmark_synthetic.py"
_embedded_suite = _embedded_dir / "run_synthetic_suite.py"
_embedded_worker.write_text(EMBEDDED_WORKER_SOURCE)
_embedded_suite.write_text(EMBEDDED_SUITE_SOURCE)


def run_benchmark(
    name,
    *,
    sizes,
    solvers,
    topology="constant-degree",
    degree=8,
    degrees=None,
    densities=None,
    rhs_counts=None,
    matrix_family="lca-random",
    blocks=8,
    rtol=1e-4,
    worker_timeout=None,
    total_budget=None,
    memory_guard_mib=None,
    suite_timeout=None,
):
    # Run an isolated benchmark grid and return its records.
    output = _embedded_dir / f"{name}.json"
    command = [
        sys.executable,
        str(_embedded_suite),
        "--python", sys.executable,
        "--worker", str(_embedded_worker),
        "--output", str(output),
        "--sizes", *map(str, sizes),
        "--solvers", *solvers,
        "--topology", topology,
        "--degree", str(degree),
        "--matrix-family", matrix_family,
        "--blocks", str(blocks),
        "--rtol", str(rtol),
    ]
    optional_lists = {
        "--degrees": degrees,
        "--densities": densities,
        "--rhs-counts": rhs_counts,
    }
    for flag, values in optional_lists.items():
        if values:
            command.extend([flag, *map(str, values)])
    optional_values = {
        "--run-timeout": worker_timeout,
        "--total-budget": total_budget,
        "--max-estimated-construction-mib": memory_guard_mib,
    }
    for flag, value in optional_values.items():
        if value is not None:
            command.extend([flag, str(value)])
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=suite_timeout)
    return json.loads(output.read_text())
""".replace("EMBEDDED_WORKER_SOURCE", repr(embedded_worker_code)).replace("EMBEDDED_SUITE_SOURCE", repr(embedded_suite_code))),
    markdown(r"""
# From 8 minutes to 4 seconds
## Solving large systems with Jacobi + GMRES

**Romain Sacchi (PSI) · Brightcon 2026 · 25 September 2026 · 8 minutes**  
Open Tools and Development · Aalborg University & online

> Direct sparse factorisation is an excellent default—until fill-in makes memory and runtime scale much faster than the matrix itself. An iterative solve makes that trade-off explicit and controllable.
"""),
    markdown(r"""
## Run of show

1. Show how size and density create direct-solver fill-in.
2. Use a large banded counterexample to show that size alone is not decisive.
3. Reuse one fixed matrix and watch factorization reuse reverse the winner.
4. Rebuild paired matrices across 500 Monte Carlo iterations, with and without warm starts.
5. Translate the evidence back to `bw2calc`.

Every benchmark runs live in an isolated worker. Unsafe cases are marked **SKIPPED**, and failures
or timeouts remain visible instead of silently disappearing.
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
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import psutil
import scipy
from IPython.display import Markdown, display
from scipy.sparse.linalg import LinearOperator, gmres, spsolve


ROOT = Path.cwd()
(ROOT / "results").mkdir(exist_ok=True)

SEED = 2026
RTOL = 1e-4
available_mib = psutil.virtual_memory().available / 2**20
memory_guard_mib = int(min(8_192, max(512, available_mib * 0.25)))
try:
    import pypardiso  # noqa: F401
    PARDISO_AVAILABLE = True
except ImportError:
    PARDISO_AVAILABLE = False
DIRECT_SOLVERS = ["umfpack"] + (["pardiso"] if PARDISO_AVAILABLE else [])
colors = {
    "numpy-dense": "#7F8C8D", "superlu": "#C44E52", "umfpack": "#DD8452",
    "pardiso": "#937860", "gmres": "#4C72B0", "jacobi-gmres": "#12A594",
    "jacobi-gmres-no-guess": "#8172B2",
}
solver_names = {
    "numpy-dense": "NumPy dense",
    "superlu": "SuperLU",
    "umfpack": "UMFPACK",
    "pardiso": "Pardiso",
    "gmres": "GMRES",
    "jacobi-gmres": "Jacobi + GMRES",
    "jacobi-gmres-no-guess": "Jacobi + GMRES (no warm start)",
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "figure.dpi": 120,
})


{
    "benchmark mode": "live, self-contained notebook",
    "worker Python": sys.executable,
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

The first matrix is deliberately ill-scaled. This makes the benefit of simple diagonal scaling visible without adding other LCA matrices.
"""),
    code(r"""
A_tiny = constant_degree_matrix(n=12, degree=2, seed=SEED, diagonal_span=4.0)
b_tiny = np.zeros(12)
b_tiny[0] = 1.0

fig, axis = plt.subplots(figsize=(5, 4))
axis.spy(A_tiny, markersize=7, color="#315B7D")
axis.set_title(f"Sparse technosphere structure ({A_tiny.nnz} nonzeros)")
axis.set_xlabel("activity"); axis.set_ylabel("product")
fig.tight_layout()
plt.show()
"""),
    markdown(r"""
## Five approaches, one answer

- **NumPy dense** stores and solves the complete matrix, including all its zeros.
- **SuperLU** is SciPy's general sparse direct solver.
- **UMFPACK** is a sparse direct solver designed to limit unnecessary work and memory.
- **GMRES** approaches the answer iteratively, stopping when the requested tolerance is reached.
- **Jacobi + GMRES** first rescales the equations using the matrix diagonal, which can help GMRES converge faster.
- **Pardiso**, when installed, appears as another high-performance sparse direct solver.

Runtime is only half the story: the residual confirms that each result satisfies $Ax=b$.
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

tiny_frame = pd.DataFrame(tiny_runs)
label_colors = {
    "NumPy dense": colors["numpy-dense"],
    "SuperLU": colors["superlu"],
    "UMFPACK": colors["umfpack"],
    "Pardiso": colors["pardiso"],
    "GMRES": colors["gmres"],
    "Jacobi + GMRES": colors["jacobi-gmres"],
}
bar_colors = [label_colors[name] for name in tiny_frame["solver"]]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(tiny_frame["solver"], tiny_frame["seconds"], color=bar_colors)
axes[1].bar(tiny_frame["solver"], tiny_frame["residual"], color=bar_colors)
axes[0].set_yscale("log"); axes[0].set_ylabel("runtime [s]")
axes[1].set_yscale("log"); axes[1].set_ylabel(r"relative residual $||Ax-b||/||b||$")
axes[0].set_title("Runtime on a tiny system")
axes[1].set_title("All methods satisfy the equation")
for axis in axes:
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25, which="both")
fig.tight_layout(); plt.show()
"""),
    markdown(r"""
# 2 · Scale the matrix, isolate every solver

Each solver runs in a fresh subprocess. This prevents one factorisation from contaminating the next solver's peak memory. A 2 ms RSS sampler captures allocations made by NumPy, SciPy, UMFPACK, and BLAS—not only Python objects. Incremental RSS is measured above the post-matrix-construction baseline.

Main series: approximately eight inputs per activity. Dense solving stops at 2,500 activities; the remaining isolated workers run to completion without a benchmark timeout.
"""),
    code(r"""
synthetic_payload = run_benchmark(
    "scaling",
    sizes=[500, 1_000, 2_500, 5_000, 7_500, 10_000, 20_000, 50_000],
    solvers=["numpy-dense", "superlu", *DIRECT_SOLVERS, "gmres", "jacobi-gmres"],
    rtol=RTOL,
    worker_timeout=120,
    total_budget=850,
    suite_timeout=900,
)
synthetic = pd.json_normalize(synthetic_payload["results"])
synthetic["memory_mib"] = synthetic["incremental_peak_rss_bytes"] / 2**20
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
    label = solver_names[solver]
    axes[0].plot(subset["size"], subset["solve_seconds"], "o-", label=label, color=colors[solver])
    axes[1].plot(subset["size"], subset["memory_mib"], "o-", label=label, color=colors[solver])

axes[0].set_yscale("log")
axes[0].set_ylabel("solve time [s]")
axes[1].set_yscale("log")
axes[1].set_ylabel("additional peak memory [MiB]")
for axis in axes:
    axis.set_xlabel("matrix rows / columns")
    axis.grid(alpha=0.25, which="both")
axes[0].set_title("Runtime separates as the system grows")
axes[1].set_title("Direct factorization can increase memory")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=[0, 0.12, 1, 1])
plt.show()
"""),
    markdown(r"""
### Density versus size

**Matrix density** is the share of matrix entries that are nonzero. This sweep varies both system
size and density from 0.1% to 10%. The later stress test extends the same comparison to 15% and
shows which combinations exceed the memory guard.

Each heatmap cell answers one question:
**how many times faster is one solver than the other?**
"""),
    code(r"""
density_size_payload = run_benchmark(
    "density versus size",
    sizes=[1_000, 2_500, 5_000, 7_500, 10_000],
    solvers=[*DIRECT_SOLVERS, "jacobi-gmres"],
    topology="fixed-density",
    densities=[0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1],
    rtol=RTOL,
    worker_timeout=90,
    total_budget=850,
    memory_guard_mib=memory_guard_mib,
    suite_timeout=900,
)
density_size = pd.json_normalize(density_size_payload["results"])
density_size["speedup"] = np.nan
for (size, density), subset in density_size.groupby(["size", "target_density"]):
    direct = subset.loc[subset.solver == "umfpack", "solve_seconds"]
    iterative = subset.loc[subset.solver == "jacobi-gmres", "solve_seconds"]
    if len(direct) == len(iterative) == 1:
        density_size.loc[subset.index, "speedup"] = direct.iloc[0] / iterative.iloc[0]

heat = density_size[density_size.solver == "jacobi-gmres"].pivot(
    index="target_density", columns="size", values="speedup"
)
fig, axis = plt.subplots(figsize=(8.5, 4.5))
image = axis.imshow(
    np.log10(heat), aspect="auto", origin="lower",
    cmap="PiYG", vmin=-2, vmax=2,
)
axis.set_xticks(range(len(heat.columns)), [f"{n:,}" for n in heat.columns])
axis.set_yticks(range(len(heat.index)), [f"{density:.1%}" for density in heat.index])
axis.set_xlabel("matrix size"); axis.set_ylabel("matrix density")
axis.set_title("UMFPACK time ÷ Jacobi + GMRES time (>1× favors Jacobi)")
for row, density in enumerate(heat.index):
    for column, size in enumerate(heat.columns):
        value = heat.loc[density, size]
        if np.isfinite(value):
            label = f"{value:.0f}×" if value >= 1 else "<1×"
            axis.text(
                column, row, label,
                ha="center", va="center", fontsize=8,
                color="white", fontweight="bold",
            )
bar = fig.colorbar(image, ax=axis, label="time ratio (log colour scale)")
bar.set_ticks([-2, -1, 0, 1, 2]); bar.set_ticklabels(["0.01×", "0.1×", "1×", "10×", "100×"])
fig.tight_layout(); plt.show()
"""),
    markdown(r"""
### Large synthetic systems

Large size alone does not guarantee that an iterative solver wins. These banded matrices keep LU
fill-in low while scaling from 50,000 to 300,000 rows. This provides a completed 50,000-row UMFPACK
reference and acts as a counterexample to the random high-fill matrices above.
"""),
    code(r"""
large_payload = run_benchmark(
    "large systems",
    sizes=[50_000, 100_000, 200_000, 300_000],
    solvers=[*DIRECT_SOLVERS, "jacobi-gmres"],
    topology="banded",
    degree=8,
    rtol=RTOL,
    worker_timeout=120,
    total_budget=900,
    memory_guard_mib=memory_guard_mib,
    suite_timeout=960,
)
large_results = pd.json_normalize(large_payload["results"])
large_results["memory_mib"] = large_results["incremental_peak_rss_bytes"] / 2**20
completed_large = large_results[large_results.status == "COMPLETED"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for solver, subset in completed_large.groupby("solver"):
    subset = subset.sort_values("size")
    label = solver_names[solver]
    axes[0].plot(subset["size"], subset["solve_seconds"], "o-", label=label, color=colors[solver])
    axes[1].plot(subset["size"], subset["memory_mib"], "o-", label=label, color=colors[solver])
for axis in axes:
    axis.set_xlabel("matrix size"); axis.grid(alpha=0.25)
    axis.ticklabel_format(style="plain", axis="both")
axes[0].set_ylabel("solve time [s]"); axes[1].set_ylabel("additional peak memory [MiB]")
axes[0].set_title("Which solvers still finish?"); axes[1].set_title("What does completion cost in memory?")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=[0, 0.1, 1, 1]); plt.show()

not_completed = large_results[large_results.status != "COMPLETED"]
if not not_completed.empty:
    display(not_completed[["solver", "size", "status"]].style.hide(axis="index"))
"""),
    markdown(r"""
## Guarded size–density stress test

With constant density, nonzeros grow with $n^2$, not $n$. The grid now includes intermediate
densities so the crossover is visible rather than implied by two distant points. A memory guard
marks unsafe combinations **SKIPPED** before allocating them.
"""),
    code(r"""
density_payload = run_benchmark(
    "density grid",
    sizes=[1_000, 2_500, 5_000, 10_000, 20_000],
    solvers=[*DIRECT_SOLVERS, "jacobi-gmres"],
    topology="fixed-density",
    densities=[0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15],
    rtol=RTOL,
    worker_timeout=90,
    total_budget=900,
    memory_guard_mib=memory_guard_mib,
    suite_timeout=960,
)
density_results = pd.json_normalize(density_payload["results"])
density_results["incremental_peak_MiB"] = density_results["incremental_peak_rss_bytes"] / 2**20
density_results["runtime_status"] = density_results.get("status", "COMPLETED")

completed = density_results[density_results.runtime_status == "COMPLETED"].copy()
completed["log10_speedup_umfpack_over_jacobi"] = np.nan
for (size, density), subset in completed.groupby(["size", "target_density"]):
    direct = subset.loc[subset.solver == "umfpack", "solve_seconds"]
    iterative = subset.loc[subset.solver == "jacobi-gmres", "solve_seconds"]
    if len(direct) == 1 and len(iterative) == 1:
        value = np.log10(direct.iloc[0] / iterative.iloc[0])
        completed.loc[subset.index, "log10_speedup_umfpack_over_jacobi"] = value

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
density_coverage = completed.groupby("size")["target_density"].nunique()
plot_size = density_coverage.idxmax()
runtime_slice = completed[completed["size"] == plot_size]
for solver, subset in runtime_slice.groupby("solver"):
    subset = subset.sort_values("density")
    axes[0].plot(subset["density"], subset["solve_seconds"], "o-", label=solver_names[solver], color=colors[solver])
axes[0].set_xlabel("realized matrix density")
axes[0].set_ylabel("solve time [s]")
axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
axes[0].set_title(f"Density effect at n = {plot_size:,}")
axes[0].grid(alpha=0.25, which="both")
axes[0].legend(frameon=False, fontsize=8)

pivot = completed[completed.solver == "jacobi-gmres"].pivot_table(index="size", columns="target_density", values="log10_speedup_umfpack_over_jacobi")
image = axes[1].imshow(pivot, aspect="auto", cmap="PiYG", vmin=-2, vmax=2)
axes[1].set_xticks(range(len(pivot.columns)), [f"{value:.1%}" for value in pivot.columns], rotation=45, ha="right")
axes[1].set_yticks(range(len(pivot.index)), pivot.index)
axes[1].set_xlabel("target density")
axes[1].set_ylabel("matrix size")
axes[1].set_title("Where does Jacobi + GMRES become faster? (>1×)")
for row, size in enumerate(pivot.index):
    for column, density in enumerate(pivot.columns):
        value = pivot.loc[size, density]
        if np.isfinite(value):
            ratio = 10 ** value
            label = f"{ratio:.0f}×" if ratio >= 1 else "<1×"
            axes[1].text(
                column, row, label,
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold",
            )
bar = fig.colorbar(image, ax=axes[1], label="UMFPACK time ÷ Jacobi time")
bar.set_ticks([-2, -1, 0, 1, 2]); bar.set_ticklabels(["0.01×", "0.1×", "1×", "10×", "100×"])
fig.tight_layout()
plt.show()
"""),
    markdown(r"""
### Matrix structure sensitivity

Two matrices can have the same size and density but very different patterns. This comparison asks
whether block structure changes the UMFPACK/Jacobi crossover.
"""),
    code(r"""
ioblock_payload = run_benchmark(
    "block structure",
    sizes=[1_000, 2_500, 5_000, 7_500, 10_000],
    solvers=[*DIRECT_SOLVERS, "jacobi-gmres"],
    topology="fixed-density",
    densities=[0.01, 0.03, 0.05, 0.1, 0.15],
    matrix_family="io-block",
    blocks=8,
    rtol=RTOL,
    worker_timeout=90,
    total_budget=600,
    memory_guard_mib=memory_guard_mib,
    suite_timeout=660,
)
ioblock = pd.json_normalize(ioblock_payload["results"])
ioblock = ioblock[ioblock.status == "COMPLETED"].copy()
ioblock["family"] = "block structured"

random_structure = completed[
    completed["target_density"].isin([0.01, 0.03, 0.05, 0.1, 0.15])
    & completed["size"].isin([1000, 2500, 5000, 7500, 10000])
].copy()
random_structure["family"] = "random links"
plot_data = pd.concat([random_structure, ioblock], ignore_index=True)

ratios = []
for (family, size, density), subset in plot_data.groupby(["family", "size", "target_density"]):
    direct = subset.loc[subset.solver == "umfpack", "solve_seconds"]
    iterative = subset.loc[subset.solver == "jacobi-gmres", "solve_seconds"]
    if len(direct) == len(iterative) == 1:
        ratios.append({"family": family, "size": size, "density": density, "speedup": direct.iloc[0] / iterative.iloc[0]})

structure_ratio = pd.DataFrame(ratios)
fig, axis = plt.subplots(figsize=(8, 4))
family_colors = {"random links": "#4C72B0", "block structured": "#12A594"}
for family, subset in structure_ratio.groupby("family"):
    summary = subset.groupby("density")["speedup"].agg(["min", "median", "max"])
    axis.fill_between(summary.index, summary["min"], summary["max"], color=family_colors[family], alpha=0.15)
    axis.plot(summary.index, summary["median"], "o-", label=f"{family} (median)", color=family_colors[family])
axis.axhline(1, color="black", linewidth=1, linestyle="--", label="equal time")
axis.set_xscale("log"); axis.set_yscale("log")
axis.set_xlabel("target density"); axis.set_ylabel("UMFPACK time ÷ Jacobi time")
axis.set_title("Structure changes the result at the same density")
axis.grid(alpha=0.25, which="both"); axis.legend(frameon=False)
fig.tight_layout(); plt.show()
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

1. **Is the matrix fixed?** Reuse favors UMFPACK or Pardiso.
2. **Does the matrix change every iteration?** Avoiding repeated factorization can favor Jacobi + GMRES.
3. **Does factorization fit in memory?** Size, density, and structure all affect fill-in.
4. **Can you verify the approximation?** Always report tolerance, convergence, residual, and agreement.

The large banded counterexample shows why **structure matters**: UMFPACK remains practical at
50,000 rows when fill-in stays low, even though it struggles on smaller random high-fill matrices.

> The solver choice follows the workload—not matrix size alone.
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

One fixed 25,000 × 25,000 matrix can serve many demands. The left plot separates the one-time
factorization from the first solve. The right plot then shows total time as more demands reuse that
same matrix.
"""),
        code("""
rhs_payload = run_benchmark(
    "fixed matrix demands",
    sizes=[25_000],
    solvers=[*DIRECT_SOLVERS, "jacobi-gmres"],
    topology="banded",
    degree=50,
    rhs_counts=[1, 2, 3, 5, 10, 25, 50, 100, 250, 500],
    rtol=RTOL,
    worker_timeout=120,
    suite_timeout=600,
)
rhs = pd.json_normalize(rhs_payload["results"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

first_demand = rhs[
    (rhs.rhs_count == 1) & rhs.solver.isin(["umfpack", "jacobi-gmres"])
].set_index("solver")
bar_solvers = [solver for solver in ["umfpack", "jacobi-gmres"] if solver in first_demand.index]
x = np.arange(len(bar_solvers))
setup_time = [first_demand.loc[solver, "factorization_seconds"] for solver in bar_solvers]
solve_time = [first_demand.loc[solver, "rhs_solve_seconds"] for solver in bar_solvers]
axes[0].bar(x, setup_time, label="one-time factorization", color="#B0B0B0")
axes[0].bar(x, solve_time, bottom=setup_time, label="solve first demand", color=[colors[solver] for solver in bar_solvers])
axes[0].set_xticks(x, [solver_names[solver] for solver in bar_solvers])
axes[0].set_ylabel("time [s]")
axes[0].set_title("Cost of the first demand")
axes[0].legend(frameon=False, fontsize=8)

for solver, subset in rhs.groupby("solver"):
    axes[1].plot(
        subset["rhs_count"], subset["solve_seconds"], "o-",
        label=solver_names[solver], color=colors[solver],
    )

comparison = rhs[rhs.solver.isin(["umfpack", "jacobi-gmres"])].pivot(
    index="rhs_count", columns="solver", values="solve_seconds"
)
if {"umfpack", "jacobi-gmres"}.issubset(comparison.columns):
    direct_wins = comparison.index[comparison["umfpack"] < comparison["jacobi-gmres"]]
    if len(direct_wins):
        crossover = direct_wins.min()
        axes[1].axvline(crossover, color="black", linestyle="--", linewidth=1)
        axes[1].text(crossover, axes[1].get_ylim()[1], f"  direct faster from ~{crossover:,}", va="top", fontsize=8)

axes[1].set_xlabel("demands solved with one fixed matrix")
axes[1].set_ylabel("total solve time [s]")
axes[1].set_title("Factorization reuse changes the winner")
axes[1].legend(frameon=False, fontsize=8)
for axis in axes:
    axis.grid(alpha=0.25)
fig.tight_layout(); plt.show()
"""),
        markdown("""
# 4 · Changing matrix, repeated solves

This is a 500-iteration synthetic **Monte Carlo** analogue: the technosphere matrix changes slightly
at every sample. UMFPACK must refactorize each matrix. Jacobi + GMRES is shown both with a warm start
from the previous solution and without one, matching the role of `use_guess` in `bw2calc`.
"""),
        code("""
demand = np.zeros(5000)
demand[0] = 1.0
records = []
coupling_rng = np.random.default_rng(SEED)
sample_couplings = np.clip(0.8 + coupling_rng.normal(0, 0.015, 500), 0.7, 0.9)
solver_cases = [
    *((solver, solver, False) for solver in DIRECT_SOLVERS),
    ("jacobi-gmres", "jacobi-gmres", True),
    ("jacobi-gmres-no-guess", "jacobi-gmres", False),
]

for label, solver, reuse_guess in solver_cases:
    started = perf_counter()
    residuals = []
    iterations = []
    previous_solution = None
    for sample in range(500):
        matrix = banded_matrix(
            5_000, 4, SEED, 4.0,
            coupling=float(sample_couplings[sample]),
        )
        solve_started = perf_counter()
        solution, info, count = solve(
            matrix, demand, solver, RTOL, 50, 300,
            x0=previous_solution if reuse_guess else None,
        )
        elapsed = perf_counter() - solve_started
        residual = float(np.linalg.norm(matrix @ solution - demand) / np.linalg.norm(demand))
        if reuse_guess:
            previous_solution = solution
        residuals.append(residual)
        iterations.append(count)
        records.append({"solver": label, "sample": sample, "solve_seconds": elapsed, "iterations": count, "relative_residual": residual, "info": info})
    records.append({"solver": label, "sample": "TOTAL", "solve_seconds": perf_counter() - started, "iterations": int(np.median(iterations)), "relative_residual": max(residuals), "info": None})
changing = pd.DataFrame(records)
per_sample = changing[changing["sample"] != "TOTAL"].copy()
per_sample["cumulative_seconds"] = per_sample.groupby("solver")["solve_seconds"].cumsum()
per_sample["monte_carlo_iteration"] = per_sample["sample"].astype(int) + 1

fig, axis = plt.subplots(figsize=(8.5, 4.5))
for solver, subset in per_sample.groupby("solver"):
    label = "Jacobi + GMRES (warm start)" if solver == "jacobi-gmres" else solver_names[solver]
    axis.plot(
        subset["monte_carlo_iteration"],
        subset["cumulative_seconds"],
        "o-",
        label=label,
        color=colors[solver],
        markersize=3,
    )
    final = subset.iloc[-1]
    axis.annotate(
        f"{final['cumulative_seconds']:.2f} s",
        (final["monte_carlo_iteration"], final["cumulative_seconds"]),
        xytext=(5, 0), textcoords="offset points", va="center", fontsize=8,
    )
axis.set_xlabel("Monte Carlo iteration")
axis.set_ylabel("cumulative solve time [s]")
axis.set_title("Cumulative cost across changing matrices")
axis.set_xlim(1, per_sample["monte_carlo_iteration"].max() * 1.05)
axis.grid(alpha=0.25)
axis.legend(frameon=False)
fig.tight_layout(); plt.show()
"""),
        markdown("""
# 5 · JacobiGMRESLCA in bw2calc

In Brightway, `bw2calc.JacobiGMRESLCA` changes the technosphere solve while leaving biosphere and
characterization processing unchanged. Main arguments are `demand`, `method`, `rtol`, `use_guess`,
`restart`, and `maxiter`.

For repeated solves, `use_guess=True` starts from the previous supply array; `False` starts each
solve without that warm start.

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
# When to use **Jacobi + GMRES**

**Good fit when:**

- The system is **very large and sparse**.
- The matrix **changes repeatedly**.
- A direct factorization is close to the machine's **memory limit**.
- A **controlled tolerance** is acceptable.

**Choose UMFPACK or Pardiso when:**

- The matrix is small or moderate.
- Many demands reuse one fixed matrix.
- You need a direct answer and factorization fits comfortably in memory.

*Measure runtime, memory, convergence, and agreement before choosing.*
"""),
    ]

# Start directly with scaling; retain only the short solver explainer from the former Section 1.
section_one_start = next((
    index for index, cell in enumerate(cells)
    if cell.cell_type == "markdown" and cell.source.startswith("# 1 · Strip LCA")
), None)
scaling_start = next((
    index for index, cell in enumerate(cells)
    if cell.cell_type == "markdown" and cell.source.startswith("# 2 · Scale")
), None)
if section_one_start is not None and scaling_start is not None:
    cells[section_one_start:scaling_start] = [markdown("""
## Solver toolbox

- **NumPy dense** stores and solves the complete matrix, including all its zeros.
- **SuperLU** is SciPy's general sparse direct solver.
- **UMFPACK** is a sparse direct solver designed to limit unnecessary work and memory.
- **GMRES** approaches the answer iteratively, stopping when the requested tolerance is reached.
- **Jacobi + GMRES** rescales the equations using the diagonal before iterating.
- **Pardiso**, when installed, appears as another high-performance sparse direct solver.

Every comparison also checks convergence and the relative residual $||Ax-b||/||b||$.
""")]

for cell in cells:
    if cell.cell_type != "markdown":
        continue
    cell.source = cell.source.replace("# 2 · Scale the matrix", "# 1 · Scale the matrix")
    cell.source = cell.source.replace("# 3 · Fixed matrix", "# 2 · Fixed matrix")
    cell.source = cell.source.replace("# 4 · Changing matrix", "# 3 · Changing matrix")
    cell.source = cell.source.replace("# 5 · JacobiGMRESLCA", "# 4 · JacobiGMRESLCA")

# Keep the presentation focused on the audience rather than speaker setup.
cells = [cell for cell in cells if not (
    cell.cell_type == "markdown" and cell.source.startswith("## Presenter preflight")
)]

structure_start = next((
    index for index, cell in enumerate(cells)
    if cell.cell_type == "markdown" and cell.source.startswith("### Matrix structure sensitivity")
), None)
if structure_start is not None:
    del cells[structure_start:structure_start + 2]

# Show the random density stress test before the structured large-matrix counterexample.
large_start = next((
    index for index, cell in enumerate(cells)
    if cell.cell_type == "markdown" and cell.source.startswith("### Large synthetic systems")
), None)
if large_start is not None:
    large_cells = cells[large_start:large_start + 2]
    del cells[large_start:large_start + 2]
    density_start = next((
        index for index, cell in enumerate(cells)
        if cell.cell_type == "markdown" and cell.source.startswith("## Guarded size–density stress test")
    ), None)
    if density_start is not None:
        cells[density_start + 2:density_start + 2] = large_cells

# Put the title and storyline before the hidden benchmark machinery.
cells.insert(2, cells.pop(0))

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

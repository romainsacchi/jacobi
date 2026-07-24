# AGENTS.md

This is the handoff document for future Codex instances. Read it before changing code or the
presentation notebook. It records the intended scientific story, benchmark regimes, known
Brightway caveat, committed artifacts, and the current stopping point.

## Purpose

This repository supports an 8-minute Brightcon 2026 oral presentation:

> **From 8 minutes to 4 seconds: solving large systems with Jacobi+GMRES**

The audience knows LCA and Python but may not know sparse direct factorization or Krylov methods.
The notebook must be concise, visual, and reproducible on a conference VM.

The central message is not that Jacobi+GMRES is always faster:

1. For one large sparse right-hand side, direct factorization can dominate and an iterative solve
   can win.
2. For many demands on one fixed matrix, direct factorization can be amortized and UMFPACK can
   win.
3. In Monte Carlo, the technosphere changes each iteration, so direct factorization is paid
   repeatedly and Jacobi+GMRES can win again.
4. Iterative results require tolerance, residual, convergence status, and score agreement—not
   runtime alone.

## Repository map

- `output/jupyter-notebook/brightcon-2026-jacobi-gmres.ipynb`: generated presentation notebook.
- `dev/build_notebook.py`: source of truth for notebook cells, narrative, plots, and fallbacks.
- `dev/benchmark_synthetic.py`: one isolated bare-matrix benchmark.
- `dev/run_synthetic_suite.py`: isolated synthetic-worker orchestrator.
- `dev/benchmark_bafu.py`: direct/Jacobi BAFU Monte Carlo worker.
- `dev/compare_bafu_runs.py`: paired Monte Carlo fingerprint and score comparison.
- `dev/benchmark_bafu_all_activities.py`: fixed-matrix first-N-activity workload.
- `dev/compare_bafu_all_activities.py`: fixed-matrix score-vector comparison.
- `data/lci-bafu.xlsx`: supplied BAFU workbook; do not export proprietary inventory data.
- `results/`: committed calibration results and notebook fallbacks.
- `results/README.md`: result provenance.
- `pyproject.toml`: environment definition.

## Current state and commits

Latest relevant commits:

- `7b42b4a`: 500 paired BAFU Monte Carlo iterations.
- `664f9b2`: all-activity comparison with warm starts disabled.
- `e33a82d`: fixed-matrix first-100-activity comparison and notebook section.
- `056680e`: uncapped synthetic scaling, runtime line graph, and tolerance comparison.
- `3b06bcc`: earlier 200-sample expansion.

The worktree was clean after the latest commit. The live full-notebook rehearsal was deliberately
stopped on the slower development machine after the 500-sample update. The notebook was then
rendered successfully from stored results with zero cell errors. A faster conference machine
should be used for live rehearsal.

## Notebook narrative

The notebook proceeds top-to-bottom:

1. Bare `Ax=b`: dense NumPy, SuperLU, UMFPACK, GMRES, and Jacobi+GMRES.
2. Synthetic runtime/RSS/iteration/residual scaling at fixed connectivity.
3. Connectivity-versus-size, guarded density, and block-structured stress cases.
4. Fixed synthetic matrix with many right-hand sides.
5. Synthetic changing-matrix repeated solves.
7. Takeaways and presenter preflight.

The synthetic iterative configuration is fixed at `rtol=1e-4`, seed 2026, with paired matrices and
demands. The BAFU Monte Carlo configuration remains available in the worker scripts as optional
supporting material, but is no longer executed by the main presentation notebook.

## Benchmark regimes

### Synthetic one-shot benchmark

Each solver runs in a fresh subprocess. UMFPACK timing includes a new LU factorization for that
one right-hand side; Jacobi avoids factorization. This is why large synthetic cases can show a
dramatic Jacobi advantage.

Current fixed-density calibrations use `rtol=1e-4`:

- 5,000, 10,000, 15,000, and 20,000 rows at 0.1% density.
- 20,000 rows at 0.2% density.
- The 20,000-row cases took roughly 40 s and 54 s for UMFPACK respectively.

A future size/density grid must be safe. A 200,000 x 200,000 matrix at 10% density implies
about 4 billion nonzeros and can exhaust memory during construction before a solve timeout.
Add an estimated-nnz/memory guard and label cells `SKIPPED`, `TIMEOUT`, or completed.

The live notebook now separates two effects:

- fixed-connectivity scaling at 5 and 25 inputs per activity, across 1,000, 5,000, and 10,000
  rows;
- a large fixed-connectivity extension at 50,000, 100,000, and 300,000 rows, using only UMFPACK,
  GMRES, and Jacobi + GMRES with isolated 120-second workers and a 15-minute suite budget;
- a fixed-density grid at 0.1%, 0.3%, 1%, 3%, 5%, 10%, and 15% for bounded matrix sizes.

The density grid is run in isolated workers with a machine-aware estimated construction-memory cap
equal to 25% of currently available RAM, bounded between 512 MiB and 8 GiB. Each worker has a
45-second timeout and emits explicit `COMPLETED`/`SKIPPED`/`TIMEOUT`/`FAILED` statuses. This allows direct
factorization to run when the host has sufficient RAM while still preventing unbounded matrix
construction. A 20,000 × 20,000 matrix at 15% density would contain roughly 60 million nonzeros
before sparse construction and LU fill-in overhead, so completion still depends on actual memory
and fill-in behavior.

The notebook also runs a small `io-block` family at 1%, 5%, and 15% density. Density is not a
complete structural descriptor: block structure, diagonal dominance, coefficient scaling, and LU
fill-in can change solver behavior at the same nominal density.

Synthetic records include matrix construction time, estimated and actual storage, peak RSS, solver
time, factorization time, repeated-RHS time, LU fill ratio when available, iterations, residual,
convergence status, and environment metadata. The notebook visualizes runtime-versus-density and
`log10(UMFPACK time / Jacobi time)` heatmaps; positive heatmap values favor Jacobi.

### Fixed-matrix all-activity benchmark

`benchmark_bafu_all_activities.py` scores the first 100 BAFU activities as one-unit demands:

- technosphere and characterization matrices are loaded once;
- direct `lca.lci(..., factorize=True)` factorizes once;
- later direct demands reuse `lca.solver`;
- later `lcia(demand=...)` calls reuse the characterization matrix;
- Jacobi uses `rtol=1e-4` and `use_guess=True`.

Observed: direct scoring loop about 0.68 s; Jacobi about 7.80 s; maximum relative score
deviation about `1.58e-4`. With `use_guess=False`, Jacobi took about 8.20 s and the maximum
relative deviation improved to about `1.45e-5`, at higher RSS. Keep warm starts enabled in the
main presentation because they are faster.

This explains why direct UMFPACK can beat Jacobi inside Brightway even when Jacobi wins the
synthetic one-shot test: direct factorization is amortized over many right-hand sides, while
Jacobi performs a new GMRES solve for every demand.

### Monte Carlo benchmark

Committed main artifacts:

- `results/bafu_direct_500.json`
- `results/bafu_jacobi_500.json`
- `results/bafu_pair_summary_500.json`

The workload uses BAFU activity `bafu-219622`, IPCC 2021 GWP100, 500 samples, seed 2026,
`rtol=1e-4`, `use_guess=True`, and paired matrix fingerprints. Observed results:

- UMFPACK: about 38.1 s.
- Jacobi+GMRES: about 31.0 s.
- Maximum relative score deviation: about 0.163%.
- Median relative score deviation: about 0.043%.
- All 500 technosphere and biosphere fingerprints match.

Direct is slower here because every sampled technosphere requires a new factorization. Jacobi
avoids that direct factorization, though it rebuilds matrix-dependent solver state as needed.

## bw2calc matrix issue

Installed `bw2calc 2.5.0` lets `JacobiGMRESLCA._prepare_matrix()` replace
`self.technosphere_matrix` with a prepared CSC matrix. During Monte Carlo, the matrix manager is
resampled but that detached CSC object can remain stale. This affects stochastic Jacobi; direct
`LCA` does not take this preparation path, and deterministic Jacobi does not advance the manager.

PR #155 and commit `049f892` were intended to fix exactly this by keeping the prepared CSC matrix
private. The PR was closed without merging, and the fix is not in the 2.5.0 tag. This repository
uses a local subclass in `dev/benchmark_bafu.py` that rebinds
`self.technosphere_matrix = self.technosphere_mm.matrix` in `after_matrix_iteration` and
clears caches. Keep this safeguard until the installed upstream release demonstrably includes
the private prepared-matrix fix.

## Reproducibility rules

For every measurement record project, database, activity/order, method tuple and unit, matrix
shape/nnz/density, solver settings, wall time, per-solve time, RSS, residual, GMRES info/iterations,
score deviation, and Python/NumPy/SciPy/bw2calc/bw2data/scikit-umfpack versions.

Never silently overwrite a result with a different seed, tolerance, activity set, or solver.
Use descriptive filenames such as `*_rtol1e-3.json`. Pair runs by activity/iteration order and
assert technosphere and biosphere fingerprints before comparing scores.

Never commit proprietary ecoinvent data, credentials, private project names, or local paths.

### Validated environment

The live project was reproduced on macOS arm64 in a Conda environment named `jacobi-bw` with:

- Python 3.14.6;
- NumPy 2.5.1;
- SciPy 1.18.0;
- scikit-umfpack 0.4.2 and SuiteSparse 7.10.1;
- bw2calc 2.5.0, bw2data 4.7, and bw2io 0.9.17.

`pip check`, a direct `scikits.umfpack.spsolve`, and a BAFU LCIA all passed. The validated score
for one kilowatt hour of `bafu-219622` with the required IPCC method was approximately
`0.09707145849913332 kg CO2-Eq`. Treat this as an installation smoke test, not a replacement for
the committed benchmark results.

## Validation commands

From the repository root:

```bash
black --target-version py311 --fast --check dev
conda run -n jacobi-bw python -m compileall -q dev
find results -maxdepth 1 -name '*.json' -print0 | xargs -0 -n1 jq empty
git diff --check
```

Render using stored results without a Brightway live run:

```bash
BRIGHTCON_LIVE=0 \
  conda run -n jacobi-bw python -m jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 \
  output/jupyter-notebook/brightcon-2026-jacobi-gmres.ipynb
```

For live rehearsal on a faster VM:

```bash
BRIGHTCON_BW_PROJECT=<project-name> \
  conda run -n jacobi-bw python -m jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=360 \
  output/jupyter-notebook/brightcon-2026-jacobi-gmres.ipynb
```

The notebook labels sources as `LIVE`, `STORED RESULTS`, or `STORED FALLBACK (...)`. Never
present stored results as live.

## Reconstructing the live Brightway project

The repository is self-contained for synthetic benchmarks and stored-result notebook rendering,
but not for live BAFU calculations. The imported Brightway project and ecoinvent 3.10 biosphere
are licensed/external data and must be prepared on the target computer.

### Install dependencies

The tested setup uses Conda for the compiled numerical stack and PyPI for current Brightway
releases. Conda-forge only offered bw2calc 2.4.0 during the validated installation, so do not add
the Brightway packages to the `conda create` command if reproducing bw2calc 2.5.0 exactly:

```bash
conda create -n jacobi-bw -c conda-forge -y \
  python=3.14.6 numpy=2.5.1 scipy=1.18.0 scikit-umfpack=0.4.2 pip

conda run -n jacobi-bw python -m pip install \
  'bw2calc==2.5.0' 'bw2data==4.7' 'bw2io==0.9.17' \
  'ipykernel>=6.29,<8' 'jupyterlab>=4.2,<5' 'matplotlib>=3.9,<4' \
  'nbformat>=5.10,<6' 'openpyxl>=3.1,<4' 'pandas>=3.0.5,<4' 'psutil>=6,<8'

conda run -n jacobi-bw python -m pip check
```

The exact core constraints are also recorded in `pyproject.toml`. `scikit-umfpack` requires a
working SuiteSparse/UMFPACK installation; installing both through Conda avoids a local source
build. A machine using Pardiso may substitute that backend, but notebook labels and metadata must
be changed accordingly.

### Create/select a project

Prepare the project before starting a timed demonstration; the notebook does not import data live:

```python
import bw2data as bd

PROJECT = "brightcon-2026"
if PROJECT not in bd.projects:
    bd.projects.create_project(PROJECT)
bd.projects.set_current(PROJECT)
```

Any project name is acceptable if passed through `BRIGHTCON_BW_PROJECT`.

If a local project named `ecoinvent-3.10-cutoff` already contains the licensed cutoff database,
its matching biosphere, and LCIA methods, the tested and fastest reconstruction path is to copy it:

```python
import bw2data as bd

SOURCE = "ecoinvent-3.10-cutoff"
TARGET = "brightcon-2026"
if TARGET not in {project.name for project in bd.projects}:
    bd.projects.set_current(SOURCE)
    bd.projects.copy_project(TARGET, switch=True)
else:
    bd.projects.set_current(TARGET)

assert "ecoinvent-3.10-biosphere" in bd.databases
```

Copying preserves the licensed database and methods in Brightway's external data directory; it
does not place them in this repository. A new bw2data version can perform a one-time project
metadata/datapackage migration on first access. Verify the source environment can still open the
project after such a migration.

### Import the licensed ecoinvent 3.10 biosphere

Use the licensed ecospold2 download and keep its path outside the repository:

```python
import bw2data as bd
import bw2io as bi

bd.projects.set_current("brightcon-2026")
source = "/path/to/licensed/ecoinvent-3.10/datasets"
importer = bi.SingleOutputEcospold2Importer(source, "ecoinvent-3.10-cutoff")
importer.apply_strategies()
importer.statistics()
importer.write_database()
```

If a compatible ecoinvent 3.10 biosphere is already present, reuse it rather than importing a
duplicate. Record its actual database name; do not guess `biosphere3`.

### Import the BAFU workbook

`data/lci-bafu.xlsx` is in the generic `bw2io.ExcelImporter` format and declares the database name
`bafu`. Match its biosphere exchanges against the actual local ecoinvent 3.10 biosphere before
writing the database:

```python
import bw2data as bd
import bw2io as bi

bd.projects.set_current("brightcon-2026")
importer = bi.ExcelImporter("data/lci-bafu.xlsx")
importer.apply_strategies()

BIOSPHERE_DATABASE = "ecoinvent-3.10-biosphere"  # replace with the local name
# Link the workbook's technosphere and production exchanges to its own activities.
importer.match_database(
    fields=["name", "reference product", "unit", "location"]
)
# Link elementary flows to the external ecoinvent 3.10 biosphere.
importer.match_database(BIOSPHERE_DATABASE, fields=["name", "categories", "unit"])
importer.statistics()
unlinked = list(importer.unlinked)
if unlinked:
    raise ValueError(f"Refusing to import with {len(unlinked)} unlinked exchanges")
importer.write_database()
```

The internal match is required: matching only the biosphere leaves all BAFU technosphere and
production exchanges unlinked. Review any unmatched exchanges before `write_database()`; resolve
custom or renamed flows explicitly. The resulting database must be named `bafu` unless every
worker/notebook reference is changed consistently.

### Install the notebook-compatible LCIA method name

The ecoinvent 3.10 import stores the required method as a four-part tuple prefixed by
`"ecoinvent-3.10"`, while the notebook deliberately requests a three-part tuple. Register an
alias containing the same characterization factors:

```python
import bw2data as bd

bd.projects.set_current("brightcon-2026")
source = (
    "ecoinvent-3.10",
    "IPCC 2021",
    "climate change",
    "global warming potential (GWP100)",
)
target = ("IPCC 2021", "climate change", "global warming potential (GWP100)")
if target not in bd.methods:
    if source not in bd.methods:
        raise ValueError(f"Missing source LCIA method: {source}")
    method = bd.Method(target)
    method.register(**dict(bd.Method(source).metadata))
    method.write(bd.Method(source).load())
```

### Validate and run

Audit the workbook without importing it:

```bash
uv run python dev/audit_bafu_workbook.py data/lci-bafu.xlsx \
  --output results/bafu_workbook_audit.json
```

Before benchmarking, confirm project and database existence, unique activity `bafu-219622`, and:

```text
name: Electricity, low voltage, at grid
reference product: Electricity, low voltage, at grid
location: CH
unit: kilowatt hour
method: ("IPCC 2021", "climate change", "global warming potential (GWP100)")
```

The notebook performs this preflight. For a standalone live 500-sample run:

```bash
BRIGHTCON_BW_PROJECT=brightcon-2026 uv run python dev/benchmark_bafu.py \
  --solver direct --iterations 500 --output /tmp/bafu_direct_500.json

BRIGHTCON_BW_PROJECT=brightcon-2026 uv run python dev/benchmark_bafu.py \
  --solver jacobi-gmres --iterations 500 --rtol 1e-4 --use-guess \
  --output /tmp/bafu_jacobi_500.json

uv run python dev/compare_bafu_runs.py /tmp/bafu_direct_500.json \
  /tmp/bafu_jacobi_500.json --output /tmp/bafu_pair_summary_500.json
```

Never commit ecoinvent source files, Brightway project directories, credentials, or private paths.

## Future work and publication checklist

- Revisit the proposed two-dimensional synthetic size/density grid with memory guards.
- Run the full live notebook on the conference VM, not the slow development laptop.
- Keep the 500-sample MC configuration unless explicitly changed.
- Recheck PR #155/upstream bw2calc before removing the local workaround.
- Confirm the Brightcon date, time, venue, and programme entry.
- Distinguish abstract-reported values from locally reproduced measurements.
- Choose and add a repository license before public release.

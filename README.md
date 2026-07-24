# Jacobi + GMRES for large Brightway systems

Materials for Romain Sacchi's 8-minute Brightcon 2026 presentation, **“From 8 minutes to 4 seconds: solving large systems with Jacobi+GMRES.”** The talk is scheduled for 25 September 2026 in the *Open Tools and Development* track at Aalborg University and online.

The repository demonstrates [`JacobiGMRESLCA`](https://docs.brightway.dev/en/latest/content/api/bw2calc/jacobi_gmres_lca/index.html), introduced in `bw2calc` 2.4.0. It combines:

- two timing observations reported in the conference abstract;
- a small, reproducible SciPy example that checks convergence and numerical agreement;
- the equivalent Brightway 2.5 API call for use with a local Brightway project.

## Repository layout

```text
.
├── AGENTS.md
├── data/
│   ├── README.md
│   ├── lci-bafu.xlsx
│   └── reported_benchmarks.csv
├── dev/
│   ├── benchmark_bafu.py
│   ├── benchmark_synthetic.py
│   └── run_synthetic_suite.py
├── output/jupyter-notebook/
│   └── brightcon-2026-jacobi-gmres.ipynb
├── results/
└── pyproject.toml
```

## Clone and handoff status

The clone contains the source notebook builder, all benchmark workers, the public BAFU workbook,
machine-specific calibration JSON, and enough metadata for another Codex instance to continue.
Read `AGENTS.md` first; it is the detailed handoff document and records the benchmark regimes,
known bw2calc issue, current commits, and pending work.

There are two reproducibility levels:

1. **Fully self-contained:** synthetic matrix generation, solver comparisons, plotting, validation,
   notebook construction, and stored-result notebook rendering. No Brightway project or network is
   needed for this path.
2. **Live Brightway:** requires a local Brightway project containing the imported `bafu` database,
   the BAFU workbook linked to the ecoinvent 3.10 biosphere, and the IPCC 2021 GWP100 method. The
   ecoinvent biosphere and the imported project are intentionally not stored here. The repository
   cannot recreate that live project from public files alone.

The committed JSON results are reproducibility records and emergency fallbacks, not a substitute
for the external Brightway project. They include software/environment metadata and aggregate
scores/timings but do not contain restricted inventory exchanges.

## Run the notebook

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run jupyter lab output/jupyter-notebook/brightcon-2026-jacobi-gmres.ipynb
```

Or install the dependencies from `pyproject.toml` in an existing Python 3.11+ environment and launch Jupyter Lab.

The synthetic demonstration is self-contained. The Brightway section expects a project named `brightcon-2026`, a database named `bafu`, and the IPCC 2021 GWP100 method. For development, override the project without editing the notebook:

```bash
BRIGHTCON_BW_PROJECT=clic-bafu-2025-ef31 uv run jupyter lab output/jupyter-notebook/brightcon-2026-jacobi-gmres.ipynb
```

Set `BRIGHTCON_LIVE=0` to use the committed calibration results instead of launching live workers.

To rebuild the notebook after editing `dev/build_notebook.py`:

```bash
uv run python dev/build_notebook.py
```

To audit the included workbook without importing it into Brightway:

```bash
uv run python dev/audit_bafu_workbook.py data/lci-bafu.xlsx \
  --output results/bafu_workbook_audit.json
```

For a live run, first create/select the Brightway project externally, import `data/lci-bafu.xlsx`
with `bw2io.ExcelImporter`, link it to the locally available ecoinvent 3.10 biosphere, and ensure
the database is named `bafu` and the method is exactly:

```text
("IPCC 2021", "climate change", "global warming potential (GWP100)")
```

The notebook validates the project, database, selected activity (`bafu-219622`), functional unit,
and method before launching live workers. Set `BRIGHTCON_BW_PROJECT` to the local project name;
do not edit the notebook to insert a private project name.

## Data and reproducibility

`data/reported_benchmarks.csv` transcribes approximate timings from the [Brightcon contribution abstract](https://indico.d-d-s.ch/event/2/contributions/59/). These values are presentation inputs, not measurements reproduced by this repository. New benchmark results should record hardware, software versions, matrix provenance, solver settings, and residuals.

Do not commit proprietary inventory data. Keep ecoinvent and other restricted databases in the normal external Brightway data directory.

## Status

The synthetic and BAFU fallback workflows are implemented. The main BAFU Monte Carlo fallback is
500 paired samples at `rtol=1e-4` with `use_guess=True`; the fixed-matrix all-activity fallback
contains the first 100 activities. Before publication, rehearse live on the faster conference VM,
decide whether to implement the proposed guarded size/density grid, add final benchmark exports if
needed, and choose an explicit license.

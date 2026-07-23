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

## Data and reproducibility

`data/reported_benchmarks.csv` transcribes approximate timings from the [Brightcon contribution abstract](https://indico.d-d-s.ch/event/2/contributions/59/). These values are presentation inputs, not measurements reproduced by this repository. New benchmark results should record hardware, software versions, matrix provenance, solver settings, and residuals.

Do not commit proprietary inventory data. Keep ecoinvent and other restricted databases in the normal external Brightway data directory.

## Status

This is an initial, runnable presentation scaffold. Before publication, add the final benchmark exports and choose an explicit license.

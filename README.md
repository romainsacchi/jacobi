# Jacobi + GMRES for large Brightway systems

This repo contains the notebook for the Brightcon 2026 talk **“From 8 minutes to 4 seconds: solving large systems with Jacobi+GMRES.”** It compares sparse direct solvers with Jacobi-preconditioned GMRES, including the cases where direct factorization is still the better choice.

## What is here

- `output/jupyter-notebook/brightcon-2026-jacobi-gmres.ipynb` — the talk notebook.
- `dev/benchmark_synthetic.py` — one isolated solver run.
- `dev/run_synthetic_suite.py` — guarded benchmark grids.
- `dev/build_notebook.py` — rebuilds the notebook from source.
- `pyproject.toml` — the tested Python environment.

The notebook is self-contained: its benchmark workers are embedded when it is built. It creates synthetic matrices only; no Brightway project or inventory data is needed.

## Run it

Python 3.14 and SuiteSparse/UMFPACK are required. Conda is the easiest way to install the compiled solver stack:

```bash
conda create -n jacobi -c conda-forge -y \
  python=3.14.6 numpy=2.5.1 scipy=1.18.0 scikit-umfpack=0.4.2 pip
conda activate jacobi
python -m pip install \
  'jupyterlab>=4.2,<5' 'matplotlib>=3.9,<4' 'nbformat>=5.10,<6' \
  'pandas>=3.0.5,<4' 'psutil>=6,<8'
jupyter lab output/jupyter-notebook/brightcon-2026-jacobi-gmres.ipynb
```

The full notebook runs every benchmark live. Large cases are isolated, time-limited, and skipped when their estimated construction memory is unsafe.

After changing the notebook source, rebuild it with:

```bash
python dev/build_notebook.py
```

Quick checks:

```bash
black --target-version py311 --fast --check dev
python -m compileall -q dev
git diff --check
```

# AGENTS.md

Read this before changing the presentation.

## Goal

This repo supports an 8-minute Brightcon 2026 talk: **“From 8 minutes to 4 seconds: solving large systems with Jacobi+GMRES.”** The audience knows LCA and Python, but may not know sparse factorization or Krylov methods.

The story is deliberately balanced:

1. Direct factorization can become expensive on large, high-fill matrices.
2. Jacobi + GMRES can avoid that cost when its convergence is good enough.
3. Direct solvers often win when one factorization serves many demands.
4. Changing matrices, as in Monte Carlo, make repeated factorization expensive again.
5. Iterative results need convergence, residual, tolerance, and agreement checks.

## Files

- `dev/build_notebook.py` is the notebook source.
- `dev/benchmark_synthetic.py` runs one matrix/solver case.
- `dev/run_synthetic_suite.py` runs cases in isolated, guarded workers.
- `output/jupyter-notebook/brightcon-2026-jacobi-gmres.ipynb` is generated and committed.
- `pyproject.toml` records the tested environment.

The presentation uses synthetic matrices only. Do not add licensed inventories, Brightway projects, credentials, or machine-specific benchmark dumps.

## Benchmark defaults

- Seed: `2026`
- GMRES tolerance: `rtol=1e-4`
- Unsafe fixed-density matrices are skipped using an estimated construction-memory guard.
- Workers report `COMPLETED`, `SKIPPED`, `TIMEOUT`, or `FAILED` instead of hiding missing cases.

The main comparisons are one-shot scaling, size versus density, a low-fill banded counterexample, many demands on one fixed matrix, and 500 changing-matrix solves with and without warm starts.

## Keep it reproducible

Record matrix shape, nonzeros, solver settings, runtime, peak RSS, residual, convergence status, and package versions. Do not relabel stored or earlier measurements as a live run.

After changes:

```bash
black --target-version py311 --fast --check dev
python -m compileall -q dev
python dev/build_notebook.py
git diff --check
```

Run the notebook top-to-bottom on the conference machine before the talk; the larger live grids can be slow on a laptop.

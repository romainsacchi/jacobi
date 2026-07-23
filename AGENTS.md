# AGENTS.md

## Purpose

This repository contains the notebook and public data for an 8-minute Brightcon 2026 oral presentation: “From 8 minutes to 4 seconds: solving large systems with Jacobi+GMRES.” Keep the material concise, reproducible, and suitable for a live demonstration.

## Project map

- `output/jupyter-notebook/`: the presentation notebook.
- `data/reported_benchmarks.csv`: approximate values transcribed from the conference abstract.
- `data/README.md`: data provenance and schema notes.
- `dev/`: isolated workbook-audit and benchmark workers.
- `results/`: traceable calibration results and live-run fallbacks.
- `pyproject.toml`: the reproducible presentation environment.

## Working rules

- Preserve a top-to-bottom notebook narrative that fits in 8 minutes.
- Keep code cells small and deterministic; set random seeds explicitly.
- Run the notebook from a clean kernel before committing changes.
- Distinguish reported results from locally reproduced measurements in labels, captions, filenames, and prose.
- For new measurements, record matrix dimensions and density, solver settings, wall time, residuals, hardware, operating system, Python, SciPy, and `bw2calc` versions.
- Do not silently replace or round benchmark values. Update provenance with every data change.
- Avoid hidden dependence on a named Brightway project. Optional project-backed examples must validate the project, database, activity, functional unit, and LCIA method explicitly.
- Never commit proprietary ecoinvent or other restricted inventory data. Aggregated timings and non-sensitive metadata are acceptable.
- Use `bw2calc.JacobiGMRESLCA` for the Brightway iterative example and keep the minimum supported version at `bw2calc>=2.4.0` unless the notebook is deliberately updated for a newer API.
- Run each timed solver in an isolated subprocess. Record total and incremental peak RSS because SciPy, UMFPACK, and BLAS allocate outside Python's memory allocator.
- For paired Monte Carlo comparisons, assert matching technosphere and biosphere fingerprints for every sample before comparing scores.
- Keep generated notebook output compact. Prefer a small table, one chart, and short diagnostics over raw matrices or verbose logs.

## Validation

From the repository root:

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace output/jupyter-notebook/brightcon-2026-jacobi-gmres.ipynb
git diff --check
```

Before a live presentation, restart the kernel and rehearse once in the actual conference environment without network access.

## Publication checklist

- Confirm the final talk date, time, and venue against the conference programme.
- Replace provisional benchmark inputs only with traceable exports.
- Choose and add a repository license before making the repository public.
- Remove local paths, credentials, private project names, and large transient outputs.

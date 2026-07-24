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
2. Synthetic runtime/RSS/iteration/residual scaling.
3. Fixed-density stress cases and a runtime line graph.
4. Deterministic BAFU direct versus Jacobi.
5. 500 paired BAFU Monte Carlo samples.
6. First 100 BAFU activities on one fixed matrix.
7. Takeaways and presenter preflight.

The main Monte Carlo configuration is fixed at `rtol=1e-4`, `use_guess=True`, seed 2026,
and the local stochastic matrix-rebinding safeguard. The exploratory `rtol=1e-3` results remain
separate and must not replace the main setting.

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

## Validation commands

From the repository root:

```bash
black --target-version py311 --fast --check dev
/opt/homebrew/Caskroom/miniforge/base/envs/bw/bin/python -m compileall -q dev
find results -maxdepth 1 -name '*.json' -print0 | xargs -0 -n1 jq empty
git diff --check
```

Render using stored results without a Brightway live run:

```bash
BRIGHTCON_LIVE=0 \
  /opt/homebrew/Caskroom/miniforge/base/envs/bw/bin/python -m jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 \
  output/jupyter-notebook/brightcon-2026-jacobi-gmres.ipynb
```

For live rehearsal on a faster VM:

```bash
BRIGHTCON_BW_PROJECT=<project-name> \
  /opt/homebrew/Caskroom/miniforge/base/envs/bw/bin/python -m jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=360 \
  output/jupyter-notebook/brightcon-2026-jacobi-gmres.ipynb
```

The notebook labels sources as `LIVE`, `STORED RESULTS`, or `STORED FALLBACK (...)`. Never
present stored results as live.

## Future work and publication checklist

- Revisit the proposed two-dimensional synthetic size/density grid with memory guards.
- Run the full live notebook on the conference VM, not the slow development laptop.
- Keep the 500-sample MC configuration unless explicitly changed.
- Recheck PR #155/upstream bw2calc before removing the local workaround.
- Confirm the Brightcon date, time, venue, and programme entry.
- Distinguish abstract-reported values from locally reproduced measurements.
- Choose and add a repository license before public release.

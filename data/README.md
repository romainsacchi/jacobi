# Data

## `reported_benchmarks.csv`

Two approximate benchmark observations transcribed from the Brightcon 2026 contribution abstract. They support the presentation narrative; they are not reproduced measurements from this repository.

Columns:

- `benchmark_id`: stable row identifier.
- `system`: short description of the matrix.
- `dimension_rows`, `dimension_columns`: matrix shape.
- `density`: reported non-zero fraction when stated; blank when not stated.
- `direct_seconds`: approximate direct-solver wall time.
- `jacobi_gmres_seconds`: approximate Jacobi+GMRES wall time.
- `source`: provenance URL.
- `status`: distinguishes reported inputs from reproduced results.
- `notes`: qualification needed for interpretation.

The abstract does not specify hardware or full solver settings. Do not present these values as a controlled benchmark comparison without adding that metadata.

Future reproduced benchmark data should be stored in a separate CSV with one row per run and include machine, operating system, package versions, matrix provenance, tolerances, restart/max-iteration settings, residuals, and peak memory.

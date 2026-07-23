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

## `lci-bafu.xlsx`

Public BAFU inventory workbook in the generic `bw2io.ExcelImporter` block format. Its internal database name is `bafu`. It contains 11,747 activities and uncertainty data suitable for stochastic calculations. The presentation assumes this workbook has already been imported and linked to the ecoinvent 3.10 biosphere before the timed demonstration.

The selected functional unit is exactly:

- name: `Electricity, low voltage, at grid`
- reference product: `Electricity, low voltage, at grid`
- location: `CH`
- unit: `kilowatt hour`
- code: `bafu-219622`

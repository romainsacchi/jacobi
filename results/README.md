# Benchmark results

These files are reproducibility records and emergency presentation fallbacks.

- `bafu_workbook_audit.json`: aggregate workbook structure and uncertainty counts.
- `synthetic_calibration.json`: uncapped constant-degree solver scaling on the development machine at `rtol=1e-4`.
- `synthetic_density_calibration.json`: uncapped 5,000–20,000-row, 0.1%-density stress sweep at `rtol=1e-4`.
- `synthetic_dense_large_calibration.json`: uncapped 20,000-row, 0.2%-density stress test at `rtol=1e-4`.
- `bafu_direct_200.json` and `bafu_jacobi_200.json`: 200 paired stochastic runs using seed 2026; Jacobi+GMRES uses `rtol=1e-4` and `use_guess=True`.
- `bafu_pair_summary_200.json`: fingerprint validation, performance, memory, residual, and score-agreement summary.
- `bafu_jacobi_200_rtol1e-3.json` and `bafu_pair_summary_200_rtol1e-3.json`: exploratory 200-sample comparison at `rtol=1e-3`; retained separately because score differences are larger.

Results are machine-specific. The notebook labels committed values as stored results whenever live workers are disabled or fail. Full BAFU inventory exchanges are not exported here.

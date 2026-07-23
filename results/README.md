# Benchmark results

These files are reproducibility records and emergency presentation fallbacks.

- `bafu_workbook_audit.json`: aggregate workbook structure and uncertainty counts.
- `synthetic_calibration.json`: constant-degree solver scaling on the development machine.
- `synthetic_density_calibration.json`: 15,000-row, 0.1%-density stress test.
- `synthetic_dense_large_calibration.json`: 20,000-row, 0.2%-density stress test.
- `bafu_direct_200.json` and `bafu_jacobi_200.json`: 200 paired stochastic runs using seed 2026.
- `bafu_pair_summary_200.json`: fingerprint validation, performance, memory, residual, and score-agreement summary.

Results are machine-specific. The notebook labels committed values as stored results whenever live workers are disabled or fail. Full BAFU inventory exchanges are not exported here.

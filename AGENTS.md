# AGENTS.md — TrustKAN-Climate

## Mission
Build a reproducible research codebase supporting a strong journal paper on trustworthy KAN-based climate forecasting.

## Non-negotiable research rules
1. Never fabricate experimental results.
2. Fit preprocessing/scalers on training data only.
3. Preserve chronological train/validation/test splits; no random time-series leakage.
4. Use identical splits and forecast horizons for all comparable models.
5. Main neural-model comparisons should use >=5 seeds unless explicitly marked preliminary.
6. Save raw predictions and per-seed metrics before aggregation.
7. Report mean ± standard deviation and statistical comparisons where appropriate.
8. Separate calibration data from final test data for conformal prediction.
9. Treat uncertainty quality, calibration, robustness, and selective risk as first-class outcomes—not decorative additions.
10. Every manuscript claim must map to a reproducible experiment/artifact.

## Engineering rules
- Python + PyTorch preferred.
- Configuration-driven experiments.
- Device-agnostic CPU/CUDA code.
- Deterministic seeds where possible.
- Modular datasets, models, metrics, UQ, drift, and interpretability components.
- Unit tests should prioritize leakage, tensor shapes, metrics, and temporal splits.

## Paper strategy
The central contribution should not be merely an incremental KAN architecture with lower RMSE. The intended contribution is an integrated trustworthy forecasting framework combining intrinsic KAN interpretability, calibrated uncertainty, distribution-shift awareness, explanation stability, and selective prediction.

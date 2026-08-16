# TrustKAN-Climate

**TrustKAN-Climate: Interpretable and Reliability-Aware Kolmogorov–Arnold Networks for Trustworthy Climate Forecasting under Distribution Shift**

This repository is a research-grade project for developing, evaluating, and documenting a KAN-based trustworthy climate forecasting framework suitable for a top-tier journal submission.

## Scientific goal

Move beyond point-forecast accuracy by jointly studying:

- interpretable-by-design temporal KAN forecasting,
- calibrated predictive uncertainty,
- temporal drift and out-of-distribution detection,
- explanation stability,
- selective forecasting / abstention,
- robustness and computational efficiency.

## Proposed evidence chain

Climate data → Temporal KAN → intrinsic interpretability → conformal uncertainty → drift/OOD awareness → reliability score → selective forecasting → trustworthy decision support.

## Phase 1 status

The repository now includes:

- leakage-safe chronological train/validation/calibration/test utilities,
- training-only normalization,
- multi-horizon CET benchmark configuration,
- Persistence, MLP, LSTM, GRU, Transformer and plain KAN baselines,
- initial TrustKAN temporal model,
- conformal interval utilities,
- selective-risk metrics,
- reproducibility/unit tests,
- publication experiment and claim-evidence plans.

The benchmark reports final MAE/RMSE in the original temperature units after inverse scaling.

## Quick start

```bash
pip install -r requirements.txt
pytest -q
python scripts/run_cet_benchmark.py --config configs/cet.yaml
```

The CET runner caches the existing public CET CSV from the earlier research repository into `data/raw/`. Raw data and generated experimental outputs are excluded from git by default.

## Planned datasets

- Central England Temperature (CET)
- additional public climate/weather datasets with documented provenance and frozen evaluation protocols

## Planned baselines

Persistence, ARIMA/SARIMA, SVR, XGBoost, MLP, LSTM, GRU, TCN, Transformer, Mamba, standard KAN, Tem2-KAN, and TrustKAN.

## Project structure

- `docs/` research plan, experiment matrix, claim-evidence matrix and Codex task roadmap
- `configs/` dataset/model configurations
- `src/data/` temporal split and preprocessing utilities
- `src/models/` baseline KAN, modern baselines and TrustKAN
- `src/uncertainty/` conformal calibration
- `src/metrics/` forecasting and selective-risk metrics
- `src/training/` reproducible PyTorch engine
- `scripts/` experiment entry points
- `tests/` leakage and model/unit tests
- `results/` machine-readable experiment outputs

## Reproducibility principles

All reported values must be generated from saved experiment outputs. No manually invented numbers, hidden failed runs, or test-set leakage are allowed. Main comparisons should use multiple random seeds and fair tuning effort. CET-only evidence must not be generalized to broad climate forecasting claims until additional datasets have been evaluated.

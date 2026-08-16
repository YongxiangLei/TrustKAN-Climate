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

## Planned datasets

- Central England Temperature (CET)
- additional public climate/weather datasets to be added with documented provenance

## Planned baselines

Persistence, ARIMA, SVR, XGBoost, MLP, LSTM, GRU, TCN, Transformer, Mamba, standard KAN, Tem2-KAN, and TrustKAN.

## Project structure

- `docs/` research plan, experiments, claims, reviewer checklist
- `configs/` dataset/model/ablation configurations
- `src/` reusable research code
- `scripts/` experiment entry points
- `tests/` leakage and reproducibility checks
- `results/` machine-readable outputs
- `paper/` LaTeX manuscript assets

## Reproducibility principles

All reported values must be generated from saved experiment outputs. No manually invented numbers, hidden failed runs, or test-set leakage are allowed. Main comparisons should use multiple random seeds and fair tuning effort.

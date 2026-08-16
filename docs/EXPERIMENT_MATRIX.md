# Experiment Matrix

| ID | Question | Models / variants | Main outputs |
|---|---|---|---|
| E01 | Does TrustKAN improve deterministic forecasting? | Persistence, ARIMA, SVR/XGBoost, LSTM, GRU, TCN, Transformer, Mamba, KAN, Tem2-KAN, TrustKAN | MAE/RMSE per horizon, mean±std |
| E02 | Is improvement consistent across horizons? | same | horizon-wise curves |
| E03 | Is improvement consistent across datasets? | same | dataset × model table |
| E04 | What does each TrustKAN module contribute? | full model and component removals | ablation table |
| E05 | Are intervals calibrated? | quantile/deep ensemble/conformal variants | coverage, width, interval score |
| E06 | Does adaptive calibration help under temporal shift? | static vs adaptive conformal | rolling coverage plots |
| E07 | Can the model identify distribution shift? | representation/residual/explanation scores | AUROC/AUPRC when labels are definable; temporal diagnostics otherwise |
| E08 | Does abstention improve reliability? | confidence/reliability variants | risk–coverage and AURC |
| E09 | Is performance robust to missing inputs? | all key models | error vs missing-rate |
| E10 | Is performance robust to noise? | all key models | error vs noise level |
| E11 | Are explanations stable? | KAN/Tem2-KAN/TrustKAN | lag/function stability statistics |
| E12 | Does reliability correlate with actual error? | TrustKAN | reliability-error correlation/calibration |
| E13 | What happens on extremes? | key baselines + TrustKAN | tail/extreme-event subset metrics |
| E14 | What is computational cost? | neural baselines | params, latency, training time, memory |
| E15 | Are gains statistically credible? | main models | forecast-origin moving-block bootstrap, paired effect sizes, corrected sensitivity tests |

## Main split policy
Use chronological splits. Hyperparameter selection uses training/validation only. Final test is untouched until model selection is frozen. Conformal calibration uses a dedicated calibration segment or rigorously defined rolling protocol.

## Seeds
Main neural results: minimum 5 seeds. Prefer 10 for headline comparisons if computationally feasible.

## Reporting
Save one row per dataset/model/horizon/seed. Aggregate only downstream. Retain raw predictions for paired statistical analysis and reproducibility.

Primary pairwise inference follows `docs/STATISTICAL_PROTOCOL.md`. Overlapping
forecast origins are handled with moving-block rather than IID bootstrap.

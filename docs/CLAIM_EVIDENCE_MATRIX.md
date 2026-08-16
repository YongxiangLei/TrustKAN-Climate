# Claim–Evidence Matrix

This file prevents manuscript claims from outrunning experimental evidence.

| Candidate claim | Required evidence | Status |
|---|---|---|
| TrustKAN improves deterministic climate forecasting | Multi-dataset, multi-horizon comparison against strong baselines; >=5 seeds | Planned |
| Temporal KAN component adds value beyond a plain KAN | Controlled KAN vs TrustKAN ablation under identical budgets | Planned |
| TrustKAN provides calibrated uncertainty | Calibration/test separation; marginal and simultaneous coverage, width and interval score vs UQ baselines | Pipeline validated; full evidence pending |
| Adaptive calibration is robust to temporal shift | Rolling coverage and shift experiments vs static conformal | Planned |
| KAN explanations are stable | Across-seed and perturbation stability metrics, not only visual examples | Planned |
| Representation changes help reveal climate shift | Quantitative drift protocol or carefully defined temporal diagnostics | Planned |
| Reliability score predicts forecast failure | Error-vs-reliability association and top-error AUROC/AUPRC against width-only and shift-only components | Pipeline validated; full evidence pending |
| Selective forecasting reduces risk | Origin-wise risk–coverage curves, AURC, retained-set error and width-only/shift-only ablations | Pipeline validated; full evidence pending |
| Method is robust to corrupted inputs | Noise, random missingness and block missingness experiments | Planned |
| Method is computationally practical | Parameter count, memory, training time and CPU/GPU inference latency | Planned |
| Method is useful on climate extremes | Pre-defined extreme subsets and event-tail metrics with uncertainty coverage | Planned |

## Rule
Do not mark a row Supported until raw experiment outputs, configuration, seeds and an analysis artifact all exist. A supported claim must link to its generated result files before inclusion in the abstract or conclusion.

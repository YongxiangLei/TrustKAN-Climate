# Research Plan

## Working title
TrustKAN-Climate: Interpretable and Reliability-Aware Kolmogorov–Arnold Networks for Trustworthy Climate Forecasting under Distribution Shift

## Research question
Can a temporal KAN forecasting system provide competitive predictive skill while producing calibrated uncertainty, stable intrinsic explanations, and reliable warnings/abstention under climate distribution shift?

## Hypotheses
H1. Temporal KAN representations improve or remain competitive with strong sequence baselines on multi-horizon climate forecasting.

H2. Calibration-aware uncertainty estimation yields prediction intervals with better empirical coverage/width trade-offs than uncalibrated uncertainty approaches.

H3. KAN functional representations provide measurable explanation stability and can reveal changes associated with temporal distribution shift.

H4. Reliability-aware selective forecasting reduces error/risk on accepted predictions under shift and corrupted inputs.

## Proposed TrustKAN components
1. Temporal encoder for multiscale lag structure.
2. KAN nonlinear functional mapping for interpretable feature interactions.
3. Point/quantile forecasting heads.
4. Split/adaptive conformal calibration layer.
5. Drift/OOD score based on representation and/or residual shift.
6. Explanation-stability score from KAN functional changes.
7. Reliability fusion and selective prediction rule.

## Evaluation dimensions
- Accuracy: MAE, RMSE, MAPE where scientifically valid, R2/correlation where useful.
- Probabilistic: coverage, MPIW, Winkler/interval score, CRPS where distributional outputs permit.
- Reliability: calibration error and coverage deviation.
- Selective prediction: risk–coverage curve, AURC, retained-sample RMSE.
- Robustness: missingness, noise, temporal shift, extreme subsets.
- Interpretability: feature/lag attribution consistency and functional stability.
- Efficiency: parameters, FLOPs where measurable, training/inference time, memory.

## Publication-grade requirements
- Multiple datasets rather than CET-only evidence.
- Strong modern baselines and original KAN variants.
- Multi-horizon evaluation.
- At least five seeds for main neural comparisons.
- Statistical significance/effect-size analysis.
- Ablations isolating every claimed module.
- Failure cases and limitations.
- Fully reproducible data splits and configurations.

## Milestones
M1: infrastructure + CET benchmark.
M2: baseline suite + Temporal KAN.
M3: uncertainty/conformal calibration.
M4: drift/OOD + selective prediction.
M5: interpretability and explanation stability.
M6: multi-dataset experiments + ablations.
M7: statistical analysis, figures and tables.
M8: manuscript completion and reviewer-style audit.

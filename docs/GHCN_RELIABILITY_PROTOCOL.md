# GHCN uncertainty and reliability protocol

## Scope

This protocol evaluates TrustKAN uncertainty and selective forecasting on the
same frozen within-station and leave-one-region-out tasks as
`docs/GHCN_BENCHMARK_PROTOCOL.md`. Model fitting and early stopping use only
training and validation windows. The calibration split is never used to update
model parameters, and the final test split is never used to choose conformal
radii, fusion weights, or abstention thresholds.

In leave-one-region-out experiments, model parameters are learned from the four
source regions. Uncertainty is then calibrated on the held-out target station's
chronologically earlier calibration segment. The task is therefore called
**source-only parameter transfer with target calibration**, not fully
calibration-free transfer.

## Predictive intervals

TrustKAN emits 0.05, 0.50, and 0.95 quantiles. For nominal 90% coverage, two
split-conformal constructions are reported:

1. **Horizonwise marginal calibration:** one finite-sample radius is estimated
   per forecast lead from target-calibration origins. Report marginal coverage,
   coverage at every lead, joint trajectory coverage, mean width, and Winkler
   interval score.
2. **Simultaneous trajectory calibration:** the conformity score is the maximum
   miss over all leads at an origin. A single finite-sample radius targets joint
   coverage of the whole forecast trajectory. Report joint and marginal
   coverage, width, and interval score.

Forecast leads are never flattened into pseudo-independent calibration
samples. The calibration and test timestamp arrays are stored, and strict
aggregation rejects any overlap.

## Reliability score

The pre-registered reliability components are:

- interval-width reliability: the survival percentile of mean conformal width
  relative to target-calibration widths;
- embedding-shift reliability: a Mahalanobis score relative to deterministic
  training-embedding samples, mapped through the target-calibration shift-score
  distribution.

The fused score is their equal-weight geometric mean. The 0.5/0.5 weights are
fixed in configuration, not fitted to test errors. Reference embeddings are a
chronologically uniform deterministic subset of at most 4,096 training
windows.

## Selective forecasting and diagnostics

For fused, width-only, and shift-only reliability, an abstention threshold is
chosen independently on calibration data by minimizing retained-set RMSE while
requiring at least 50% calibration coverage. Test risk--coverage curves retain
whole forecast origins, including all leads, and AURC is computed without
flattening horizons.

Primary selective evidence compares fused AURC and calibration-selected test
risk against the width-only and shift-only ablations. Test-set Spearman
association and top-error AUROC/AUPRC are descriptive diagnostics only; they do
not alter a threshold or model.

## Reporting and gates

Report each region separately and use equal-region macro averages. Formal
aggregation requires five seeds and five target regions for every
protocol/horizon. Every artifact contains predictions, intervals, calibration
state, reliability components, risk curves, timestamps, split origins, raw
data/config/code hashes, and its own ledger checksum.

`configs/ghcn_reliability_smoke.yaml` uses two regions, one horizon, one epoch,
and one seed for software validation only. Its values are prohibited from
manuscript result tables. `configs/ghcn_reliability.yaml` is the frozen full
protocol.

Adaptive/rolling conformal inference is not part of this static experiment. It
must use a separate causal, origin-wise sequential protocol before supporting
claims about adaptation under drift.

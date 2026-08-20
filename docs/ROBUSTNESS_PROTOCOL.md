# Robustness and extreme-subset protocol

These rules are frozen before test rankings are inspected. They apply to CET,
GHCN and Jena once each dataset has raw prediction artifacts. Corruption and
extreme labels never enter model selection, conformal radii or abstention
thresholds.

## Input corruptions

Corruptions are applied to **test histories only**, after train-only
standardization. Targets are never altered. In standardized space the training
mean is 0, so missing inputs are filled with 0.

The pre-registered grid in `configs/robustness.yaml` is:

| Kind | Daily levels | Hourly levels |
|---|---|---|
| Clean reference | — | — |
| Gaussian sensor noise | 0.05, 0.10, 0.20 train-std | same |
| Random timestep dropout | 10%, 20%, 40% of history | same |
| Recent block dropout | 1, 3, 7 days | 1, 6, 24 hours |

Random dropout removes entire timestamps (all features at that lag), not
isolated scalar entries. Block dropout always removes the most recent
contiguous history, which is the operational missing-sensor case.

A single robustness seed (`11`) is used to generate the corruption masks. Do
not retune the grid after seeing which model degrades least.

## Extreme subsets

Cold and warm tails are defined from **training-period target quantiles**
(5% and 95%). Validation, calibration and test targets are forbidden for
threshold estimation.

The primary label uses the first forecast lead: an origin is cold if
`y[t+1] < q_train(0.05)` and warm if `y[t+1] > q_train(0.95)`. An any-lead
diagnostic may be reported but is not the headline definition.

Report subset size, fraction, MAE and RMSE versus the complement. If a subset
has fewer than 30 origins, mark it underpowered and do not claim extreme-event
skill. Interval coverage and width on the same first-lead subsets are allowed
only when conformal radii were already frozen on calibration data.

## What may not be claimed

- Robustness from a single corruption level chosen after looking at the
  ranking.
- Extreme-event forecasting from an unthresholded tail of point errors.
- That a model “detects extremes” unless a pre-registered detection metric is
  computed on the same training-quantile labels.

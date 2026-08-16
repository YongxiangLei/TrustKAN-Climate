# GHCN multi-region benchmark protocol

## Frozen panel and target

The benchmark uses the five stations in
`configs/datasets/ghcn_frozen.yaml`. The target is quality-controlled daily
midrange temperature, `(TMAX + TMIN) / 2`, in degrees Celsius from 1950-01-01
through 2024-12-31. Station selection is independent of model performance.

Prepared arrays must be produced by `scripts/prepare_ghcn_panel.py`. The
benchmark refuses artifacts whose station identity, frozen-config hash,
preparation-code hash, raw archive hash, or target identity differs from the
tracked panel.

## Temporal construction

Each station is split chronologically by observation index into 60% training,
15% validation, 10% calibration, and 15% final test. Forecast targets may not
cross a split boundary. Standardization parameters are fitted separately for
each station using only its training period. History/target windows must be
strictly daily-contiguous; no interpolation or bidirectional filling is used.

The full experiment uses 30 history days and horizons of 1, 7, and 30 days.
This shorter context than CET is pre-registered because even eligible GHCN
stations contain occasional missing days. A feasibility audit confirmed
non-empty training, validation, calibration, and test windows for every frozen
station at all three horizons.

Run `python scripts/audit_ghcn_windows.py --config configs/ghcn.yaml` to
reproduce the checksum-bound window counts without fitting a model.

## Evaluation tasks

### Within-station temporal generalization

A separate model is trained on each station's training windows, selected on
that station's validation windows, and evaluated on its final test period. This
tests temporal generalization while preserving geographic identity.

### Leave-one-region-out parameter transfer

For each target region, standardized training windows from the other four
stations are pooled. Model fitting and early stopping use source-region
training and validation windows only. The held-out station contributes no model
training or validation samples. Its training-period observations are used only
to estimate its own normalization transform; final predictions are inverted
with that transform. Test windows may use immediately preceding held-out
history because those observations are available at forecast time.

Source regions contribute equally: for each split, every source is reduced to
the smallest available source-window count by deterministic, chronologically
evenly spaced sampling before pooling. This prevents a more complete station
from dominating model fitting solely because it supplies more windows.

This task is therefore described as zero-shot **parameter transfer with
target-history normalization**, not as a claim that no target observations are
ever seen. Persistence is evaluated on the same held-out targets as a local
information baseline but has no fitted parameters.

## Model selection and reporting

Classical hyperparameters and neural early stopping are selected on validation
data only. The calibration split remains untouched for later uncertainty and
reliability experiments. Final neural runs use five fixed seeds. Failed runs
remain in the ledger.

Report per-region MAE/RMSE and equal-region macro averages. Do not pool all
forecast origins across stations as if they were IID. Paired within-region
model comparisons use the moving-block protocol in
`docs/STATISTICAL_PROTOCOL.md`. Across-region macro values are descriptive
unless a separately pre-registered hierarchical analysis is used.
Publication aggregation must use both `--min-seeds 5` and `--min-regions 5`;
the command fails if a model/protocol/horizon group has an incomplete panel.

`configs/ghcn_smoke.yaml` uses two regions, one seed, one epoch, a shorter
history, and a recent tail subset. It validates software only and is prohibited
from manuscript result tables. `configs/ghcn.yaml` is the full frozen protocol.

# Skeptical reviewer audit

Date: 2026-08-16

## Provisional recommendation

**Reject in the current evidence state; encourage resubmission after a complete,
pre-specified multi-dataset study.** The repository contains a promising
trustworthy-forecasting agenda and increasingly strong reproducibility controls,
but it does not yet contain evidence supporting its central scientific claims.

This is an internal adversarial review, not an assessment of unpublished full
results that may exist elsewhere.

## Major concerns

### 1. CET alone cannot support broad climate-forecasting claims

CET is a valuable long-record temperature benchmark, but a single univariate
station series cannot establish generality across variables, spatial regimes,
sampling frequencies, geography or reanalysis products. At minimum, the main
paper needs pre-specified experiments on the repository's proposed GHCN, Jena
and ERA5 settings, with the same leakage-safe protocol and no post-hoc dataset
removal.

### 2. The central novelty is not yet isolated experimentally

The manuscript strategy combines a temporal KAN, uncertainty calibration,
shift awareness, explanation stability and selective forecasting. Without
controlled component ablations, reviewers cannot tell whether gains arise from
the KAN component, parameter count, training objective, conformal wrapper or
reliability fusion. Every module needs an identical-budget removal or replacement
experiment.

### 3. Baseline coverage and tuning fairness remain inadequate

The formal CET configuration now includes validation-selected SVR and random
forest candidates plus TCN, but classical time-series baselines and additional
competitive modern forecasters remain absent from completed results. More
important than model count is a documented and comparable validation-only
tuning budget. The new classical selection traces are a good start; equivalent
neural-model tuning budgets and full executions are still missing.

### 4. Tem2-KAN fidelity needs experimental completion

The repository now pins the source commit and dependency, enforces the verified
300→20 architecture, and documents all deliberate leakage-safe protocol
differences. However, the dedicated five-seed experiment has not yet been run or
compared with source behavior. It should remain a reference implementation and
should not anchor a novelty claim until those artifacts exist.

### 5. The trustworthy-AI claims are still planned rather than demonstrated

The claim–evidence matrix correctly marks calibration, adaptive shift handling,
explanation stability, failure prediction and selective risk as planned. These
cannot be presented as contributions in the abstract until raw multi-seed
artifacts and analyses exist. Point-forecast RMSE alone would not validate the
paper's stated contribution.

### 6. Distribution shift needs an operational definition

Chronological train/test separation is not automatically a labeled OOD study.
The paper must define shift sources and evaluation units in advance: temporal
regime changes, geography transfer, sensor corruption, variable shift, or
synthetic interventions. Detection metrics require defensible positive/negative
labels; otherwise the analysis should be framed as temporal diagnostics rather
than OOD detection accuracy.

### 7. Climate extremes require pre-defined thresholds

Extreme-event evaluation is vulnerable to post-hoc threshold selection. Define
extreme subsets from domain conventions or training-period quantiles before
viewing test performance. Report sample counts, tail errors, interval coverage
and width, and uncertainty in estimates. Avoid claiming event forecasting from
a point-temperature tail analysis alone.

### 8. Dependence-aware inference needs formal execution

The code now uses forecast-origin circular moving-block bootstrap intervals,
keeps multi-horizon elements together, verifies paired timestamps and exposes
Holm/BH correction utilities. The protocol correctly demotes Wilcoxon to a
sensitivity analysis. Remaining work is to pre-specify comparison families and
block-length sensitivity checks, then execute the protocol on full multi-seed
artifacts rather than smoke outputs.

### 9. Compute reporting is not yet publication-grade

The runner now records parameter count, training time and synchronized
per-sample inference latency, which closes an important schema gap. Formal
reporting still needs hardware details, warm-up/repetition policy, memory usage,
consistent batch sizes and separation of hyperparameter-search cost from final
fit cost.

### 10. Missing-data handling needs quantitative reporting

The CET audit found 14 non-daily intervals, including one 235-day gap. The
pipeline now excludes every window whose history or target crosses an irregular
interval, preserving day-based horizon semantics. The paper must still report
gap counts, excluded-window counts and whether this filtering changes the
evaluation-period composition.

## Evidence already in good shape

- Training-only standardization and chronological four-way splitting.
- Complete-target boundary isolation for multi-step windows.
- Calendar-continuity filtering for day-based windows.
- Separate calibration and final-test regions at the data-structure level.
- Immutable per-run ledger design with raw prediction paths and configuration
  fingerprints.
- Raw artifact validation before aggregation.
- Explicit smoke/full isolation and a five-seed publication gate.
- A claim–evidence matrix that currently avoids marking planned claims as
  supported.

## Minimum evidence package before result writing

1. Apply and report calendar-gap semantics for every dataset and horizon.
2. Freeze dataset inclusion, horizons, primary metrics and tuning budgets.
3. Verify Tem2-KAN fidelity and distinctness from TrustKAN.
4. Run all primary neural comparisons with at least five seeds.
5. Add validation-tuned classical and modern baselines under identical splits.
6. Produce all raw predictions before aggregation; retain failed-run records.
7. Use dependence-aware paired statistical comparisons with effect sizes.
8. Complete calibrated uncertainty, drift, selective-risk and explanation-
   stability experiments on more than one dataset.
9. Complete pre-defined robustness and extreme-subset analyses.
10. Update the claim–evidence matrix with direct artifact links only after every
    required item exists.

## Immediate next action

Run a resource-estimation pilot for one seed per formal model/horizon, then use
the measured runtime and memory to schedule the five-seed sweep. Do not launch
the full sweep until the formal config and baseline tuning budgets are frozen.
